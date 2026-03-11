from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .analysis import (
    analyze_orderbook,
    analyze_trend,
    analyze_volatility,
    build_candles,
    derive_indicators,
    MomentumIndicators,
    support_resistance,
)
from .config import BotConfig
from .indodax_client import IndodaxClient
from .persistence import StatePersistence
from .strategies import StrategyDecision, make_trade_decision
from .tracking import PortfolioTracker

logger = logging.getLogger(__name__)


class Trader:
    def __init__(self, config: BotConfig, client: Optional[IndodaxClient] = None) -> None:
        self.config = config
        self.client = client or IndodaxClient(config.api_key)
        self._all_pairs: Optional[List[str]] = None
        self.tracker = PortfolioTracker(
            initial_capital=config.initial_capital,
            target_profit_pct=config.target_profit_pct,
            max_loss_pct=config.max_loss_pct,
        )
        self.persistence = StatePersistence(Path(config.state_file))
        self._restored_state: Optional[Dict[str, Any]] = None
        if config.auto_resume:
            restored = self.persistence.load()
            if restored and isinstance(restored, dict):
                portfolio_state = restored.get("portfolio")
                if isinstance(portfolio_state, dict):
                    self.tracker.load_state(portfolio_state)
                restored_pair = restored.get("pair")
                if restored_pair:
                    self.config.pair = str(restored_pair)
                self._restored_state = restored

    @property
    def restored_state(self) -> Optional[Dict[str, Any]]:
        return self._restored_state

    def _extract_price(self, ticker: Dict[str, Any]) -> float:
        if "ticker" in ticker:
            last = ticker["ticker"].get("last")
            if last is None:
                last = ticker["ticker"].get("last_price")
            if last is None:
                raise ValueError("Ticker missing price fields")
            return float(last)
        last_value = ticker.get("last")
        if last_value is None:
            last_value = ticker.get("last_price")
        if last_value is None:
            raise ValueError("Ticker missing price fields")
        return float(last_value)

    def _decision_to_dict(self, decision: StrategyDecision) -> Dict[str, Any]:
        return {
            "mode": decision.mode,
            "action": decision.action,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "target_price": decision.target_price,
            "amount": decision.amount,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "support": decision.support,
            "resistance": decision.resistance,
        }

    def _persist_state(
        self,
        snapshot: Dict[str, Any],
        decision: StrategyDecision,
        reference_price: float,
        outcome: Dict[str, Any],
    ) -> None:
        if not self.config.auto_resume:
            return
        state = {
            "pair": snapshot.get("pair"),
            "price": reference_price,
            "decision": self._decision_to_dict(decision),
            "portfolio": self.tracker.to_state(),
            "outcome": outcome.get("status"),
            "reason": outcome.get("reason"),
            "timestamp": time.time(),
        }
        self.persistence.save(state)

    def _score_snapshot(self, snapshot: Dict[str, Any]) -> float:
        decision: StrategyDecision = snapshot["decision"]
        vol = snapshot.get("volatility")
        orderbook = snapshot.get("orderbook")
        score = decision.confidence
        if vol:
            score += min(vol.volatility * 5, 0.1)
        if orderbook:
            spread_bonus = max(0.0, 0.001 - orderbook.spread_pct) * 50
            score += spread_bonus
        return max(0.0, min(1.5, score))

    def _staged_amounts(self, decision: StrategyDecision, snapshot: Dict[str, Any]) -> List[float]:
        total = decision.amount
        if total <= 0:
            return []
        vol = snapshot.get("volatility")
        volatility = getattr(vol, "volatility", 0.0) if vol else 0.0
        confidence = decision.confidence
        max_steps = max(1, self.config.staged_entry_steps)

        if volatility < 0.01 and confidence >= 0.75:
            fractions = [1.0]
        elif volatility < 0.02:
            fractions = [0.6, 0.4]
        else:
            fractions = [0.5, 0.3, 0.2]
        fractions = fractions[:max_steps]

        # Normalize to ensure full allocation within total
        total_frac = sum(fractions)
        fractions = [f / total_frac for f in fractions] if total_frac else [1.0]

        staged = [round(total * frac, 12) for frac in fractions]
        # Ensure rounding does not exceed total and the final leg absorbs rounding drift
        if staged:
            staged[-1] = max(0.0, total - sum(staged[:-1]))
        return [s for s in staged if s > 0]

    def _scale_staged_amounts(self, decision_amount: float, effective_amount: float, staged: List[float]) -> List[float]:
        if effective_amount <= 0:
            return []
        if not staged:
            return [effective_amount]
        if decision_amount <= 0:
            return [effective_amount]
        scale = effective_amount / decision_amount
        return [max(0.0, amt * scale) for amt in staged]

    def analyze_market(self, pair: Optional[str] = None) -> Dict[str, Any]:
        pair = pair or self.config.pair
        ticker = self.client.get_ticker(pair)
        depth = self.client.get_depth(pair, count=200)
        trades = self.client.get_trades(pair, count=400)

        candles = build_candles(trades, interval_seconds=self.config.interval_seconds, limit=96)
        trend = analyze_trend(candles, self.config.fast_window, self.config.slow_window)
        orderbook = analyze_orderbook(depth)
        vol = analyze_volatility(candles)
        levels = support_resistance(candles)
        price = self._extract_price(ticker)
        indicators: MomentumIndicators = derive_indicators(candles)

        decision = make_trade_decision(trend, orderbook, vol, price, self.config, levels, indicators)
        return {
            "pair": pair,
            "price": price,
            "trend": trend,
            "orderbook": orderbook,
            "volatility": vol,
            "levels": levels,
            "indicators": indicators,
            "decision": decision,
            "candles": candles,
        }

    def maybe_execute(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        decision: StrategyDecision = snapshot["decision"]
        price = snapshot["price"]

        stop_reason = self.tracker.stop_reason(price)
        if stop_reason:
            logger.info("Stop triggered (%s) equity=%s", stop_reason, self.tracker.as_dict(price))
            outcome = {"status": "stopped", "reason": stop_reason, "portfolio": self.tracker.as_dict(price)}
            self._persist_state(snapshot, decision, price, outcome)
            return outcome

        if decision.action == "hold":
            logger.info("Hold action | reason=%s | portfolio=%s", decision.reason, self.tracker.as_dict(price))
            outcome = {"status": "hold", "reason": decision.reason, "portfolio": self.tracker.as_dict(price)}
            self._persist_state(snapshot, decision, price, outcome)
            return outcome

        if decision.confidence < self.config.min_confidence:
            logger.info(
                "Skip low confidence action=%s conf=%.3f min=%.3f",
                decision.action,
                decision.confidence,
                self.config.min_confidence,
            )
            outcome = {
                "status": "skipped",
                "reason": f"confidence {decision.confidence} below threshold {self.config.min_confidence}",
                "portfolio": self.tracker.as_dict(price),
            }
            self._persist_state(snapshot, decision, price, outcome)
            return outcome

        # simple slippage guard using top of book
        depth = self.client.get_depth(snapshot["pair"], count=5)
        bids = depth.get("buy") or []
        asks = depth.get("sell") or []
        top_bid = float(bids[0][0]) if bids else price
        top_ask = float(asks[0][0]) if asks else price
        reference_price = top_ask if decision.action == "buy" else top_bid
        allowed_max = price * (1 + self.config.max_slippage_pct)
        allowed_min = price * (1 - self.config.max_slippage_pct)

        if decision.action == "buy" and reference_price > allowed_max:
            logger.info("Skip buy due to slippage price=%s allowed_max=%s", reference_price, allowed_max)
            outcome = {
                "status": "skipped",
                "reason": "slippage too high for buy",
                "portfolio": self.tracker.as_dict(reference_price),
            }
            self._persist_state(snapshot, decision, reference_price, outcome)
            return outcome
        if decision.action == "sell" and reference_price < allowed_min:
            logger.info("Skip sell due to slippage price=%s allowed_min=%s", reference_price, allowed_min)
            outcome = {
                "status": "skipped",
                "reason": "slippage too high for sell",
                "portfolio": self.tracker.as_dict(reference_price),
            }
            self._persist_state(snapshot, decision, reference_price, outcome)
            return outcome

        # capital and position guards (risk management)
        effective_amount = decision.amount
        if decision.action == "buy":
            max_affordable = max(0.0, self.tracker.cash / reference_price)
            effective_amount = min(decision.amount, max_affordable)
        elif decision.action == "sell":
            max_sellable = max(0.0, self.tracker.base_position)
            effective_amount = min(decision.amount, max_sellable)

        if effective_amount <= 0:
            logger.info("Skip due to insufficient balance/position | action=%s", decision.action)
            outcome = {
                "status": "skipped",
                "reason": "insufficient balance or position",
                "portfolio": self.tracker.as_dict(reference_price),
            }
            self._persist_state(snapshot, decision, reference_price, outcome)
            return outcome

        staged = self._scale_staged_amounts(decision.amount, effective_amount, self._staged_amounts(decision, snapshot))

        executed_steps: List[Dict[str, Any]] = []
        remaining_amount = effective_amount
        if self.config.dry_run:
            for amt in staged:
                step_amount = min(amt, remaining_amount)
                if step_amount <= 0:
                    continue
                logger.info("DRY-RUN %s %s @ %s (staged)", decision.action, step_amount, reference_price)
                self.tracker.record_trade(decision.action, reference_price, step_amount)
                remaining_amount -= step_amount
                executed_steps.append({"amount": step_amount, "price": reference_price})
            outcome = {
                "status": "simulated",
                "action": decision.action,
                "price": reference_price,
                "amount": sum(step["amount"] for step in executed_steps),
                "executed_steps": executed_steps,
                "mode": decision.mode,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "portfolio": self.tracker.as_dict(reference_price),
            }
            self._persist_state(snapshot, decision, reference_price, outcome)
            return outcome

        # live trading path
        # Guard against programmatic use without running through CLI validation.
        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        for amt in staged:
            step_amount = min(amt, remaining_amount)
            if step_amount <= 0:
                continue
            order_resp = self.client.create_order(snapshot["pair"], decision.action, reference_price, step_amount)
            self.tracker.record_trade(decision.action, reference_price, step_amount)
            remaining_amount -= step_amount
            executed_steps.append({"amount": step_amount, "price": reference_price, "order": order_resp})
            logger.info(
                "Placed order action=%s amount=%s price=%s response=%s",
                decision.action,
                step_amount,
                reference_price,
                order_resp,
            )
        outcome = {
            "status": "placed",
            "action": decision.action,
            "price": reference_price,
            "amount": sum(step["amount"] for step in executed_steps),
            "executed_steps": executed_steps,
            "mode": decision.mode,
            "portfolio": self.tracker.as_dict(reference_price),
        }
        self._persist_state(snapshot, decision, reference_price, outcome)
        return outcome

    def scan_and_choose(self) -> Tuple[str, Dict[str, Any]]:
        pairs: List[str] = []
        if self.config.scan_pairs:
            pairs = list(self.config.scan_pairs)
        else:
            if self._all_pairs is None:
                try:
                    pairs_data = self.client.get_pairs()
                    names = []
                    for p in pairs_data:
                        if "name" in p:
                            names.append(p["name"])
                        elif "ticker_id" in p:
                            names.append(p["ticker_id"])
                    self._all_pairs = [n.lower() for n in names if n]
                # pragma: no cover - guard for pair listing/parsing failures
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    logger.warning("Failed to load pairs; fallback to default %s", exc)
                    self._all_pairs = [self.config.pair]
            pairs = self._all_pairs or [self.config.pair]

        best_pair = self.config.pair
        best_snapshot: Optional[Dict[str, Any]] = None
        best_score = -1.0
        failed_pairs: List[str] = []

        for pair in pairs:
            try:
                snapshot = self.analyze_market(pair)
            # pragma: no cover - guard for per-pair API/parse failures
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                logger.warning("Failed to analyze %s: %s", pair, exc)
                failed_pairs.append(pair)
                continue
            decision: StrategyDecision = snapshot["decision"]
            if decision.action == "hold":
                continue
            score = self._score_snapshot(snapshot)
            if score > best_score:
                best_score = score
                best_snapshot = snapshot
                best_pair = pair

        if failed_pairs:
            logger.warning("Skipped %s pairs due to errors: %s", len(failed_pairs), ",".join(failed_pairs))

        if best_snapshot:
            return best_pair, best_snapshot

        # fallback to default pair if nothing tradable
        return best_pair, self.analyze_market(best_pair)
