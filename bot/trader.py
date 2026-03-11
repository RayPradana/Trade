from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import requests

from .analysis import (
    analyze_orderbook,
    analyze_trend,
    analyze_volatility,
    build_candles,
    support_resistance,
)
from .config import BotConfig
from .indodax_client import IndodaxClient
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

        decision = make_trade_decision(trend, orderbook, vol, price, self.config, levels)
        return {
            "pair": pair,
            "price": price,
            "trend": trend,
            "orderbook": orderbook,
            "volatility": vol,
            "levels": levels,
            "decision": decision,
            "candles": candles,
        }

    def maybe_execute(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        decision: StrategyDecision = snapshot["decision"]
        price = snapshot["price"]

        stop_reason = self.tracker.stop_reason(price)
        if stop_reason:
            logger.info("Stop triggered (%s) equity=%s", stop_reason, self.tracker.as_dict(price))
            return {"status": "stopped", "reason": stop_reason, "portfolio": self.tracker.as_dict(price)}

        if decision.action == "hold":
            logger.info("Hold action | reason=%s | portfolio=%s", decision.reason, self.tracker.as_dict(price))
            return {"status": "hold", "reason": decision.reason, "portfolio": self.tracker.as_dict(price)}

        if decision.confidence < self.config.min_confidence:
            logger.info(
                "Skip low confidence action= %s conf=%.3f min=%.3f",
                decision.action,
                decision.confidence,
                self.config.min_confidence,
            )
            return {
                "status": "skipped",
                "reason": f"confidence {decision.confidence} below threshold {self.config.min_confidence}",
                "portfolio": self.tracker.as_dict(price),
            }

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
            return {
                "status": "skipped",
                "reason": "slippage too high for buy",
                "portfolio": self.tracker.as_dict(reference_price),
            }
        if decision.action == "sell" and reference_price < allowed_min:
            logger.info("Skip sell due to slippage price=%s allowed_min=%s", reference_price, allowed_min)
            return {
                "status": "skipped",
                "reason": "slippage too high for sell",
                "portfolio": self.tracker.as_dict(reference_price),
            }

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
            return {
                "status": "skipped",
                "reason": "insufficient balance or position",
                "portfolio": self.tracker.as_dict(reference_price),
            }

        if self.config.dry_run:
            logger.info("DRY-RUN %s %s @ %s", decision.action, effective_amount, reference_price)
            self.tracker.record_trade(decision.action, reference_price, effective_amount)
            return {
                "status": "simulated",
                "action": decision.action,
                "price": reference_price,
                "amount": effective_amount,
                "mode": decision.mode,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "portfolio": self.tracker.as_dict(reference_price),
            }

        # live trading path
        # Guard against programmatic use without running through CLI validation.
        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        order_resp = self.client.create_order(
            self.config.pair, decision.action, reference_price, effective_amount
        )
        self.tracker.record_trade(decision.action, reference_price, effective_amount)
        logger.info(
            "Placed order action=%s amount=%s price=%s response=%s",
            decision.action,
            effective_amount,
            reference_price,
            order_resp,
        )
        return {
            "status": "placed",
            "order": order_resp,
            "action": decision.action,
            "price": reference_price,
            "amount": effective_amount,
            "mode": decision.mode,
            "portfolio": self.tracker.as_dict(reference_price),
        }

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
                except Exception as exc:  # pragma: no cover - network guard
                    logger.warning("Failed to load pairs; fallback to default %s", exc)
                    self._all_pairs = [self.config.pair]
            pairs = self._all_pairs or [self.config.pair]

        best_pair = self.config.pair
        best_snapshot: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for pair in pairs:
            try:
                snapshot = self.analyze_market(pair)
            except (requests.RequestException, RuntimeError, ValueError) as exc:  # pragma: no cover - guard for flaky pairs
                logger.warning("Failed to analyze %s: %s", pair, exc)
                continue
            decision: StrategyDecision = snapshot["decision"]
            if decision.action == "hold":
                continue
            if decision.confidence > best_score:
                best_score = decision.confidence
                best_snapshot = snapshot
                best_pair = pair

        if best_snapshot:
            self.config.pair = best_pair
            return best_pair, best_snapshot

        # fallback to default pair if nothing tradable
        self.config.pair = best_pair
        return best_pair, self.analyze_market(best_pair)
