from __future__ import annotations

import logging
import time
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
from .persistence import StatePersistence
from .rate_limit import RateLimitedOrderQueue
from .realtime import MultiPairFeed, RealtimeFeed
from .grid import GridPlan, build_grid_plan
from .indodax_client import IndodaxClient
from .strategies import StrategyDecision, make_trade_decision
from .tracking import PortfolioTracker

logger = logging.getLogger(__name__)


class Trader:
    def __init__(self, config: BotConfig, client: Optional[IndodaxClient] = None) -> None:
        self.config = config
        self.order_queue = (
            RateLimitedOrderQueue(min_interval=config.order_min_interval) if config.order_queue_enabled else None
        )
        self.client = client or IndodaxClient(
            config.api_key,
            api_secret=config.api_secret,
            order_queue=self.order_queue,
            order_min_interval=config.order_min_interval,
            enable_queue=config.order_queue_enabled,
        )
        self._all_pairs: Optional[List[str]] = None
        self._scan_offset: int = 0  # rotating window index for pairs_per_cycle
        self._multi_feed: Optional[MultiPairFeed] = None
        self.tracker = PortfolioTracker(
            initial_capital=config.initial_capital,
            target_profit_pct=config.target_profit_pct,
            max_loss_pct=config.max_loss_pct,
        )
        # ── Auto-resume: persistence and state recovery ──────────────────────
        self.persistence = StatePersistence(config.state_path)
        self.restored_pair: Optional[str] = None  # set when state is loaded from disk
        self._try_restore_state()
        # ── Real-time feed ───────────────────────────────────────────────────
        self.realtime: Optional[RealtimeFeed] = None
        if config.real_time:
            self.realtime = RealtimeFeed(
                pair=self.config.pair,
                client=self.client,
                websocket_url=config.websocket_url,
                poll_interval=max(0.5, float(self.config.interval_seconds)),
                websocket_enabled=config.websocket_enabled,
                subscribe_message=config.websocket_subscribe_message,
            )
            self.realtime.start()

    # ------------------------------------------------------------------
    # Auto-resume helpers
    # ------------------------------------------------------------------

    def _try_restore_state(self) -> None:
        """Load persisted state on startup and restore PortfolioTracker.

        State is only restored when:
        - The state file exists and is valid JSON
        - The saved ``dry_run`` flag matches the current config (prevents mixing
          virtual and live state after the user toggles DRY_RUN)
        - The saved position is > 0 (a position=0 file is stale and cleared)
        """
        state = self.persistence.load()
        if state is None:
            return
        # Reject cross-mode state (dry_run live vs dry_run virtual)
        saved_dry_run = state.get("dry_run")
        if saved_dry_run is not None and bool(saved_dry_run) != self.config.dry_run:
            logger.info(
                "Ignoring saved state (dry_run mismatch: saved=%s  current=%s)",
                saved_dry_run,
                self.config.dry_run,
            )
            return
        portfolio = state.get("portfolio")
        if portfolio is None:
            return
        # Check position BEFORE mutating tracker to avoid partial load of stale state
        saved_pos = float((portfolio.get("base_position") or 0))
        if saved_pos <= 0:
            # Stale state with no open position — remove to keep things clean
            self.persistence.clear()
            return
        self.tracker.load_state(portfolio)
        pair = state.get("pair")
        if pair:
            self.restored_pair = str(pair)
        logger.info(
            "🔄 Resumed state: pair=%s  pos=%.8f  avg_cost=%s  pnl=%s",
            pair,
            self.tracker.base_position,
            self.tracker.avg_cost,
            self.tracker.realized_pnl,
        )
        # For live trading reconcile saved position against real API balance
        if not self.config.dry_run:
            self._reconcile_with_api(str(pair) if pair else self.config.pair)

    def _reconcile_with_api(self, pair: str) -> None:
        """Cross-check the saved position against the real Indodax account balance.

        Called once on startup when ``dry_run=False`` and a saved position exists.
        If the real coin balance differs from the saved value by more than 5 %
        the tracker is updated to match reality so the bot never acts on stale data.
        """
        try:
            info = self.client.get_account_info()
            # Indodax returns {"success": 1, "return": {"balance": {"btc": "0.001", ...}}}
            balance_dict = (info.get("return") or {}).get("balance") or {}
            base_coin = pair.split("_")[0].lower()
            real_balance = float(balance_dict.get(base_coin) or "0")
            saved_position = self.tracker.base_position
            tolerance = max(saved_position * 0.05, 1e-8)
            if abs(real_balance - saved_position) > tolerance:
                logger.warning(
                    "Reconciliation mismatch for %s: saved=%.8f  real=%.8f — using real balance",
                    pair,
                    saved_position,
                    real_balance,
                )
                self.tracker.base_position = real_balance
                if real_balance <= 0:
                    self.tracker.avg_cost = 0.0
                    self.persistence.clear()
                    self.restored_pair = None
                    logger.info("Position cleared after reconciliation (no real balance for %s)", base_coin)
                else:
                    self._save_state(pair)
            else:
                logger.info(
                    "Reconciliation OK for %s: saved=%.8f  real=%.8f",
                    pair,
                    saved_position,
                    real_balance,
                )
        except Exception as exc:
            logger.warning(
                "Reconciliation failed for %s: %s — trusting saved state",
                pair,
                exc,
            )

    def _save_state(self, pair: str) -> None:
        """Persist the current PortfolioTracker state to disk (fire-and-forget)."""
        try:
            self.persistence.save(
                {
                    "portfolio": self.tracker.to_state(),
                    "pair": pair,
                    "dry_run": self.config.dry_run,
                }
            )
        except Exception as exc:
            logger.warning("Failed to save state: %s", exc)

    def _clear_state(self) -> None:
        """Remove the state file when a position is fully closed."""
        try:
            self.persistence.clear()
        except Exception as exc:
            logger.warning("Failed to clear state: %s", exc)

    def _persist_after_trade(self, pair: str) -> None:
        """Save state if still in a position, or clear it if the position is closed."""
        if self.tracker.base_position <= 0:
            self._clear_state()
        else:
            self._save_state(pair)

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

    def _pair_volume(self, pair: str) -> float:
        """Return the cached 24-h IDR trading volume for *pair*, or 0.0.

        Used for priority sorting so high-liquidity pairs are scanned first.
        Key lookup order:
        1. ``vol_idr``     – standard Indodax summaries field for all IDR pairs
        2. ``idr_volume``  – alternative spelling used by some endpoints
        3. ``volume``      – generic fallback
        Pairs with no cached ticker (absent from summaries) return 0.0 so they
        are placed at the end of the priority-sorted list.
        """
        if self._multi_feed is None:
            return 0.0
        ticker = self._multi_feed.get_ticker(pair)
        if ticker is None:
            return 0.0
        for key in ("vol_idr", "idr_volume", "volume"):
            val = ticker.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return 0.0

    def _sort_pairs_by_priority(self, pairs: List[str]) -> List[str]:
        """Return *pairs* sorted by 24-h IDR volume, highest first (stable).

        Pairs with no cached ticker are placed at the end so the most liquid
        — and typically most volatile / profitable — coins are always analyzed
        first in the serial scan loop.
        """
        return sorted(pairs, key=self._pair_volume, reverse=True)

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

    def analyze_market(
        self,
        pair: Optional[str] = None,
        prefetched_ticker: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pair = pair or self.config.pair
        ticker: Dict[str, Any]
        depth: Dict[str, Any]
        trades: List[Dict[str, Any]]
        if self.realtime and self.realtime.has_snapshot and pair == self.config.pair:
            snap = self.realtime.snapshot()
            ticker = snap.get("ticker") or prefetched_ticker or self.client.get_ticker(pair)
            depth = snap.get("depth") or self.client.get_depth(pair, count=200)
            trades = snap.get("trades") or self.client.get_trades(pair, count=400)
        else:
            ticker = prefetched_ticker or self.client.get_ticker(pair)
            depth = self.client.get_depth(pair, count=200)
            trades = self.client.get_trades(pair, count=400)

        candles = build_candles(trades, interval_seconds=self.config.interval_seconds, limit=96)
        trend = analyze_trend(candles, self.config.fast_window, self.config.slow_window)
        orderbook = analyze_orderbook(depth)
        vol = analyze_volatility(candles)
        levels = support_resistance(candles)
        price = self._extract_price(ticker)
        indicators: MomentumIndicators = derive_indicators(candles)

        grid_plan: Optional[GridPlan] = None
        decision: StrategyDecision
        if self.config.grid_enabled:
            grid_plan = build_grid_plan(price, self.config)
            total_amount = sum(order.amount for order in grid_plan.orders)
            decision = StrategyDecision(
                mode="grid_trading",
                action="grid",
                confidence=1.0,
                reason=f"grid levels={len(grid_plan.orders)} spacing={self.config.grid_spacing_pct}",
                target_price=price,
                amount=total_amount,
                stop_loss=None,
                take_profit=None,
            )
        else:
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
            "grid_plan": grid_plan,
        }

    def maybe_execute(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        decision: StrategyDecision = snapshot["decision"]
        price = snapshot["price"]

        # Update trailing stop before checking stop conditions
        if self.config.trailing_stop_pct > 0:
            self.tracker.update_trailing_stop(price, self.config.trailing_stop_pct)

        stop_reason = self.tracker.stop_reason(price)
        if stop_reason:
            logger.info("Stop triggered (%s) equity=%s", stop_reason, self.tracker.as_dict(price))
            outcome = {"status": "stopped", "reason": stop_reason, "portfolio": self.tracker.as_dict(price)}
            return outcome

        if self.config.grid_enabled and snapshot.get("grid_plan"):
            return self._execute_grid(snapshot)

        if decision.action == "hold":
            logger.info("Hold action | reason=%s | portfolio=%s", decision.reason, self.tracker.as_dict(price))
            outcome = {"status": "hold", "reason": decision.reason, "portfolio": self.tracker.as_dict(price)}
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
            return outcome
        if decision.action == "sell" and reference_price < allowed_min:
            logger.info("Skip sell due to slippage price=%s allowed_min=%s", reference_price, allowed_min)
            outcome = {
                "status": "skipped",
                "reason": "slippage too high for sell",
                "portfolio": self.tracker.as_dict(reference_price),
            }
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
            self._persist_after_trade(snapshot["pair"])
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
        self._persist_after_trade(snapshot["pair"])
        return outcome

    def _execute_grid(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        price = snapshot["price"]
        decision: StrategyDecision = snapshot["decision"]
        plan: GridPlan = snapshot["grid_plan"]
        orders = plan.orders

        if not orders:
            outcome = {
                "status": "skipped",
                "reason": "no grid orders generated",
                "portfolio": self.tracker.as_dict(price),
            }
            return outcome

        pair = snapshot["pair"]
        if self.config.dry_run:
            orders_payload = [{"side": o.side, "price": o.price, "amount": o.amount} for o in orders]
            outcome = {
                "status": "grid_simulated",
                "anchor_price": plan.anchor_price,
                "orders": orders_payload,
                "portfolio": self.tracker.as_dict(price),
            }
            return outcome

        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        executed: List[Dict[str, Any]] = []
        for order in orders:
            resp = self.client.create_order(pair, order.side, order.price, order.amount)
            executed.append({"side": order.side, "price": order.price, "amount": order.amount, "response": resp})
            logger.info("Placed grid order %s %s @ %s resp=%s", order.side, order.amount, order.price, resp)

        outcome = {
            "status": "grid_placed",
            "anchor_price": plan.anchor_price,
            "orders": executed,
            "portfolio": self.tracker.as_dict(price),
        }
        return outcome

    def force_sell(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        """Sell the entire open base position at current market price.

        Used when a stop condition or an explicit exit signal requires immediate
        liquidation before searching for the next trading opportunity.  Safe to
        call even when there is no open position – returns ``status: no_position``
        in that case.
        """
        price = snapshot["price"]
        pair = snapshot["pair"]
        amount = self.tracker.base_position

        if amount <= 0:
            return {"status": "no_position", "pair": pair, "price": price}

        # Use top-of-book bid as reference price for the sell; fall back to
        # the snapshot price if the order-book fetch fails.
        reference_price = price
        try:
            depth = self.client.get_depth(pair, count=5)
            bids = depth.get("buy") or []
            if bids:
                reference_price = float(bids[0][0])
        except Exception:
            pass

        if self.config.dry_run:
            logger.info("DRY-RUN force-sell %.8f %s @ %s", amount, pair, reference_price)
            self.tracker.record_trade("sell", reference_price, amount)
            outcome: Dict[str, Any] = {
                "status": "force_sold",
                "action": "sell",
                "pair": pair,
                "price": reference_price,
                "amount": amount,
                "portfolio": self.tracker.as_dict(reference_price),
            }
            self._persist_after_trade(pair)
            return outcome

        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        order_resp = self.client.create_order(pair, "sell", reference_price, amount)
        self.tracker.record_trade("sell", reference_price, amount)
        outcome = {
            "status": "force_sold",
            "action": "sell",
            "pair": pair,
            "price": reference_price,
            "amount": amount,
            "order": order_resp,
            "portfolio": self.tracker.as_dict(reference_price),
        }
        self._persist_after_trade(pair)
        return outcome

    _MAX_SCAN_RETRIES = 3
    _SCAN_BACKOFF_BASE = 2.0  # seconds for the first retry; doubles each attempt (2 → 4 → 8 …)
    _SCAN_BACKOFF_MAX = 30.0  # hard cap so a long run of 429s never blocks indefinitely

    def _analyze_with_retry(
        self,
        pair: str,
        prefetched_ticker: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze a single pair with exponential back-off on HTTP 429 responses.

        The client raises ``RuntimeError("HTTP error: 429 …")`` (wrapping the
        underlying ``requests.HTTPError``).  Both exception types are handled so
        the retry logic works regardless of whether tests mock at the client
        layer or the ``analyze_market`` layer.
        """
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(self._MAX_SCAN_RETRIES):
            try:
                return self.analyze_market(pair, prefetched_ticker=prefetched_ticker)
            except (requests.HTTPError, RuntimeError) as exc:
                # Detect 429 from both the raw requests.HTTPError (test path) and
                # from the RuntimeError wrapper that IndodaxClient raises (production).
                if isinstance(exc, requests.HTTPError):
                    is_429 = exc.response is not None and exc.response.status_code == 429
                else:
                    # IndodaxClient wraps requests.HTTPError as the __cause__
                    cause = exc.__cause__
                    is_429 = (
                        isinstance(cause, requests.HTTPError)
                        and cause.response is not None
                        and cause.response.status_code == 429
                    )
                if is_429:
                    backoff = min(self._SCAN_BACKOFF_BASE * (2 ** attempt), self._SCAN_BACKOFF_MAX)
                    logger.warning(
                        "Rate-limited on %s (attempt %d/%d); backing off %.1fs",
                        pair, attempt + 1, self._MAX_SCAN_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    last_exc = exc
                    # Don't rely on a stale prefetched ticker on retry; let the
                    # call fetch it fresh via REST.
                    prefetched_ticker = None
                else:
                    raise
        raise last_exc

    def scan_and_choose(self) -> Tuple[str, Dict[str, Any]]:
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

        all_pairs = self._all_pairs or [self.config.pair]

        # ── Multi-pair feed (persistent ticker cache) ────────────────────────
        # Start the feed once after the full pair list is known.  It seeds the
        # cache synchronously on first start via /api/summaries and then keeps
        # it fresh in a background thread (WebSocket or periodic summaries poll).
        if self._multi_feed is None:
            self._multi_feed = MultiPairFeed(
                pairs=all_pairs,
                client=self.client,
                websocket_url=self.config.websocket_url,
                websocket_enabled=self.config.websocket_enabled,
                batch_size=self.config.websocket_batch_size,
            )
            self._multi_feed.start()
            logger.info(
                "MultiPairFeed started for %d pairs (batch_size=%d)",
                len(all_pairs),
                self.config.websocket_batch_size,
            )

        # ── Rotating pair window ─────────────────────────────────────────────
        # When pairs_per_cycle > 0 we analyse a rotating window of that many
        # pairs per call instead of the full list.  This spreads REST calls
        # across cycles and reduces peak request rate.
        n = len(all_pairs)
        ppc = self.config.pairs_per_cycle
        if ppc > 0 and n > ppc:
            start = self._scan_offset % n
            # Slice wrapping around the end of the list
            if start + ppc <= n:
                pairs = all_pairs[start : start + ppc]
            else:
                pairs = all_pairs[start:] + all_pairs[: (start + ppc) - n]
            self._scan_offset = (start + ppc) % n
            logger.debug(
                "Rotating scan: window=[%d, %d) of %d total pairs",
                start,
                (start + ppc) % n,
                n,
            )
        else:
            pairs = all_pairs

        # ── Priority sort ────────────────────────────────────────────────────
        # Re-order by 24-h IDR trading volume (highest first) so the serial
        # loop reaches the most liquid — and most likely tradeable — pairs
        # before touching low-volume tail coins.  Pairs absent from the cache
        # stay at the end.
        pairs = self._sort_pairs_by_priority(pairs)

        best_pair = pairs[0] if pairs else self.config.pair
        best_snapshot: Optional[Dict[str, Any]] = None
        best_score = -1.0
        failed_pairs: List[str] = []
        skipped_pairs: List[str] = []

        feed_seeded = self._multi_feed.is_seeded

        for scan_idx, pair in enumerate(pairs):
            if self.config.scan_request_delay > 0:
                time.sleep(self.config.scan_request_delay)
            # Use the multi-pair feed's cached ticker to skip the per-pair REST
            # ticker call entirely.  When the feed is seeded but this specific
            # pair has no cached data (absent from /api/summaries — typically
            # inactive or very-new pairs), skip it rather than falling through
            # to a REST call that would trigger a 429.
            prefetched_ticker = self._multi_feed.get_ticker(pair)
            if prefetched_ticker is None and feed_seeded:
                skipped_pairs.append(pair)
                continue
            try:
                snapshot = self._analyze_with_retry(pair, prefetched_ticker=prefetched_ticker)
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

            # ── Serial early exit ─────────────────────────────────────────
            # Stop scanning as soon as the first pair whose signal meets the
            # confidence threshold is found.  Because pairs are sorted by
            # liquidity (liquid-first), this exits on the highest-quality
            # opportunity available without analysing all 500+ pairs.
            # Lower-volume pairs are deferred to later cycle windows.
            if decision.confidence >= self.config.min_confidence:
                logger.debug(
                    "Serial scan: early exit on %s (conf=%.3f, scanned %d/%d pairs)",
                    pair,
                    decision.confidence,
                    scan_idx + 1,
                    len(pairs),
                )
                break

        if skipped_pairs:
            logger.debug(
                "Skipped %d pairs not in feed cache: %s",
                len(skipped_pairs),
                ",".join(skipped_pairs[:10]) + ("…" if len(skipped_pairs) > 10 else ""),
            )

        if failed_pairs:
            logger.warning("Skipped %s pairs due to errors: %s", len(failed_pairs), ",".join(failed_pairs))

        if best_snapshot:
            return best_pair, best_snapshot

        if failed_pairs:
            message = "No pairs could be analyzed successfully"
            message = f"{message} (failed: {','.join(failed_pairs)})"
            raise RuntimeError(message)

        # fallback to default pair if nothing tradable but no analysis errors
        return best_pair, self.analyze_market(best_pair)
