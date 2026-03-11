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
    candles_from_ohlc,
    derive_indicators,
    detect_spoofing,
    detect_whale_activity,
    interval_to_ohlc_tf,
    multi_timeframe_confirm,
    MomentumIndicators,
    MultiTimeframeResult,
    SpoofingResult,
    WhaleActivity,
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

# Extra candles requested beyond slow_window when fetching OHLC history so
# that EMA-based indicators (MACD, Bollinger) have enough warm-up data.
_OHLC_BUFFER_CANDLES = 100


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
        self._scan_cycle_count: int = 0  # total completed full-scan cycles (for dynamic refresh)
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

    def _fetch_candles(
        self,
        pair: str,
        trades: List[Dict[str, Any]],
    ) -> List[Any]:
        """Return OHLCV candles for *pair*, preferring the official OHLC endpoint.

        Tries ``/tradingview/history_v2`` first (reliable pre-formed candles).
        Falls back to :func:`~bot.analysis.build_candles` from raw trades when
        the OHLC endpoint is unavailable or returns no data.
        """
        try:
            tf = interval_to_ohlc_tf(self.config.interval_seconds)
            # Request enough candles to seed all indicators (slow_window + buffer).
            limit = max(200, self.config.slow_window + _OHLC_BUFFER_CANDLES)
            ohlc_data = self.client.get_ohlc(pair, tf=tf, limit=limit)
            candles = candles_from_ohlc(ohlc_data)
            if candles:
                logger.debug(
                    "OHLC candles for %s: %d candles (tf=%s)", pair, len(candles), tf
                )
                return candles
        except Exception as exc:
            logger.debug(
                "OHLC fetch failed for %s (%s); falling back to trades", pair, exc
            )
        # Legacy fallback: build candles by bucketing raw trade ticks.
        return build_candles(
            trades, interval_seconds=self.config.interval_seconds, limit=200
        )

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

    def _get_reference_pair_trend(self, current_pair: str) -> Optional[str]:
        """Return the trend direction of the highest-volume reference pair.

        Used for correlated-pair confirmation: when BTC/IDR (or whatever pair
        has the most liquidity in the feed cache) is trending up, alt-coin buy
        signals receive a small confidence boost in :meth:`analyze_market`.

        Returns ``"up"``, ``"down"``, or ``None`` when no reference can be
        determined (feed not started, same as current pair, no data, etc.).
        """
        if self._multi_feed is None:
            return None
        try:
            all_known = list(self._multi_feed._cache.keys())
            if not all_known:
                return None
            ranked = sorted(all_known, key=self._pair_volume, reverse=True)
            # Pick the top-volume pair that is NOT the current pair
            ref_pair = next((p for p in ranked if p != current_pair), None)
            if ref_pair is None:
                return None
            ticker = self._multi_feed.get_ticker(ref_pair)
            if ticker is None:
                return None
            # Determine direction from cached last/high/low values
            last = float(ticker.get("last") or ticker.get("last_price") or 0)
            high = float(ticker.get("high") or 0)
            low = float(ticker.get("low") or 0)
            if last <= 0 or high <= 0 or low <= 0:
                return None
            mid = (high + low) / 2
            if last > mid * 1.001:
                return "up"
            if last < mid * 0.999:
                return "down"
            return None
        except Exception:
            return None

    def _liquidity_depth_idr(self, depth: Dict[str, Any], price: float) -> float:
        """Return the total IDR value of the top-20 bid and ask levels.

        Used to filter out thin markets where the effective spread would make
        a trade unprofitable.  Returns 0.0 on any parse error.
        """
        try:
            bids = (depth.get("buy") or [])[:20]
            asks = (depth.get("sell") or [])[:20]
            total = 0.0
            for level in bids + asks:
                total += float(level[0]) * float(level[1])
            return total
        except (IndexError, TypeError, ValueError):
            return 0.0

    def _effective_interval(self, snapshot: Optional[Dict[str, Any]] = None) -> int:
        """Return the effective sleep interval (seconds) between scan cycles.

        When :attr:`~BotConfig.adaptive_interval_enabled` is ``True``, a
        higher volatility in the most recent snapshot reduces the sleep to
        :attr:`~BotConfig.adaptive_interval_min_seconds`.  Otherwise the
        configured :attr:`~BotConfig.interval_seconds` is returned unchanged.
        """
        if not self.config.adaptive_interval_enabled:
            return self.config.interval_seconds
        vol_value = 0.0
        if snapshot:
            vol = snapshot.get("volatility")
            if vol:
                vol_value = getattr(vol, "volatility", 0.0)
        # Adaptive logic: if volatility > 1.5× the typical threshold (0.01),
        # use the minimum interval; otherwise keep the normal interval.
        if vol_value > 0.015:
            return self.config.adaptive_interval_min_seconds
        return self.config.interval_seconds

    def _refresh_dynamic_pairs(self) -> None:
        """Replace the pair watchlist with the top-N most liquid pairs.

        Called every ``dynamic_pairs_refresh_cycles`` full-scan cycles when the
        feature is enabled (``dynamic_pairs_refresh_cycles > 0``).  Reads
        current volume data from the multi-pair feed cache to rank pairs and
        updates ``_all_pairs`` in-place so that the next scan uses the fresh
        watchlist.

        The refresh is best-effort: any exception is logged and the existing
        watchlist is kept unchanged.
        """
        if self._multi_feed is None:
            return
        try:
            all_known = list(self._multi_feed._cache.keys())
            if not all_known:
                return
            ranked = sorted(all_known, key=self._pair_volume, reverse=True)
            top_n = self.config.dynamic_pairs_top_n
            new_pairs = ranked[:top_n] if top_n > 0 else ranked
            if new_pairs:
                self._all_pairs = new_pairs
                logger.info(
                    "Dynamic pairs: refreshed watchlist → %d pairs (top=%d by volume)",
                    len(new_pairs),
                    top_n,
                )
        except Exception:
            logger.warning("Dynamic pair refresh failed", exc_info=True)

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
            trades = snap.get("trades") or self.client.get_trades(pair, count=self.config.trade_count)
        else:
            ticker = prefetched_ticker or self.client.get_ticker(pair)
            depth = self.client.get_depth(pair, count=200)
            trades = self.client.get_trades(pair, count=self.config.trade_count)

        # ── Candle data ──────────────────────────────────────────────────────
        # Prefer the official OHLCV history endpoint which returns pre-formed
        # candles and covers enough history for all indicators even for
        # high-volume pairs.  Fall back to building candles from raw trades
        # (the legacy path) when the OHLC call fails.
        candles = self._fetch_candles(pair, trades)

        insufficient_data = len(candles) < self.config.min_candles
        if insufficient_data:
            logger.debug(
                "Pair %s: only %d candle(s) available (need ≥%d for reliable indicators)",
                pair,
                len(candles),
                self.config.min_candles,
            )

        trend = analyze_trend(candles, self.config.fast_window, self.config.slow_window)
        orderbook = analyze_orderbook(depth)
        vol = analyze_volatility(candles)
        levels = support_resistance(candles)
        price = self._extract_price(ticker)
        indicators: MomentumIndicators = derive_indicators(candles)

        # ── Multi-timeframe analysis ─────────────────────────────────────────
        mtf: Optional[MultiTimeframeResult] = None
        if self.config.mtf_timeframes:
            candles_by_tf: Dict[str, Any] = {}
            for tf in self.config.mtf_timeframes:
                try:
                    ohlc = self.client.get_ohlc(pair, tf=tf, limit=max(200, self.config.slow_window + _OHLC_BUFFER_CANDLES))
                    candles_by_tf[tf] = candles_from_ohlc(ohlc)
                except Exception:
                    logger.debug("MTF: failed to fetch tf=%s for %s", tf, pair, exc_info=True)
            if candles_by_tf:
                mtf = multi_timeframe_confirm(candles_by_tf, self.config.fast_window, self.config.slow_window)
                logger.debug("MTF %s: direction=%s aligned=%s tf=%s", pair, mtf.direction, mtf.aligned, mtf.tf_directions)

        # ── Whale / smart-money detection ────────────────────────────────────
        whale: WhaleActivity = detect_whale_activity(depth)
        if whale.detected:
            logger.debug("Whale detected on %s: side=%s ratio=%.1f×", pair, whale.side, whale.ratio)

        # ── Spoofing / manipulation detection ────────────────────────────────
        spoofing: SpoofingResult = detect_spoofing(depth)
        if spoofing.detected:
            logger.debug(
                "Spoofing detected on %s: side=%s distance=%.2f%%",
                pair, spoofing.side, spoofing.distance_pct * 100,
            )

        # ── Correlated pair boost (BTC/ETH reference trend) ──────────────────
        # When the highest-volume reference pair (typically BTC/IDR) is trending
        # in the same direction as the primary signal, add a small confidence
        # boost in `make_trade_decision` by passing the correlation note via the
        # whale argument augmentation below.  We do this via the `mtf` mechanism
        # in `make_trade_decision` by synthesising a very simple cross-pair trend
        # note stored in the snapshot and applied in _score_snapshot.
        reference_trend = self._get_reference_pair_trend(pair)

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
            decision = make_trade_decision(
                trend, orderbook, vol, price, self.config, levels, indicators,
                mtf=mtf, whale=whale, spoofing=spoofing,
                effective_capital=self.tracker.effective_capital(),
            )
            # Apply correlated-pair confidence boost when reference trend aligns
            if reference_trend == "up" and decision.action == "buy":
                boosted_conf = min(1.0, decision.confidence + 0.04)
                decision = StrategyDecision(
                    **{**decision.__dict__, "confidence": round(boosted_conf, 3),
                       "reason": decision.reason + " corr_confirm"}
                )
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
            "insufficient_data": insufficient_data,
            "mtf": mtf,
            "whale": whale,
            "spoofing": spoofing,
            "reference_trend": reference_trend,
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

        # ── Daily loss cap ────────────────────────────────────────────────────
        if self.config.max_daily_loss_pct > 0 and decision.action == "buy":
            daily_loss_pct = self.tracker.daily_loss_pct(price)
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                logger.warning(
                    "Daily loss cap reached: %.2f%% ≥ %.2f%% — skipping buy",
                    daily_loss_pct * 100,
                    self.config.max_daily_loss_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"daily_loss_cap {daily_loss_pct:.2%} ≥ {self.config.max_daily_loss_pct:.2%}",
                    "portfolio": self.tracker.as_dict(price),
                }

        # ── Per-coin exposure cap ─────────────────────────────────────────────
        if self.config.max_exposure_per_coin_pct > 0 and decision.action == "buy":
            current_equity = self.tracker.snapshot(price).equity
            current_exposure = self.tracker.base_position * price
            exposure_pct = current_exposure / current_equity if current_equity > 0 else 0.0
            if exposure_pct >= self.config.max_exposure_per_coin_pct:
                logger.info(
                    "Exposure cap reached: %.2f%% ≥ %.2f%% — skipping buy on %s",
                    exposure_pct * 100,
                    self.config.max_exposure_per_coin_pct * 100,
                    snapshot["pair"],
                )
                return {
                    "status": "skipped",
                    "reason": (
                        f"exposure_cap {exposure_pct:.2%} ≥ {self.config.max_exposure_per_coin_pct:.2%}"
                    ),
                    "portfolio": self.tracker.as_dict(price),
                }

        # ── Portfolio-wide risk cap ───────────────────────────────────────────
        if self.config.max_portfolio_risk_pct > 0 and decision.action == "buy":
            current_equity = self.tracker.snapshot(price).equity
            total_position_value = self.tracker.base_position * price
            portfolio_risk_pct = total_position_value / current_equity if current_equity > 0 else 0.0
            if portfolio_risk_pct >= self.config.max_portfolio_risk_pct:
                logger.info(
                    "Portfolio risk cap reached: %.2f%% ≥ %.2f%% — skipping buy",
                    portfolio_risk_pct * 100,
                    self.config.max_portfolio_risk_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"portfolio_risk_cap {portfolio_risk_pct:.2%} ≥ {self.config.max_portfolio_risk_pct:.2%}",
                    "portfolio": self.tracker.as_dict(price),
                }

        # ── Profit-buffer drawdown guard ──────────────────────────────────────
        if self.config.profit_buffer_drawdown_pct > 0 and decision.action == "buy":
            pb_drawdown = self.tracker.profit_buffer_drawdown_pct()
            if pb_drawdown >= self.config.profit_buffer_drawdown_pct:
                logger.warning(
                    "Profit-buffer drawdown guard: buffer dropped %.1f%% from peak (limit=%.1f%%) — skipping buy",
                    pb_drawdown * 100,
                    self.config.profit_buffer_drawdown_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"profit_buffer_drawdown {pb_drawdown:.2%} ≥ {self.config.profit_buffer_drawdown_pct:.2%}",
                    "portfolio": self.tracker.as_dict(price),
                }

        # ── Re-entry cooldown / dip check ─────────────────────────────────────
        if decision.action == "buy" and (
            self.config.re_entry_cooldown_seconds > 0 or self.config.re_entry_dip_pct > 0
        ):
            if not self.tracker.re_entry_allowed(
                price,
                cooldown_seconds=self.config.re_entry_cooldown_seconds,
                dip_pct=self.config.re_entry_dip_pct,
            ):
                logger.info(
                    "Re-entry blocked: cooldown or dip condition not met (last_sell=%.2f, now=%.2f)",
                    self.tracker.last_sell_price,
                    price,
                )
                return {
                    "status": "skipped",
                    "reason": "re_entry_condition_not_met",
                    "portfolio": self.tracker.as_dict(price),
                }

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
        depth = self.client.get_depth(snapshot["pair"], count=20)
        bids = depth.get("buy") or []
        asks = depth.get("sell") or []
        top_bid = float(bids[0][0]) if bids else price
        top_ask = float(asks[0][0]) if asks else price
        reference_price = top_ask if decision.action == "buy" else top_bid
        allowed_max = price * (1 + self.config.max_slippage_pct)
        allowed_min = price * (1 - self.config.max_slippage_pct)

        # ── Minimum liquidity depth check ─────────────────────────────────────
        if self.config.min_liquidity_depth_idr > 0 and decision.action == "buy":
            total_depth = self._liquidity_depth_idr(depth, price)
            if total_depth < self.config.min_liquidity_depth_idr:
                logger.info(
                    "Thin market on %s: depth Rp%.0f < min Rp%.0f — skipping buy",
                    snapshot["pair"], total_depth, self.config.min_liquidity_depth_idr,
                )
                return {
                    "status": "skipped",
                    "reason": f"thin_market depth Rp{total_depth:.0f} < Rp{self.config.min_liquidity_depth_idr:.0f}",
                    "portfolio": self.tracker.as_dict(price),
                }

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

    def partial_take_profit(self, snapshot: Dict[str, Any], fraction: float) -> Dict[str, Any]:
        """Sell *fraction* of the current base position as a partial take-profit.

        This is triggered from the position-monitoring loop in ``main.py`` when
        price crosses the decision's TP level and
        :attr:`~BotConfig.partial_tp_fraction` is configured.

        :param snapshot: Current market snapshot for the held pair.
        :param fraction: Fraction of position to sell (0 < fraction < 1).
        :returns: Outcome dict similar to :meth:`force_sell`.
        """
        price = snapshot["price"]
        pair = snapshot["pair"]
        total_position = self.tracker.base_position

        if total_position <= 0:
            return {"status": "no_position", "pair": pair, "price": price}
        if not (0 < fraction < 1):
            return {"status": "invalid_fraction", "pair": pair, "price": price}

        amount = total_position * fraction

        # Use top-of-book bid as reference; fall back to snapshot price
        reference_price = price
        try:
            depth = self.client.get_depth(pair, count=5)
            bids = depth.get("buy") or []
            if bids:
                reference_price = float(bids[0][0])
        except Exception:
            pass

        if self.config.dry_run:
            logger.info(
                "DRY-RUN partial-TP sell %.8f (%.0f%%) %s @ %s",
                amount, fraction * 100, pair, reference_price,
            )
            self.tracker.record_trade("sell", reference_price, amount)
            self.tracker.partial_tp_taken = True
            outcome: Dict[str, Any] = {
                "status": "partial_tp",
                "action": "sell",
                "pair": pair,
                "price": reference_price,
                "amount": amount,
                "fraction": fraction,
                "portfolio": self.tracker.as_dict(reference_price),
            }
            self._persist_after_trade(pair)
            return outcome

        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        order_resp = self.client.create_order(pair, "sell", reference_price, amount)
        self.tracker.record_trade("sell", reference_price, amount)
        self.tracker.partial_tp_taken = True
        outcome = {
            "status": "partial_tp",
            "action": "sell",
            "pair": pair,
            "price": reference_price,
            "amount": amount,
            "fraction": fraction,
            "order": order_resp,
            "portfolio": self.tracker.as_dict(reference_price),
        }
        self._persist_after_trade(pair)
        return outcome

    def _conditions_allow_holding(self, snapshot: Dict[str, Any]) -> bool:
        """Return ``True`` when market indicators suggest holding past the TP target.

        Used by :meth:`evaluate_dynamic_tp` to decide whether to postpone
        taking profit.  Returns ``True`` when all *configured* conditions pass
        (any condition with a threshold of 0 is disabled and always passes).
        """
        config = self.config
        trend = snapshot.get("trend")
        orderbook = snapshot.get("orderbook")
        indicators = snapshot.get("indicators")

        # Trend strength must be strong enough to justify holding
        if config.conditional_tp_min_trend_strength > 0:
            strength = getattr(trend, "strength", 0.0) if trend else 0.0
            if strength < config.conditional_tp_min_trend_strength:
                return False

        # Order-book must still show bullish imbalance
        if config.conditional_tp_min_ob_imbalance > 0:
            imbalance = getattr(orderbook, "imbalance", 0.0) if orderbook else 0.0
            if imbalance < config.conditional_tp_min_ob_imbalance:
                return False

        # RSI must not be overbought
        if config.conditional_tp_max_rsi > 0:
            rsi = getattr(indicators, "rsi", 50.0) if indicators else 50.0
            if rsi >= config.conditional_tp_max_rsi:
                return False

        return True

    def evaluate_dynamic_tp(self, snapshot: Dict[str, Any]) -> Optional[str]:
        """Decide what to do when equity has hit the take-profit target.

        Called from the position-monitoring loop in ``main.py`` whenever
        :meth:`~PortfolioTracker.stop_reason` returns
        ``"target_profit_reached"``.

        Returns
        -------
        ``"target_profit_reached"``
            Close the position now — conditions don't support holding.
        ``"trailing_tp_triggered"``
            Close now — the trailing TP floor was hit.
        ``None``
            Hold — either trailing TP is active and price is above the floor,
            or conditional checks say the trend is still bullish.
        """
        config = self.config
        price = snapshot["price"]

        # If neither feature is configured → standard TP behaviour (close now)
        dynamic_tp_enabled = (
            config.trailing_tp_pct > 0
            or config.conditional_tp_min_trend_strength > 0
            or config.conditional_tp_max_rsi > 0
            or config.conditional_tp_min_ob_imbalance > 0
        )
        if not dynamic_tp_enabled:
            return "target_profit_reached"

        # If trailing TP is already active and price fell through the floor → close
        if (
            self.tracker.trailing_tp_stop is not None
            and price <= self.tracker.trailing_tp_stop
        ):
            return "trailing_tp_triggered"

        # Check whether market conditions allow holding
        conditions_ok = self._conditions_allow_holding(snapshot)

        if not conditions_ok:
            # Conditions no longer support holding → take profit now
            logger.debug(
                "Conditional TP: conditions failed at price=%.2f — taking profit",
                price,
            )
            return "target_profit_reached"

        # Conditions still bullish — activate / advance the trailing TP floor
        if config.trailing_tp_pct > 0:
            self.tracker.activate_trailing_tp(price, config.trailing_tp_pct)
            logger.debug(
                "Dynamic TP: trailing floor updated to %.2f (price=%.2f)",
                self.tracker.trailing_tp_stop or 0.0,
                price,
            )

        # Hold the position — let profits run
        return None

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

        # ── Minimum volume filter ─────────────────────────────────────────
        # Skip pairs whose 24-h IDR trading volume is below the configured
        # threshold so the scan focuses only on liquid instruments.
        low_volume_pairs: List[str] = []
        if self.config.min_volume_idr > 0:
            filtered: List[str] = []
            for p in pairs:
                if self._pair_volume(p) >= self.config.min_volume_idr:
                    filtered.append(p)
                else:
                    low_volume_pairs.append(p)
            pairs = filtered
            if not pairs:
                # All pairs below threshold — fall back to unfiltered list to
                # avoid analysing nothing.
                logger.warning(
                    "All %d pairs are below MIN_VOLUME_IDR=%.0f; ignoring filter for this cycle",
                    len(low_volume_pairs),
                    self.config.min_volume_idr,
                )
                pairs = self._sort_pairs_by_priority(
                    self._all_pairs or [self.config.pair]
                )

        best_pair = pairs[0] if pairs else self.config.pair
        best_snapshot: Optional[Dict[str, Any]] = None
        best_score = -1.0
        # Track the best "hold" result separately so we can return it as the
        # fallback snapshot without making a redundant REST call at the end.
        best_hold_pair: Optional[str] = None
        best_hold_snapshot: Optional[Dict[str, Any]] = None
        best_hold_score = -1.0
        failed_pairs: List[str] = []
        skipped_pairs: List[str] = []
        insufficient_data_pairs: List[str] = []

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
            # Skip pairs where we could not build enough candles for reliable
            # indicators – treating them as "hold" would produce misleading
            # default values (RSI=50, MACD=0, BB=[0/0/0]).
            if snapshot.get("insufficient_data"):
                insufficient_data_pairs.append(pair)
                continue
            decision: StrategyDecision = snapshot["decision"]
            if decision.action == "hold":
                # Keep the best hold so we can return real data in the fallback
                # instead of triggering another REST round-trip.
                score = self._score_snapshot(snapshot)
                if score > best_hold_score:
                    best_hold_score = score
                    best_hold_pair = pair
                    best_hold_snapshot = snapshot
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

        if insufficient_data_pairs:
            logger.debug(
                "Skipped %d pairs with insufficient candle data (need ≥%d candles): %s",
                len(insufficient_data_pairs),
                self.config.min_candles,
                ",".join(insufficient_data_pairs[:5])
                + ("…" if len(insufficient_data_pairs) > 5 else ""),
            )

        if skipped_pairs:
            logger.debug(
                "Skipped %d pairs not in feed cache: %s",
                len(skipped_pairs),
                ",".join(skipped_pairs[:10]) + ("…" if len(skipped_pairs) > 10 else ""),
            )

        if low_volume_pairs:
            logger.debug(
                "Skipped %d pairs below MIN_VOLUME_IDR=%.0f",
                len(low_volume_pairs),
                self.config.min_volume_idr,
            )

        if failed_pairs:
            logger.warning("Skipped %s pairs due to errors: %s", len(failed_pairs), ",".join(failed_pairs))

        # ── Dynamic pair refresh ──────────────────────────────────────────────
        self._scan_cycle_count += 1
        refresh = self.config.dynamic_pairs_refresh_cycles
        if refresh > 0 and self._scan_cycle_count % refresh == 0:
            self._refresh_dynamic_pairs()

        if best_snapshot:
            return best_pair, best_snapshot

        if failed_pairs:
            message = "No pairs could be analyzed successfully"
            message = f"{message} (failed: {','.join(failed_pairs)})"
            raise RuntimeError(message)

        # Return the best "hold" result captured during the scan – this avoids
        # a redundant REST round-trip and shows real indicator values rather
        # than the default neutrals that a fresh re-analysis of an arbitrary
        # pair might produce.
        if best_hold_snapshot is not None and best_hold_pair is not None:
            return best_hold_pair, best_hold_snapshot

        # Last resort: re-analyse the highest-volume pair.  Log a warning if
        # every scanned pair lacked sufficient candle data so the operator can
        # diagnose the root cause (API issues, wrong TRADE_COUNT, etc.).
        if insufficient_data_pairs and not best_hold_snapshot:
            logger.warning(
                "All %d analysed pairs had insufficient candle data. "
                "Check the Indodax OHLC API or increase TRADE_COUNT (current: %d).",
                len(insufficient_data_pairs),
                self.config.trade_count,
            )
        return best_pair, self.analyze_market(best_pair)
