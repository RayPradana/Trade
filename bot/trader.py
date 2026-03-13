from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple, Iterable

import requests

from .analysis import (
    analyze_orderbook,
    analyze_trade_flow,
    analyze_trend,
    analyze_volatility,
    build_candles,
    candles_from_ohlc,
    derive_indicators,
    detect_flash_dump,
    detect_liquidity_sweep,
    detect_liquidity_trap,
    detect_liquidity_vacuum,
    detect_micro_trend,
    detect_orderbook_absorption,
    detect_rug_pull_risk,
    detect_smart_money_footprint,
    detect_spread_anomaly,
    detect_spread_expansion,
    detect_spoofing,
    detect_volume_acceleration,
    detect_whale_activity,
    detect_market_regime,
    interval_to_ohlc_tf,
    multi_timeframe_confirm,
    smart_entry_filter,
    LiquiditySweep,
    LiquidityTrap,
    LiquidityVacuum,
    MicroTrend,
    MomentumIndicators,
    MultiTimeframeResult,
    OrderbookAbsorption,
    RugPullRisk,
    SmartEntryResult,
    SmartMoneyFootprint,
    SpreadExpansion,
    SpoofingResult,
    TradeFlowResult,
    VolumeAcceleration,
    WhaleActivity,
    support_resistance,
)
from .config import BotConfig
from .persistence import StatePersistence
from .rate_limit import RateLimitedOrderQueue
from .realtime import MultiPairFeed, RealtimeFeed
from .grid import GridOrder, GridPlan, build_grid_plan
from .indodax_client import IndodaxClient
from .strategies import StrategyDecision, adaptive_max_positions, adaptive_risk_per_trade, make_trade_decision
from .tracking import MultiPositionManager, PortfolioTracker
from .journal import TradeJournal
from .market_data import MarketDataFeed

logger = logging.getLogger(__name__)

_WHALE_EVENT_MAX = 100
_ADAPTIVE_CONF_STRENGTH_THRESHOLD = 0.01
_ADAPTIVE_CONF_STRONG = 0.30
_ADAPTIVE_CONF_WEAK = 0.40

# Extra candles requested beyond slow_window when fetching OHLC history so
# that EMA-based indicators (MACD, Bollinger) have enough warm-up data.
_OHLC_BUFFER_CANDLES = 100

# Number of order-book levels fetched for execution guards (slippage, spread,
# sell-wall, liquidity depth).  Kept as a constant so all guards are consistent.
_EXECUTION_DEPTH_LEVELS = 20


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
        # Apply private API response cache TTLs from config (no-op for stub clients).
        _configure_caches = getattr(self.client, "configure_caches", None)
        if callable(_configure_caches):
            _configure_caches(
                account_info_ttl=config.account_info_cache_ttl,
                open_orders_ttl=config.open_orders_cache_ttl,
            )
        self._all_pairs: Optional[List[str]] = None
        self._scan_offset: int = 0  # rotating window index for pairs_per_cycle
        self._multi_feed: Optional[MultiPairFeed] = None
        self._scan_cycle_count: int = 0  # total completed full-scan cycles (for dynamic refresh)
        # ── Pump-protection price history ────────────────────────────────────
        # Per-pair dict of (unix_timestamp, price) buffers.  Keyed by pair name
        # so that prices from different pairs are never compared against each
        # other (which would produce nonsensical percentage rises like 46M%).
        # Only populated when pump_protection_pct > 0.
        self._price_history: Dict[str, List[Tuple[float, float]]] = {}
        self.tracker = PortfolioTracker(
            initial_capital=config.initial_capital,
            target_profit_pct=config.target_profit_pct,
            max_loss_pct=config.max_loss_pct,
            continue_after_target=config.continue_after_target,
        )
        # ── Multi-position manager ────────────────────────────────────────────
        self.multi_manager: Optional[MultiPositionManager] = None
        if config.multi_position_enabled:
            self.multi_manager = MultiPositionManager(
                initial_capital=config.initial_capital,
                max_positions=config.multi_position_max,
                target_profit_pct=config.target_profit_pct,
                max_loss_pct=config.max_loss_pct,
                continue_after_target=config.continue_after_target,
            )
        # Whale wallet tracking (recent events, limited to prevent unbounded growth)
        self._whale_events: Deque[Dict[str, object]] = deque(maxlen=_WHALE_EVENT_MAX)
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
        # ── Per-pair position feeds (realtime for actively held pairs) ────────
        # When a position is opened on a pair that differs from self.config.pair
        # a dedicated RealtimeFeed is started so that holding-loop analysis uses
        # WebSocket data rather than polling REST for every held pair.
        self._position_feeds: Dict[str, RealtimeFeed] = {}
        # Start feeds for any positions that were already restored from state.
        if config.real_time:
            for _restored_pair in list(self.active_positions.keys()):
                if _restored_pair != self.config.pair:
                    self._ensure_position_feed(_restored_pair)
        # New feature state
        self._consecutive_errors: int = 0
        self._circuit_breaker_until: float = 0.0
        self._volatility_cooldown_until: float = 0.0
        self.journal: Optional[TradeJournal] = None
        self._spread_history: Dict[str, List[float]] = {}
        # Previous depth snapshot per pair for orderbook absorption detection
        self._prev_depth: Dict[str, Dict[str, list]] = {}
        # Per-pair cooldown: maps pair → unix timestamp of last executed trade
        self._pair_last_trade: Dict[str, float] = {}
        # ── Per-pair minimum order cache ─────────────────────────────────────
        # Populated by _ensure_pair_min_order_cache().  Tracks how many scan
        # cycles have elapsed since the last cache refresh.
        self._pair_min_order_cache_cycles: int = 0
        # ── Per-pair OHLC candle cache ────────────────────────────────────────
        # Maps pair → (fetch_timestamp, candles).  Used during scan to avoid
        # calling get_ohlc() on every cycle for every pair, which would
        # saturate the REST rate limit.  Entries are reused until
        # config.scan_candle_cache_seconds have elapsed.
        self._candle_cache: Dict[str, Tuple[float, List[Any]]] = {}

        # ── Market Data Feed ──────────────────────────────────────────────────
        self.market_data_feed: Optional[MarketDataFeed] = None
        if config.market_data_enabled:
            self.market_data_feed = MarketDataFeed.from_config(config)

    def _min_confidence_threshold(self, snapshot: Dict[str, Any]) -> float:
        """Return the effective minimum confidence threshold for a snapshot.

        Starts from the configured minimum (respecting confidence-tier skip when
        enabled) and lowers the threshold when a trend strength value is present
        to make the bot more responsive during stronger momentum.
        """
        base_min = (
            min(self.config.min_confidence, self.config.confidence_tier_skip)
            if self.config.confidence_position_sizing_enabled
            else self.config.min_confidence
        )
        trend = snapshot.get("trend")
        strength = getattr(trend, "strength", None)
        if strength is not None:
            adaptive_min = (
                _ADAPTIVE_CONF_STRONG
                if strength > _ADAPTIVE_CONF_STRENGTH_THRESHOLD
                else _ADAPTIVE_CONF_WEAK
            )
            base_min = min(base_min, adaptive_min)
        return base_min

    # ── Multi-position helpers ─────────────────────────────────────────────

    def _active_tracker(self, pair: str) -> PortfolioTracker:
        """Return the :class:`PortfolioTracker` that owns *pair*'s position.

        In **single-position mode** (``multi_manager is None``) this always
        returns ``self.tracker``.

        In **multi-position mode** it returns the pair-specific sub-tracker
        from :attr:`multi_manager` when one exists, otherwise returns a
        lightweight empty tracker (``base_position=0``) so that position-based
        guards (stop-loss, time-exit, etc.) do not fire on pairs that have not
        been entered yet.
        """
        if self.multi_manager is not None:
            t = self.multi_manager.get_tracker(pair)
            if t is not None:
                return t
            # Return a zero-position placeholder so guards don't misfiring.
            # Capital is allocated only when the BUY actually executes.
            return PortfolioTracker(
                initial_capital=0.0,
                target_profit_pct=0.0,
                max_loss_pct=0.0,
            )
        return self.tracker

    def _ensure_position_feed(self, pair: str) -> None:
        """Start a dedicated :class:`RealtimeFeed` for a held-position pair.

        Called when a new position is opened on *pair* so that the holding
        loop can use WebSocket market data instead of polling REST on every
        cycle.  No-op when real-time mode is disabled, when the pair is
        already covered by the primary :attr:`realtime` feed, or when a feed
        for this pair is already running.
        """
        if not self.config.real_time:
            return
        if pair == self.config.pair and self.realtime is not None:
            return  # primary feed already covers this pair
        if pair in self._position_feeds:
            return  # already running
        feed = RealtimeFeed(
            pair=pair,
            client=self.client,
            websocket_url=self.config.websocket_url,
            poll_interval=max(0.5, float(self.config.interval_seconds)),
            websocket_enabled=self.config.websocket_enabled,
            subscribe_message=self.config.websocket_subscribe_message,
        )
        feed.start()
        self._position_feeds[pair] = feed
        logger.info("Started realtime feed for held position: %s", pair)

    def _remove_position_feed(self, pair: str) -> None:
        """Stop and remove the :class:`RealtimeFeed` for a closed position.

        Called when a position on *pair* is fully closed so that the
        background WebSocket thread is cleaned up promptly.
        """
        feed = self._position_feeds.pop(pair, None)
        if feed is not None:
            feed.stop()
            logger.debug("Stopped realtime feed for closed position: %s", pair)

    @property
    def active_positions(self) -> Dict[str, PortfolioTracker]:
        """Return ``{pair: tracker}`` for all currently held positions.

        In single-position mode returns the primary pair if held, otherwise
        an empty dict.  In multi-position mode delegates to
        :attr:`MultiPositionManager.active_positions`.
        """
        if self.multi_manager is not None:
            return self.multi_manager.active_positions
        if self.tracker.base_position > 0 or getattr(self.tracker, "has_pending_buy", False):
            return {self.config.pair: self.tracker}
        return {}

    def at_max_positions(self) -> bool:
        """``True`` when no additional positions can be opened.

        In single-position mode this is ``True`` when one position is already
        held (the classic "one trade at a time" behaviour).  In multi-position
        mode it checks against :attr:`~MultiPositionManager.max_positions`.
        """
        if self.multi_manager is not None:
            return not self.multi_manager.can_open_position()
        return self.tracker.base_position > 0 or getattr(self.tracker, "has_pending_buy", False)

    def portfolio_snapshot(self, pair: str, price: float) -> Dict[str, object]:
        """Return a portfolio dict suitable for logging and display.

        In **multi-position mode** returns aggregate stats across the entire
        multi-position portfolio (total equity = unallocated cash + mark-to-market
        value of all open positions), so callers always see the overall bot
        financial state rather than a zero-placeholder for pairs not yet entered.

        In **single-position mode** delegates to the primary tracker's
        :meth:`~PortfolioTracker.as_dict`.
        """
        if self.multi_manager is not None:
            # Build a price map for mark-to-market valuation.
            # Use the current scan price for the scanned pair; fall back to
            # avg_cost for other held pairs so we don't need extra API calls.
            prices: Dict[str, float] = {pair: price}
            for p, t in self.multi_manager.active_positions.items():
                if p not in prices:
                    prices[p] = t.avg_cost or 0.0
            equity = self.multi_manager.total_equity(prices)
            cash = self.multi_manager.cash
            pnl = self.multi_manager.total_realized_pnl()
            total_trades = sum(t.trade_count for t in self.multi_manager._trackers.values())
            winning = sum(
                round(t.trade_count * t.win_rate)
                for t in self.multi_manager._trackers.values()
            )
            win_rate = winning / total_trades if total_trades > 0 else 0.0
            initial = self.multi_manager.initial_capital
            target_profit_pct = getattr(self.multi_manager, "_target_profit_pct", 0.0)
            max_loss_pct = getattr(self.multi_manager, "_max_loss_pct", 0.0)
            pb = max(0.0, equity - initial)
            peak_pb = max(pb, getattr(self.multi_manager, "_peak_profit_buffer", pb))
            pb_dd = max(0.0, (peak_pb - pb) / peak_pb) if peak_pb > 0 else 0.0
            return {
                "equity": equity,
                "cash": cash,
                "base_position": 0.0,
                "realized_pnl": pnl,
                "trade_count": total_trades,
                "win_rate": win_rate,
                "trailing_stop": None,
                "target_equity": initial * (1 + target_profit_pct),
                "min_equity": initial * (1 - max_loss_pct),
                "avg_cost": 0.0,
                "principal": initial,
                "profit_buffer": pb,
                "effective_capital": equity,
                "peak_profit_buffer": peak_pb,
                "profit_buffer_drawdown": round(pb_dd, 4),
                "tp_activated": False,
                "trailing_tp_stop": None,
            }
        return self.tracker.as_dict(price)

    # ------------------------------------------------------------------
    # Auto-resume helpers
    # ------------------------------------------------------------------

    def _ensure_pair_min_order_cache(self) -> None:
        """Lazily load (and optionally refresh) the per-pair minimum order cache.

        Called once at the start of each scan cycle.  The first call always
        triggers a ``/api/pairs`` fetch.  Subsequent calls only refresh when
        ``pair_min_order_refresh_cycles > 0`` and the configured number of
        cycles has elapsed, OR when the TTL-based stale flag is set on the
        client (i.e. the cache is older than ``_pair_min_order_cache_ttl``).
        """
        if not self.config.pair_min_order_cache_enabled:
            return
        if not hasattr(self.client, "load_pair_min_orders"):
            return
        refresh = self.config.pair_min_order_refresh_cycles
        cached: dict = getattr(self.client, "_pair_min_order", {})
        already_loaded = bool(cached)
        cycle_due = refresh > 0 and self._pair_min_order_cache_cycles >= refresh
        ttl_stale = hasattr(self.client, "is_pair_min_order_cache_stale") and self.client.is_pair_min_order_cache_stale()
        if not already_loaded or cycle_due or ttl_stale:
            self.client.load_pair_min_orders()
            self._pair_min_order_cache_cycles = 0
        else:
            self._pair_min_order_cache_cycles += 1

    def _restore_whale_events(self, state: Dict[str, Any]) -> None:
        events = state.get("whale_events")
        if isinstance(events, list):
            self._whale_events = deque(events[-_WHALE_EVENT_MAX:], maxlen=_WHALE_EVENT_MAX)
        else:
            self._whale_events = deque(maxlen=_WHALE_EVENT_MAX)

    def _try_restore_state(self) -> None:
        """Load persisted state on startup and restore PortfolioTracker.

        State is only restored when:
        - The state file exists and is valid JSON
        - The saved ``dry_run`` flag matches the current config (prevents mixing
          virtual and live state after the user toggles DRY_RUN)
        - The saved position is > 0 (a position=0 file is stale and cleared)

        In multi-position mode the state may contain a ``multi_positions`` dict
        with per-pair tracker snapshots.  Those are restored into
        :attr:`multi_manager` so that :attr:`active_positions` reflects held
        pairs immediately on the next cycle and the main holding loop monitors
        them correctly.  A legacy single-position state file (no
        ``multi_positions`` key) is migrated transparently: the single entry is
        registered in ``multi_manager`` as if it were opened there.
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
        self._restore_whale_events(state)

        # ── Multi-position restore path ───────────────────────────────────────
        if self.multi_manager is not None:
            multi_positions = state.get("multi_positions")
            pool_cash = float(state.get("multi_cash", self.multi_manager.cash))
            pair = state.get("pair")

            if multi_positions:
                # New format: restore all per-pair positions into multi_manager.
                restored_pairs = self.multi_manager.restore_from_state(
                    multi_positions, pool_cash  # type: ignore[arg-type]
                )
                if restored_pairs:
                    self.restored_pair = restored_pairs[0]
                    logger.info(
                        "🔄 Resumed %d multi-position(s): %s",
                        len(restored_pairs),
                        ", ".join(restored_pairs),
                    )
                    for p in restored_pairs:
                        t = self.multi_manager.get_tracker(p)
                        if t is not None:
                            logger.info(
                                "   ├─ %s: pos=%.8f  avg_cost=%s  pnl=%s",
                                p,
                                t.base_position,
                                t.avg_cost,
                                t.realized_pnl,
                            )
                    if not self.config.dry_run:
                        self._reconcile_with_api(restored_pairs[0])
                else:
                    # multi_positions was non-empty but all entries had
                    # base_position=0 (stale/corrupted state).  Clear the file
                    # and reset the cash pool so the bot starts fresh.
                    logger.warning(
                        "Stale multi-position state (no valid positions found) — "
                        "clearing state and resetting cash to initial_capital"
                    )
                    self.persistence.clear()
                    self.multi_manager.cash = self.multi_manager.initial_capital
                return

            # Backward-compat: single-position save format used with multi-position
            # mode.  Migrate the lone entry into multi_manager so it is picked up
            # by active_positions on the next cycle.
            portfolio = state.get("portfolio")
            if portfolio is None:
                return
            saved_pos = float((portfolio.get("base_position") or 0))
            if saved_pos <= 0:
                self.persistence.clear()
                return
            primary_pair = str(pair) if pair else self.config.pair
            self.multi_manager.restore_from_state(
                {primary_pair: portfolio},  # type: ignore[arg-type]
                pool_cash,
            )
            self.restored_pair = primary_pair
            t = self.multi_manager.get_tracker(primary_pair)
            logger.info(
                "🔄 Resumed state (compat): pair=%s  pos=%.8f  avg_cost=%s  pnl=%s",
                primary_pair,
                t.base_position if t else 0,
                t.avg_cost if t else 0,
                t.realized_pnl if t else 0,
            )
            if not self.config.dry_run:
                self._reconcile_with_api(primary_pair)
            return

        # ── Single-position restore path (original behaviour) ─────────────────
        portfolio = state.get("portfolio")
        if portfolio is None:
            return
        # Check position BEFORE mutating tracker to avoid partial load of stale state
        saved_pos = float((portfolio.get("base_position") or 0))
        pending = portfolio.get("pending_orders") or []
        has_pending = isinstance(pending, list) and len(pending) > 0
        if saved_pos <= 0 and not has_pending:
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

    def set_journal(self, journal: TradeJournal) -> None:
        self.journal = journal

    def _validate_balance(self, pair: str, action: str, amount: float, price: float) -> bool:
        if not self.config.balance_check_enabled:
            return True
        try:
            info = self.client.get_account_info()
            balance_dict = (info.get("return") or {}).get("balance") or {}
            if action == "buy":
                idr_needed = amount * price
                available_idr = float(balance_dict.get("idr") or "0")
                if available_idr < idr_needed:
                    logger.warning(
                        "Balance check FAILED for buy on %s: need IDR %.2f, have IDR %.2f",
                        pair, idr_needed, available_idr,
                    )
                    return False
            elif action == "sell":
                base_coin = pair.split("_")[0].lower()
                available_coin = float(balance_dict.get(base_coin) or "0")
                if available_coin < amount:
                    logger.warning(
                        "Balance check FAILED for sell on %s: need %.8f %s, have %.8f",
                        pair, amount, base_coin, available_coin,
                    )
                    return False
        except Exception as exc:
            logger.warning("Balance check failed for %s: %s — proceeding anyway", pair, exc)
        return True

    def _check_volatility_cooldown(self, current_price: float, pair: str) -> None:
        if self.config.volatility_cooldown_pct <= 0:
            return
        history = self._price_history.get(pair, [])
        if len(history) < 2:
            return
        window_start = history[0][1]
        if window_start <= 0:
            return
        spike = abs(current_price - window_start) / window_start
        if spike >= self.config.volatility_cooldown_pct:
            self._volatility_cooldown_until = time.time() + self.config.volatility_cooldown_seconds
            logger.warning(
                "Volatility cooldown triggered on %s: spike=%.2f%% — pausing for %.0fs",
                pair, spike * 100, self.config.volatility_cooldown_seconds,
            )

    def _reconcile_with_api(self, pair: str) -> None:
        """Cross-check the saved position against the real Indodax account balance.

        Called once on startup when ``dry_run=False`` and a saved position exists.
        If the real coin balance differs from the saved value by more than 5 %
        the tracker is updated to match reality so the bot never acts on stale data.
        """
        try:
            info = self.client.get_account_info()
            # Indodax returns {"success": 1, "return": {"balance": {"btc": "0.001", ...}}}
            if info.get("success") != 1:
                logger.warning("Reconciliation skipped for %s: API success=%s", pair, info.get("success"))
                return
            balance_dict = (info.get("return") or {}).get("balance") or {}
            base_coin = pair.split("_")[0].lower()
            if base_coin not in balance_dict:
                logger.warning("Reconciliation skipped for %s: balance key %s missing", pair, base_coin)
                return
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
        """Persist the current PortfolioTracker state to disk (fire-and-forget).

        In multi-position mode all active per-pair trackers are written to a
        ``multi_positions`` dict so that :meth:`_try_restore_state` can restore
        every open position on the next startup, not just the most recently
        traded one.
        """
        try:
            payload: dict = {
                "portfolio": self.tracker.to_state(),
                "pair": pair,
                "dry_run": self.config.dry_run,
                "whale_events": list(self._whale_events),
            }
            if self.multi_manager is not None:
                payload["multi_positions"] = {
                    p: t.to_state()
                    for p, t in self.multi_manager._trackers.items()
                    if t.base_position > 0 or t.has_pending_buy
                }
                payload["multi_cash"] = self.multi_manager.cash
            self.persistence.save(payload)
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
        if self.config.pair_cooldown_seconds > 0:
            now = time.time()
            self._pair_last_trade[pair] = now
            # Prune entries that are already past their cooldown window to prevent
            # the dict from growing without bound over long bot runs.
            cutoff = now - self.config.pair_cooldown_seconds
            expired = [p for p, ts in self._pair_last_trade.items() if ts < cutoff]
            for p in expired:
                del self._pair_last_trade[p]
        if self.multi_manager is not None:
            # Multi-position: save while any position remains; clear when all closed.
            if self.multi_manager.position_count() > 0:
                self._save_state(pair)
            else:
                self._clear_state()
        else:
            if self.tracker.base_position <= 0 and not getattr(self.tracker, "has_pending_buy", False):
                self._clear_state()
            else:
                self._save_state(pair)

    def cleanup_stale_data(self) -> None:
        """Remove stale entries from per-pair caches to prevent memory leaks.

        Called periodically by the autonomous task scheduler to prune:
        - Price history entries for pairs no longer in the watchlist
        - Spread history entries for inactive pairs
        - Previous depth snapshots for inactive pairs
        - Expired candle cache entries
        """
        active_pairs = set()
        if self._all_pairs:
            active_pairs.update(self._all_pairs)
        # Always keep pairs with open positions
        active_pairs.update(self.active_positions.keys())
        active_pairs.add(self.config.pair)

        # Prune price history for pairs no longer in watchlist
        stale_price = [p for p in self._price_history if p not in active_pairs]
        for p in stale_price:
            del self._price_history[p]

        # Prune spread history for inactive pairs
        stale_spread = [p for p in self._spread_history if p not in active_pairs]
        for p in stale_spread:
            del self._spread_history[p]

        # Prune previous depth snapshots
        stale_depth = [p for p in self._prev_depth if p not in active_pairs]
        for p in stale_depth:
            del self._prev_depth[p]

        # Prune expired candle cache entries (older than 2× the configured TTL)
        now = time.time()
        cache_ttl = getattr(self.config, "scan_candle_cache_seconds", 120)
        stale_candles = [
            p for p, (ts, _) in self._candle_cache.items()
            if now - ts > cache_ttl * 2
        ]
        for p in stale_candles:
            del self._candle_cache[p]

        if stale_price or stale_spread or stale_depth or stale_candles:
            logger.debug(
                "Cleanup: removed price=%d spread=%d depth=%d candle=%d stale entries",
                len(stale_price), len(stale_spread), len(stale_depth), len(stale_candles),
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

    def _extract_volume_idr(self, ticker: Dict[str, Any]) -> float:
        """Extract 24-h IDR trading volume from a raw ticker dict."""
        src = ticker.get("ticker", ticker)
        for key in ("vol_idr", "volume_idr", "volume", "vol"):
            val = src.get(key)
            if val is not None:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass
        return 0.0

    def _extract_trade_count_24h(self, ticker: Dict[str, Any]) -> int:
        """Extract 24-h trade count from a raw ticker dict."""
        src = ticker.get("ticker", ticker)
        for key in ("trade_count", "count", "trades_24h"):
            val = src.get(key)
            if val is not None:
                try:
                    return int(val)
                except (TypeError, ValueError):
                    pass
        return 0

    def _fetch_candles(
        self,
        pair: str,
        trades: List[Dict[str, Any]],
        *,
        use_cache: bool = False,
    ) -> List[Any]:
        """Return OHLCV candles for *pair*.

        Priority (highest to lowest):

        1. **WS trade buffer** – When the ``MultiPairFeed`` has accumulated
           enough real-time trades for *pair*, candles are built directly from
           that buffer via :func:`~bot.analysis.build_candles`.  This path
           requires **no REST call and no cache**, giving truly live indicator
           data that updates with every WS push.

        2. **Passed ``trades`` list** – Candles built from the trades already
           fetched for this analysis cycle (e.g. from a position-feed snapshot).

        3. **REST OHLC endpoint** – ``/tradingview/history_v2`` is tried when
           neither WS nor live trades provide enough history.  The result may be
           stored in the in-memory cache when *use_cache* is ``True`` and
           ``config.scan_candle_cache_seconds > 0`` to avoid redundant REST calls
           across repeated scan cycles for the same pair.

        The legacy REST path is used only as a last resort so that the bot can
        still compute indicators for pairs that are not yet receiving WS trade
        data (e.g. on first startup before the feed is fully seeded).
        """
        min_candles = max(2, self.config.slow_window // 2)

        # ── 1. Try real-time WS trade buffer from MultiPairFeed ─────────────
        if self._multi_feed is not None:
            ws_trades = self._multi_feed.get_trades(pair)
            if ws_trades and len(ws_trades) >= min_candles:
                ws_candles = build_candles(
                    ws_trades,
                    interval_seconds=self.config.interval_seconds,
                    limit=max(200, self.config.slow_window + _OHLC_BUFFER_CANDLES),
                )
                if len(ws_candles) >= min_candles:
                    logger.debug(
                        "WS candles for %s: %d candles from %d trades",
                        pair,
                        len(ws_candles),
                        len(ws_trades),
                    )
                    return ws_candles

        # ── 2. Build from trades already fetched this cycle ──────────────────
        if trades and len(trades) >= min_candles:
            live_candles = build_candles(
                trades,
                interval_seconds=self.config.interval_seconds,
                limit=200,
            )
            if len(live_candles) >= min_candles:
                return live_candles

        # ── 3. REST OHLC (fallback only) ─────────────────────────────────────
        ttl = self.config.scan_candle_cache_seconds
        if use_cache and ttl > 0:
            cached = self._candle_cache.get(pair)
            if cached is not None:
                ts, candles = cached
                if time.time() - ts < ttl:
                    logger.debug(
                        "Candle cache hit for %s (age=%.0fs)", pair, time.time() - ts
                    )
                    return candles
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
                if use_cache and ttl > 0:
                    self._candle_cache[pair] = (time.time(), candles)
                return candles
        except Exception as exc:
            logger.debug(
                "OHLC fetch failed for %s (%s); falling back to trades", pair, exc
            )
        # Legacy fallback: build candles by bucketing raw trade ticks.
        result = build_candles(
            trades, interval_seconds=self.config.interval_seconds, limit=200
        )
        # Cache the fallback result too so repeated scan calls don't hammer the
        # OHLC endpoint for pairs that genuinely have no data within the TTL.
        if use_cache and ttl > 0:
            self._candle_cache[pair] = (time.time(), result)
        return result

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

    def _record_price(self, pair: str, price: float) -> None:
        """Append *(now, price)* to the per-pair pump-detection history buffer.

        Each pair maintains its own independent price buffer so that prices
        from different pairs (e.g. BTC/IDR at ~1.5B IDR vs a 100-IDR altcoin)
        are **never** compared against each other, which would produce wildly
        incorrect rise percentages.

        Old entries outside the lookback window are pruned to keep the buffer
        small.  This method is a no-op when both pump protection and fake-pump
        reversal detection are disabled.
        """
        if self.config.pump_protection_pct <= 0 and self.config.fake_pump_reversal_pct <= 0:
            return
        now = time.time()
        buf = self._price_history.setdefault(pair, [])
        buf.append((now, price))
        # Prune entries older than the lookback window (keep a small tail buffer
        # so the oldest sample inside the window is always available)
        cutoff = now - self.config.pump_lookback_seconds
        while len(buf) > 1 and buf[0][0] < cutoff:
            buf.pop(0)

    def _record_whale_event(self, pair: str, whale: WhaleActivity, price: float) -> None:
        """Track recent whale detections for observability / debugging."""
        if not whale.detected:
            return
        self._whale_events.append(
            {
                "pair": pair,
                "side": whale.side,
                "ratio": round(whale.ratio, 3),
                "price": price,
                "ts": time.time(),
            }
        )

    def whale_events(self) -> List[Dict[str, object]]:
        """Return recent whale-detection events (newest last)."""
        return list(self._whale_events)

    def _is_pumped(self, pair: str, current_price: float) -> bool:
        """Return *True* when *pair*'s price has risen above the pump threshold.

        Compares *current_price* to the oldest recorded price for *pair* inside
        the ``pump_lookback_seconds`` window.  Returns ``False`` when pump
        protection is disabled or when there is insufficient price history for
        the pair.
        """
        if self.config.pump_protection_pct <= 0:
            return False
        buf = self._price_history.get(pair)
        if not buf:
            return False
        oldest_price = buf[0][1]
        if oldest_price <= 0:
            return False
        rise = (current_price - oldest_price) / oldest_price
        return rise >= self.config.pump_protection_pct

    def _is_fake_pump(self, pair: str, current_price: float) -> bool:
        """Detect a pump-then-dump (fake pump) pattern in the price history.

        On Indodax, manipulative actors frequently spike a price by several
        percent and then dump it back down within ≈20 seconds.  This method
        looks for that two-phase pattern inside the per-pair rolling buffer:

        * **Phase 1 – spike**: the in-window peak price rose ≥
          ``pump_protection_pct`` versus the oldest buffer entry.
        * **Phase 2 – reversal / dump**: *current_price* has since fallen ≥
          ``fake_pump_reversal_pct`` from that peak.

        Both conditions must hold simultaneously to return ``True``.

        Returns ``False`` when either feature is disabled, there is
        insufficient history, or the pattern is not present.
        """
        if self.config.pump_protection_pct <= 0 or self.config.fake_pump_reversal_pct <= 0:
            return False
        buf = self._price_history.get(pair)
        if not buf or len(buf) < 2:
            return False
        oldest_price = buf[0][1]
        if oldest_price <= 0:
            return False
        # Find the highest price recorded within the window.
        peak_price = max(entry[1] for entry in buf)
        # Phase 1: was there a genuine spike (pump)?
        spike = (peak_price - oldest_price) / oldest_price
        if spike < self.config.pump_protection_pct:
            return False
        # Phase 2: has current price reversed (dump) significantly from peak?
        if peak_price <= 0:
            return False
        reversal = (peak_price - current_price) / peak_price
        return reversal >= self.config.fake_pump_reversal_pct

    def _pair_composite_score(self, pair: str) -> float:
        """Return a composite ranking score for *pair* used to select the top-N watchlist.

        The score combines 24-h IDR trading volume with 24-h price volatility
        (high-low range as a fraction of last price).  This surfaces pairs that
        are both liquid *and* actively moving — the best candidates for
        short-term trading:

            score = vol_idr × ((high − low) / last)

        A pair with no cached ticker or zero/invalid price data scores 0.0 and
        is placed at the end (excluded from the top-N watchlist).
        """
        if self._multi_feed is None:
            return 0.0
        ticker = self._multi_feed.get_ticker(pair)
        if ticker is None:
            return 0.0
        vol_idr = 0.0
        for key in ("vol_idr", "idr_volume", "volume"):
            val = ticker.get(key)
            if val is not None:
                try:
                    vol_idr = float(val)
                    break
                except (ValueError, TypeError):
                    pass
        try:
            last = float(ticker.get("last") or ticker.get("last_price") or 0)
            high = float(ticker.get("high") or 0)
            low = float(ticker.get("low") or 0)
            if last > 0 and high > low:
                daily_range_pct = (high - low) / last
                return vol_idr * daily_range_pct
        except (ValueError, TypeError):
            pass
        # Fall back to volume-only when price data is missing/invalid
        return vol_idr

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

    def _liquidity_depth_idr(self, depth: Dict[str, Any], price: float) -> Optional[float]:
        """Return the total IDR value of the top-N bid and ask levels.

        Used to filter out thin markets where the effective spread would make
        a trade unprofitable.

        Returns ``None`` when the depth dict contains no orderbook keys at all
        (e.g. the API returned an error dict or an unexpected format).  The
        caller **must** treat ``None`` as "data unavailable" and skip the
        liquidity check rather than blocking the trade, to avoid false positives
        when the depth endpoint fails transiently.

        Returns ``0.0`` only when the ``buy`` and ``sell`` keys exist but the
        levels are genuinely empty (no active orders).

        Returns a positive float when levels are present and parseable.
        """
        has_buy = "buy" in depth
        has_sell = "sell" in depth
        if not has_buy and not has_sell:
            # Depth response missing both orderbook keys — treat as unavailable,
            # not as a thin market (the API may have returned an error dict).
            logger.debug(
                "_liquidity_depth_idr: depth response has no buy/sell keys "
                "(keys=%s) — skipping liquidity check",
                list(depth.keys())[:5],
            )
            return None
        try:
            bids = (depth.get("buy") or [])[:_EXECUTION_DEPTH_LEVELS]
            asks = (depth.get("sell") or [])[:_EXECUTION_DEPTH_LEVELS]
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
        """Replace the pair watchlist with the top-N pairs by composite score.

        Called every ``dynamic_pairs_refresh_cycles`` full-scan cycles when the
        feature is enabled (``dynamic_pairs_refresh_cycles > 0``).  Reads
        current data from the multi-pair feed cache to rank pairs by a composite
        score of 24-h IDR volume × daily price range (volatility proxy) and
        updates ``_all_pairs`` in-place so that the next scan uses the fresh
        watchlist.

        Using volume × volatility (instead of volume alone) ensures the
        watchlist contains pairs that are both liquid *and* actively moving,
        which produces more consistent trading opportunities and avoids
        scanning illiquid or stale coins.

        Optional pre-filters applied before ranking:

        * ``top_volume_min_volume_idr`` – exclude pairs whose 24-h IDR volume
          is below this threshold so ultra-low-liquidity coins never enter the
          watchlist regardless of their volatility score.
        * ``top_volume_min_price_change_24h_pct`` – exclude stagnant pairs
          whose 24-h absolute price change is below this fraction.  This drops
          dead coins (e.g. DENT at 4 IDR sitting unchanged for hours) before
          the ranking step.

        The refresh is best-effort: any exception is logged and the existing
        watchlist is kept unchanged.
        """
        if self._multi_feed is None:
            return
        try:
            all_known = list(self._multi_feed._cache.keys())
            if not all_known:
                return

            # ── Pre-filter: minimum volume ────────────────────────────────────
            min_vol = self.config.top_volume_min_volume_idr
            min_chg = self.config.top_volume_min_price_change_24h_pct
            min_price = self.config.min_buy_price_idr
            candidates = all_known
            dropped_vol: List[str] = []
            dropped_stagnant: List[str] = []
            dropped_low_price: List[str] = []

            if min_vol > 0 or min_chg > 0 or min_price > 0:
                filtered: List[str] = []
                for p in candidates:
                    ticker = self._multi_feed.get_ticker(p)
                    if ticker is None:
                        filtered.append(p)  # no data → keep and let score sort it out
                        continue

                    # Price check – only drop cheap coins that are also stuck or dead.
                    # A coin is "stuck" when its 24h high equals its low (no movement
                    # at all).  A coin is "dead" when it has no volume.  Active cheap
                    # coins (e.g. SHIB at 50 IDR but with regular trades) are kept on
                    # the watchlist and face an orderbook quality check at execution.
                    if min_price > 0:
                        try:
                            last_price = float(ticker.get("last") or ticker.get("last_price") or 0)
                            if 0 < last_price < min_price:
                                high = float(ticker.get("high") or 0)
                                low = float(ticker.get("low") or 0)
                                vol_idr = float(
                                    ticker.get("vol_idr")
                                    or ticker.get("idr_volume")
                                    or ticker.get("volume")
                                    or 0
                                )
                                is_stuck = high > 0 and low > 0 and high == low
                                is_dead = vol_idr <= 0
                                if is_stuck or is_dead:
                                    dropped_low_price.append(p)
                                    continue
                        except (ValueError, TypeError):
                            pass  # price missing → keep pair

                    # Volume check
                    if min_vol > 0:
                        vol = 0.0
                        for key in ("vol_idr", "idr_volume", "volume"):
                            val = ticker.get(key)
                            if val is not None:
                                try:
                                    vol = float(val)
                                    break
                                except (ValueError, TypeError):
                                    pass
                        if vol < min_vol:
                            dropped_vol.append(p)
                            continue

                    # Price-change (momentum) check
                    if min_chg > 0:
                        try:
                            last = float(ticker.get("last") or ticker.get("last_price") or 0)
                            open_ = float(ticker.get("open") or ticker.get("open_price") or 0)
                            if last > 0 and open_ > 0:
                                chg = abs(last - open_) / open_
                                if chg < min_chg:
                                    dropped_stagnant.append(p)
                                    continue
                        except (ValueError, TypeError):
                            pass  # data missing → keep pair

                    filtered.append(p)
                candidates = filtered

            if dropped_low_price:
                logger.debug(
                    "Top-volume selector: dropped %d stuck/dead cheap pairs (price < Rp%.0f)",
                    len(dropped_low_price),
                    min_price,
                )
            if dropped_vol:
                logger.debug(
                    "Top-volume selector: dropped %d low-volume pairs (< Rp%.0f)",
                    len(dropped_vol),
                    min_vol,
                )
            if dropped_stagnant:
                logger.debug(
                    "Top-volume selector: dropped %d stagnant pairs (< %.2f%% 24h change)",
                    len(dropped_stagnant),
                    min_chg * 100,
                )

            if not candidates:
                logger.warning(
                    "Top-volume selector: all pairs filtered out — keeping existing watchlist"
                )
                return

            # ── Rank remaining candidates by composite score ──────────────────
            ranked = sorted(candidates, key=self._pair_composite_score, reverse=True)
            top_n = self.config.dynamic_pairs_top_n
            new_pairs = ranked[:top_n] if top_n > 0 else ranked
            if new_pairs:
                self._all_pairs = new_pairs
                # Subscribe to real-time orderbook + trades WS channels for the
                # updated watchlist so scan-phase analysis has live depth data.
                if self._multi_feed is not None:
                    self._multi_feed.subscribe_depth_pairs(new_pairs)
                logger.info(
                    "Top-volume selector: watchlist updated → %d pairs "
                    "(top=%d, filtered_vol=%d, filtered_stagnant=%d, filtered_low_price=%d): %s",
                    len(new_pairs),
                    top_n,
                    len(dropped_vol),
                    len(dropped_stagnant),
                    len(dropped_low_price),
                    ", ".join(new_pairs[:10]) + (" …" if len(new_pairs) > 10 else ""),
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

        # For small portfolios, collapse to a single-step entry so that
        # individual staged tranches don't fall below the exchange minimum
        # order size and cause the entire trade to be skipped.
        min_eq = self.config.staged_entry_min_equity
        if min_eq > 0 and self.tracker.cash < min_eq:
            max_steps = 1

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
        skip_depth: bool = False,
        skip_trades: bool = False,
    ) -> Dict[str, Any]:
        pair = pair or self.config.pair
        ticker: Dict[str, Any]
        depth: Dict[str, Any]
        trades: List[Dict[str, Any]]
        # When ``skip_depth`` is True (used during the multi-pair scan loop
        # when a WebSocket ticker is already available) we avoid the per-pair
        # REST /depth call entirely.  Orderbook-based signals (whale/spoofing
        # detection, imbalance) will return neutral defaults for the scan,
        # which is acceptable because depth is only needed for the final trade
        # execution decision, not for pair selection.
        # When ``skip_trades`` is True (also used during scanning when WebSocket
        # ticker is available) we skip the per-pair REST /trades call and pass
        # an empty list.  analyze_trade_flow will return a neutral result, and
        # _fetch_candles will use the OHLC cache instead.
        _empty_depth: Dict[str, Any] = {"buy": [], "sell": []}
        # Determine the best available realtime snapshot for this pair:
        # 1. Per-pair position feed (started when a position on this pair is opened)
        # 2. Primary realtime feed (only covers self.config.pair)
        # 3. Fall back to REST API calls
        _pos_feed = self._position_feeds.get(pair)
        if _pos_feed and _pos_feed.has_snapshot:
            _rt_snap = _pos_feed.snapshot()
        elif self.realtime and self.realtime.has_snapshot and pair == self.config.pair:
            _rt_snap = self.realtime.snapshot()
        else:
            _rt_snap = None

        if _rt_snap is not None:
            # Ticker priority: position-feed WS → prefetched (scan) → multi-feed WS → REST
            _multi_ticker = self._multi_feed.get_ticker(pair) if self._multi_feed else None
            ticker = (
                _rt_snap.get("ticker")
                or prefetched_ticker
                or _multi_ticker
                or self.client.get_ticker(pair)
            )
            if skip_depth:
                # Prefer real-time WS orderbook from MultiPairFeed; empty only as
                # last resort so OB signals are never silently neutral.
                _ws_depth = self._multi_feed.get_depth(pair) if self._multi_feed else None
                depth = _ws_depth or _rt_snap.get("depth") or _empty_depth
            else:
                snap_depth = _rt_snap.get("depth")
                _ws_depth = self._multi_feed.get_depth(pair) if self._multi_feed else None
                depth = snap_depth or _ws_depth
                if not depth:
                    try:
                        depth = self.client.get_depth(pair, count=200)
                    except RuntimeError as exc:
                        if "429" in str(exc):
                            logger.warning(
                                "Depth request rate-limited for %s — using cached/empty depth",
                                pair,
                            )
                            depth = snap_depth or _ws_depth or _empty_depth
                        else:
                            raise
            if skip_trades:
                _ws_trades = self._multi_feed.get_trades(pair) if self._multi_feed else None
                trades = _ws_trades or _rt_snap.get("trades") or []
            else:
                trades = _rt_snap.get("trades") or self.client.get_trades(pair, count=self.config.trade_count)
        else:
            # Ticker priority: prefetched (scan) → multi-feed WS → REST
            _multi_ticker = self._multi_feed.get_ticker(pair) if self._multi_feed else None
            ticker = prefetched_ticker or _multi_ticker or self.client.get_ticker(pair)
            if skip_depth:
                _ws_depth = self._multi_feed.get_depth(pair) if self._multi_feed else None
                depth = _ws_depth or _empty_depth
            else:
                _ws_depth = self._multi_feed.get_depth(pair) if self._multi_feed else None
                try:
                    depth = _ws_depth or self.client.get_depth(pair, count=200)
                except RuntimeError as exc:
                    if "429" in str(exc):
                        logger.warning(
                            "Depth request rate-limited for %s — using cached/empty depth",
                            pair,
                        )
                        depth = _ws_depth or _empty_depth
                    else:
                        raise
            if skip_trades:
                _ws_trades = self._multi_feed.get_trades(pair) if self._multi_feed else None
                trades = _ws_trades or []
            else:
                _ws_trades = self._multi_feed.get_trades(pair) if self._multi_feed else None
                trades = _ws_trades or self.client.get_trades(pair, count=self.config.trade_count)

        # ── Candle data ──────────────────────────────────────────────────────
        # Prefer the official OHLCV history endpoint which returns pre-formed
        # candles and covers enough history for all indicators even for
        # high-volume pairs.  Fall back to building candles from raw trades
        # (the legacy path) when the OHLC call fails.
        # When skip_trades is True (scan context) enable the candle cache so
        # repeated scan calls reuse recently fetched OHLC data instead of
        # making a new REST request every cycle for every pair.
        candles = self._fetch_candles(pair, trades, use_cache=skip_trades)

        # ── Rug-pull / dead coin filter ──────────────────────────────────────
        # Check the ticker for extreme 24-h price drops or near-zero volume
        # before spending time on indicators.  When a rug-pull risk is detected
        # the snapshot is returned immediately with a hard "hold" decision so
        # the pair is skipped without burning API quota on depth / candles.
        rug_pull_risk: Optional[RugPullRisk] = None
        _rug_enabled = (
            self.config.rug_pull_max_drop_24h_pct > 0
            or self.config.rug_pull_min_volume_idr > 0
            or self.config.rug_pull_min_trades_24h > 0
        )
        if _rug_enabled:
            rug_pull_risk = detect_rug_pull_risk(
                ticker,
                max_drop_24h_pct=self.config.rug_pull_max_drop_24h_pct,
                min_volume_24h_idr=self.config.rug_pull_min_volume_idr,
                min_trades_24h=self.config.rug_pull_min_trades_24h,
            )
            if rug_pull_risk.detected:
                # logger.warning(
                #     "Rug-pull/dead-coin risk on %s: %s — skipping",
                #     pair,
                #     rug_pull_risk.reason,
                # )
                price = self._extract_price(ticker)
                _hold_decision = StrategyDecision(
                    mode="hold",
                    action="hold",
                    confidence=0.0,
                    reason=f"rug_pull_risk:{rug_pull_risk.reason}",
                    target_price=price,
                    amount=0.0,
                    stop_loss=None,
                    take_profit=None,
                )
                # Still compute basic market data from already-fetched
                # depth/candles so that the log shows meaningful spread,
                # imbalance, volatility, support and resistance values
                # instead of nan/N/A.
                _rug_orderbook = analyze_orderbook(depth)
                _rug_vol = analyze_volatility(candles)
                _rug_levels = support_resistance(candles)
                _rug_indicators = derive_indicators(candles)
                return {
                    "pair": pair,
                    "price": price,
                    "decision": _hold_decision,
                    "rug_pull_risk": rug_pull_risk,
                    "insufficient_data": False,
                    "orderbook": _rug_orderbook,
                    "volatility": _rug_vol,
                    "levels": _rug_levels,
                    "indicators": _rug_indicators,
                    "volume_24h_idr": self._extract_volume_idr(ticker),
                    "trades_24h": self._extract_trade_count_24h(ticker),
                }

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
        # Record price in the per-pair pump-protection rolling buffer.
        self._record_price(pair, price)
        indicators: MomentumIndicators = derive_indicators(candles)
        trades_24h = self._extract_trade_count_24h(ticker)
        regime = detect_market_regime(candles, trend, vol)

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
            self._record_whale_event(pair, whale, price)

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

        # ── Smart Entry Engine ────────────────────────────────────────────────
        smart_entry: Optional[SmartEntryResult] = None
        if self.config.see_enabled:
            smart_entry = smart_entry_filter(
                candles,
                depth,
                price,
                levels,
                volume_surge_ratio=self.config.see_volume_surge_ratio,
                pump_sniper_enabled=self.config.see_pump_sniper_enabled,
                pump_sniper_price_ratio=self.config.see_pump_sniper_price_ratio,
                pump_sniper_volume_ratio=self.config.see_pump_sniper_volume_ratio,
                pump_sniper_short=self.config.see_pump_sniper_short,
                pump_sniper_long=self.config.see_pump_sniper_long,
                whale_pressure_min=self.config.see_whale_pressure_min,
                breakout_volume_min=self.config.see_breakout_volume_min,
                early_breakout_proximity_pct=self.config.early_breakout_proximity_pct,
                early_breakout_min_volume_ratio=self.config.early_breakout_min_volume_ratio,
            )
            if smart_entry.pre_pump.detected:
                logger.debug(
                    "SEE pre-pump on %s: surge_ratio=%.2f score=%.2f",
                    pair, smart_entry.pre_pump.volume_surge_ratio, smart_entry.pre_pump.score,
                )
            if smart_entry.whale_pressure.detected:
                logger.debug(
                    "SEE whale pressure on %s: side=%s pressure=%.2f",
                    pair, smart_entry.whale_pressure.side, smart_entry.whale_pressure.pressure,
                )
            if smart_entry.fake_breakout.detected:
                logger.debug(
                    "SEE fake breakout on %s: vol_ratio=%.2f score=%.2f",
                    pair, smart_entry.fake_breakout.volume_ratio, smart_entry.fake_breakout.score,
                )
            if smart_entry.early_breakout.detected:
                logger.debug(
                    "SEE early breakout on %s: vol_ratio=%.2f score=%.2f",
                    pair, smart_entry.early_breakout.volume_ratio, smart_entry.early_breakout.score,
                )

        # ── Trade flow analysis ───────────────────────────────────────────────
        trade_flow: TradeFlowResult = analyze_trade_flow(trades)
        if trade_flow.aggressive_buyers:
            logger.debug(
                "Trade flow on %s: buy_ratio=%.2f (aggressive buyers)",
                pair, trade_flow.buy_ratio,
            )
        else:
            logger.debug(
                "Trade flow on %s: buy_ratio=%.2f sell_vol=%.2f",
                pair, trade_flow.buy_ratio, trade_flow.sell_volume,
            )

        # ── Liquidity sweep detection ─────────────────────────────────────────
        liquidity_sweep: Optional[LiquiditySweep] = None
        if self.config.liquidity_sweep_enabled and candles:
            liquidity_sweep = detect_liquidity_sweep(
                candles,
                lookback=self.config.liquidity_sweep_lookback,
                min_sweep_pct=self.config.liquidity_sweep_min_pct,
                reversal_pct=self.config.liquidity_sweep_reversal_pct,
            )
            if liquidity_sweep.detected:
                logger.debug(
                    "Liquidity sweep on %s: dir=%s sweep=%.2f%% rev=%.2f%%",
                    pair, liquidity_sweep.direction,
                    liquidity_sweep.sweep_pct * 100, liquidity_sweep.reversal_pct * 100,
                )

        # ── Liquidity trap detection ──────────────────────────────────────────
        liquidity_trap: Optional[LiquidityTrap] = None
        if self.config.liquidity_trap_enabled and candles:
            liquidity_trap = detect_liquidity_trap(
                candles,
                lookback=self.config.liquidity_sweep_lookback,
                breakout_pct=self.config.liquidity_trap_breakout_pct,
                reversal_pct=self.config.liquidity_trap_reversal_pct,
            )
            if liquidity_trap.detected:
                logger.debug(
                    "Liquidity trap on %s: dir=%s breakout=%.2f%% rev=%.2f%%",
                    pair, liquidity_trap.direction,
                    liquidity_trap.breakout_pct * 100, liquidity_trap.reversal_pct * 100,
                )

        # ── Liquidity vacuum detection ────────────────────────────────────────
        liquidity_vacuum: Optional[LiquidityVacuum] = None
        if self.config.liquidity_vacuum_min_gap_pct > 0 and depth:
            liquidity_vacuum = detect_liquidity_vacuum(
                depth,
                min_gap_pct=self.config.liquidity_vacuum_min_gap_pct,
                depth_levels=self.config.liquidity_vacuum_depth_levels,
            )
            if liquidity_vacuum.detected:
                logger.debug(
                    "Liquidity vacuum on %s: gap=%.2f%% at price=%.2f",
                    pair, liquidity_vacuum.gap_pct * 100, liquidity_vacuum.gap_price,
                )

        # ── Smart money footprint detection ───────────────────────────────────
        smart_money: Optional[SmartMoneyFootprint] = None
        if self.config.smart_money_enabled and candles:
            smart_money = detect_smart_money_footprint(
                candles,
                volume_factor=self.config.smart_money_volume_factor,
                divergence_lookback=self.config.smart_money_divergence_lookback,
            )
            if smart_money.detected:
                logger.debug(
                    "Smart money on %s: bias=%s vol_ratio=%.2f",
                    pair, smart_money.bias, smart_money.volume_ratio,
                )

        # ── Volume acceleration detection ─────────────────────────────────────
        volume_accel: Optional[VolumeAcceleration] = None
        if self.config.volume_accel_enabled and candles:
            volume_accel = detect_volume_acceleration(
                candles,
                window=self.config.volume_accel_window,
                min_ratio=self.config.volume_accel_min_ratio,
            )
            if volume_accel.detected:
                logger.debug(
                    "Volume acceleration on %s: ratio=%.2f",
                    pair, volume_accel.acceleration_ratio,
                )

        # ── Micro trend detection ─────────────────────────────────────────────
        micro_trend: Optional[MicroTrend] = None
        if self.config.micro_trend_enabled and candles:
            micro_trend = detect_micro_trend(
                candles,
                window=self.config.micro_trend_window,
            )
            if micro_trend.direction != "flat":
                logger.debug(
                    "Micro trend on %s: dir=%s strength=%.4f",
                    pair, micro_trend.direction, micro_trend.strength,
                )

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
                smart_entry=smart_entry,
                trade_flow=trade_flow,
                liquidity_sweep=liquidity_sweep,
                liquidity_trap=liquidity_trap,
                liquidity_vacuum=liquidity_vacuum,
                smart_money=smart_money,
                volume_accel=volume_accel,
                micro_trend=micro_trend,
                regime=regime,
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
            "regime": regime,
            "decision": decision,
            "candles": candles,
            "grid_plan": grid_plan,
            "insufficient_data": insufficient_data,
            "mtf": mtf,
            "whale": whale,
            "spoofing": spoofing,
            "reference_trend": reference_trend,
            "smart_entry": smart_entry,
            "trade_flow": trade_flow,
            "liquidity_sweep": liquidity_sweep,
            "liquidity_trap": liquidity_trap,
            "liquidity_vacuum": liquidity_vacuum,
            "smart_money": smart_money,
            "volume_accel": volume_accel,
            "micro_trend": micro_trend,
            "volume_24h_idr": self._extract_volume_idr(ticker),
            "trades_24h": trades_24h,
        }

    def _check_small_coin_ob_quality(
        self,
        bids: List[Any],
        top_bid: float,
        top_ask: float,
    ) -> Optional[str]:
        """Return a skip reason if orderbook quality is insufficient for a cheap coin.

        Called when ``price < config.min_buy_price_idr``.  Checks:

        1. **Bid levels** – fewer than ``small_coin_min_bid_levels`` indicates a
           thin/stuck book.
        2. **Bid depth** – total IDR value of all resting bids below
           ``small_coin_min_depth_idr`` suggests negligible liquidity.
        3. **Spread** – bid-ask spread exceeding ``small_coin_max_spread_pct``
           signals poor execution quality.

        Returns ``None`` when all active checks pass (coin is liquid and tradeable).
        """
        # 1. Minimum bid levels
        min_levels = self.config.small_coin_min_bid_levels
        if min_levels > 0 and len(bids) < min_levels:
            return (
                f"small_coin_thin_book {len(bids)} < {min_levels} bid levels"
            )

        # 2. Minimum IDR bid depth
        min_depth = self.config.small_coin_min_depth_idr
        if min_depth > 0 and top_bid > 0:
            total_depth = 0.0
            for bid in bids:
                try:
                    total_depth += float(bid[0]) * float(bid[1])
                except (ValueError, TypeError, IndexError):
                    pass
            if total_depth < min_depth:
                return (
                    f"small_coin_illiquid depth_idr={total_depth:.0f} < {min_depth:.0f}"
                )

        # 3. Maximum spread (overrides global max_spread_pct for cheap coins)
        spread_threshold = self.config.small_coin_max_spread_pct
        if spread_threshold > 0 and top_bid > 0 and top_ask > top_bid:
            spread_pct = (top_ask - top_bid) / top_bid
            if spread_pct > spread_threshold:
                return (
                    f"small_coin_wide_spread {spread_pct:.4%} > {spread_threshold:.4%}"
                )

        return None  # all checks pass → allow the cheap coin

    def _small_coin_low_trade_reason(self, trades_24h: int) -> Optional[str]:
        """Return skip reason for cheap coins with insufficient reported trades.

        Indodax summaries often omit ``trade_count``; a missing/zero value is
        treated as "unknown" rather than blocking by default.  A non-positive
        configured threshold also disables this check.
        """
        min_trades = self.config.small_coin_min_trades_24h
        if min_trades <= 0 or trades_24h <= 0:
            # Trade count missing/unknown — do not block solely on absence.
            return None
        if trades_24h < min_trades:
            return f"small_coin_low_trades {trades_24h} < {min_trades}"
        return None

    def _cancel_open_orders(
        self,
        pair: str,
        order_types: Iterable[str],
        reason: Optional[str] = None,
    ) -> int:
        """Cancel open orders for *pair* whose type is included in ``order_types``."""
        if self.config.dry_run or self.config.api_key is None:
            return 0
        try:
            open_resp = self.client.open_orders(pair)
            orders = (open_resp.get("return") or {}).get("orders") or []
        except Exception as exc:
            logger.warning("Unable to fetch open orders for %s: %s", pair, exc)
            return 0

        order_types = {t.lower() for t in order_types}
        cancelled = 0
        for order in orders if isinstance(orders, list) else []:
            order_type = str(order.get("type", "")).lower()
            if order_type not in order_types:
                continue
            order_id = str(order.get("order_id"))
            try:
                self.client.cancel_order(pair, order_id, order_type)
                cancelled += 1
                logger.info(
                    "Cancelled open %s order %s for %s%s",
                    order_type,
                    order_id,
                    pair,
                    f" ({reason})" if reason else "",
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Failed to cancel %s order %s for %s: %s",
                    order_type,
                    order_id,
                    pair,
                    exc,
                )

        if cancelled:
            getattr(self.client, "invalidate_open_orders_cache", lambda p: None)(pair)
        return cancelled

    def _cancel_open_buy_orders(
        self,
        pair: str,
        tracker: "PortfolioTracker",
        reason: Optional[str] = None,
    ) -> int:
        """Cancel any open BUY orders for *pair* when conditions turn unfavourable.

        Returns the number of cancelled orders.  When no position is held the
        per-pair tracker is rolled back and its capital slice (if any) is
        returned to the multi-position pool so the bot can seek another pair.
        """
        cancelled = self._cancel_open_orders(pair, ("buy",), reason=reason)

        if cancelled:
            if tracker.base_position <= 0:
                tracker.cancel_pending_buy()
                if self.multi_manager is not None:
                    self.multi_manager.return_position_cash(pair)
        return cancelled

    def _cancel_stale_orders(self, pair: str) -> int:
        """Cancel open orders older than ``config.stale_order_seconds``.

        Returns the number of cancelled orders.  Skipped when
        ``stale_order_seconds`` is 0 (disabled), in dry-run mode, or when
        API credentials are not configured.
        """
        if self.config.stale_order_seconds <= 0:
            return 0
        if self.config.dry_run or self.config.api_key is None:
            return 0
        try:
            open_resp = self.client.open_orders(pair)
            orders = (open_resp.get("return") or {}).get("orders") or []
        except Exception as exc:
            logger.warning("Unable to fetch open orders for stale check on %s: %s", pair, exc)
            return 0

        now = time.time()
        cancelled = 0
        for order in orders if isinstance(orders, list) else []:
            submit_time = 0.0
            for ts_key in ("submit_time", "order_time", "time"):
                raw = order.get(ts_key)
                if raw is not None:
                    try:
                        submit_time = float(raw)
                        break
                    except (TypeError, ValueError):
                        pass
            if submit_time <= 0:
                continue
            age = now - submit_time
            if age >= self.config.stale_order_seconds:
                order_id = str(order.get("order_id"))
                order_type = str(order.get("type", "")).lower()
                try:
                    self.client.cancel_order(pair, order_id, order_type)
                    cancelled += 1
                    logger.info(
                        "Cancelled stale %s order %s for %s (age=%.0fs ≥ %.0fs)",
                        order_type, order_id, pair, age, self.config.stale_order_seconds,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to cancel stale %s order %s for %s: %s",
                        order_type, order_id, pair, exc,
                    )
        if cancelled:
            getattr(self.client, "invalidate_open_orders_cache", lambda p: None)(pair)
        return cancelled

    def maybe_execute(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        _pair = snapshot["pair"]
        decision: StrategyDecision = snapshot["decision"]
        price = snapshot["price"]
        # Route to the per-pair tracker in multi-position mode; otherwise use
        # self.tracker (classic single-position behaviour).
        _tracker = self._active_tracker(_pair)

        # ── Stale order cancellation ─────────────────────────────────────────
        # Cancel open orders that have been sitting unfilled for too long.
        self._cancel_stale_orders(_pair)

        # Circuit breaker
        if self.config.circuit_breaker_max_errors > 0 and time.time() < self._circuit_breaker_until:
            return {"status": "circuit_breaker", "reason": "circuit_breaker_active", "portfolio": _tracker.as_dict(price)}

        # Volatility cooldown
        if time.time() < self._volatility_cooldown_until:
            return {"status": "volatility_cooldown", "reason": "volatility_cooldown_active", "portfolio": _tracker.as_dict(price)}

        # Update volatility cooldown state
        self._check_volatility_cooldown(price, snapshot["pair"])

        # Update trailing stop before checking stop conditions
        if self.config.trailing_stop_pct > 0:
            _tracker.update_trailing_stop(price, self.config.trailing_stop_pct)

        # Stop conditions are only meaningful when a position is actually held.
        stop_reason = _tracker.stop_reason(price) if _tracker.base_position > 0 else None
        if stop_reason:
            logger.info("Stop triggered (%s) equity=%s", stop_reason, _tracker.as_dict(price))
            outcome = {"status": "stopped", "reason": stop_reason, "portfolio": _tracker.as_dict(price)}
            return outcome

        # ── Time-based exit (anti-stagnation for illiquid coins) ─────────────
        # Force-sell an open position that has been held longer than
        # max_hold_seconds without reaching the profit threshold.  This
        # prevents capital from being tied up in slow/illiquid pairs.
        # When volume_high_threshold_idr > 0, the effective hold limit is
        # chosen adaptively: 90 min for high-volume pairs, 30 min otherwise.
        _effective_max_hold = self.config.max_hold_seconds
        if self.config.volume_high_threshold_idr > 0:
            _vol_24h = snapshot.get("volume_24h_idr", 0.0)
            if _vol_24h >= self.config.volume_high_threshold_idr:
                _effective_max_hold = self.config.max_hold_seconds_volume_high
            else:
                _effective_max_hold = self.config.max_hold_seconds_volume_low
        if (
            _effective_max_hold > 0
            and _tracker.base_position > 0
            and _tracker.avg_cost > 0
        ):
            hold_secs = _tracker.position_hold_seconds
            if hold_secs >= _effective_max_hold:
                unrealised_pct = (price - _tracker.avg_cost) / _tracker.avg_cost
                if unrealised_pct < self.config.max_hold_profit_pct:
                    logger.info(
                        "Time-based exit: held %.0fs ≥ %.0fs, profit=%.2f%% < %.2f%% — force selling",
                        hold_secs,
                        _effective_max_hold,
                        unrealised_pct * 100,
                        self.config.max_hold_profit_pct * 100,
                    )
                    result = self.force_sell(snapshot)
                    result["status"] = "time_exit"
                    result["reason"] = (
                        f"max_hold_seconds exceeded: held {hold_secs:.0f}s, "
                        f"profit {unrealised_pct:.2%} < {self.config.max_hold_profit_pct:.2%}"
                    )
                    return result

        # ── Daily loss cap ────────────────────────────────────────────────────
        if self.config.max_daily_loss_pct > 0 and decision.action == "buy":
            daily_loss_pct = _tracker.daily_loss_pct(price)
            if daily_loss_pct >= self.config.max_daily_loss_pct:
                logger.warning(
                    "Daily loss cap reached: %.2f%% ≥ %.2f%% — skipping buy",
                    daily_loss_pct * 100,
                    self.config.max_daily_loss_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"daily_loss_cap {daily_loss_pct:.2%} ≥ {self.config.max_daily_loss_pct:.2%}",
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Per-coin exposure cap ─────────────────────────────────────────────
        if self.config.max_exposure_per_coin_pct > 0 and decision.action == "buy":
            current_equity = _tracker.snapshot(price).equity
            current_exposure = _tracker.base_position * price
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
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Portfolio-wide risk cap ───────────────────────────────────────────
        if self.config.max_portfolio_risk_pct > 0 and decision.action == "buy":
            current_equity = _tracker.snapshot(price).equity
            total_position_value = _tracker.base_position * price
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
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Profit-buffer drawdown guard ──────────────────────────────────────
        if self.config.profit_buffer_drawdown_pct > 0 and decision.action == "buy":
            pb_drawdown = _tracker.profit_buffer_drawdown_pct()
            if pb_drawdown >= self.config.profit_buffer_drawdown_pct:
                logger.warning(
                    "Profit-buffer drawdown guard: buffer dropped %.1f%% from peak (limit=%.1f%%) — skipping buy",
                    pb_drawdown * 100,
                    self.config.profit_buffer_drawdown_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"profit_buffer_drawdown {pb_drawdown:.2%} ≥ {self.config.profit_buffer_drawdown_pct:.2%}",
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Re-entry cooldown / dip check ─────────────────────────────────────
        if decision.action == "buy" and (
            self.config.re_entry_cooldown_seconds > 0 or self.config.re_entry_dip_pct > 0
        ):
            if not _tracker.re_entry_allowed(
                price,
                cooldown_seconds=self.config.re_entry_cooldown_seconds,
                dip_pct=self.config.re_entry_dip_pct,
            ):
                logger.info(
                    "Re-entry blocked: cooldown or dip condition not met (last_sell=%.2f, now=%.2f)",
                    _tracker.last_sell_price,
                    price,
                )
                return {
                    "status": "skipped",
                    "reason": "re_entry_condition_not_met",
                    "portfolio": _tracker.as_dict(price),
                }

        # Consecutive loss protection
        if self.config.max_consecutive_losses > 0 and decision.action == "buy":
            if _tracker.loss_streak >= self.config.max_consecutive_losses:
                return {"status": "skipped", "reason": "max_consecutive_losses", "portfolio": _tracker.as_dict(price)}

        # ── Per-pair trade cooldown ───────────────────────────────────────────
        if self.config.pair_cooldown_seconds > 0 and decision.action == "buy":
            last_trade = self._pair_last_trade.get(_pair, 0.0)
            elapsed = time.time() - last_trade
            if elapsed < self.config.pair_cooldown_seconds:
                remaining = self.config.pair_cooldown_seconds - elapsed
                logger.info(
                    "Pair cooldown active for %s — %.0fs remaining",
                    _pair, remaining,
                )
                return {
                    "status": "skipped",
                    "reason": f"pair_cooldown remaining={remaining:.0f}s",
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Max open positions check ──────────────────────────────────────────
        # In multi-position mode: reject BUY when the position limit is reached.
        # In single-position mode: use the adaptive-sizing max-position guard.
        if decision.action == "buy":
            if self.multi_manager is not None:
                if self.at_max_positions():
                    logger.info(
                        "Max open positions reached (%d) — skipping buy on %s",
                        self.config.multi_position_max,
                        _pair,
                    )
                    return {
                        "status": "skipped",
                        "reason": f"max_open_positions={self.config.multi_position_max}",
                        "portfolio": _tracker.as_dict(price),
                    }
            else:
                equity = _tracker.effective_capital()
                eff_max_pos = adaptive_max_positions(equity, self.config)
                if eff_max_pos > 0 and _tracker.base_position > 0:
                    if eff_max_pos <= 1:
                        logger.info(
                            "Max open positions reached (%d) — skipping buy", eff_max_pos
                        )
                        return {
                            "status": "skipped",
                            "reason": f"max_open_positions={eff_max_pos}",
                            "portfolio": _tracker.as_dict(price),
                        }

        # Flash dump protection
        if self.config.flash_dump_pct > 0 and decision.action == "buy":
            pair_history = self._price_history.get(snapshot["pair"], [])
            dump = detect_flash_dump(pair_history, self.config.flash_dump_lookback_seconds, self.config.flash_dump_pct)
            if dump.detected:
                return {"status": "skipped", "reason": f"flash_dump drop={dump.drop_pct:.2%}", "portfolio": _tracker.as_dict(price)}

        # Strategy disabled check
        if decision.action == "buy" and _tracker.is_strategy_disabled(decision.mode):
            return {"status": "skipped", "reason": "strategy_disabled", "portfolio": _tracker.as_dict(price)}

        if self.config.grid_enabled and snapshot.get("grid_plan"):
            return self._execute_grid(snapshot)

        if decision.action == "hold":
            self._cancel_open_buy_orders(_pair, _tracker, decision.reason)
            _portfolio = self.portfolio_snapshot(_pair, price)
            logger.info("Hold action | reason=%s | portfolio=%s", decision.reason, _portfolio)
            outcome = {"status": "hold", "reason": decision.reason, "portfolio": _portfolio}
            return outcome

        # When confidence-based position sizing is active, honour the tier_skip
        # threshold as an additional lower bound on the minimum confidence, so
        # setting CONFIDENCE_POSITION_SIZING_ENABLED=true is sufficient without
        # requiring a separate MIN_CONFIDENCE adjustment.
        _effective_min_conf = self._min_confidence_threshold(snapshot)
        if decision.confidence < _effective_min_conf:
            logger.info(
                "Skip low confidence action=%s conf=%.3f min=%.3f",
                decision.action,
                decision.confidence,
                _effective_min_conf,
            )
            outcome = {
                "status": "skipped",
                "reason": f"confidence {decision.confidence} below threshold {_effective_min_conf}",
                "portfolio": _tracker.as_dict(price),
            }
            return outcome

        # simple slippage guard using top of book
        depth = self.client.get_depth(snapshot["pair"], count=_EXECUTION_DEPTH_LEVELS)
        bids = depth.get("buy") or []
        asks = depth.get("sell") or []
        try:
            top_bid = float(bids[0][0]) if bids else price
        except (ValueError, TypeError):
            top_bid = price
        try:
            top_ask = float(asks[0][0]) if asks else price
        except (ValueError, TypeError):
            top_ask = price
        reference_price = top_ask if decision.action == "buy" else top_bid
        # Make entries slightly more aggressive (within slippage guard) to
        # improve the likelihood of an immediate fill instead of sitting at the
        # back of the queue at the same best price.
        entry_aggr = max(0.0, self.config.entry_aggressiveness_pct)
        allowed_max = price * (1 + self.config.max_slippage_pct)
        allowed_min = price * (1 - self.config.max_slippage_pct)
        if entry_aggr > 0:
            if decision.action == "buy" and top_ask > 0:
                reference_price = min(top_ask * (1 + entry_aggr), allowed_max)
            elif decision.action == "sell" and top_bid > 0:
                reference_price = max(top_bid * (1 - entry_aggr), allowed_min)

        # ── Spread filter ─────────────────────────────────────────────────────
        # Skip any trade (buy or sell) when the bid-ask spread is too wide.
        # Wide spreads increase execution cost and indicate thin liquidity.
        if self.config.max_spread_pct > 0 and top_bid > 0 and top_ask > 0:
            live_spread_pct = (top_ask - top_bid) / top_bid
            if live_spread_pct >= self.config.max_spread_pct:
                logger.info(
                    "Spread too wide on %s: %.4f%% ≥ %.4f%% — skipping %s",
                    snapshot["pair"],
                    live_spread_pct * 100,
                    self.config.max_spread_pct * 100,
                    decision.action,
                )
                return {
                    "status": "skipped",
                    "reason": (
                        f"spread_too_wide {live_spread_pct:.4%} ≥ {self.config.max_spread_pct:.4%}"
                    ),
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Minimum price filter (hard floor) ────────────────────────────────
        if (
            self.config.min_coin_price_idr > 0
            and decision.action == "buy"
            and price < self.config.min_coin_price_idr
        ):
            logger.info(
                "Price below hard floor on %s: %.6g IDR < %.6g — skipping buy",
                snapshot["pair"],
                price,
                self.config.min_coin_price_idr,
            )
            return {
                "status": "skipped",
                "reason": (
                    f"price_below_min_coin {price:.6g} < {self.config.min_coin_price_idr:.6g}"
                ),
                "portfolio": _tracker.as_dict(price),
            }

        # ── Minimum price filter (soft quality check) ────────────────────────
        # When price is below min_buy_price_idr, evaluate orderbook quality
        # instead of hard-skipping.  Coins with a thin/stuck book are blocked;
        # active cheap coins with adequate depth and a tight spread are allowed.
        if self.config.min_buy_price_idr > 0 and decision.action == "buy":
            if price < self.config.min_buy_price_idr:
                skip_reason = self._check_small_coin_ob_quality(
                    bids, top_bid, top_ask
                )
                if skip_reason:
                    logger.info(
                        "Cheap coin %s quality check failed (%.6g IDR < %.6g): %s — skipping buy",
                        snapshot["pair"],
                        price,
                        self.config.min_buy_price_idr,
                        skip_reason,
                    )
                    return {
                        "status": "skipped",
                        "reason": skip_reason,
                        "portfolio": _tracker.as_dict(price),
                    }
                # Quiet/dead coin guard: require minimum 24-h volume/trade count
                vol_24h_idr = float(snapshot.get("volume_24h_idr") or 0.0)
                trades_24h = int(snapshot.get("trades_24h") or 0)
                if (
                    self.config.small_coin_min_volume_24h_idr > 0
                    and vol_24h_idr < self.config.small_coin_min_volume_24h_idr
                ):
                    logger.info(
                        "Cheap coin %s blocked by low 24h volume: %.0f < %.0f",
                        snapshot["pair"],
                        vol_24h_idr,
                        self.config.small_coin_min_volume_24h_idr,
                    )
                    return {
                        "status": "skipped",
                        "reason": (
                            f"small_coin_low_volume {vol_24h_idr:.0f} < {self.config.small_coin_min_volume_24h_idr:.0f}"
                        ),
                        "portfolio": _tracker.as_dict(price),
                    }
                trade_reason = self._small_coin_low_trade_reason(trades_24h)
                if trade_reason:
                    logger.info(
                        "Cheap coin %s blocked by low trade count: %s",
                        snapshot["pair"],
                        trade_reason,
                    )
                    return {
                        "status": "skipped",
                        "reason": trade_reason,
                        "portfolio": _tracker.as_dict(price),
                    }

        # ── Tick-move filter ──────────────────────────────────────────────────
        # Skip buy when the minimum possible price increment (tick) is a
        # disproportionately large fraction of the current price.  This
        # catches illiquid integer-priced coins (e.g. 4→5 IDR = 25%) where
        # a profitable exit requires the price to make an unusually large jump.
        # The tick is estimated from the gap between the best and second-best
        # bid in the orderbook; the bid-ask spread is used as a lower-bound
        # fallback when only one bid level is available.
        if self.config.max_tick_move_pct > 0 and decision.action == "buy" and top_bid > 0:
            tick_pct: Optional[float] = None
            if len(bids) >= 2:
                try:
                    second_bid = float(bids[1][0])
                except (ValueError, TypeError):
                    second_bid = 0.0
                if 0 < second_bid < top_bid:
                    tick_pct = (top_bid - second_bid) / top_bid
            if tick_pct is None and top_ask > top_bid:
                tick_pct = (top_ask - top_bid) / top_bid
            # When the book is flat (e.g. 4 to 4) we still want to approximate
            # the discrete tick.  Use a 1-unit tick relative to the best bid to
            # catch tiny integer-priced coins that require outsized moves.
            if tick_pct is None and top_bid > 0:
                tick_pct = 1.0 / top_bid
            if tick_pct is not None and tick_pct > self.config.max_tick_move_pct:
                logger.info(
                    "Tick too large on %s: %.4f%% > %.4f%% — skipping buy",
                    snapshot["pair"],
                    tick_pct * 100,
                    self.config.max_tick_move_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": (
                        f"tick_too_large {tick_pct:.4%} > {self.config.max_tick_move_pct:.4%}"
                    ),
                    "portfolio": _tracker.as_dict(price),
                }

        # Spread anomaly detection
        if self.config.spread_anomaly_multiplier > 0 and top_bid > 0 and top_ask > 0:
            live_spread_pct = (top_ask - top_bid) / top_bid
            pair_key = snapshot["pair"]
            if pair_key not in self._spread_history:
                self._spread_history[pair_key] = []
            self._spread_history[pair_key].append(live_spread_pct)
            self._spread_history[pair_key] = self._spread_history[pair_key][-50:]
            recent_spreads = self._spread_history[pair_key][:-1]  # exclude current
            anomaly = detect_spread_anomaly(live_spread_pct, recent_spreads, self.config.spread_anomaly_multiplier)
            if anomaly.detected:
                return {"status": "skipped", "reason": f"spread_anomaly ratio={anomaly.ratio:.2f}x", "portfolio": _tracker.as_dict(price)}

        # ── Spread expansion detection ───────────────────────────────────────
        if self.config.spread_expansion_enabled and top_bid > 0 and top_ask > 0 and decision.action == "buy":
            live_spread_pct = (top_ask - top_bid) / top_bid
            pair_key = snapshot["pair"]
            if pair_key not in self._spread_history:
                self._spread_history[pair_key] = []
            hist = self._spread_history[pair_key][-self.config.spread_expansion_window:]
            expansion = detect_spread_expansion(
                live_spread_pct, hist,
                multiplier=self.config.spread_expansion_multiplier,
            )
            if expansion.detected:
                logger.info(
                    "Spread expansion on %s: ratio=%.2f× — skipping buy",
                    pair_key, expansion.expansion_ratio,
                )
                return {
                    "status": "skipped",
                    "reason": f"spread_expansion ratio={expansion.expansion_ratio:.2f}x",
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Orderbook absorption detection ───────────────────────────────────
        if self.config.orderbook_absorption_threshold > 0 and decision.action == "buy":
            pair_key = snapshot["pair"]
            current_depth = {"buy": bids, "sell": asks}
            prev_depth = self._prev_depth.get(pair_key)
            if prev_depth is not None:
                absorption = detect_orderbook_absorption(
                    prev_depth, current_depth,
                    threshold=self.config.orderbook_absorption_threshold,
                )
                if absorption.detected and absorption.side == "bid":
                    logger.info(
                        "Orderbook absorption on %s: side=%s ratio=%.2f — skipping buy (bid wall consumed)",
                        pair_key, absorption.side, absorption.absorption_ratio,
                    )
                    self._prev_depth[pair_key] = current_depth
                    return {
                        "status": "skipped",
                        "reason": f"ob_absorption side={absorption.side} ratio={absorption.absorption_ratio:.2f}",
                        "portfolio": _tracker.as_dict(price),
                    }
            self._prev_depth[pair_key] = current_depth

        # ── Liquidity vacuum guard ───────────────────────────────────────────
        if self.config.liquidity_vacuum_min_gap_pct > 0 and decision.action == "buy":
            vacuum = snapshot.get("liquidity_vacuum")
            if vacuum is not None and vacuum.detected:
                logger.info(
                    "Liquidity vacuum on %s: gap=%.2f%% — skipping buy",
                    snapshot["pair"], vacuum.gap_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"liquidity_vacuum gap={vacuum.gap_pct:.2%}",
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Liquidity sweep guard ─────────────────────────────────────────────
        if self.config.liquidity_sweep_enabled and decision.action == "buy":
            sweep = snapshot.get("liquidity_sweep")
            if sweep is not None and sweep.detected and sweep.direction == "up":
                logger.info(
                    "Liquidity sweep (up) on %s: sweep=%.2f%% — skipping buy (stop-hunt risk)",
                    snapshot["pair"], sweep.sweep_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"liquidity_sweep_up sweep={sweep.sweep_pct:.2%}",
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Liquidity trap guard ──────────────────────────────────────────────
        if self.config.liquidity_trap_enabled and decision.action == "buy":
            trap = snapshot.get("liquidity_trap")
            if trap is not None and trap.detected and trap.direction == "up":
                logger.info(
                    "Liquidity trap (up) on %s: breakout=%.2f%% — skipping buy (false breakout)",
                    snapshot["pair"], trap.breakout_pct * 100,
                )
                return {
                    "status": "skipped",
                    "reason": f"liquidity_trap_up breakout={trap.breakout_pct:.2%}",
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Sell-wall / orderbook wall guard ─────────────────────────────────
        # Skip buy when aggregate ask-side volume dominates bid-side volume by
        # the configured multiple.  This protects against entering a market
        # where persistent sell-wall pressure will suppress price recovery.
        if self.config.orderbook_wall_threshold > 0 and decision.action == "buy":
            def _parse_vol(entries: list) -> float:
                total = 0.0
                for entry in entries[:_EXECUTION_DEPTH_LEVELS]:
                    if len(entry) >= 2:
                        try:
                            total += float(entry[1])
                        except (ValueError, TypeError):
                            pass
                return total
            bid_vol = _parse_vol(bids)
            ask_vol = _parse_vol(asks)
            if bid_vol > 0 and ask_vol / bid_vol >= self.config.orderbook_wall_threshold:
                logger.info(
                    "Sell-wall detected on %s: ask/bid=%.1f× ≥ %.1f× — skipping buy",
                    snapshot["pair"],
                    ask_vol / bid_vol,
                    self.config.orderbook_wall_threshold,
                )
                return {
                    "status": "skipped",
                    "reason": (
                        f"sell_wall ask/bid={ask_vol / bid_vol:.2f}× ≥ {self.config.orderbook_wall_threshold:.1f}×"
                    ),
                    "portfolio": _tracker.as_dict(price),
                }

        # ── Pump protection ───────────────────────────────────────────────────
        # Skip entry when price has spiked by more than pump_protection_pct
        # within the last pump_lookback_seconds.  Prevents FOMO buying after
        # a sharp pump that is likely to retrace.
        current_pair = snapshot["pair"]
        if decision.action == "buy" and self._is_pumped(current_pair, price):
            pair_buf = self._price_history.get(current_pair, [])
            oldest_price = pair_buf[0][1] if pair_buf else price
            rise = (price - oldest_price) / oldest_price if oldest_price > 0 else 0.0
            logger.info(
                "Pump detected on %s: price rose %.2f%% in ≤%.0fs — skipping buy",
                snapshot["pair"],
                rise * 100,
                self.config.pump_lookback_seconds,
            )
            return {
                "status": "skipped",
                "reason": (
                    f"pump_detected rise={rise:.2%} ≥ {self.config.pump_protection_pct:.2%} "
                    f"in {self.config.pump_lookback_seconds:.0f}s"
                ),
                "portfolio": _tracker.as_dict(price),
            }

        # ── Anti-fake-pump detection (spike → dump within ~20 s) ─────────────
        # On Indodax, pump-and-dump cycles often complete within ~20 seconds.
        # This guard checks whether price already spiked (pump) AND has since
        # reversed (dump) within the rolling window.  Buying at the post-dump
        # price is highly risky because further downside is likely.
        if decision.action == "buy" and self._is_fake_pump(current_pair, price):
            pair_buf = self._price_history.get(current_pair, [])
            peak_price = max(entry[1] for entry in pair_buf) if pair_buf else price
            reversal = (peak_price - price) / peak_price if peak_price > 0 else 0.0
            logger.info(
                "Fake-pump detected on %s: peak=%.2f current=%.2f reversal=%.2f%% — skipping buy",
                current_pair,
                peak_price,
                price,
                reversal * 100,
            )
            return {
                "status": "skipped",
                "reason": (
                    f"fake_pump_detected reversal={reversal:.2%} ≥ "
                    f"{self.config.fake_pump_reversal_pct:.2%} after spike"
                ),
                "portfolio": _tracker.as_dict(price),
            }

        # ── Minimum liquidity depth check ─────────────────────────────────────
        if self.config.min_liquidity_depth_idr > 0 and decision.action == "buy":
            total_depth = self._liquidity_depth_idr(depth, price)
            if total_depth is None:
                # Depth data unavailable (API error / unexpected format).
                # Do NOT treat as thin market — skip the filter and proceed.
                logger.warning(
                    "Depth data unavailable for %s — skipping liquidity check",
                    snapshot["pair"],
                )
            elif total_depth < self.config.min_liquidity_depth_idr:
                logger.info(
                    "Thin market on %s: depth Rp%.0f < min Rp%.0f — skipping buy",
                    snapshot["pair"], total_depth, self.config.min_liquidity_depth_idr,
                )
                return {
                    "status": "skipped",
                    "reason": f"thin_market depth Rp{total_depth:.0f} < Rp{self.config.min_liquidity_depth_idr:.0f}",
                    "portfolio": _tracker.as_dict(price),
                }

        if decision.action == "buy" and reference_price > allowed_max:
            logger.info("Skip buy due to slippage price=%s allowed_max=%s", reference_price, allowed_max)
            outcome = {
                "status": "skipped",
                "reason": "slippage too high for buy",
                "portfolio": _tracker.as_dict(reference_price),
            }
            return outcome
        if decision.action == "sell" and reference_price < allowed_min:
            logger.info("Skip sell due to slippage price=%s allowed_min=%s", reference_price, allowed_min)
            outcome = {
                "status": "skipped",
                "reason": "slippage too high for sell",
                "portfolio": _tracker.as_dict(reference_price),
            }
            return outcome

        # capital and position guards (risk management)
        effective_amount = decision.amount
        if decision.action == "buy":
            # In multi-position mode the tracker for a new pair is a zero-cash
            # placeholder; use the multi_manager's prospective allocation instead.
            if self.multi_manager is not None and not self.multi_manager.has_position(_pair):
                available_cash = self.multi_manager.capital_per_new_position()
            else:
                available_cash = _tracker.cash
            max_affordable = max(0.0, available_cash / reference_price)
            effective_amount = min(decision.amount, max_affordable)
            # Log adaptive sizing tier when it's active
            if self.config.adaptive_sizing_enabled:
                equity = _tracker.effective_capital() or available_cash
                eff_risk = adaptive_risk_per_trade(equity, self.config)
                eff_max = adaptive_max_positions(equity, self.config)
                logger.debug(
                    "Adaptive sizing: equity=%.0f risk=%.0f%% max_pos=%d",
                    equity, eff_risk * 100, eff_max,
                )
        elif decision.action == "sell":
            max_sellable = max(0.0, _tracker.base_position)
            effective_amount = min(decision.amount, max_sellable)

        if effective_amount <= 0:
            logger.info("Skip due to insufficient balance/position | action=%s", decision.action)
            outcome = {
                "status": "skipped",
                "reason": "insufficient balance or position",
                "portfolio": _tracker.as_dict(reference_price),
            }
            return outcome

        # ── Indodax minimum order value guard ─────────────────────────────────
        # Indodax rejects orders whose total IDR value (price × amount) is below
        # the exchange minimum.  The bot defaults to 30,000 IDR (configurable via
        # MIN_ORDER_IDR) to stay safely above the threshold.  Check upfront so
        # the error is surfaced as a clean "skipped" outcome rather than a
        # RuntimeError from the exchange.
        total_order_value_idr = effective_amount * reference_price
        if total_order_value_idr < self.config.min_order_idr:
            logger.info(
                "Order value Rp%.0f < min Rp%.0f for %s — skipping %s",
                total_order_value_idr,
                self.config.min_order_idr,
                snapshot["pair"],
                decision.action,
            )
            return {
                "status": "skipped",
                "reason": (
                    f"order_below_minimum Rp{total_order_value_idr:.0f} "
                    f"< Rp{self.config.min_order_idr:.0f}"
                ),
                "portfolio": _tracker.as_dict(reference_price),
            }

        staged = self._scale_staged_amounts(decision.amount, effective_amount, self._staged_amounts(decision, snapshot))

        if not self._validate_balance(snapshot["pair"], decision.action, effective_amount, reference_price):
            return {"status": "skipped", "reason": "balance_check_failed", "portfolio": _tracker.as_dict(price)}

        # ── Multi-position: allocate capital for new BUY before execution ─────
        # Up to this point _tracker was a zero-cash placeholder; create the real
        # per-pair tracker now so record_trade() has a properly funded account.
        if decision.action == "buy" and self.multi_manager is not None:
            if not self.multi_manager.has_position(_pair):
                _tracker = self.multi_manager.allocate_capital(_pair)
                # Tighten effective_amount against the newly allocated cash
                max_affordable = max(0.0, _tracker.cash / reference_price)
                effective_amount = min(effective_amount, max_affordable)
                # Re-derive staged after adjustment
                staged = self._scale_staged_amounts(decision.amount, effective_amount, self._staged_amounts(decision, snapshot))

        _pre_trade_avg_cost = _tracker.avg_cost

        executed_steps: List[Dict[str, Any]] = []
        remaining_amount = effective_amount
        if self.config.dry_run:
            for amt in staged:
                step_amount = min(amt, remaining_amount)
                if step_amount <= 0:
                    continue
                # Skip individual staged steps that fall below the minimum order
                # value (can happen when staging splits a borderline-size order).
                if step_amount * reference_price < self.config.min_order_idr:
                    logger.debug(
                        "DRY-RUN skipping staged step: Rp%.0f < min Rp%.0f",
                        step_amount * reference_price,
                        self.config.min_order_idr,
                    )
                    continue
                logger.info("DRY-RUN %s %s @ %s (staged)", decision.action, step_amount, reference_price)
                _tracker.record_trade(decision.action, reference_price, step_amount)
                remaining_amount -= step_amount
                executed_steps.append({"amount": step_amount, "price": reference_price})
            # Guard: if every staged step was below min_order, nothing was bought.
            # Return skipped so the caller never logs a false "PLACED" / "✅".
            if not executed_steps:
                # Roll back allocated capital in multi-position mode (nothing was bought).
                if decision.action == "buy" and self.multi_manager is not None:
                    self.multi_manager.return_position_cash(_pair)
                logger.info(
                    "All staged steps below min Rp%.0f — skipping %s on %s",
                    self.config.min_order_idr,
                    decision.action,
                    snapshot["pair"],
                )
                return {
                    "status": "skipped",
                    "reason": f"all_steps_below_min_order (min=Rp{self.config.min_order_idr:.0f})",
                    "portfolio": _tracker.as_dict(reference_price),
                }
            # Multi-position: return cash to pool after a full sell-close.
            if decision.action == "sell" and self.multi_manager is not None and _tracker.base_position <= 0:
                self.multi_manager.return_position_cash(_pair)
            # Manage per-pair realtime feeds: start on first buy, stop on full close.
            if decision.action == "buy" and _tracker.base_position > 0:
                self._ensure_position_feed(_pair)
            elif decision.action == "sell" and _tracker.base_position <= 0:
                self._remove_position_feed(_pair)
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
                "portfolio": _tracker.as_dict(reference_price),
            }
            self._persist_after_trade(snapshot["pair"])
            # Journal logging
            if self.journal is not None:
                import datetime as _dt
                _ts = time.time()
                _dt_str = _dt.datetime.fromtimestamp(_ts).strftime("%Y-%m-%d %H:%M:%S")
                _pnl = (reference_price - _pre_trade_avg_cost) * effective_amount if decision.action == "sell" else 0.0
                self.journal.log_trade(
                    timestamp=_ts,
                    datetime_str=_dt_str,
                    pair=snapshot["pair"],
                    action=decision.action,
                    price=reference_price,
                    amount=sum(step["amount"] for step in executed_steps),
                    idr_value=reference_price * sum(step["amount"] for step in executed_steps),
                    pnl=_pnl,
                    strategy=decision.mode,
                    confidence=decision.confidence,
                    reason=decision.reason,
                    avg_cost=_tracker.avg_cost,
                    equity=_tracker.snapshot(reference_price).equity,
                )
            self._consecutive_errors = 0
            return outcome

        # live trading path
        # Guard against programmatic use without running through CLI validation.
        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        base_coin = snapshot["pair"].split("_")[0].lower()
        receive_key = f"receive_{base_coin}"

        def _calc_received_amount(order_resp: Dict[str, Any], pre_step_position: float) -> float:
            if decision.action != "buy":
                return step_amount
            received = step_amount
            try:
                received = float((order_resp.get("return") or {}).get(receive_key) or 0.0)
            except (TypeError, ValueError):
                received = 0.0

            if received <= 0:
                # Fallback: some exchange responses return receive_<coin>=0
                # even when the order is filled.  Verify the live balance
                # and treat any increase as a filled amount so we don't
                # abandon a real position.
                try:
                    acct_info = self.client.get_account_info()
                    balance_dict = (acct_info.get("return") or {}).get("balance") or {}
                    actual_balance = float(balance_dict.get(base_coin) or "0")
                    balance_delta = actual_balance - pre_step_position
                    if balance_delta > 0:
                        received = min(step_amount, balance_delta)
                        logger.info(
                            "Buy response showed receive_%s=0 but account balance increased by %.8f — "
                            "recording filled buy",
                            base_coin.upper(),
                            received,
                        )
                except Exception:
                    # Do not block the trade; fall back to pending logic below.
                    pass
            return received

        for amt in staged:
            step_amount = min(amt, remaining_amount)
            if step_amount <= 0:
                continue
            pre_step_position = _tracker.base_position
            # Skip steps whose IDR value would be rejected by the exchange.
            if step_amount * reference_price < self.config.min_order_idr:
                logger.info(
                    "Skipping staged step: Rp%.0f < min Rp%.0f (pair=%s)",
                    step_amount * reference_price,
                    self.config.min_order_idr,
                    snapshot["pair"],
                )
                continue
            try:
                order_resp = self.client.create_order(snapshot["pair"], decision.action, reference_price, step_amount)
                self._consecutive_errors = 0
                # Invalidate balance cache so next getInfo reflects the new order.
                getattr(self.client, "invalidate_account_info_cache", lambda: None)()
            except Exception as exc:
                self._consecutive_errors += 1
                if self.config.circuit_breaker_max_errors > 0 and self._consecutive_errors >= self.config.circuit_breaker_max_errors:
                    self._circuit_breaker_until = time.time() + self.config.circuit_breaker_pause_seconds
                    logger.warning("Circuit breaker triggered after %d errors: %s", self._consecutive_errors, exc)
                raise

            # Indodax returns ``receive_<coin>`` in the trade response.  When a
            # buy limit order is merely placed (not filled) this value is 0.
            # Avoid opening a phantom position in that case; cancel and retry
            # once with a slightly more aggressive price to avoid missing the move.
            received_amount = _calc_received_amount(order_resp, pre_step_position)

            if decision.action == "buy" and received_amount <= 0:
                order_id = str((order_resp.get("return") or {}).get("order_id") or order_resp.get("order_id") or "")
                if order_id:
                    try:
                        self.client.cancel_order(snapshot["pair"], order_id, decision.action)
                        logger.info("Cancelled unfilled buy order %s on %s before retrying", order_id, snapshot["pair"])
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("Failed to cancel unfilled buy order %s on %s: %s", order_id, snapshot["pair"], exc)

                retry_bump = max(entry_aggr, self.config.entry_retry_aggressiveness_pct)
                retry_price = reference_price
                if retry_bump > 0:
                    retry_price = min(reference_price * (1 + retry_bump), allowed_max)
                    _fmt_price = getattr(self.client, "format_price", None)
                    if callable(_fmt_price):
                        # format_price returns (price, precision) tuple
                        retry_price, _ = _fmt_price(snapshot["pair"], retry_price)
                if retry_price > reference_price:
                    logger.info(
                        "Retrying buy at more aggressive price: %.10f → %.10f (pair=%s)",
                        reference_price,
                        retry_price,
                        snapshot["pair"],
                    )
                    order_resp = self.client.create_order(snapshot["pair"], decision.action, retry_price, step_amount)
                    getattr(self.client, "invalidate_account_info_cache", lambda: None)()
                    reference_price = retry_price
                    received_amount = _calc_received_amount(order_resp, pre_step_position)

                if received_amount <= 0:
                    logger.info(
                        "Buy order not filled after retry (pair=%s) — will re-evaluate on next cycle",
                        snapshot["pair"],
                    )
                    remaining_amount -= step_amount
                    executed_steps.append(
                        {"amount": 0.0, "price": reference_price, "order": order_resp, "pending": True}
                    )
                    continue

            step_amount = min(step_amount, received_amount)
            _tracker.record_trade(decision.action, reference_price, step_amount)
            remaining_amount -= step_amount
            executed_steps.append({"amount": step_amount, "price": reference_price, "order": order_resp})
            logger.info(
                "Placed order action=%s amount=%s price=%s response=%s",
                decision.action,
                step_amount,
                reference_price,
                order_resp,
            )
        # Guard: if every staged step was below min_order, nothing was placed.
        if not executed_steps:
            if decision.action == "buy" and self.multi_manager is not None:
                self.multi_manager.return_position_cash(_pair)
            logger.info(
                "All staged steps below min Rp%.0f — skipping %s on %s",
                self.config.min_order_idr,
                decision.action,
                snapshot["pair"],
            )
            return {
                "status": "skipped",
                "reason": f"all_steps_below_min_order (min=Rp{self.config.min_order_idr:.0f})",
                "portfolio": _tracker.as_dict(reference_price),
            }
        # Multi-position: return cash to pool after a full sell-close (live path).
        if decision.action == "sell" and self.multi_manager is not None and _tracker.base_position <= 0:
            self.multi_manager.return_position_cash(_pair)
        # Manage per-pair realtime feeds: start on first buy, stop on full close.
        if decision.action == "buy" and _tracker.base_position > 0:
            self._ensure_position_feed(_pair)
        elif decision.action == "sell" and _tracker.base_position <= 0:
            self._remove_position_feed(_pair)
        outcome = {
            "status": "placed",
            "action": decision.action,
            "price": reference_price,
            "amount": sum(step["amount"] for step in executed_steps),
            "executed_steps": executed_steps,
            "mode": decision.mode,
            "portfolio": _tracker.as_dict(reference_price),
        }
        self._persist_after_trade(snapshot["pair"])
        # Journal logging
        if self.journal is not None:
            import datetime as _dt
            _ts = time.time()
            _dt_str = _dt.datetime.fromtimestamp(_ts).strftime("%Y-%m-%d %H:%M:%S")
            _pnl = (reference_price - _pre_trade_avg_cost) * effective_amount if decision.action == "sell" else 0.0
            self.journal.log_trade(
                timestamp=_ts,
                datetime_str=_dt_str,
                pair=snapshot["pair"],
                action=decision.action,
                price=reference_price,
                amount=sum(step["amount"] for step in executed_steps),
                idr_value=reference_price * sum(step["amount"] for step in executed_steps),
                pnl=_pnl,
                strategy=decision.mode,
                confidence=decision.confidence,
                reason=decision.reason,
                avg_cost=_tracker.avg_cost,
                equity=_tracker.snapshot(reference_price).equity,
            )
        # Strategy auto-disable after sell
        if decision.action == "sell" and self.config.strategy_auto_disable_losses > 0:
            strat_data = _tracker._strategy_stats.get(decision.mode, {})
            losses_count = strat_data.get("consecutive_losses", 0)
            if losses_count >= self.config.strategy_auto_disable_losses:
                _tracker.disable_strategy(decision.mode)
                logger.warning("Strategy %s auto-disabled after %d consecutive losses", decision.mode, losses_count)
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
        min_idr = self.config.min_order_idr
        filtered_orders: List[GridOrder] = []
        for order in orders:
            order_value_idr = order.amount * order.price
            if min_idr > 0 and order_value_idr < min_idr:
                logger.debug(
                    "Skipping grid order %s %s @ %s: Rp%.0f < min Rp%.0f",
                    order.side,
                    order.amount,
                    order.price,
                    order_value_idr,
                    min_idr,
                )
                continue
            filtered_orders.append(order)

        if not filtered_orders:
            return {
                "status": "skipped",
                "reason": f"all_grid_orders_below_min_order (min=Rp{min_idr:.0f})",
                "portfolio": self.tracker.as_dict(price),
            }

        for order in filtered_orders:
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
        _tracker = self._active_tracker(pair)
        total_position = _tracker.base_position

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
            _tracker.record_trade("sell", reference_price, amount)
            _tracker.partial_tp_taken = True
            outcome: Dict[str, Any] = {
                "status": "partial_tp",
                "action": "sell",
                "pair": pair,
                "price": reference_price,
                "amount": amount,
                "fraction": fraction,
                "portfolio": _tracker.as_dict(reference_price),
            }
            self._persist_after_trade(pair)
            return outcome

        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        order_resp = self.client.create_order(pair, "sell", reference_price, amount)
        _tracker.record_trade("sell", reference_price, amount)
        _tracker.partial_tp_taken = True
        outcome = {
            "status": "partial_tp",
            "action": "sell",
            "pair": pair,
            "price": reference_price,
            "amount": amount,
            "fraction": fraction,
            "order": order_resp,
            "portfolio": _tracker.as_dict(reference_price),
        }
        self._persist_after_trade(pair)
        return outcome

    def check_momentum_exit(self, snapshot: Dict[str, Any]) -> bool:
        """Return ``True`` when weakening momentum justifies an early exit.

        This is the *adaptive* counterpart to the conditional-TP logic: instead
        of deciding whether to *hold past* a profit target, this method decides
        whether to *exit early* (before the target) when market conditions
        deteriorate.  It protects open profits by closing the position as soon
        as the book turns bearish, even if the fixed TP has not been reached.

        Both thresholds must be configured for the check to activate.
        ``momentum_exit_min_profit_pct > 0`` is the enabling condition;
        ``momentum_exit_ob_threshold`` may be ``0.0`` (exit when imbalance
        turns negative, i.e. seller dominant).

        Parameters
        ----------
        snapshot:
            Market snapshot as returned by :meth:`analyze_market`.

        Returns
        -------
        bool
            ``True`` when momentum has faded and the position has enough
            unrealised profit to justify an early close.  ``False`` otherwise
            (hold, or feature disabled).
        """
        config = self.config
        # Feature is off unless the minimum-profit threshold is configured.
        # momentum_exit_ob_threshold can be 0.0 (meaning exit when imbalance
        # goes negative / seller dominant), so only the profit guard enables/
        # disables the feature.
        if config.momentum_exit_min_profit_pct == 0:
            return False

        price = snapshot.get("price", 0.0)
        orderbook = snapshot.get("orderbook")
        _tracker = self._active_tracker(snapshot.get("pair", self.config.pair))
        if not price or _tracker.avg_cost == 0:
            return False

        # Require minimum unrealised profit before triggering
        unrealised_pct = (price - _tracker.avg_cost) / _tracker.avg_cost
        if unrealised_pct < config.momentum_exit_min_profit_pct:
            return False

        # Check order-book imbalance — exit when sellers have taken over
        imbalance = getattr(orderbook, "imbalance", 0.0) if orderbook else 0.0
        if imbalance < config.momentum_exit_ob_threshold:
            logger.info(
                "Momentum exit: imbalance=%.3f < threshold=%.3f at profit=%.2f%% — exiting early",
                imbalance, config.momentum_exit_ob_threshold, unrealised_pct * 100,
            )
            return True

        return False

    def check_post_entry_dump(self, tracker: "PortfolioTracker", price: float) -> bool:
        """Return ``True`` when price dumps shortly after entry and an immediate exit is desired."""
        cfg = self.config
        if cfg.post_entry_dump_pct <= 0 or tracker.avg_cost <= 0 or tracker.base_position <= 0:
            return False
        drop_pct = (tracker.avg_cost - price) / tracker.avg_cost
        if drop_pct < cfg.post_entry_dump_pct:
            return False
        window = cfg.post_entry_dump_window_seconds
        if window > 0 and tracker.position_hold_seconds > window:
            return False
        logger.info(
            "Post-entry dump detected: price=%.6g avg_cost=%.6g drop=%.2f%% age=%.0fs window=%.0fs",
            price,
            tracker.avg_cost,
            drop_pct * 100,
            tracker.position_hold_seconds,
            window,
        )
        return True

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
        _tracker = self._active_tracker(snapshot["pair"])

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
            _tracker.trailing_tp_stop is not None
            and price <= _tracker.trailing_tp_stop
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
            _tracker.activate_trailing_tp(price, config.trailing_tp_pct)
            logger.debug(
                "Dynamic TP: trailing floor updated to %.2f (price=%.2f)",
                _tracker.trailing_tp_stop or 0.0,
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
        _tracker = self._active_tracker(pair)
        amount = _tracker.base_position

        if amount <= 0:
            return {"status": "no_position", "pair": pair, "price": price}

        # Cancel any open orders (buy or sell) for this pair to avoid conflicts
        # that would make the exit loop retry without actually liquidating.
        self._cancel_open_orders(pair, ("buy", "sell"))

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
            _tracker.record_trade("sell", reference_price, amount)
            if self.multi_manager is not None and _tracker.base_position <= 0:
                self.multi_manager.return_position_cash(pair)
            if _tracker.base_position <= 0:
                self._remove_position_feed(pair)
            outcome: Dict[str, Any] = {
                "status": "force_sold",
                "action": "sell",
                "pair": pair,
                "price": reference_price,
                "amount": amount,
                "portfolio": _tracker.as_dict(reference_price),
            }
            self._persist_after_trade(pair)
            return outcome

        if self.config.api_key is None:
            raise ValueError("API credentials required for live trading")

        # Verify the actual coin balance before placing the sell order.
        # A buy limit order may have been placed and recorded internally but not
        # yet filled by the exchange (receive_<coin>=0 in the order response).
        # In that case the exchange balance is zero while the tracker thinks a
        # position is held.  We detect this by fetching the live account balance
        # and, when the discrepancy is larger than 1 %, cancel any open buy
        # orders so the IDR is returned to the account and then either sell the
        # actual available coins or roll back the phantom position cleanly.
        base_coin = pair.split("_")[0].lower()
        try:
            acct_info = self.client.get_account_info()
            balance_dict = (acct_info.get("return") or {}).get("balance") or {}
            actual_balance = float(balance_dict.get(base_coin) or "0")
            if actual_balance < amount * 0.99:
                logger.warning(
                    "force_sell: exchange balance %.8f %s < tracked %.8f — "
                    "cancelling open buy orders for %s",
                    actual_balance,
                    base_coin,
                    amount,
                    pair,
                )
                # Cancel any pending buy orders so the reserved IDR is freed.
                try:
                    open_resp = self.client.open_orders(pair)
                    orders = (open_resp.get("return") or {}).get("orders") or []
                    for order in (orders if isinstance(orders, list) else []):
                        order_type = str(order.get("type", "")).lower()
                        if order_type == "buy":
                            self.client.cancel_order(pair, str(order["order_id"]), order_type)
                            logger.info(
                                "Cancelled pending buy order %s for %s",
                                order["order_id"],
                                pair,
                            )
                except Exception as exc:
                    logger.warning(
                        "force_sell: failed to cancel open orders for %s: %s",
                        pair,
                        exc,
                    )
                if actual_balance <= 0:
                    # No coins available — the buy was never filled.  Roll back
                    # the phantom position so the tracker reflects reality.
                    _tracker.cancel_pending_buy()
                    if self.multi_manager is not None:
                        self.multi_manager.return_position_cash(pair)
                    outcome: Dict[str, Any] = {
                        "status": "no_position",
                        "pair": pair,
                        "price": reference_price,
                        "amount": 0.0,
                        "portfolio": _tracker.as_dict(reference_price),
                    }
                    self._persist_after_trade(pair)
                    return outcome
                # Use the actual available balance for the sell.
                amount = actual_balance
        except Exception as exc:
            logger.warning(
                "force_sell: could not verify balance for %s: %s — "
                "proceeding with tracked amount",
                pair,
                exc,
            )

        # ── Per-pair minimum order check (auto-adjust / skip dust) ───────────
        # Check the cached per-pair minimum before sending the sell order.
        # If the amount falls below the exchange floor, treat it as unsellable
        # "dust" and clear the position without submitting an API call (which
        # would raise "Minimum order X COIN" and cause repeated error loops).
        _get_min = getattr(self.client, "get_pair_min_order", None)
        min_info = _get_min(pair) if callable(_get_min) else {}
        min_idr_per_pair = min_info.get("min_idr", 0.0)
        min_coin_per_pair = min_info.get("min_coin", 0.0)
        sell_idr_value = amount * reference_price

        # Indodax enforces minimum order size by total IDR value, not coin amount.
        # Keep the effective threshold in rupiah to mirror that rule.
        effective_min_idr = max(self.config.min_order_idr, min_idr_per_pair)
        is_below_coin_min = min_coin_per_pair > 0 and amount < min_coin_per_pair
        is_below_idr_min = effective_min_idr > 0 and sell_idr_value < effective_min_idr

        # ── Dust check (coin + IDR) ─────────────────────────────────────
        # When both the coin amount and the IDR value are below their
        # respective minimums, the position is truly unsellable dust —
        # clear it so the bot can move on.
        if is_below_coin_min and is_below_idr_min:
            logger.warning(
                "force_sell: amount %.8f %s (Rp %.0f) is below exchange minimums "
                "(min_coin=%.8f, min_idr=%.0f) — clearing dust position",
                amount,
                pair.split("_")[0].upper(),
                sell_idr_value,
                min_coin_per_pair,
                effective_min_idr,
            )
            _tracker.cancel_pending_buy()
            if self.multi_manager is not None:
                self.multi_manager.return_position_cash(pair)
            outcome = {
                "status": "dust_cleared",
                "pair": pair,
                "price": reference_price,
                "amount": amount,
                "min_idr": effective_min_idr,
                "portfolio": _tracker.as_dict(reference_price),
            }
            self._persist_after_trade(pair)
            return outcome

        # ── IDR-level minimum check (exchange-reported) ──────────────────
        # When the exchange has a per-pair IDR minimum and the sell value
        # is below it, keep the position for monitoring rather than
        # discarding it — the position may grow or be aggregated later.
        if min_idr_per_pair > 0 and sell_idr_value < min_idr_per_pair:
            logger.warning(
                "force_sell: sell amount %.8f %s (Rp %.0f) is below exchange minimum "
                "(min_idr=%.0f) — skipping sell and keeping position to monitor",
                amount,
                pair.split("_")[0].upper(),
                sell_idr_value,
                min_idr_per_pair,
            )
            return {
                "status": "below_minimum",
                "pair": pair,
                "price": reference_price,
                "amount": amount,
                "min_idr": min_idr_per_pair,
                "portfolio": _tracker.as_dict(reference_price),
            }

        # ── Place the sell order ─────────────────────────────────────────────
        try:
            order_resp = self.client.create_order(pair, "sell", reference_price, amount)
        except RuntimeError as exc:
            # If the exchange still rejects with a "Minimum order" error
            # (e.g. min_coin cache is stale or not populated), parse the
            # exchange-reported minimum and clear the position as dust.
            exc_str = str(exc)
            _parse_min = getattr(self.client, "parse_minimum_order_error", None)
            parsed_min = _parse_min(exc_str) if callable(_parse_min) else None
            if parsed_min is None:
                # Fallback: detect via the error text directly
                import re as _re
                _m = _re.search(r"Minimum order\s+([\d.]+)", exc_str, _re.IGNORECASE)
                if _m:
                    try:
                        parsed_min = float(_m.group(1))
                    except ValueError:
                        pass
            if parsed_min is not None:
                parsed_min_idr = parsed_min * reference_price
                if parsed_min_idr > 0:
                    effective_min_idr = max(effective_min_idr, parsed_min_idr)
                    if sell_idr_value < effective_min_idr:
                        logger.warning(
                            "force_sell: caught minimum order error for %s "
                            "(value=Rp%.0f < min_idr=Rp%.0f) — clearing dust position",
                            pair,
                            sell_idr_value,
                            effective_min_idr,
                        )
                        # Update the cache with the exchange-reported minimum so
                        # future attempts won't hit the same error.
                        _cache = getattr(self.client, "_pair_min_order", None)
                        if isinstance(_cache, dict):
                            cached = _cache.setdefault(pair.lower(), {})
                            if effective_min_idr > cached.get("min_idr", 0.0):
                                cached["min_idr"] = effective_min_idr
                        _tracker.cancel_pending_buy()
                        if self.multi_manager is not None:
                            self.multi_manager.return_position_cash(pair)
                        outcome = {
                            "status": "dust_cleared",
                            "pair": pair,
                            "price": reference_price,
                            "amount": amount,
                            "portfolio": _tracker.as_dict(reference_price),
                        }
                        self._persist_after_trade(pair)
                        return outcome
            # Amount was not below the parsed minimum — bubble up so the caller
            # sees the real failure instead of silently clearing the position.
            raise

        _tracker.record_trade("sell", reference_price, amount)
        if self.multi_manager is not None and _tracker.base_position <= 0:
            self.multi_manager.return_position_cash(pair)
        if _tracker.base_position <= 0:
            self._remove_position_feed(pair)
        # Invalidate balance cache so next getInfo reflects the sell proceeds.
        getattr(self.client, "invalidate_account_info_cache", lambda: None)()
        # Invalidate open-orders cache for this pair; the order is now closed.
        getattr(self.client, "invalidate_open_orders_cache", lambda p: None)(pair)
        outcome = {
            "status": "force_sold",
            "action": "sell",
            "pair": pair,
            "price": reference_price,
            "amount": amount,
            "order": order_resp,
            "portfolio": _tracker.as_dict(reference_price),
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
        skip_depth: bool = False,
        skip_trades: bool = False,
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
                return self.analyze_market(
                    pair,
                    prefetched_ticker=prefetched_ticker,
                    skip_depth=skip_depth,
                    skip_trades=skip_trades,
                )
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
                    last_exc = exc
                    # Don't rely on a stale prefetched ticker on retry; let the
                    # call fetch it fresh via REST.
                    prefetched_ticker = None
                    # On retry, disable skip_trades so fresh trades data is
                    # fetched with the fresh ticker.
                    skip_trades = False
                    # Only sleep when there is a subsequent retry; sleeping after
                    # the final attempt would block the scan for no benefit since
                    # the exception is raised immediately after the loop.
                    if attempt < self._MAX_SCAN_RETRIES - 1:
                        time.sleep(backoff)
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
                # Populate the per-pair min-order cache from the already-fetched
                # pairs data so we avoid a second /api/pairs request.  This
                # prevents the duplicate REST call that previously triggered 429
                # errors on the first cycle (one from _ensure_pair_min_order_cache
                # and one here).
                if self.config.pair_min_order_cache_enabled and hasattr(self.client, "load_pair_min_orders"):
                    try:
                        self.client.load_pair_min_orders(pairs_data)
                        self._pair_min_order_cache_cycles = 0
                    except Exception as exc:  # pragma: no cover
                        logger.debug("Failed to populate min order cache from pairs data: %s", exc)
            # pragma: no cover - guard for pair listing/parsing failures
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                logger.warning("Failed to load pairs; fallback to default %s", exc)
                self._all_pairs = [self.config.pair]

        # ── Per-pair minimum order cache ─────────────────────────────────────
        # On the first cycle this is a no-op when the cache was populated above
        # from the same /api/pairs response.  On subsequent cycles it handles
        # periodic refresh according to pair_min_order_refresh_cycles.
        self._ensure_pair_min_order_cache()

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
            # Apply dynamic pair selection immediately after seeding so the
            # very first scan cycle already uses the top-N watchlist instead
            # of all 500+ pairs.  This prevents the rate-limit burst that
            # would otherwise occur on the first cycle.
            if self.config.dynamic_pairs_refresh_cycles > 0 and self._multi_feed.is_seeded:
                self._refresh_dynamic_pairs()
                all_pairs = self._all_pairs or all_pairs
            else:
                # No dynamic refresh yet — subscribe to depth/trades for the
                # initial watchlist so orderbook data is available from the
                # very first scan cycle.
                self._multi_feed.subscribe_depth_pairs(all_pairs)

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
            # Multi-position: skip pairs where we already hold a position.
            # Those are monitored separately in the main holding loop.
            if self.multi_manager is not None and self.multi_manager.has_position(pair):
                skipped_pairs.append(pair)
                continue
            # Use the multi-pair feed's cached ticker to skip the per-pair REST
            # ticker call entirely.  When the feed is seeded but this specific
            # pair has no cached data (absent from /api/summaries — typically
            # inactive or very-new pairs), skip it rather than falling through
            # to a REST call that would trigger a 429.
            prefetched_ticker = self._multi_feed.get_ticker(pair)
            if prefetched_ticker is None and feed_seeded:
                skipped_pairs.append(pair)
                continue
            # ── Hard floor: drop ultra-cheap coins before analysis ────────────
            if prefetched_ticker is not None and self.config.min_coin_price_idr > 0:
                try:
                    _last_price = float(
                        prefetched_ticker.get("last")
                        or prefetched_ticker.get("last_price")
                        or 0
                    )
                    if 0 < _last_price < self.config.min_coin_price_idr:
                        skipped_pairs.append(pair)
                        continue
                except (ValueError, TypeError):
                    pass

            # ── Pre-scan cheap coin filter ────────────────────────────────────
            # For coins priced below min_buy_price_idr, check real-time WS
            # orderbook quality before running the full analysis.  Sepi/stuck
            # coins (thin book, wide spread, too few levels) are dropped from
            # the scan entirely so they never appear in the best-hold result.
            # Only applied when WS depth data is already available for the pair.
            if self.config.min_buy_price_idr > 0 and prefetched_ticker is not None:
                try:
                    _last_price = float(
                        prefetched_ticker.get("last")
                        or prefetched_ticker.get("last_price")
                        or 0
                    )
                    if 0 < _last_price < self.config.min_buy_price_idr:
                        _vol_24h = self._extract_volume_idr(prefetched_ticker)
                        if (
                            self.config.small_coin_min_volume_24h_idr > 0
                            and _vol_24h < self.config.small_coin_min_volume_24h_idr
                        ):
                            logger.debug(
                                "Pre-scan: skipping cheap low-volume coin %s "
                                "(price=%.6g IDR < %.6g, vol24h=%.0f < %.0f)",
                                pair,
                                _last_price,
                                self.config.min_buy_price_idr,
                                _vol_24h,
                                self.config.small_coin_min_volume_24h_idr,
                            )
                            skipped_pairs.append(pair)
                            continue
                        _trades_24h = self._extract_trade_count_24h(prefetched_ticker)
                        trade_reason = self._small_coin_low_trade_reason(_trades_24h)
                        if trade_reason:
                            logger.debug(
                                "Pre-scan: skipping cheap low-trade coin %s "
                                "(price=%.6g IDR < %.6g, %s)",
                                pair,
                                _last_price,
                                self.config.min_buy_price_idr,
                                trade_reason,
                            )
                            skipped_pairs.append(pair)
                            continue
                        _ws_depth = self._multi_feed.get_depth(pair)
                        if _ws_depth is not None:
                            _bids = _ws_depth.get("buy", [])
                            _asks = _ws_depth.get("sell", [])
                            _top_bid = float(_bids[0][0]) if _bids else 0.0
                            _top_ask = float(_asks[0][0]) if _asks else 0.0
                            _skip_reason = self._check_small_coin_ob_quality(
                                _bids, _top_bid, _top_ask
                            )
                            if _skip_reason:
                                logger.debug(
                                    "Pre-scan: skipping sepi/stuck cheap coin %s "
                                    "(price=%.6g IDR < %.6g): %s",
                                    pair,
                                    _last_price,
                                    self.config.min_buy_price_idr,
                                    _skip_reason,
                                )
                                skipped_pairs.append(pair)
                                continue
                except (ValueError, TypeError):
                    pass  # price missing → keep pair and let full analysis decide
            # When WebSocket ticker data is available, the skip_depth / skip_trades
            # flags suppress per-pair REST calls during the scan loop.
            # analyze_market() now uses real-time WS depth and trades from
            # MultiPairFeed when available (market:order-book-{pair} and
            # market:trade-activity-{pair} channels), so orderbook imbalance,
            # spread, whale detection, and trade-flow are all live during scanning.
            # Candles are built from the WS trade buffer when enough data exists,
            # falling back to REST OHLC only when the buffer is too small.
            # This eliminates stale-cache and empty-signal issues in pair selection.
            scan_skip_depth = prefetched_ticker is not None
            scan_skip_trades = prefetched_ticker is not None
            try:
                snapshot = self._analyze_with_retry(
                    pair,
                    prefetched_ticker=prefetched_ticker,
                    skip_depth=scan_skip_depth,
                    skip_trades=scan_skip_trades,
                )
            # pragma: no cover - guard for per-pair API/parse failures
            except (requests.RequestException, RuntimeError, ValueError) as exc:
                logger.warning("Failed to analyze %s: %s", pair, exc)
                failed_pairs.append(pair)
                continue
            except Exception as exc:  # noqa: BLE001 — isolate unexpected per-pair failures
                # Catch KeyError, AttributeError, TypeError, etc. so one bad pair
                # doesn't abort the entire scan cycle.
                logger.warning("Unexpected error analyzing %s: %s", pair, exc, exc_info=True)
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
            # A SELL signal is only actionable when we actually hold a position.
            # Without one, it would be immediately skipped in maybe_execute, so
            # treat it as "hold" here so the scanner continues looking for BUY
            # opportunities on other pairs instead of returning early.
            if decision.action == "sell" and self._active_tracker(pair).base_position <= 0:
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
            _scan_min_conf = self._min_confidence_threshold(snapshot)
            if decision.confidence >= _scan_min_conf:
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
        # Use the retry wrapper here as well so 429s during the fallback
        # analysis are handled consistently with the main scan loop.  If the
        # final retries still fail, the exception is propagated so the caller
        # can back off rather than silently using stale data.
        return best_pair, self._analyze_with_retry(best_pair)
