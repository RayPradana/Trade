"""Professional low-latency trading engine architecture.

Splits the monolithic trading loop into six separate, concurrent engines so
that slow I/O (REST / WebSocket market-data fetching) never blocks strategy
evaluation or order execution:

    ┌──────────────────────────────────────────────────────────────────┐
    │  MarketDataEngine  (daemon thread)                               │
    │    – calls trader.analyze_market() for every watched pair        │
    │    – stores snapshots in DataCache                               │
    ├──────────────────────────────────────────────────────────────────┤
    │  DataCache  (thread-safe in-memory store)                        │
    │    – keyed by pair; each entry holds the latest snapshot + ts    │
    ├──────────────────────────────────────────────────────────────────┤
    │  StrategyEngine  (daemon thread)                                 │
    │    – reads fresh snapshots from DataCache                        │
    │    – evaluates buy / hold / exit / resume conditions             │
    │    – publishes TradingSignal to SignalQueue                      │
    ├──────────────────────────────────────────────────────────────────┤
    │  SignalQueue  (thread-safe queue.Queue)                          │
    │    – decouples strategy evaluation from order execution          │
    ├──────────────────────────────────────────────────────────────────┤
    │  ExecutionEngine  (daemon thread)                                │
    │    – consumes TradingSignal objects                              │
    │    – asks RiskEngine for pre-execution approval                  │
    │    – calls trader.maybe_execute() / force_sell() / resume_*()   │
    ├──────────────────────────────────────────────────────────────────┤
    │  RiskEngine  (synchronous validator)                             │
    │    – circuit-breaker, daily-loss cap, price sanity check         │
    └──────────────────────────────────────────────────────────────────┘

Usage::

    orchestrator = TradingOrchestrator(trader, config, notify_fn=_notify)
    orchestrator.start()          # launch all background threads
    orchestrator.run(shutdown)    # blocking monitor loop; returns on shutdown
    orchestrator.stop()           # gracefully join all threads
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------

@dataclass
class MarketSnapshot:
    """A cached result of ``trader.analyze_market()`` for one pair."""

    pair: str
    snapshot: Dict[str, Any]
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class TradingSignal:
    """A signal produced by StrategyEngine and consumed by ExecutionEngine.

    ``signal_type`` is one of:
      * ``"buy"``          – open a new position
      * ``"exit"``         – close the current position
      * ``"resume_buy"``   – retry an unfilled buy order
      * ``"resume_sell"``  – retry an unfilled sell order
    """

    pair: str
    snapshot: Dict[str, Any]
    signal_type: str  # "buy" | "exit" | "resume_buy" | "resume_sell"
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# 1. Data Cache Layer
# ---------------------------------------------------------------------------

class DataCache:
    """Thread-safe in-memory cache for per-pair market snapshots.

    ``MarketDataEngine`` writes here; ``StrategyEngine`` reads here.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._store: Dict[str, MarketSnapshot] = {}

    # -- writes ---------------------------------------------------------------

    def put(self, pair: str, snapshot: Dict[str, Any]) -> None:
        """Store (or replace) the latest snapshot for *pair*."""
        with self._lock:
            self._store[pair] = MarketSnapshot(pair=pair, snapshot=snapshot)

    # -- reads ----------------------------------------------------------------

    def get(self, pair: str) -> Optional[Dict[str, Any]]:
        """Return the cached snapshot for *pair*, or ``None`` if absent."""
        with self._lock:
            entry = self._store.get(pair)
            return entry.snapshot if entry is not None else None

    def get_all(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of ``{pair: snapshot}`` for all cached pairs."""
        with self._lock:
            return {p: e.snapshot for p, e in self._store.items()}

    def age_seconds(self, pair: str) -> float:
        """Seconds since the cached snapshot for *pair* was last updated.

        Returns ``float("inf")`` when *pair* has no cached entry.
        """
        with self._lock:
            entry = self._store.get(pair)
            if entry is None:
                return float("inf")
            return time.monotonic() - entry.timestamp

    def pairs(self) -> List[str]:
        """List of pairs currently stored in the cache."""
        with self._lock:
            return list(self._store.keys())


# ---------------------------------------------------------------------------
# 2. Signal Queue
# ---------------------------------------------------------------------------

class SignalQueue:
    """Thread-safe queue of :class:`TradingSignal` objects.

    Only the *latest* signal for each pair is retained; if a newer signal
    for the same pair arrives before the previous one is consumed the older
    signal is silently discarded.  This prevents the execution engine from
    acting on stale decisions when the strategy engine is faster than the
    exchange.
    """

    def __init__(self, maxsize: int = 200) -> None:
        self._queue: queue.Queue[TradingSignal] = queue.Queue(maxsize=maxsize)
        # latest signal type seen per pair (for deduplication)
        self._latest: Dict[str, str] = {}
        self._lock = threading.Lock()

    def put(self, signal: TradingSignal) -> None:
        """Enqueue *signal*, evicting the oldest entry when the queue is full."""
        with self._lock:
            self._latest[signal.pair] = signal.signal_type
        try:
            self._queue.put_nowait(signal)
        except queue.Full:
            # Drop the oldest signal to make room for the new one so that
            # the execution engine never falls behind by more than maxsize.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(signal)
            except queue.Full:
                logger.warning(
                    "SignalQueue still full after eviction — dropping signal for %s",
                    signal.pair,
                )

    def get(self, timeout: float = 0.5) -> Optional[TradingSignal]:
        """Block for up to *timeout* seconds and return the next signal, or
        ``None`` when the queue is empty."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def size(self) -> int:
        """Approximate number of signals currently in the queue."""
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# 3. Market Data Engine
# ---------------------------------------------------------------------------

class MarketDataEngine:
    """Continuously fetches market data for all watched pairs.

    Runs as a daemon thread.  On every iteration it calls
    ``trader.analyze_market(pair)`` for each pair and writes the result into
    *cache*.  The I/O-heavy REST / WebSocket calls happen here so that
    :class:`StrategyEngine` can always read *fresh* data without blocking.

    Args:
        trader: The :class:`~bot.trader.Trader` instance.
        config: The :class:`~bot.config.BotConfig` instance.
        cache:  The shared :class:`DataCache`.
        interval: Seconds to wait between full-pass refresh cycles.
                  Defaults to ``config.market_data_engine_interval`` when set,
                  otherwise ``2.0``.
    """

    def __init__(
        self,
        trader: Any,
        config: Any,
        cache: DataCache,
        *,
        interval: Optional[float] = None,
    ) -> None:
        self._trader = trader
        self._config = config
        self._cache = cache
        self._interval: float = (
            interval
            if interval is not None
            else float(getattr(config, "market_data_engine_interval", 2.0))
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error_count: int = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the background data-fetch thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="market-data-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "▶ MarketDataEngine started  (interval=%.1fs)", self._interval
        )

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)

    def is_alive(self) -> bool:
        """``True`` when the background thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # -- internals ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            pairs = self._get_pairs()
            for pair in pairs:
                if self._stop.is_set():
                    break
                try:
                    snap = self._trader.analyze_market(pair)
                    self._cache.put(pair, snap)
                    self.error_count = 0
                except Exception as exc:
                    self.error_count += 1
                    logger.debug(
                        "MarketDataEngine: analyze_market(%s) failed — %s",
                        pair,
                        exc,
                    )
            self._stop.wait(timeout=self._interval)

    def _get_pairs(self) -> List[str]:
        """Return the list of pairs to refresh this iteration.

        Priority order:
        1. Pairs with open / pending positions (always refreshed first).
        2. All other pairs from ``_all_pairs`` / configured default pair.
        """
        try:
            active = list(self._trader.active_positions.keys())
        except Exception:
            active = []

        try:
            all_pairs: List[str] = list(self._trader._all_pairs or [self._config.pair])
        except Exception:
            all_pairs = [self._config.pair]

        # Active pairs first (higher priority), then the rest deduplicated.
        seen = set(active)
        ordered = list(active)
        for p in all_pairs:
            if p not in seen:
                seen.add(p)
                ordered.append(p)
        return ordered or [self._config.pair]


# ---------------------------------------------------------------------------
# 4. Strategy Engine
# ---------------------------------------------------------------------------

class StrategyEngine:
    """Evaluates trading decisions from cached market data and emits signals.

    Runs as a daemon thread.  On every iteration it reads all cached
    snapshots from :class:`DataCache`, evaluates conditions (exit, resume,
    buy), and pushes :class:`TradingSignal` objects to :class:`SignalQueue`.

    Args:
        trader:  The :class:`~bot.trader.Trader` instance.
        config:  The :class:`~bot.config.BotConfig` instance.
        cache:   The shared :class:`DataCache` (read-only from this engine).
        signals: The :class:`SignalQueue` to publish signals on.
        interval: Seconds between evaluation passes.  Defaults to
                  ``config.strategy_engine_interval`` when set, otherwise
                  ``3.0``.
    """

    def __init__(
        self,
        trader: Any,
        config: Any,
        cache: DataCache,
        signals: SignalQueue,
        *,
        interval: Optional[float] = None,
    ) -> None:
        self._trader = trader
        self._config = config
        self._cache = cache
        self._signals = signals
        self._interval: float = (
            interval
            if interval is not None
            else float(getattr(config, "strategy_engine_interval", 3.0))
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error_count: int = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the background strategy-evaluation thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="strategy-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "▶ StrategyEngine started  (interval=%.1fs)", self._interval
        )

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5.0)

    def is_alive(self) -> bool:
        """``True`` when the background thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # -- internals ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            snapshots = self._cache.get_all()
            for pair, snap in snapshots.items():
                if self._stop.is_set():
                    break
                try:
                    self._evaluate(pair, snap)
                    self.error_count = 0
                except Exception as exc:
                    self.error_count += 1
                    logger.debug(
                        "StrategyEngine: evaluate(%s) failed — %s", pair, exc
                    )
            self._stop.wait(timeout=self._interval)

    def _evaluate(self, pair: str, snap: Dict[str, Any]) -> None:
        """Emit at most one :class:`TradingSignal` for *pair* based on *snap*."""
        decision = snap.get("decision")
        if decision is None:
            return

        price: float = snap.get("price") or 0.0
        tracker = self._trader._active_tracker(pair)

        # ── Update trailing stops for open positions (non-blocking) ─────────
        if tracker.base_position > 0:
            try:
                if self._config.trailing_stop_pct > 0:
                    tracker.update_trailing_stop(price, self._config.trailing_stop_pct)
                if self._config.trailing_tp_pct > 0:
                    tracker.activate_trailing_tp(price, self._config.trailing_tp_pct)
            except Exception:
                pass

        # ── Pending buy resume ───────────────────────────────────────────────
        if getattr(tracker, "has_pending_buy", False):
            self._signals.put(
                TradingSignal(pair=pair, snapshot=snap, signal_type="resume_buy")
            )
            return

        # ── Pending sell resume ──────────────────────────────────────────────
        if getattr(tracker, "has_pending_sell", False):
            self._signals.put(
                TradingSignal(pair=pair, snapshot=snap, signal_type="resume_sell")
            )
            return

        # ── Exit conditions (held position) ──────────────────────────────────
        if tracker.base_position > 0:
            stop_reason = tracker.stop_reason(price)
            # Momentum / post-entry-dump checks
            if stop_reason is None:
                try:
                    if self._trader.check_momentum_exit(snap):
                        stop_reason = "momentum_exit"
                except Exception:
                    pass
            if stop_reason is None:
                try:
                    if self._trader.check_post_entry_dump(tracker, price):
                        stop_reason = "post_entry_dump"
                except Exception:
                    pass

            # Dynamic TP override
            if stop_reason == "target_profit_reached":
                try:
                    dynamic_reason = self._trader.evaluate_dynamic_tp(snap)
                    if dynamic_reason is None:
                        return  # hold past TP
                    stop_reason = dynamic_reason
                except Exception:
                    pass

            if stop_reason is not None or decision.action == "sell":
                self._signals.put(
                    TradingSignal(pair=pair, snapshot=snap, signal_type="exit")
                )
                return

            return  # holding, no exit condition

        # ── New entry opportunity ────────────────────────────────────────────
        if decision.action == "buy" and not self._trader.at_max_positions():
            self._signals.put(
                TradingSignal(pair=pair, snapshot=snap, signal_type="buy")
            )


# ---------------------------------------------------------------------------
# 5. Risk Engine
# ---------------------------------------------------------------------------

class RiskEngine:
    """Synchronous pre-execution risk validator.

    Called by :class:`ExecutionEngine` before every order.  Returns a
    ``(approved, reason)`` tuple.  When *approved* is ``False`` the
    execution engine skips the signal.

    This engine is intentionally stateless and has no background thread —
    it is a pure validator invoked synchronously by the execution engine.
    """

    def __init__(self, config: Any) -> None:
        self._config = config

    def approve(
        self, signal: TradingSignal, trader: Any
    ) -> Tuple[bool, str]:
        """Return ``(True, "approved")`` or ``(False, <reason>)``."""
        price: float = signal.snapshot.get("price") or 0.0

        # -- Basic sanity check -----------------------------------------------
        if price <= 0.0:
            return False, "invalid_price"

        # -- Circuit breaker --------------------------------------------------
        cb_max: int = getattr(self._config, "circuit_breaker_max_errors", 0)
        if cb_max > 0:
            consecutive: int = getattr(trader, "_consecutive_errors", 0)
            if consecutive >= cb_max:
                return False, "circuit_breaker_tripped"

        # -- Daily loss cap ---------------------------------------------------
        daily_loss_pct: float = getattr(self._config, "max_daily_loss_pct", 0.0)
        if daily_loss_pct > 0.0:
            try:
                tracker = trader._active_tracker(signal.pair)
                daily_pnl: float = getattr(tracker, "_daily_pnl", 0.0)
                limit = -(abs(self._config.initial_capital) * daily_loss_pct)
                if daily_pnl <= limit:
                    return False, "daily_loss_cap_reached"
            except Exception:
                pass

        return True, "approved"


# ---------------------------------------------------------------------------
# 6. Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """Consumes signals and executes orders through the trader.

    Runs as a daemon thread.  On each loop iteration it pulls one
    :class:`TradingSignal` from the :class:`SignalQueue`, asks the
    :class:`RiskEngine` for pre-trade approval, then dispatches to the
    appropriate trader method (``maybe_execute``, ``force_sell``,
    ``resume_pending_buy``, ``resume_pending_sell``).

    Args:
        trader:    The :class:`~bot.trader.Trader` instance.
        config:    The :class:`~bot.config.BotConfig` instance.
        signals:   The :class:`SignalQueue` to read from.
        risk:      The :class:`RiskEngine` to validate signals.
        notify_fn: Optional callable ``(text: str) -> None`` for notifications.
        on_outcome: Optional callback ``(signal, outcome) -> None`` invoked
                    after every executed signal; used by
                    :class:`TradingOrchestrator` for display/logging.
    """

    def __init__(
        self,
        trader: Any,
        config: Any,
        signals: SignalQueue,
        risk: RiskEngine,
        *,
        notify_fn: Optional[Callable[[str], None]] = None,
        on_outcome: Optional[Callable[[TradingSignal, Dict[str, Any]], None]] = None,
    ) -> None:
        self._trader = trader
        self._config = config
        self._signals = signals
        self._risk = risk
        self._notify = notify_fn or (lambda _text: None)
        self._on_outcome = on_outcome
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.error_count: int = 0

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Start the background execution thread."""
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="execution-engine",
            daemon=True,
        )
        self._thread.start()
        logger.info("▶ ExecutionEngine started")

    def stop(self) -> None:
        """Signal the thread to stop and wait for it to finish."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def is_alive(self) -> bool:
        """``True`` when the background thread is running."""
        return self._thread is not None and self._thread.is_alive()

    # -- internals ------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            signal = self._signals.get(timeout=0.5)
            if signal is None:
                continue
            try:
                self._process(signal)
                self.error_count = 0
            except Exception as exc:
                self.error_count += 1
                logger.warning(
                    "ExecutionEngine: error processing %s signal for %s — %s",
                    signal.signal_type,
                    signal.pair,
                    exc,
                )

    def _process(self, signal: TradingSignal) -> None:
        """Validate and execute one *signal*."""
        approved, reason = self._risk.approve(signal, self._trader)
        if not approved:
            logger.debug(
                "ExecutionEngine: signal rejected by RiskEngine — pair=%s reason=%s",
                signal.pair,
                reason,
            )
            return

        snap = signal.snapshot
        price: float = snap.get("price") or 0.0
        outcome: Dict[str, Any] = {}

        if signal.signal_type == "resume_buy":
            outcome = self._trader.resume_pending_buy(snap)
            status = outcome.get("status", "")
            if status == "resumed":
                logger.info(
                    "🔄 RESUMED BUY  %s  @ Rp %s  amount=%.8f",
                    signal.pair,
                    f"{price:,.0f}",
                    outcome.get("amount", 0),
                )
                self._notify(
                    f"🔄 PENDING BUY FILLED {signal.pair} @ Rp {price:,.0f}\n"
                    f"Amount: {outcome.get('amount', 0):.8f}"
                )
            elif status in ("cancelled_below_min", "cancelled_zero"):
                logger.info(
                    "❌ Pending buy cancelled for %s: %s", signal.pair, status
                )

        elif signal.signal_type == "resume_sell":
            outcome = self._trader.resume_pending_sell(snap)
            status = outcome.get("status", "")
            if status in ("resumed", "partial"):
                is_partial = status == "partial"
                label = "PARTIAL SELL" if is_partial else "PENDING SELL FILLED"
                logger.info(
                    "🔄 %s  %s  @ Rp %s  amount=%.8f",
                    label,
                    signal.pair,
                    f"{price:,.0f}",
                    outcome.get("amount", 0),
                )
                tracker = self._trader._active_tracker(signal.pair)
                portfolio = tracker.as_dict(price)
                self._notify(
                    f"🔄 {label} {signal.pair} @ Rp {price:,.0f}\n"
                    f"Amount: {outcome.get('amount', 0):.8f}\n"
                    f"PnL: Rp {portfolio.get('realized_pnl', 0):,.2f}"
                )

        elif signal.signal_type == "exit":
            outcome = self._trader.force_sell(snap)
            status = outcome.get("status", "")
            if status != "pending_sell":
                tracker = self._trader._active_tracker(signal.pair)
                portfolio = tracker.as_dict(price)
                logger.info(
                    "📤 EXIT  %s  @ Rp %s  amount=%.8f  pnl=Rp %s",
                    signal.pair,
                    f"{price:,.0f}",
                    outcome.get("amount", 0),
                    f"{portfolio.get('realized_pnl', 0):,.2f}",
                )
                self._notify(
                    f"📤 EXIT {signal.pair} @ Rp {price:,.0f}\n"
                    f"Amount: {outcome.get('amount', 0):.8f}\n"
                    f"PnL: Rp {portfolio.get('realized_pnl', 0):,.2f}"
                )
            else:
                logger.info(
                    "🔄 PENDING SELL  %s  — sell not filled, will retry",
                    signal.pair,
                )

        elif signal.signal_type == "buy":
            outcome = self._trader.maybe_execute(snap)
            action = outcome.get("action", "hold")
            status = outcome.get("status", "")
            if action == "buy" and status in ("simulated", "placed"):
                logger.info(
                    "📈 BUY  %s  @ Rp %s  amount=%.8f  status=%s",
                    signal.pair,
                    f"{price:,.0f}",
                    outcome.get("amount", 0),
                    status,
                )
                tracker = self._trader._active_tracker(signal.pair)
                portfolio = tracker.as_dict(price)
                self._notify(
                    f"📈 BUY {signal.pair} @ Rp {price:,.0f}\n"
                    f"Amount: {outcome.get('amount', 0):.8f}\n"
                    f"Status: {status}\n"
                    f"Equity: Rp {portfolio.get('equity', 0):,.2f}"
                )

        if self._on_outcome is not None and outcome:
            try:
                self._on_outcome(signal, outcome)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 7. Trading Orchestrator
# ---------------------------------------------------------------------------

class TradingOrchestrator:
    """Wires all six engines together and manages their lifecycle.

    Architecture::

        MarketDataEngine → DataCache → StrategyEngine → SignalQueue
                                                             ↓
                                          RiskEngine ← ExecutionEngine

    Args:
        trader:    The :class:`~bot.trader.Trader` instance.
        config:    The :class:`~bot.config.BotConfig` instance.
        notify_fn: Optional callable for Telegram / Discord notifications.
        on_outcome: Optional hook invoked after each executed signal;
                    receives ``(TradingSignal, outcome_dict)``.
    """

    def __init__(
        self,
        trader: Any,
        config: Any,
        *,
        notify_fn: Optional[Callable[[str], None]] = None,
        on_outcome: Optional[Callable[[TradingSignal, Dict[str, Any]], None]] = None,
    ) -> None:
        self._trader = trader
        self._config = config

        # -- Shared plumbing --------------------------------------------------
        self.cache = DataCache()
        self.signal_queue = SignalQueue(maxsize=200)
        self.risk_engine = RiskEngine(config)

        # -- Recent outcome buffer (thread-safe) ------------------------------
        # ExecutionEngine pushes here on every signal outcome so that the
        # monitor loop can display placed / skipped results within the cycle.
        self._outcome_lock: threading.Lock = threading.Lock()
        self._recent_outcomes: List[Tuple[str, Dict[str, Any]]] = []

        _ext_on_outcome = on_outcome

        def _chained_on_outcome(
            signal: TradingSignal, outcome: Dict[str, Any]
        ) -> None:
            with self._outcome_lock:
                self._recent_outcomes.append((signal.pair, outcome))
            if _ext_on_outcome is not None:
                try:
                    _ext_on_outcome(signal, outcome)
                except Exception:
                    pass

        # -- Engines ----------------------------------------------------------
        self.market_data_engine = MarketDataEngine(
            trader, config, self.cache
        )
        self.strategy_engine = StrategyEngine(
            trader, config, self.cache, self.signal_queue
        )
        self.execution_engine = ExecutionEngine(
            trader,
            config,
            self.signal_queue,
            self.risk_engine,
            notify_fn=notify_fn,
            on_outcome=_chained_on_outcome,
        )

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        """Launch all background engine threads."""
        self.market_data_engine.start()
        self.strategy_engine.start()
        self.execution_engine.start()
        logger.info(
            "✅ TradingOrchestrator: all engines started\n"
            "   Market Data → Data Cache → Strategy Engine\n"
            "                                    ↓\n"
            "   Risk Engine  ← Execution Engine ← Signal Queue"
        )

    def stop(self) -> None:
        """Signal all engines to stop and wait for threads to join."""
        self.market_data_engine.stop()
        self.strategy_engine.stop()
        self.execution_engine.stop()
        logger.info("⏹  TradingOrchestrator: all engines stopped")

    def run(self, shutdown: threading.Event, *, poll_interval: float = 30.0) -> None:
        """Block until *shutdown* is set, logging periodic health reports.

        This method is intended to be called from the main thread after
        :meth:`start`.  It does *not* perform any trading logic — that
        happens in the background engine threads.  The main thread is free
        to display summaries, react to signals, or simply wait.

        Args:
            shutdown:      A :class:`threading.Event` that stops the loop.
            poll_interval: Seconds between health-check log lines.
        """
        logger.info(
            "⚙  TradingOrchestrator running — press Ctrl-C or send SIGTERM to stop"
        )
        while not shutdown.is_set():
            shutdown.wait(timeout=poll_interval)
            if not shutdown.is_set():
                self._log_health()

    # -- outcome buffer -------------------------------------------------------

    def pop_recent_outcomes(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return and clear all execution outcomes buffered since the last call.

        Thread-safe: the :class:`ExecutionEngine` pushes outcomes from its
        background thread; the monitor loop pops them once per cycle so they
        can be displayed within the structured cycle block rather than as
        free-floating log lines.

        Returns a list of ``(pair, outcome_dict)`` tuples in insertion order.
        When multiple outcomes exist for the same pair the caller should use
        the last one (most recent).
        """
        with self._outcome_lock:
            result = list(self._recent_outcomes)
            self._recent_outcomes.clear()
            return result

    # -- helpers --------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Return a snapshot of engine health metrics."""
        return {
            "cache_pairs": self.cache.pairs(),
            "signal_queue_size": self.signal_queue.size(),
            "market_data_engine": (
                "running" if self.market_data_engine.is_alive() else "stopped"
            ),
            "market_data_errors": self.market_data_engine.error_count,
            "strategy_engine": (
                "running" if self.strategy_engine.is_alive() else "stopped"
            ),
            "strategy_errors": self.strategy_engine.error_count,
            "execution_engine": (
                "running" if self.execution_engine.is_alive() else "stopped"
            ),
            "execution_errors": self.execution_engine.error_count,
        }

    def _log_health(self) -> None:
        h = self.health()
        logger.debug(
            "⚙  Engine health — "
            "market_data=%s(err=%d)  strategy=%s(err=%d)  "
            "execution=%s(err=%d)  cache=%d pairs  queue=%d signals",
            h["market_data_engine"],
            h["market_data_errors"],
            h["strategy_engine"],
            h["strategy_errors"],
            h["execution_engine"],
            h["execution_errors"],
            len(h["cache_pairs"]),
            h["signal_queue_size"],
        )
