"""Autonomous trading module.

Provides 9 autonomous trading categories for the Indodax trading bot:

 1. 24/7 autonomous trading
 2. Automatic pair rotation
 3. Automatic strategy switching
 4. Auto restart after crash
 5. State persistence
 6. Failover system
 7. Task scheduling
 8. Dynamic polling intervals
 9. Auto recovery system

Each algorithm is implemented as a pure function operating on standard
trading state data and returns typed dataclasses.
All implementations use only the Python standard library.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float:
    """Convert *value* to float, returning ``0.0`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _returns(prices: Sequence[float]) -> List[float]:
    """Simple percentage returns from a price series."""
    if len(prices) < 2:
        return []
    return [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1] != 0
    ]


def _now_ms() -> int:
    """Current epoch in milliseconds."""
    return int(time.time() * 1000)


# ═══════════════════════════════════════════════════════════════════════════
#  1. 24/7 Autonomous Trading
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AutonomousTradingState:
    """Represents the state of the 24/7 autonomous trading loop."""

    is_running: bool = False
    uptime_seconds: float = 0.0
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    last_cycle_timestamp: int = 0
    current_pair: str = ""
    current_strategy: str = ""
    health_status: str = "stopped"
    error_count: int = 0
    max_errors_before_pause: int = 10
    pause_duration_seconds: float = 60.0


@dataclass
class TradingCycleResult:
    """Result from one iteration of the trading loop."""

    cycle_number: int = 0
    pair: str = ""
    strategy: str = ""
    action: str = "hold"  # buy / sell / hold
    confidence: float = 0.0
    timestamp: int = 0
    duration_ms: float = 0.0
    success: bool = True
    error_message: str = ""


def run_autonomous_cycle(
    state: AutonomousTradingState,
    prices: Sequence[float],
    pair: str = "btc_idr",
    strategy: str = "default",
) -> Tuple[AutonomousTradingState, TradingCycleResult]:
    """Execute one cycle of the 24/7 autonomous trading loop.

    Returns an updated state and the cycle result.
    """
    now = _now_ms()
    cycle_result = TradingCycleResult(
        cycle_number=state.total_cycles + 1,
        pair=pair,
        strategy=strategy,
        timestamp=now,
    )

    if not prices or len(prices) < 2:
        cycle_result.success = False
        cycle_result.error_message = "insufficient price data"
        cycle_result.action = "hold"
        new_state = AutonomousTradingState(
            is_running=state.is_running,
            uptime_seconds=state.uptime_seconds,
            total_cycles=state.total_cycles + 1,
            successful_cycles=state.successful_cycles,
            failed_cycles=state.failed_cycles + 1,
            last_cycle_timestamp=now,
            current_pair=pair,
            current_strategy=strategy,
            health_status="degraded",
            error_count=state.error_count + 1,
            max_errors_before_pause=state.max_errors_before_pause,
            pause_duration_seconds=state.pause_duration_seconds,
        )
        return new_state, cycle_result

    rets = _returns(prices)
    avg_return = mean(rets) if rets else 0.0
    vol = pstdev(rets) if len(rets) >= 2 else 0.0

    if avg_return > 0.001 and vol < 0.05:
        cycle_result.action = "buy"
        cycle_result.confidence = min(1.0, avg_return / 0.01)
    elif avg_return < -0.001 and vol < 0.05:
        cycle_result.action = "sell"
        cycle_result.confidence = min(1.0, abs(avg_return) / 0.01)
    else:
        cycle_result.action = "hold"
        cycle_result.confidence = 0.0

    cycle_result.success = True
    health = "running"
    if state.error_count >= state.max_errors_before_pause:
        health = "paused"

    new_state = AutonomousTradingState(
        is_running=True,
        uptime_seconds=state.uptime_seconds,
        total_cycles=state.total_cycles + 1,
        successful_cycles=state.successful_cycles + 1,
        failed_cycles=state.failed_cycles,
        last_cycle_timestamp=now,
        current_pair=pair,
        current_strategy=strategy,
        health_status=health,
        error_count=0,
        max_errors_before_pause=state.max_errors_before_pause,
        pause_duration_seconds=state.pause_duration_seconds,
    )
    return new_state, cycle_result


def check_autonomous_health(state: AutonomousTradingState) -> Dict[str, Any]:
    """Return a health-check summary for the autonomous engine."""
    success_rate = (
        state.successful_cycles / state.total_cycles
        if state.total_cycles > 0
        else 0.0
    )
    return {
        "is_running": state.is_running,
        "health_status": state.health_status,
        "total_cycles": state.total_cycles,
        "success_rate": round(success_rate, 4),
        "error_count": state.error_count,
        "uptime_seconds": state.uptime_seconds,
        "needs_restart": state.error_count >= state.max_errors_before_pause,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  2. Automatic Pair Rotation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PairScore:
    """Score for a single trading pair used during rotation."""

    pair: str = ""
    volume_score: float = 0.0
    volatility_score: float = 0.0
    spread_score: float = 0.0
    momentum_score: float = 0.0
    composite_score: float = 0.0


@dataclass
class PairRotationResult:
    """Result of the pair-rotation algorithm."""

    selected_pairs: List[str] = field(default_factory=list)
    scores: List[PairScore] = field(default_factory=list)
    rotated: bool = False
    previous_pair: str = ""
    new_pair: str = ""
    reason: str = ""


def rotate_pairs(
    pair_data: Dict[str, Dict[str, Any]],
    current_pair: str = "",
    max_pairs: int = 3,
    min_volume: float = 0.0,
) -> PairRotationResult:
    """Score and rotate trading pairs based on market quality metrics.

    *pair_data* maps pair names to dicts with optional keys:
    ``volume``, ``volatility``, ``spread``, ``momentum``.
    """
    result = PairRotationResult(previous_pair=current_pair)

    if not pair_data:
        result.reason = "no pair data"
        return result

    scored: List[PairScore] = []
    for pair, metrics in pair_data.items():
        vol = _safe_float(metrics.get("volume", 0))
        if vol < min_volume:
            continue
        volatility = _safe_float(metrics.get("volatility", 0))
        spread = _safe_float(metrics.get("spread", 0))
        momentum = _safe_float(metrics.get("momentum", 0))

        vol_score = min(1.0, vol / 1_000_000) if vol > 0 else 0.0
        volatility_score = _clamp(volatility * 10, 0.0, 1.0)
        spread_score = max(0.0, 1.0 - spread * 100) if spread < 0.01 else 0.0
        mom_score = _clamp((momentum + 1) / 2, 0.0, 1.0)

        composite = (
            0.3 * vol_score
            + 0.25 * volatility_score
            + 0.25 * spread_score
            + 0.2 * mom_score
        )
        scored.append(
            PairScore(
                pair=pair,
                volume_score=round(vol_score, 4),
                volatility_score=round(volatility_score, 4),
                spread_score=round(spread_score, 4),
                momentum_score=round(mom_score, 4),
                composite_score=round(composite, 4),
            )
        )

    scored.sort(key=lambda s: s.composite_score, reverse=True)
    selected = [s.pair for s in scored[:max_pairs]]
    result.scores = scored
    result.selected_pairs = selected
    if selected and selected[0] != current_pair:
        result.rotated = True
        result.new_pair = selected[0]
        result.reason = "higher scoring pair available"
    else:
        result.rotated = False
        result.new_pair = current_pair
        result.reason = "current pair still optimal"

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  3. Automatic Strategy Switching
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyPerformance:
    """Performance summary for a single strategy."""

    name: str = ""
    win_rate: float = 0.0
    avg_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    score: float = 0.0


@dataclass
class StrategySwitchResult:
    """Result of the strategy auto-switch algorithm."""

    switched: bool = False
    previous_strategy: str = ""
    new_strategy: str = ""
    reason: str = ""
    rankings: List[StrategyPerformance] = field(default_factory=list)
    regime: str = "unknown"


def auto_switch_strategy(
    strategy_stats: Dict[str, Dict[str, float]],
    current_strategy: str = "",
    min_trades: int = 5,
    prices: Optional[Sequence[float]] = None,
) -> StrategySwitchResult:
    """Evaluate strategies and switch to the best performer.

    *strategy_stats* maps strategy names to dicts with keys:
    ``win_rate``, ``avg_return``, ``max_drawdown``, ``sharpe``, ``trades``.
    """
    result = StrategySwitchResult(previous_strategy=current_strategy)

    # Simple regime detection from prices
    if prices and len(prices) >= 10:
        rets = _returns(prices)
        avg_r = mean(rets) if rets else 0.0
        vol = pstdev(rets) if len(rets) >= 2 else 0.0
        if avg_r > 0.002:
            result.regime = "trending_up"
        elif avg_r < -0.002:
            result.regime = "trending_down"
        elif vol > 0.03:
            result.regime = "volatile"
        else:
            result.regime = "ranging"

    if not strategy_stats:
        result.reason = "no strategy data"
        return result

    rankings: List[StrategyPerformance] = []
    for name, stats in strategy_stats.items():
        trades = int(stats.get("trades", 0))
        if trades < min_trades:
            continue
        wr = _safe_float(stats.get("win_rate", 0))
        ar = _safe_float(stats.get("avg_return", 0))
        md = _safe_float(stats.get("max_drawdown", 0))
        sh = _safe_float(stats.get("sharpe", 0))

        # Composite score
        score = 0.3 * wr + 0.3 * _clamp(ar * 100, -1, 1) + 0.2 * sh + 0.2 * max(0, 1 - md)
        rankings.append(
            StrategyPerformance(
                name=name,
                win_rate=wr,
                avg_return=ar,
                max_drawdown=md,
                sharpe_ratio=sh,
                total_trades=trades,
                score=round(score, 4),
            )
        )

    rankings.sort(key=lambda s: s.score, reverse=True)
    result.rankings = rankings

    if not rankings:
        result.reason = "no strategies meet minimum trade threshold"
        return result

    best = rankings[0].name
    if best != current_strategy:
        result.switched = True
        result.new_strategy = best
        result.reason = f"strategy '{best}' has higher score"
    else:
        result.switched = False
        result.new_strategy = current_strategy
        result.reason = "current strategy still optimal"

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  4. Auto Restart After Crash
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CrashEvent:
    """Record of a single crash event."""

    timestamp: int = 0
    error_type: str = ""
    error_message: str = ""
    component: str = ""
    recoverable: bool = True


@dataclass
class RestartDecision:
    """Decision output from the auto-restart algorithm."""

    should_restart: bool = False
    delay_seconds: float = 0.0
    restart_count: int = 0
    max_restarts: int = 5
    backoff_factor: float = 2.0
    cooldown_remaining: float = 0.0
    reason: str = ""


def decide_restart(
    crash_history: Sequence[CrashEvent],
    max_restarts: int = 5,
    base_delay: float = 5.0,
    backoff_factor: float = 2.0,
    cooldown_window_seconds: float = 3600.0,
    current_time: Optional[int] = None,
) -> RestartDecision:
    """Decide whether to restart after a crash with exponential backoff."""
    now = current_time if current_time is not None else _now_ms()
    result = RestartDecision(
        max_restarts=max_restarts,
        backoff_factor=backoff_factor,
    )

    if not crash_history:
        result.should_restart = False
        result.reason = "no crash events"
        return result

    # Count recent crashes within cooldown window
    window_start = now - int(cooldown_window_seconds * 1000)
    recent = [c for c in crash_history if c.timestamp >= window_start]
    result.restart_count = len(recent)

    # Check if latest crash is recoverable
    latest = crash_history[-1]
    if not latest.recoverable:
        result.should_restart = False
        result.reason = f"unrecoverable error: {latest.error_type}"
        return result

    if len(recent) >= max_restarts:
        result.should_restart = False
        result.reason = f"max restarts ({max_restarts}) reached in cooldown window"
        return result

    # Exponential backoff
    delay = base_delay * (backoff_factor ** (len(recent) - 1))
    result.should_restart = True
    result.delay_seconds = round(delay, 2)
    result.reason = f"restart #{len(recent)} with {delay:.1f}s delay"
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  5. State Persistence
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PersistentState:
    """Serializable bot state for persistence."""

    version: int = 1
    timestamp: int = 0
    current_pair: str = ""
    current_strategy: str = ""
    positions: Dict[str, float] = field(default_factory=dict)
    balances: Dict[str, float] = field(default_factory=dict)
    active_orders: List[Dict[str, Any]] = field(default_factory=list)
    cycle_count: int = 0
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def serialize_state(state: PersistentState) -> str:
    """Serialize bot state to a JSON string."""
    data = {
        "version": state.version,
        "timestamp": state.timestamp or _now_ms(),
        "current_pair": state.current_pair,
        "current_strategy": state.current_strategy,
        "positions": state.positions,
        "balances": state.balances,
        "active_orders": state.active_orders,
        "cycle_count": state.cycle_count,
        "error_count": state.error_count,
        "metadata": state.metadata,
    }
    return json.dumps(data, indent=2)


def deserialize_state(data: str) -> PersistentState:
    """Deserialize bot state from a JSON string."""
    if not data or not data.strip():
        return PersistentState()
    try:
        parsed = json.loads(data)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse state data, returning default state")
        return PersistentState()

    return PersistentState(
        version=int(parsed.get("version", 1)),
        timestamp=int(parsed.get("timestamp", 0)),
        current_pair=str(parsed.get("current_pair", "")),
        current_strategy=str(parsed.get("current_strategy", "")),
        positions=dict(parsed.get("positions", {})),
        balances=dict(parsed.get("balances", {})),
        active_orders=list(parsed.get("active_orders", [])),
        cycle_count=int(parsed.get("cycle_count", 0)),
        error_count=int(parsed.get("error_count", 0)),
        metadata=dict(parsed.get("metadata", {})),
    )


def validate_state(state: PersistentState) -> Tuple[bool, List[str]]:
    """Validate that a persistent state is consistent and usable."""
    errors: List[str] = []
    if state.version < 1:
        errors.append("invalid version")
    if state.cycle_count < 0:
        errors.append("negative cycle count")
    if state.error_count < 0:
        errors.append("negative error count")
    for pair, amount in state.positions.items():
        if not isinstance(pair, str) or not pair:
            errors.append(f"invalid position pair: {pair}")
        if amount < 0:
            errors.append(f"negative position for {pair}")
    for asset, balance in state.balances.items():
        if balance < 0:
            errors.append(f"negative balance for {asset}")
    return len(errors) == 0, errors


# ═══════════════════════════════════════════════════════════════════════════
#  6. Failover System
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ComponentHealth:
    """Health status of a single system component."""

    name: str = ""
    is_healthy: bool = True
    last_heartbeat: int = 0
    consecutive_failures: int = 0
    max_failures: int = 3
    priority: int = 0  # lower = higher priority


@dataclass
class FailoverDecision:
    """Decision from the failover algorithm."""

    failover_triggered: bool = False
    failed_components: List[str] = field(default_factory=list)
    fallback_actions: Dict[str, str] = field(default_factory=dict)
    system_status: str = "healthy"
    degraded_mode: bool = False


def evaluate_failover(
    components: Sequence[ComponentHealth],
    heartbeat_timeout_ms: int = 30000,
    current_time: Optional[int] = None,
) -> FailoverDecision:
    """Evaluate system components and decide on failover actions."""
    now = current_time if current_time is not None else _now_ms()
    result = FailoverDecision()

    if not components:
        result.system_status = "unknown"
        return result

    failed: List[str] = []
    actions: Dict[str, str] = {}

    for comp in components:
        heartbeat_age = now - comp.last_heartbeat if comp.last_heartbeat else 0
        is_timed_out = heartbeat_age > heartbeat_timeout_ms
        is_failed = not comp.is_healthy or is_timed_out or comp.consecutive_failures >= comp.max_failures

        if is_failed:
            failed.append(comp.name)
            if comp.priority <= 1:
                actions[comp.name] = "restart_critical"
            elif comp.consecutive_failures >= comp.max_failures:
                actions[comp.name] = "restart_with_backoff"
            else:
                actions[comp.name] = "monitor"

    result.failed_components = failed
    result.fallback_actions = actions

    if not failed:
        result.system_status = "healthy"
    elif len(failed) == len(components):
        result.system_status = "critical"
        result.failover_triggered = True
        result.degraded_mode = True
    elif any(
        c.priority <= 1 for c in components if c.name in failed
    ):
        result.system_status = "degraded"
        result.failover_triggered = True
        result.degraded_mode = True
    else:
        result.system_status = "warning"
        result.failover_triggered = False

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  7. Task Scheduling
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ScheduledTask:
    """A single scheduled task."""

    name: str = ""
    interval_seconds: float = 60.0
    last_run: int = 0
    next_run: int = 0
    enabled: bool = True
    priority: int = 5  # 1 = highest
    max_runtime_seconds: float = 30.0
    run_count: int = 0
    error_count: int = 0


@dataclass
class ScheduleResult:
    """Result of the task scheduling algorithm."""

    tasks_due: List[str] = field(default_factory=list)
    next_wake_ms: int = 0
    total_scheduled: int = 0
    total_enabled: int = 0


def schedule_tasks(
    tasks: Sequence[ScheduledTask],
    current_time: Optional[int] = None,
) -> ScheduleResult:
    """Determine which tasks are due for execution.

    Returns tasks sorted by priority (lower number = higher priority).
    """
    now = current_time if current_time is not None else _now_ms()
    result = ScheduleResult()

    if not tasks:
        return result

    enabled = [t for t in tasks if t.enabled]
    result.total_scheduled = len(tasks)
    result.total_enabled = len(enabled)

    due: List[Tuple[int, str]] = []
    next_wake = float("inf")

    for task in enabled:
        next_run = task.next_run if task.next_run else task.last_run + int(task.interval_seconds * 1000)
        if task.last_run == 0 or now >= next_run:
            due.append((task.priority, task.name))
        else:
            remaining = next_run - now
            if remaining < next_wake:
                next_wake = remaining

    due.sort(key=lambda x: x[0])
    result.tasks_due = [name for _, name in due]
    result.next_wake_ms = int(next_wake) if next_wake != float("inf") else 0

    return result


def update_task_after_run(
    task: ScheduledTask,
    success: bool = True,
    current_time: Optional[int] = None,
) -> ScheduledTask:
    """Return an updated task after a run attempt."""
    now = current_time if current_time is not None else _now_ms()
    return ScheduledTask(
        name=task.name,
        interval_seconds=task.interval_seconds,
        last_run=now,
        next_run=now + int(task.interval_seconds * 1000),
        enabled=task.enabled,
        priority=task.priority,
        max_runtime_seconds=task.max_runtime_seconds,
        run_count=task.run_count + 1,
        error_count=task.error_count + (0 if success else 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. Dynamic Polling Intervals
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PollingConfig:
    """Configuration and state for dynamic polling intervals."""

    base_interval_ms: int = 5000
    min_interval_ms: int = 1000
    max_interval_ms: int = 60000
    current_interval_ms: int = 5000
    volatility_factor: float = 1.0
    volume_factor: float = 1.0
    error_factor: float = 1.0
    mode: str = "adaptive"  # adaptive / fixed / aggressive


@dataclass
class PollingAdjustment:
    """Result of a polling-interval adjustment."""

    previous_interval_ms: int = 0
    new_interval_ms: int = 0
    adjustment_reason: str = ""
    factors: Dict[str, float] = field(default_factory=dict)


def adjust_polling_interval(
    config: PollingConfig,
    recent_volatility: float = 0.0,
    recent_volume: float = 0.0,
    recent_errors: int = 0,
    has_open_orders: bool = False,
) -> Tuple[PollingConfig, PollingAdjustment]:
    """Dynamically adjust the polling interval based on market conditions."""
    adj = PollingAdjustment(previous_interval_ms=config.current_interval_ms)

    if config.mode == "fixed":
        adj.new_interval_ms = config.base_interval_ms
        adj.adjustment_reason = "fixed mode"
        new_config = PollingConfig(
            base_interval_ms=config.base_interval_ms,
            min_interval_ms=config.min_interval_ms,
            max_interval_ms=config.max_interval_ms,
            current_interval_ms=config.base_interval_ms,
            mode="fixed",
        )
        return new_config, adj

    # Volatility: high vol → shorter interval
    vol_factor = max(0.3, 1.0 - recent_volatility * 10) if recent_volatility > 0 else 1.0

    # Volume: high volume → shorter interval
    volume_factor = max(0.5, 1.0 - min(recent_volume / 1_000_000, 0.5)) if recent_volume > 0 else 1.0

    # Errors: more errors → longer interval (backoff)
    error_factor = 1.0 + recent_errors * 0.5

    # Open orders need more frequent polling
    order_factor = 0.5 if has_open_orders else 1.0

    combined = vol_factor * volume_factor * error_factor * order_factor
    new_interval = int(config.base_interval_ms * combined)
    new_interval = max(config.min_interval_ms, min(config.max_interval_ms, new_interval))

    reasons = []
    if vol_factor < 0.8:
        reasons.append("high volatility")
    if volume_factor < 0.8:
        reasons.append("high volume")
    if error_factor > 1.0:
        reasons.append("error backoff")
    if has_open_orders:
        reasons.append("open orders")
    if not reasons:
        reasons.append("normal conditions")

    adj.new_interval_ms = new_interval
    adj.adjustment_reason = "; ".join(reasons)
    adj.factors = {
        "volatility": round(vol_factor, 4),
        "volume": round(volume_factor, 4),
        "error": round(error_factor, 4),
        "order": round(order_factor, 4),
    }

    new_config = PollingConfig(
        base_interval_ms=config.base_interval_ms,
        min_interval_ms=config.min_interval_ms,
        max_interval_ms=config.max_interval_ms,
        current_interval_ms=new_interval,
        volatility_factor=round(vol_factor, 4),
        volume_factor=round(volume_factor, 4),
        error_factor=round(error_factor, 4),
        mode=config.mode,
    )
    return new_config, adj


# ═══════════════════════════════════════════════════════════════════════════
#  9. Auto Recovery System
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RecoveryAction:
    """A single recovery action to be taken."""

    action_type: str = ""  # reconnect / rebalance / cancel_orders / reset_state / pause
    target: str = ""
    priority: int = 5
    estimated_duration_seconds: float = 0.0
    description: str = ""


@dataclass
class RecoveryPlan:
    """Full recovery plan produced by the auto recovery system."""

    needs_recovery: bool = False
    severity: str = "none"  # none / low / medium / high / critical
    actions: List[RecoveryAction] = field(default_factory=list)
    estimated_downtime_seconds: float = 0.0
    can_auto_recover: bool = True
    manual_intervention_needed: bool = False
    diagnosis: str = ""


def diagnose_and_recover(
    error_log: Sequence[Dict[str, Any]],
    component_health: Optional[Sequence[ComponentHealth]] = None,
    state: Optional[PersistentState] = None,
    max_errors: int = 10,
) -> RecoveryPlan:
    """Analyse errors and component health, then produce a recovery plan."""
    plan = RecoveryPlan()

    if not error_log and not component_health:
        plan.diagnosis = "no errors detected"
        return plan

    # Classify errors
    error_types: Dict[str, int] = {}
    for entry in (error_log or []):
        etype = str(entry.get("type", "unknown"))
        error_types[etype] = error_types.get(etype, 0) + 1

    total_errors = sum(error_types.values())

    # Determine severity
    if total_errors == 0 and not component_health:
        plan.diagnosis = "no errors detected"
        return plan

    if total_errors >= max_errors:
        plan.severity = "critical"
    elif total_errors >= max_errors // 2:
        plan.severity = "high"
    elif total_errors >= max_errors // 4:
        plan.severity = "medium"
    elif total_errors > 0:
        plan.severity = "low"
    else:
        plan.severity = "none"

    plan.needs_recovery = plan.severity in ("medium", "high", "critical")
    actions: List[RecoveryAction] = []

    # Connection errors → reconnect
    conn_errors = error_types.get("connection", 0) + error_types.get("timeout", 0)
    if conn_errors > 0:
        actions.append(
            RecoveryAction(
                action_type="reconnect",
                target="exchange_api",
                priority=1,
                estimated_duration_seconds=5.0,
                description=f"reconnect after {conn_errors} connection errors",
            )
        )

    # Order errors → cancel pending orders
    order_errors = error_types.get("order", 0) + error_types.get("execution", 0)
    if order_errors > 0:
        actions.append(
            RecoveryAction(
                action_type="cancel_orders",
                target="all_pending",
                priority=2,
                estimated_duration_seconds=3.0,
                description=f"cancel orders after {order_errors} order errors",
            )
        )

    # State errors → reset state
    state_errors = error_types.get("state", 0) + error_types.get("data", 0)
    if state_errors > 0:
        actions.append(
            RecoveryAction(
                action_type="reset_state",
                target="trading_state",
                priority=3,
                estimated_duration_seconds=2.0,
                description=f"reset state after {state_errors} state errors",
            )
        )

    # Component health checks
    if component_health:
        for comp in component_health:
            if not comp.is_healthy or comp.consecutive_failures >= comp.max_failures:
                actions.append(
                    RecoveryAction(
                        action_type="reconnect",
                        target=comp.name,
                        priority=comp.priority,
                        estimated_duration_seconds=10.0,
                        description=f"restart unhealthy component '{comp.name}'",
                    )
                )

    # Critical → may need manual intervention
    if plan.severity == "critical":
        plan.manual_intervention_needed = True
        plan.can_auto_recover = False
        actions.append(
            RecoveryAction(
                action_type="pause",
                target="all_trading",
                priority=0,
                estimated_duration_seconds=0.0,
                description="pause all trading - manual review needed",
            )
        )

    actions.sort(key=lambda a: a.priority)
    plan.actions = actions
    plan.estimated_downtime_seconds = sum(a.estimated_duration_seconds for a in actions)
    plan.diagnosis = (
        f"{total_errors} errors detected ({', '.join(f'{k}:{v}' for k, v in error_types.items())}); "
        f"severity={plan.severity}"
    )

    return plan
