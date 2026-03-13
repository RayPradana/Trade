"""Tests for bot/autonomous.py – 9 autonomous trading categories."""

import json
import unittest

from bot.autonomous import (
    # 1. 24/7 Autonomous trading
    AutonomousTradingState,
    TradingCycleResult,
    run_autonomous_cycle,
    check_autonomous_health,
    # 2. Automatic pair rotation
    PairScore,
    PairRotationResult,
    rotate_pairs,
    # 3. Automatic strategy switching
    StrategyPerformance,
    StrategySwitchResult,
    auto_switch_strategy,
    # 4. Auto restart after crash
    CrashEvent,
    RestartDecision,
    decide_restart,
    # 5. State persistence
    PersistentState,
    serialize_state,
    deserialize_state,
    validate_state,
    # 6. Failover system
    ComponentHealth,
    FailoverDecision,
    evaluate_failover,
    # 7. Task scheduling
    ScheduledTask,
    ScheduleResult,
    schedule_tasks,
    update_task_after_run,
    # 8. Dynamic polling intervals
    PollingConfig,
    PollingAdjustment,
    adjust_polling_interval,
    # 9. Auto recovery system
    RecoveryAction,
    RecoveryPlan,
    diagnose_and_recover,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=30, start=100.0, trend=0.5):
    return [start + trend * i for i in range(n)]


def _make_down_prices(n=30, start=100.0, trend=-0.5):
    return [start + trend * i for i in range(n)]


def _ts(offset=0):
    """Fixed timestamp for deterministic tests."""
    return 1_700_000_000_000 + offset


# ═══════════════════════════════════════════════════════════════════════════
#  1. 24/7 Autonomous Trading
# ═══════════════════════════════════════════════════════════════════════════

class TestAutonomousTrading(unittest.TestCase):

    def test_run_cycle_basic(self):
        state = AutonomousTradingState()
        prices = _make_prices(30)
        new_state, result = run_autonomous_cycle(state, prices, "btc_idr", "trend")
        self.assertIsInstance(new_state, AutonomousTradingState)
        self.assertIsInstance(result, TradingCycleResult)
        self.assertEqual(result.pair, "btc_idr")
        self.assertEqual(result.strategy, "trend")
        self.assertTrue(result.success)
        self.assertEqual(new_state.total_cycles, 1)
        self.assertEqual(new_state.successful_cycles, 1)

    def test_run_cycle_empty_prices(self):
        state = AutonomousTradingState()
        new_state, result = run_autonomous_cycle(state, [], "btc_idr")
        self.assertFalse(result.success)
        self.assertEqual(result.action, "hold")
        self.assertEqual(new_state.failed_cycles, 1)
        self.assertEqual(new_state.health_status, "degraded")

    def test_run_cycle_single_price(self):
        state = AutonomousTradingState()
        new_state, result = run_autonomous_cycle(state, [100.0], "btc_idr")
        self.assertFalse(result.success)
        self.assertEqual(new_state.failed_cycles, 1)

    def test_run_cycle_increments(self):
        state = AutonomousTradingState(total_cycles=5, successful_cycles=4, failed_cycles=1)
        prices = _make_prices(30)
        new_state, _ = run_autonomous_cycle(state, prices)
        self.assertEqual(new_state.total_cycles, 6)
        self.assertEqual(new_state.successful_cycles, 5)

    def test_health_check(self):
        state = AutonomousTradingState(
            is_running=True, total_cycles=100,
            successful_cycles=95, failed_cycles=5,
            health_status="running", error_count=2,
        )
        health = check_autonomous_health(state)
        self.assertTrue(health["is_running"])
        self.assertEqual(health["total_cycles"], 100)
        self.assertAlmostEqual(health["success_rate"], 0.95)
        self.assertFalse(health["needs_restart"])

    def test_health_needs_restart(self):
        state = AutonomousTradingState(
            error_count=15, max_errors_before_pause=10,
            total_cycles=20, successful_cycles=5,
        )
        health = check_autonomous_health(state)
        self.assertTrue(health["needs_restart"])


# ═══════════════════════════════════════════════════════════════════════════
#  2. Automatic Pair Rotation
# ═══════════════════════════════════════════════════════════════════════════

class TestPairRotation(unittest.TestCase):

    def test_rotate_basic(self):
        data = {
            "btc_idr": {"volume": 500_000, "volatility": 0.05, "spread": 0.002, "momentum": 0.3},
            "eth_idr": {"volume": 800_000, "volatility": 0.07, "spread": 0.001, "momentum": 0.5},
        }
        result = rotate_pairs(data, current_pair="btc_idr", max_pairs=2)
        self.assertIsInstance(result, PairRotationResult)
        self.assertGreater(len(result.selected_pairs), 0)
        self.assertEqual(len(result.scores), 2)

    def test_rotate_empty(self):
        result = rotate_pairs({}, current_pair="btc_idr")
        self.assertFalse(result.rotated)
        self.assertEqual(result.reason, "no pair data")

    def test_rotate_min_volume_filter(self):
        data = {
            "btc_idr": {"volume": 100, "volatility": 0.05, "spread": 0.002, "momentum": 0.1},
            "eth_idr": {"volume": 500_000, "volatility": 0.05, "spread": 0.001, "momentum": 0.2},
        }
        result = rotate_pairs(data, min_volume=1000)
        self.assertNotIn("btc_idr", result.selected_pairs)
        self.assertIn("eth_idr", result.selected_pairs)

    def test_rotate_max_pairs(self):
        data = {f"pair_{i}": {"volume": 100_000 * (i + 1), "volatility": 0.05} for i in range(10)}
        result = rotate_pairs(data, max_pairs=3)
        self.assertLessEqual(len(result.selected_pairs), 3)

    def test_rotate_keeps_current_if_best(self):
        data = {
            "btc_idr": {"volume": 900_000, "volatility": 0.08, "spread": 0.001, "momentum": 0.9},
            "eth_idr": {"volume": 100_000, "volatility": 0.01, "spread": 0.005, "momentum": 0.1},
        }
        result = rotate_pairs(data, current_pair="btc_idr")
        self.assertFalse(result.rotated)
        self.assertEqual(result.new_pair, "btc_idr")


# ═══════════════════════════════════════════════════════════════════════════
#  3. Automatic Strategy Switching
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategySwitching(unittest.TestCase):

    def test_switch_basic(self):
        stats = {
            "momentum": {"win_rate": 0.6, "avg_return": 0.01, "max_drawdown": 0.1, "sharpe": 1.5, "trades": 20},
            "mean_rev": {"win_rate": 0.5, "avg_return": 0.005, "max_drawdown": 0.2, "sharpe": 0.8, "trades": 15},
        }
        result = auto_switch_strategy(stats, current_strategy="mean_rev")
        self.assertIsInstance(result, StrategySwitchResult)
        self.assertTrue(result.switched)
        self.assertEqual(result.new_strategy, "momentum")

    def test_switch_empty(self):
        result = auto_switch_strategy({}, current_strategy="test")
        self.assertFalse(result.switched)
        self.assertEqual(result.reason, "no strategy data")

    def test_switch_min_trades_filter(self):
        stats = {
            "strat_a": {"win_rate": 0.9, "avg_return": 0.05, "sharpe": 3.0, "trades": 2},
        }
        result = auto_switch_strategy(stats, min_trades=5)
        self.assertFalse(result.switched)

    def test_switch_regime_detection(self):
        stats = {
            "trend": {"win_rate": 0.6, "avg_return": 0.01, "sharpe": 1.0, "trades": 10},
        }
        up_prices = _make_prices(30, trend=2.0)
        result = auto_switch_strategy(stats, prices=up_prices)
        self.assertEqual(result.regime, "trending_up")

    def test_switch_keeps_current(self):
        stats = {
            "trend": {"win_rate": 0.6, "avg_return": 0.01, "sharpe": 1.5, "trades": 10},
        }
        result = auto_switch_strategy(stats, current_strategy="trend")
        self.assertFalse(result.switched)
        self.assertEqual(result.new_strategy, "trend")


# ═══════════════════════════════════════════════════════════════════════════
#  4. Auto Restart After Crash
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoRestart(unittest.TestCase):

    def test_restart_basic(self):
        crashes = [CrashEvent(timestamp=_ts(), error_type="timeout", recoverable=True)]
        result = decide_restart(crashes, current_time=_ts(1000))
        self.assertIsInstance(result, RestartDecision)
        self.assertTrue(result.should_restart)
        self.assertGreater(result.delay_seconds, 0)

    def test_restart_no_crashes(self):
        result = decide_restart([])
        self.assertFalse(result.should_restart)
        self.assertEqual(result.reason, "no crash events")

    def test_restart_unrecoverable(self):
        crashes = [CrashEvent(timestamp=_ts(), error_type="fatal", recoverable=False)]
        result = decide_restart(crashes, current_time=_ts(1000))
        self.assertFalse(result.should_restart)
        self.assertIn("unrecoverable", result.reason)

    def test_restart_max_reached(self):
        crashes = [CrashEvent(timestamp=_ts(i * 100), recoverable=True) for i in range(6)]
        result = decide_restart(crashes, max_restarts=5, current_time=_ts(1000))
        self.assertFalse(result.should_restart)
        self.assertIn("max restarts", result.reason)

    def test_restart_exponential_backoff(self):
        crashes = [CrashEvent(timestamp=_ts(i * 100), recoverable=True) for i in range(3)]
        result = decide_restart(crashes, base_delay=5.0, backoff_factor=2.0, current_time=_ts(1000))
        self.assertTrue(result.should_restart)
        # 3 crashes: delay = 5 * 2^(3-1) = 20
        self.assertEqual(result.delay_seconds, 20.0)


# ═══════════════════════════════════════════════════════════════════════════
#  5. State Persistence
# ═══════════════════════════════════════════════════════════════════════════

class TestStatePersistence(unittest.TestCase):

    def test_serialize_deserialize(self):
        state = PersistentState(
            current_pair="btc_idr",
            current_strategy="momentum",
            positions={"btc": 0.5},
            balances={"idr": 10_000_000},
            cycle_count=42,
        )
        data = serialize_state(state)
        restored = deserialize_state(data)
        self.assertEqual(restored.current_pair, "btc_idr")
        self.assertEqual(restored.current_strategy, "momentum")
        self.assertEqual(restored.positions, {"btc": 0.5})
        self.assertEqual(restored.cycle_count, 42)

    def test_deserialize_empty(self):
        result = deserialize_state("")
        self.assertIsInstance(result, PersistentState)
        self.assertEqual(result.current_pair, "")

    def test_deserialize_invalid_json(self):
        result = deserialize_state("not json")
        self.assertIsInstance(result, PersistentState)

    def test_validate_valid_state(self):
        state = PersistentState(
            version=1, cycle_count=10,
            positions={"btc": 0.5}, balances={"idr": 100},
        )
        ok, errors = validate_state(state)
        self.assertTrue(ok)
        self.assertEqual(errors, [])

    def test_validate_invalid_state(self):
        state = PersistentState(version=0, cycle_count=-1, balances={"idr": -100})
        ok, errors = validate_state(state)
        self.assertFalse(ok)
        self.assertGreater(len(errors), 0)

    def test_serialize_produces_json(self):
        state = PersistentState(current_pair="eth_idr")
        data = serialize_state(state)
        parsed = json.loads(data)
        self.assertEqual(parsed["current_pair"], "eth_idr")


# ═══════════════════════════════════════════════════════════════════════════
#  6. Failover System
# ═══════════════════════════════════════════════════════════════════════════

class TestFailover(unittest.TestCase):

    def test_failover_all_healthy(self):
        components = [
            ComponentHealth(name="api", is_healthy=True, last_heartbeat=_ts(), priority=0),
            ComponentHealth(name="ws", is_healthy=True, last_heartbeat=_ts(), priority=1),
        ]
        result = evaluate_failover(components, current_time=_ts(1000))
        self.assertEqual(result.system_status, "healthy")
        self.assertFalse(result.failover_triggered)

    def test_failover_empty_components(self):
        result = evaluate_failover([])
        self.assertEqual(result.system_status, "unknown")

    def test_failover_critical_component_down(self):
        components = [
            ComponentHealth(name="api", is_healthy=False, last_heartbeat=_ts(), priority=0),
            ComponentHealth(name="ws", is_healthy=True, last_heartbeat=_ts(), priority=2),
        ]
        result = evaluate_failover(components, current_time=_ts(1000))
        self.assertTrue(result.failover_triggered)
        self.assertIn("api", result.failed_components)

    def test_failover_heartbeat_timeout(self):
        components = [
            ComponentHealth(name="api", is_healthy=True, last_heartbeat=_ts(-60000), priority=0),
        ]
        result = evaluate_failover(components, heartbeat_timeout_ms=30000, current_time=_ts())
        self.assertIn("api", result.failed_components)

    def test_failover_all_down(self):
        components = [
            ComponentHealth(name="api", is_healthy=False, priority=0),
            ComponentHealth(name="ws", is_healthy=False, priority=1),
        ]
        result = evaluate_failover(components, current_time=_ts())
        self.assertEqual(result.system_status, "critical")
        self.assertTrue(result.degraded_mode)


# ═══════════════════════════════════════════════════════════════════════════
#  7. Task Scheduling
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskScheduling(unittest.TestCase):

    def test_schedule_basic(self):
        tasks = [
            ScheduledTask(name="scan", interval_seconds=10, last_run=_ts(-20000)),
            ScheduledTask(name="rebalance", interval_seconds=60, last_run=_ts(-70000)),
        ]
        result = schedule_tasks(tasks, current_time=_ts())
        self.assertIn("scan", result.tasks_due)
        self.assertIn("rebalance", result.tasks_due)

    def test_schedule_empty(self):
        result = schedule_tasks([])
        self.assertEqual(result.tasks_due, [])

    def test_schedule_not_due(self):
        tasks = [
            ScheduledTask(name="scan", interval_seconds=60, last_run=_ts(-10000)),
        ]
        result = schedule_tasks(tasks, current_time=_ts())
        self.assertNotIn("scan", result.tasks_due)

    def test_schedule_priority_order(self):
        tasks = [
            ScheduledTask(name="low", interval_seconds=1, priority=10, last_run=0),
            ScheduledTask(name="high", interval_seconds=1, priority=1, last_run=0),
        ]
        result = schedule_tasks(tasks, current_time=_ts())
        self.assertEqual(result.tasks_due[0], "high")

    def test_schedule_disabled_skipped(self):
        tasks = [
            ScheduledTask(name="disabled", enabled=False, last_run=0),
            ScheduledTask(name="enabled", enabled=True, last_run=0),
        ]
        result = schedule_tasks(tasks, current_time=_ts())
        self.assertNotIn("disabled", result.tasks_due)
        self.assertIn("enabled", result.tasks_due)

    def test_update_task_after_run(self):
        task = ScheduledTask(name="scan", interval_seconds=10, run_count=5, error_count=1)
        updated = update_task_after_run(task, success=True, current_time=_ts())
        self.assertEqual(updated.run_count, 6)
        self.assertEqual(updated.error_count, 1)
        self.assertEqual(updated.last_run, _ts())

    def test_update_task_after_failure(self):
        task = ScheduledTask(name="scan", run_count=5, error_count=1)
        updated = update_task_after_run(task, success=False, current_time=_ts())
        self.assertEqual(updated.error_count, 2)


# ═══════════════════════════════════════════════════════════════════════════
#  8. Dynamic Polling Intervals
# ═══════════════════════════════════════════════════════════════════════════

class TestDynamicPolling(unittest.TestCase):

    def test_polling_normal(self):
        config = PollingConfig(base_interval_ms=5000)
        new_config, adj = adjust_polling_interval(config)
        self.assertIsInstance(adj, PollingAdjustment)
        self.assertGreater(adj.new_interval_ms, 0)

    def test_polling_high_volatility(self):
        config = PollingConfig(base_interval_ms=5000, min_interval_ms=1000)
        new_config, adj = adjust_polling_interval(config, recent_volatility=0.1)
        self.assertLess(adj.new_interval_ms, config.base_interval_ms)
        self.assertIn("high volatility", adj.adjustment_reason)

    def test_polling_with_errors(self):
        config = PollingConfig(base_interval_ms=5000)
        new_config, adj = adjust_polling_interval(config, recent_errors=3)
        self.assertGreater(adj.new_interval_ms, config.base_interval_ms)
        self.assertIn("error backoff", adj.adjustment_reason)

    def test_polling_fixed_mode(self):
        config = PollingConfig(base_interval_ms=5000, mode="fixed")
        new_config, adj = adjust_polling_interval(config, recent_volatility=0.1)
        self.assertEqual(adj.new_interval_ms, 5000)
        self.assertEqual(adj.adjustment_reason, "fixed mode")

    def test_polling_open_orders(self):
        config = PollingConfig(base_interval_ms=5000, min_interval_ms=1000)
        new_config, adj = adjust_polling_interval(config, has_open_orders=True)
        self.assertLess(adj.new_interval_ms, config.base_interval_ms)
        self.assertIn("open orders", adj.adjustment_reason)

    def test_polling_respects_bounds(self):
        config = PollingConfig(base_interval_ms=5000, min_interval_ms=2000, max_interval_ms=10000)
        _, adj = adjust_polling_interval(config, recent_volatility=1.0, has_open_orders=True)
        self.assertGreaterEqual(adj.new_interval_ms, 2000)
        self.assertLessEqual(adj.new_interval_ms, 10000)


# ═══════════════════════════════════════════════════════════════════════════
#  9. Auto Recovery System
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoRecovery(unittest.TestCase):

    def test_recovery_no_errors(self):
        plan = diagnose_and_recover([])
        self.assertFalse(plan.needs_recovery)
        self.assertEqual(plan.severity, "none")

    def test_recovery_connection_errors(self):
        errors = [{"type": "connection"}, {"type": "timeout"}]
        plan = diagnose_and_recover(errors)
        self.assertIsInstance(plan, RecoveryPlan)
        any_reconnect = any(a.action_type == "reconnect" for a in plan.actions)
        self.assertTrue(any_reconnect)

    def test_recovery_order_errors(self):
        errors = [{"type": "order"}, {"type": "execution"}]
        plan = diagnose_and_recover(errors)
        any_cancel = any(a.action_type == "cancel_orders" for a in plan.actions)
        self.assertTrue(any_cancel)

    def test_recovery_critical_severity(self):
        errors = [{"type": "connection"} for _ in range(12)]
        plan = diagnose_and_recover(errors, max_errors=10)
        self.assertEqual(plan.severity, "critical")
        self.assertTrue(plan.manual_intervention_needed)
        self.assertFalse(plan.can_auto_recover)

    def test_recovery_with_unhealthy_components(self):
        errors = [{"type": "connection"}]
        components = [
            ComponentHealth(name="api", is_healthy=False, consecutive_failures=5, max_failures=3),
        ]
        plan = diagnose_and_recover(errors, component_health=components)
        comp_actions = [a for a in plan.actions if a.target == "api"]
        self.assertGreater(len(comp_actions), 0)

    def test_recovery_actions_sorted_by_priority(self):
        errors = [{"type": "connection"}, {"type": "order"}, {"type": "state"}]
        plan = diagnose_and_recover(errors)
        if len(plan.actions) >= 2:
            for i in range(len(plan.actions) - 1):
                self.assertLessEqual(plan.actions[i].priority, plan.actions[i + 1].priority)


if __name__ == "__main__":
    unittest.main()
