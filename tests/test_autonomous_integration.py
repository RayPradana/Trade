"""Integration tests for autonomous module wired into trader and main.

Verifies that the autonomous features (state tracking, pair rotation,
strategy switching, dynamic polling, failover, task scheduling, crash
recovery, cleanup) are correctly integrated into the trading pipeline.
"""

import logging
import time
import unittest
from typing import Any, Dict

from bot.autonomous import (
    AutonomousTradingState,
    TradingCycleResult,
    run_autonomous_cycle,
    check_autonomous_health,
    rotate_pairs,
    auto_switch_strategy,
    CrashEvent,
    decide_restart,
    ComponentHealth,
    evaluate_failover,
    ScheduledTask,
    schedule_tasks,
    update_task_after_run,
    PollingConfig,
    adjust_polling_interval,
    diagnose_and_recover,
)
from bot.config import BotConfig
from bot.trader import Trader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubClient:
    """Minimal client stub for Trader tests."""
    _pair_min_order: Dict[str, Any] = {}

    def __init__(self, pairs=None):
        self._pairs = pairs or ["btc_idr", "eth_idr"]

    def get_pairs(self):
        return [{"name": p} for p in self._pairs]

    def get_summaries(self):
        return {}

    def get_depth(self, pair, count=5):
        return {"buy": [["100", "1"]], "sell": [["100.05", "1"]]}

    def get_pair_min_order(self, pair):
        return {"min_coin": 0.0, "min_idr": 0.0}

    def load_pair_min_orders(self, *a, **kw):
        pass


def _make_config(**overrides):
    defaults = dict(api_key=None, dry_run=True, initial_capital=1_000_000.0)
    defaults.update(overrides)
    return BotConfig(**defaults)


# ═══════════════════════════════════════════════════════════════════════════
#  1. Autonomous state tracking in trading cycle
# ═══════════════════════════════════════════════════════════════════════════

class TestAutonomousStateCycleIntegration(unittest.TestCase):
    """Verify autonomous state updates correctly mirror trading cycles."""

    def test_state_increments_on_successful_cycle(self):
        state = AutonomousTradingState(is_running=True, total_cycles=0)
        prices = [100 + i * 0.5 for i in range(30)]
        new_state, result = run_autonomous_cycle(state, prices, pair="btc_idr")
        self.assertEqual(new_state.total_cycles, 1)
        self.assertEqual(new_state.successful_cycles, 1)
        self.assertTrue(result.success)

    def test_state_tracks_failures_on_empty_data(self):
        state = AutonomousTradingState(is_running=True, total_cycles=5)
        new_state, result = run_autonomous_cycle(state, [], pair="btc_idr")
        self.assertEqual(new_state.failed_cycles, 1)
        self.assertFalse(result.success)
        self.assertEqual(new_state.health_status, "degraded")

    def test_health_check_detects_restart_needed(self):
        state = AutonomousTradingState(
            is_running=True, error_count=10, max_errors_before_pause=10,
        )
        health = check_autonomous_health(state)
        self.assertTrue(health["needs_restart"])


# ═══════════════════════════════════════════════════════════════════════════
#  2. Pair rotation integration
# ═══════════════════════════════════════════════════════════════════════════

class TestPairRotationIntegration(unittest.TestCase):
    """Verify pair rotation works with real market-like data."""

    def test_rotation_selects_highest_scorer(self):
        pair_data = {
            "btc_idr": {"volume": 5_000_000, "volatility": 0.03, "spread": 0.001, "momentum": 0.5},
            "eth_idr": {"volume": 2_000_000, "volatility": 0.02, "spread": 0.002, "momentum": 0.3},
            "doge_idr": {"volume": 100_000, "volatility": 0.01, "spread": 0.005, "momentum": -0.1},
        }
        result = rotate_pairs(pair_data, current_pair="doge_idr", max_pairs=2)
        self.assertIn("btc_idr", result.selected_pairs)
        self.assertTrue(result.rotated)
        self.assertEqual(result.new_pair, "btc_idr")

    def test_no_rotation_when_current_optimal(self):
        pair_data = {
            "btc_idr": {"volume": 5_000_000, "volatility": 0.03, "spread": 0.001, "momentum": 0.5},
            "eth_idr": {"volume": 100_000, "volatility": 0.01, "spread": 0.005, "momentum": -0.1},
        }
        result = rotate_pairs(pair_data, current_pair="btc_idr")
        self.assertFalse(result.rotated)

    def test_rotation_filters_low_volume(self):
        pair_data = {
            "btc_idr": {"volume": 500, "volatility": 0.03, "spread": 0.001, "momentum": 0.5},
        }
        result = rotate_pairs(pair_data, current_pair="btc_idr", min_volume=1000)
        self.assertEqual(result.selected_pairs, [])
        # When all pairs are filtered out, the reason indicates no rotation occurred
        self.assertIn("current pair still optimal", result.reason)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Strategy auto-switching integration
# ═══════════════════════════════════════════════════════════════════════════

class TestStrategySwitchIntegration(unittest.TestCase):
    """Verify strategy switching works with performance data."""

    def test_switches_to_better_strategy(self):
        stats = {
            "trend_following": {"win_rate": 0.6, "avg_return": 0.02, "max_drawdown": 0.1, "sharpe": 1.5, "trades": 20},
            "mean_reversion": {"win_rate": 0.7, "avg_return": 0.03, "max_drawdown": 0.05, "sharpe": 2.0, "trades": 15},
        }
        result = auto_switch_strategy(stats, current_strategy="trend_following")
        self.assertTrue(result.switched)
        self.assertEqual(result.new_strategy, "mean_reversion")

    def test_no_switch_when_current_best(self):
        stats = {
            "trend_following": {"win_rate": 0.8, "avg_return": 0.05, "max_drawdown": 0.02, "sharpe": 3.0, "trades": 30},
            "mean_reversion": {"win_rate": 0.5, "avg_return": 0.01, "max_drawdown": 0.1, "sharpe": 0.5, "trades": 10},
        }
        result = auto_switch_strategy(stats, current_strategy="trend_following")
        self.assertFalse(result.switched)

    def test_regime_detection_from_prices(self):
        prices_up = [100 + i for i in range(20)]
        result = auto_switch_strategy({}, prices=prices_up)
        self.assertEqual(result.regime, "trending_up")


# ═══════════════════════════════════════════════════════════════════════════
#  4. Crash recovery integration
# ═══════════════════════════════════════════════════════════════════════════

class TestCrashRecoveryIntegration(unittest.TestCase):
    """Verify crash recovery and restart decisions."""

    def test_restart_decision_with_recent_crash(self):
        now = int(time.time() * 1000)
        crashes = [CrashEvent(timestamp=now, error_type="connection", recoverable=True)]
        decision = decide_restart(crashes, max_restarts=5, current_time=now)
        self.assertTrue(decision.should_restart)
        self.assertGreater(decision.delay_seconds, 0)

    def test_restart_denied_after_max_crashes(self):
        now = int(time.time() * 1000)
        crashes = [
            CrashEvent(timestamp=now - i * 1000, error_type="connection", recoverable=True)
            for i in range(5)
        ]
        decision = decide_restart(crashes, max_restarts=5, current_time=now)
        self.assertFalse(decision.should_restart)

    def test_no_restart_on_unrecoverable_error(self):
        now = int(time.time() * 1000)
        crashes = [CrashEvent(timestamp=now, error_type="fatal", recoverable=False)]
        decision = decide_restart(crashes, current_time=now)
        self.assertFalse(decision.should_restart)


# ═══════════════════════════════════════════════════════════════════════════
#  5. Failover system integration
# ═══════════════════════════════════════════════════════════════════════════

class TestFailoverIntegration(unittest.TestCase):
    """Verify failover detection and actions."""

    def test_healthy_system_no_failover(self):
        now = int(time.time() * 1000)
        components = [
            ComponentHealth(name="exchange_api", is_healthy=True, last_heartbeat=now, priority=0),
            ComponentHealth(name="market_data", is_healthy=True, last_heartbeat=now, priority=1),
        ]
        result = evaluate_failover(components, current_time=now)
        self.assertFalse(result.failover_triggered)
        self.assertEqual(result.system_status, "healthy")

    def test_critical_component_triggers_failover(self):
        now = int(time.time() * 1000)
        components = [
            ComponentHealth(name="exchange_api", is_healthy=False, last_heartbeat=now, priority=0),
            ComponentHealth(name="market_data", is_healthy=True, last_heartbeat=now, priority=1),
        ]
        result = evaluate_failover(components, current_time=now)
        self.assertTrue(result.failover_triggered)
        self.assertEqual(result.system_status, "degraded")
        self.assertIn("exchange_api", result.failed_components)


# ═══════════════════════════════════════════════════════════════════════════
#  6. Task scheduling integration
# ═══════════════════════════════════════════════════════════════════════════

class TestTaskSchedulingIntegration(unittest.TestCase):
    """Verify task scheduling with trading-like task set."""

    def test_due_tasks_returned_on_first_run(self):
        tasks = [
            ScheduledTask(name="health_check", interval_seconds=120.0, priority=1, enabled=True),
            ScheduledTask(name="pair_rotation", interval_seconds=300.0, priority=2, enabled=True),
        ]
        result = schedule_tasks(tasks)
        self.assertEqual(len(result.tasks_due), 2)
        self.assertEqual(result.tasks_due[0], "health_check")  # higher priority first

    def test_tasks_not_due_after_recent_run(self):
        now = int(time.time() * 1000)
        tasks = [
            ScheduledTask(
                name="health_check", interval_seconds=120.0, priority=1,
                last_run=now, next_run=now + 120000, enabled=True,
            ),
        ]
        result = schedule_tasks(tasks, current_time=now + 1000)
        self.assertEqual(len(result.tasks_due), 0)

    def test_task_update_after_run(self):
        task = ScheduledTask(name="health_check", interval_seconds=60.0)
        updated = update_task_after_run(task, success=True)
        self.assertEqual(updated.run_count, 1)
        self.assertEqual(updated.error_count, 0)
        self.assertGreater(updated.last_run, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  7. Dynamic polling integration
# ═══════════════════════════════════════════════════════════════════════════

class TestDynamicPollingIntegration(unittest.TestCase):
    """Verify dynamic polling adjusts intervals based on market conditions."""

    def test_high_volatility_shortens_interval(self):
        config = PollingConfig(base_interval_ms=5000, min_interval_ms=1000, max_interval_ms=60000)
        new_config, adj = adjust_polling_interval(config, recent_volatility=0.1)
        self.assertLess(new_config.current_interval_ms, 5000)

    def test_errors_increase_interval(self):
        config = PollingConfig(base_interval_ms=5000, min_interval_ms=1000, max_interval_ms=60000)
        new_config, adj = adjust_polling_interval(config, recent_errors=5)
        self.assertGreater(new_config.current_interval_ms, 5000)

    def test_fixed_mode_ignores_factors(self):
        config = PollingConfig(base_interval_ms=5000, mode="fixed")
        new_config, adj = adjust_polling_interval(config, recent_volatility=0.2, recent_errors=10)
        self.assertEqual(new_config.current_interval_ms, 5000)

    def test_open_orders_speed_up_polling(self):
        config = PollingConfig(base_interval_ms=5000, min_interval_ms=1000, max_interval_ms=60000)
        new_config, adj = adjust_polling_interval(config, has_open_orders=True)
        self.assertLess(new_config.current_interval_ms, 5000)


# ═══════════════════════════════════════════════════════════════════════════
#  8. Auto recovery integration
# ═══════════════════════════════════════════════════════════════════════════

class TestAutoRecoveryIntegration(unittest.TestCase):
    """Verify the auto recovery system produces correct recovery plans."""

    def test_connection_errors_produce_reconnect_action(self):
        errors = [{"type": "connection"}, {"type": "connection"}, {"type": "timeout"}]
        plan = diagnose_and_recover(errors, max_errors=10)
        actions = [a.action_type for a in plan.actions]
        self.assertIn("reconnect", actions)

    def test_critical_severity_flags_manual_intervention(self):
        errors = [{"type": "connection"}] * 12
        plan = diagnose_and_recover(errors, max_errors=10)
        self.assertTrue(plan.manual_intervention_needed)
        self.assertFalse(plan.can_auto_recover)

    def test_no_errors_no_recovery(self):
        plan = diagnose_and_recover([])
        self.assertFalse(plan.needs_recovery)


# ═══════════════════════════════════════════════════════════════════════════
#  9. Trader cleanup_stale_data integration
# ═══════════════════════════════════════════════════════════════════════════

class TestTraderCleanupIntegration(unittest.TestCase):
    """Verify Trader.cleanup_stale_data removes stale entries."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_cleanup_removes_inactive_pair_data(self):
        config = _make_config()
        trader = Trader(config, client=_StubClient())
        # Seed stale data for a pair not in watchlist
        trader._price_history["dead_idr"] = [(time.time(), 100.0)]
        trader._spread_history["dead_idr"] = [0.001, 0.002]
        trader._prev_depth["dead_idr"] = {"buy": [], "sell": []}
        trader._candle_cache["dead_idr"] = (time.time() - 99999, [])
        # Set the watchlist
        trader._all_pairs = ["btc_idr"]

        trader.cleanup_stale_data()

        self.assertNotIn("dead_idr", trader._price_history)
        self.assertNotIn("dead_idr", trader._spread_history)
        self.assertNotIn("dead_idr", trader._prev_depth)
        self.assertNotIn("dead_idr", trader._candle_cache)

    def test_cleanup_keeps_active_pair_data(self):
        config = _make_config()
        trader = Trader(config, client=_StubClient())
        trader._price_history["btc_idr"] = [(time.time(), 50000.0)]
        trader._spread_history["btc_idr"] = [0.001]
        trader._all_pairs = ["btc_idr"]

        trader.cleanup_stale_data()

        self.assertIn("btc_idr", trader._price_history)
        self.assertIn("btc_idr", trader._spread_history)

    def test_cleanup_keeps_position_pair_data(self):
        """Pairs with open positions must not be cleaned up."""
        config = _make_config()
        trader = Trader(config, client=_StubClient())
        # Simulate an open position by recording a buy
        trader.tracker.record_trade("buy", 50000.0, 0.01)
        trader._price_history[config.pair] = [(time.time(), 50000.0)]
        trader._all_pairs = []  # Empty watchlist

        trader.cleanup_stale_data()

        # The default pair should still be retained since it has a position
        self.assertIn(config.pair, trader._price_history)


# ═══════════════════════════════════════════════════════════════════════════
#  10. End-to-end: autonomous cycle with trader state
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEndAutonomous(unittest.TestCase):
    """End-to-end test combining autonomous state tracking with trader."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_full_cycle_state_updates(self):
        """Simulate a full trading cycle with autonomous state tracking."""
        # Initialize autonomous state
        state = AutonomousTradingState(is_running=True)

        # Simulate 5 successful cycles
        prices = [100 + i * 0.5 for i in range(30)]
        for i in range(5):
            state, result = run_autonomous_cycle(state, prices, pair="btc_idr")

        self.assertEqual(state.total_cycles, 5)
        self.assertEqual(state.successful_cycles, 5)
        self.assertEqual(state.health_status, "running")

        # Simulate error cycle
        state, result = run_autonomous_cycle(state, [], pair="btc_idr")
        self.assertEqual(state.failed_cycles, 1)
        self.assertFalse(result.success)

        # Check health
        health = check_autonomous_health(state)
        self.assertTrue(health["is_running"])

    def test_failover_with_recovery_plan(self):
        """Test that failover triggers recovery plan generation."""
        now = int(time.time() * 1000)
        components = [
            ComponentHealth(name="exchange_api", is_healthy=False, last_heartbeat=now, priority=0),
        ]
        failover = evaluate_failover(components, current_time=now)
        self.assertTrue(failover.failover_triggered)

        # Generate recovery plan
        errors = [{"type": "connection"}, {"type": "timeout"}]
        plan = diagnose_and_recover(errors, component_health=components)
        self.assertTrue(plan.needs_recovery or len(plan.actions) > 0)

    def test_dynamic_polling_with_scheduling(self):
        """Dynamic polling adjusts interval, scheduling picks due tasks."""
        polling = PollingConfig(base_interval_ms=5000, min_interval_ms=1000, max_interval_ms=60000)
        new_polling, _ = adjust_polling_interval(polling, recent_volatility=0.05)

        tasks = [
            ScheduledTask(name="health_check", interval_seconds=120.0, priority=1, enabled=True),
        ]
        result = schedule_tasks(tasks)
        self.assertIn("health_check", result.tasks_due)

        # After running the task, it shouldn't be due immediately
        tasks[0] = update_task_after_run(tasks[0], success=True)
        result2 = schedule_tasks(tasks)
        self.assertEqual(len(result2.tasks_due), 0)


if __name__ == "__main__":
    unittest.main()
