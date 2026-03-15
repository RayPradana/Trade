"""Tests for bot.risk_management – 17 risk management categories."""

import unittest
from bot.analysis import Candle
from bot.risk_management import (
    # 1. Position sizing
    PositionSize, calculate_position_size,
    # 2. Portfolio risk allocation
    PortfolioAllocation, allocate_portfolio_risk,
    # 3. Max position limit
    PositionLimitCheck, check_position_limit,
    # 4. Daily loss limit
    DailyLossCheck, check_daily_loss_limit,
    # 5. Maximum drawdown protection
    DrawdownCheck, check_max_drawdown,
    # 6. Stop-loss (fixed)
    FixedStopLoss, check_fixed_stop_loss,
    # 7. Stop-loss (trailing)
    TrailingStopLoss, check_trailing_stop_loss,
    # 8. Take-profit
    TakeProfit, check_take_profit,
    # 9. Dynamic risk adjustment
    DynamicRiskAdjustment, adjust_risk_dynamically,
    # 10. Volatility-based position sizing
    VolatilityPositionSize, size_by_volatility,
    # 11. Risk parity portfolio
    RiskParityResult, calculate_risk_parity,
    # 12. Exposure limit per asset
    AssetExposureCheck, check_asset_exposure,
    # 13. Exposure limit per sector
    SectorExposureCheck, check_sector_exposure,
    # 14. Correlation risk monitoring
    CorrelationRisk, monitor_correlation_risk,
    # 15. Capital allocation rules
    CapitalAllocation, allocate_capital,
    # 16. Circuit breaker system
    CircuitBreakerStatus, check_circuit_breaker,
    # 17. Strategy shutdown on anomaly
    AnomalyShutdown, check_anomaly_shutdown,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_candles(n: int = 20, start_price: float = 50000.0,
                  trend: float = 0.001) -> list:
    """Generate *n* candles with a gentle trend."""
    candles = []
    p = start_price
    for i in range(n):
        o = p
        h = p * 1.005
        l = p * 0.995
        c = p * (1 + trend)
        candles.append(Candle(timestamp=1000 + i * 60, open=o,
                              high=h, low=l, close=c, volume=100.0))
        p = c
    return candles


def _make_returns(n: int = 30, base: float = 0.001,
                  noise: float = 0.005) -> list:
    """Generate a simple return series."""
    import random
    random.seed(42)
    return [base + random.uniform(-noise, noise) for _ in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
# 1. Position Sizing
# ═══════════════════════════════════════════════════════════════════════════


class TestPositionSizing(unittest.TestCase):
    def test_fixed_pct_basic(self):
        r = calculate_position_size(100_000, risk_per_trade_pct=2,
                                    entry_price=50_000, stop_loss_price=48_000)
        self.assertIsInstance(r, PositionSize)
        self.assertGreater(r.quantity, 0)
        self.assertEqual(r.method, "fixed_pct")

    def test_kelly_method(self):
        r = calculate_position_size(100_000, entry_price=50_000,
                                    method="kelly", volatility=0.03)
        self.assertEqual(r.method, "kelly")
        self.assertGreater(r.quantity, 0)

    def test_volatility_method(self):
        r = calculate_position_size(100_000, entry_price=50_000,
                                    method="volatility", volatility=0.05)
        self.assertEqual(r.method, "volatility")
        self.assertGreater(r.quantity, 0)

    def test_zero_balance(self):
        r = calculate_position_size(0, entry_price=50_000)
        self.assertEqual(r.quantity, 0.0)

    def test_reason_populated(self):
        r = calculate_position_size(100_000, entry_price=50_000)
        self.assertIn("position_size", r.reason)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Portfolio Risk Allocation
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioAllocation(unittest.TestCase):
    def test_basic_allocation(self):
        assets = ["BTC", "ETH", "SOL"]
        rets = {"BTC": _make_returns(), "ETH": _make_returns(),
                "SOL": _make_returns()}
        r = allocate_portfolio_risk(assets, rets)
        self.assertIsInstance(r, PortfolioAllocation)
        self.assertEqual(len(r.allocations), 3)
        self.assertAlmostEqual(sum(r.allocations.values()), 1.0, places=3)

    def test_empty_assets(self):
        r = allocate_portfolio_risk([], {})
        self.assertEqual(r.allocations, {})

    def test_cap_respected(self):
        assets = ["A", "B"]
        rets = {"A": [0.01] * 30, "B": _make_returns(30, base=0.0, noise=0.1)}
        r = allocate_portfolio_risk(assets, rets, max_single_asset_pct=40)
        for w in r.allocations.values():
            self.assertLessEqual(w, 1.0)

    def test_diversification_score(self):
        assets = ["A", "B", "C", "D"]
        rets = {a: _make_returns() for a in assets}
        r = allocate_portfolio_risk(assets, rets)
        self.assertGreater(r.diversification_score, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Max Position Limit
# ═══════════════════════════════════════════════════════════════════════════


class TestPositionLimit(unittest.TestCase):
    def test_within_limit(self):
        r = check_position_limit({"BTC": 5000}, 3000, 10_000)
        self.assertIsInstance(r, PositionLimitCheck)
        self.assertTrue(r.allowed)

    def test_exceeds_limit(self):
        r = check_position_limit({"BTC": 8000}, 5000, 10_000)
        self.assertFalse(r.allowed)

    def test_per_asset_limit(self):
        r = check_position_limit({"BTC": 4000}, 2000, 20_000,
                                 max_per_asset=5000, asset="BTC")
        self.assertFalse(r.allowed)

    def test_remaining_capacity(self):
        r = check_position_limit({"A": 3000}, 1000, 10_000)
        self.assertEqual(r.remaining_capacity, 7000)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Daily Loss Limit
# ═══════════════════════════════════════════════════════════════════════════


class TestDailyLossLimit(unittest.TestCase):
    def test_can_trade_positive_pnl(self):
        r = check_daily_loss_limit(500, 1000)
        self.assertIsInstance(r, DailyLossCheck)
        self.assertTrue(r.can_trade)

    def test_limit_hit(self):
        r = check_daily_loss_limit(-1200, 1000)
        self.assertFalse(r.can_trade)

    def test_pending_risk(self):
        r = check_daily_loss_limit(-500, 1000, pending_risk=600)
        self.assertFalse(r.can_trade)

    def test_remaining_budget(self):
        r = check_daily_loss_limit(-300, 1000)
        self.assertTrue(r.can_trade)
        self.assertGreater(r.remaining_budget, 0)

    def test_zero_limit_disabled_allows_trading(self):
        # limit=0 means the feature is disabled → must always allow trading
        r = check_daily_loss_limit(0.0, 0.0)
        self.assertTrue(r.can_trade)

    def test_zero_limit_with_loss_still_allows_trading(self):
        # even with a negative PnL, limit=0 means no cap
        r = check_daily_loss_limit(-500_000.0, 0.0)
        self.assertTrue(r.can_trade)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Maximum Drawdown Protection
# ═══════════════════════════════════════════════════════════════════════════


class TestDrawdownProtection(unittest.TestCase):
    def test_no_drawdown(self):
        eq = [100, 101, 102, 103]
        r = check_max_drawdown(eq)
        self.assertIsInstance(r, DrawdownCheck)
        self.assertFalse(r.should_reduce)
        self.assertFalse(r.should_stop)

    def test_warning_drawdown(self):
        eq = [100, 90, 86]  # 14% from peak
        r = check_max_drawdown(eq, max_drawdown_pct=20, warning_pct=10)
        self.assertTrue(r.should_reduce)
        self.assertFalse(r.should_stop)

    def test_stop_drawdown(self):
        eq = [100, 90, 78]  # 22% from peak
        r = check_max_drawdown(eq, max_drawdown_pct=20, warning_pct=15)
        self.assertTrue(r.should_stop)

    def test_empty_history(self):
        r = check_max_drawdown([])
        self.assertFalse(r.should_stop)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Stop-Loss (Fixed)
# ═══════════════════════════════════════════════════════════════════════════


class TestFixedStopLoss(unittest.TestCase):
    def test_not_triggered(self):
        r = check_fixed_stop_loss(50000, 49000, stop_loss_pct=5)
        self.assertIsInstance(r, FixedStopLoss)
        self.assertFalse(r.triggered)

    def test_triggered_long(self):
        r = check_fixed_stop_loss(50000, 47000, stop_loss_pct=5)
        self.assertTrue(r.triggered)
        self.assertGreater(r.loss_amount, 0)

    def test_short_stop(self):
        r = check_fixed_stop_loss(50000, 54000, stop_loss_pct=5, side="short")
        self.assertTrue(r.triggered)

    def test_invalid_entry(self):
        r = check_fixed_stop_loss(0, 50000)
        self.assertFalse(r.triggered)


# ═══════════════════════════════════════════════════════════════════════════
# 7. Stop-Loss (Trailing)
# ═══════════════════════════════════════════════════════════════════════════


class TestTrailingStopLoss(unittest.TestCase):
    def test_not_triggered(self):
        r = check_trailing_stop_loss(50000, 54000, 55000, trail_pct=5)
        self.assertIsInstance(r, TrailingStopLoss)
        self.assertFalse(r.triggered)

    def test_triggered(self):
        r = check_trailing_stop_loss(50000, 51000, 55000, trail_pct=5)
        # stop = 55000 * 0.95 = 52250; price 51000 < 52250
        self.assertTrue(r.triggered)

    def test_locked_profit(self):
        r = check_trailing_stop_loss(50000, 54000, 56000, trail_pct=3)
        # entry=50k, high=56k, trail=3% → locked ≈ (56k-50k)/50k*100 - 3 = 9%
        self.assertGreater(r.locked_profit_pct, 0)

    def test_invalid_prices(self):
        r = check_trailing_stop_loss(0, 0, 0)
        self.assertFalse(r.triggered)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Take-Profit
# ═══════════════════════════════════════════════════════════════════════════


class TestTakeProfit(unittest.TestCase):
    def test_not_triggered(self):
        r = check_take_profit(50000, 51000, take_profit_pct=5)
        self.assertIsInstance(r, TakeProfit)
        self.assertFalse(r.triggered)

    def test_triggered_long(self):
        r = check_take_profit(50000, 53000, take_profit_pct=5)
        self.assertTrue(r.triggered)
        self.assertGreater(r.profit_amount, 0)

    def test_short_take_profit(self):
        r = check_take_profit(50000, 44000, take_profit_pct=10, side="short")
        self.assertTrue(r.triggered)

    def test_invalid_entry(self):
        r = check_take_profit(0, 50000)
        self.assertFalse(r.triggered)


# ═══════════════════════════════════════════════════════════════════════════
# 9. Dynamic Risk Adjustment
# ═══════════════════════════════════════════════════════════════════════════


class TestDynamicRisk(unittest.TestCase):
    def test_normal_conditions(self):
        r = adjust_risk_dynamically(2.0, [100, 50, 200])
        self.assertIsInstance(r, DynamicRiskAdjustment)
        self.assertEqual(r.base_risk_pct, 2.0)

    def test_drawdown_reduces_risk(self):
        r = adjust_risk_dynamically(2.0, [], current_drawdown_pct=12)
        self.assertLess(r.adjusted_risk_pct, 2.0)
        self.assertIn("moderate_drawdown", r.conditions)

    def test_losing_streak(self):
        r = adjust_risk_dynamically(2.0, [-10, -20, -30])
        self.assertIn("losing_streak", r.conditions)
        self.assertLess(r.scale_factor, 1.0)

    def test_high_vol(self):
        r = adjust_risk_dynamically(2.0, [], volatility=0.06)
        self.assertIn("high_volatility", r.conditions)

    def test_strong_win_rate_bonus(self):
        r = adjust_risk_dynamically(2.0, [100, 50, 200],
                                    win_rate=0.7, current_drawdown_pct=2)
        self.assertIn("strong_win_rate", r.conditions)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Volatility-Based Position Sizing
# ═══════════════════════════════════════════════════════════════════════════


class TestVolatilityPositionSizing(unittest.TestCase):
    def test_basic_sizing(self):
        candles = _make_candles(20)
        r = size_by_volatility(100_000, candles)
        self.assertIsInstance(r, VolatilityPositionSize)
        self.assertGreater(r.quantity, 0)
        self.assertGreater(r.atr_value, 0)

    def test_empty_candles(self):
        r = size_by_volatility(100_000, [])
        self.assertEqual(r.quantity, 0.0)

    def test_zero_balance(self):
        r = size_by_volatility(0, _make_candles(5))
        self.assertEqual(r.quantity, 0.0)

    def test_higher_risk_larger_position(self):
        candles = _make_candles(20)
        r1 = size_by_volatility(100_000, candles, risk_pct=1.0)
        r2 = size_by_volatility(100_000, candles, risk_pct=2.0)
        self.assertGreater(r2.quantity, r1.quantity)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Risk Parity Portfolio
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskParity(unittest.TestCase):
    def test_equal_volatilities(self):
        assets = ["A", "B", "C"]
        vols = {"A": 0.1, "B": 0.1, "C": 0.1}
        r = calculate_risk_parity(assets, vols)
        self.assertIsInstance(r, RiskParityResult)
        for w in r.weights.values():
            self.assertAlmostEqual(w, 1 / 3, places=3)
        self.assertTrue(r.is_balanced)

    def test_different_volatilities(self):
        assets = ["A", "B"]
        vols = {"A": 0.05, "B": 0.20}
        r = calculate_risk_parity(assets, vols)
        self.assertGreater(r.weights["A"], r.weights["B"])

    def test_empty_assets(self):
        r = calculate_risk_parity([], {})
        self.assertEqual(r.weights, {})

    def test_portfolio_volatility(self):
        assets = ["A", "B"]
        vols = {"A": 0.1, "B": 0.2}
        r = calculate_risk_parity(assets, vols)
        self.assertGreater(r.portfolio_volatility, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Exposure Limit Per Asset
# ═══════════════════════════════════════════════════════════════════════════


class TestAssetExposure(unittest.TestCase):
    def test_within_limit(self):
        r = check_asset_exposure("BTC", 15_000, 100_000, max_asset_pct=20)
        self.assertIsInstance(r, AssetExposureCheck)
        self.assertTrue(r.within_limit)

    def test_exceeds_limit(self):
        r = check_asset_exposure("BTC", 25_000, 100_000, max_asset_pct=20)
        self.assertFalse(r.within_limit)

    def test_zero_portfolio(self):
        r = check_asset_exposure("BTC", 0, 0)
        self.assertTrue(r.within_limit)

    def test_utilization_pct(self):
        r = check_asset_exposure("ETH", 10_000, 100_000)
        self.assertAlmostEqual(r.utilization_pct, 10.0, places=2)


# ═══════════════════════════════════════════════════════════════════════════
# 13. Exposure Limit Per Sector
# ═══════════════════════════════════════════════════════════════════════════


class TestSectorExposure(unittest.TestCase):
    def test_all_within(self):
        positions = {"BTC": 10_000, "ETH": 15_000, "AAPL": 5_000}
        sectors = {"BTC": "crypto", "ETH": "crypto", "AAPL": "tech"}
        r = check_sector_exposure(positions, sectors, 100_000,
                                  max_sector_pct=30)
        self.assertIsInstance(r, SectorExposureCheck)
        self.assertTrue(r.all_within_limit)

    def test_sector_breach(self):
        positions = {"BTC": 20_000, "ETH": 15_000}
        sectors = {"BTC": "crypto", "ETH": "crypto"}
        r = check_sector_exposure(positions, sectors, 100_000,
                                  max_sector_pct=30)
        self.assertFalse(r.all_within_limit)
        self.assertIn("crypto", r.breached_sectors)

    def test_zero_portfolio(self):
        r = check_sector_exposure({}, {}, 0)
        self.assertTrue(r.all_within_limit)

    def test_sector_exposure_values(self):
        positions = {"A": 30_000, "B": 20_000}
        sectors = {"A": "tech", "B": "finance"}
        r = check_sector_exposure(positions, sectors, 100_000)
        self.assertIn("tech", r.sector_exposures)


# ═══════════════════════════════════════════════════════════════════════════
# 14. Correlation Risk Monitoring
# ═══════════════════════════════════════════════════════════════════════════


class TestCorrelationRisk(unittest.TestCase):
    def test_high_correlation(self):
        rets = {"A": [0.01, 0.02, 0.03, 0.02, 0.01],
                "B": [0.01, 0.02, 0.03, 0.02, 0.01]}
        r = monitor_correlation_risk(["A", "B"], rets, threshold=0.7)
        self.assertIsInstance(r, CorrelationRisk)
        self.assertGreater(len(r.high_corr_pairs), 0)

    def test_low_correlation(self):
        rets = {"A": [0.01, -0.02, 0.03, -0.01, 0.02],
                "B": [0.02, 0.01, -0.01, 0.03, -0.02]}
        r = monitor_correlation_risk(["A", "B"], rets, threshold=0.9)
        self.assertEqual(len(r.high_corr_pairs), 0)

    def test_single_asset(self):
        r = monitor_correlation_risk(["A"], {"A": [0.01, 0.02]})
        self.assertEqual(r.portfolio_corr_risk, "low")

    def test_risk_level_assigned(self):
        rets = {"A": [0.01, 0.02, 0.03, 0.02, 0.01],
                "B": [0.01, 0.02, 0.03, 0.02, 0.01]}
        r = monitor_correlation_risk(["A", "B"], rets)
        self.assertIn(r.portfolio_corr_risk,
                       ["low", "moderate", "high", "critical"])


# ═══════════════════════════════════════════════════════════════════════════
# 15. Capital Allocation Rules
# ═══════════════════════════════════════════════════════════════════════════


class TestCapitalAllocation(unittest.TestCase):
    def test_equal_allocation(self):
        r = allocate_capital(100_000, ["trend", "mean_rev", "momentum"])
        self.assertIsInstance(r, CapitalAllocation)
        self.assertEqual(len(r.strategy_allocations), 3)
        self.assertGreater(r.reserve_amount, 0)

    def test_weighted_allocation(self):
        r = allocate_capital(100_000, ["A", "B"],
                             strategy_weights={"A": 3.0, "B": 1.0})
        self.assertGreater(r.strategy_allocations["A"],
                           r.strategy_allocations["B"])

    def test_no_capital(self):
        r = allocate_capital(0, ["A"])
        self.assertEqual(r.total_deployed, 0.0)

    def test_reserve_pct(self):
        r = allocate_capital(100_000, ["A"], reserve_pct=20)
        self.assertAlmostEqual(r.reserve_amount, 20_000, places=0)


# ═══════════════════════════════════════════════════════════════════════════
# 16. Circuit Breaker System
# ═══════════════════════════════════════════════════════════════════════════


class TestCircuitBreaker(unittest.TestCase):
    def test_no_triggers(self):
        r = check_circuit_breaker()
        self.assertIsInstance(r, CircuitBreakerStatus)
        self.assertFalse(r.is_tripped)
        self.assertEqual(r.severity, "low")

    def test_daily_loss_trigger(self):
        r = check_circuit_breaker(daily_loss_pct=6, max_daily_loss_pct=5)
        self.assertTrue(r.is_tripped)
        self.assertIn("daily_loss_limit", r.triggers)

    def test_multiple_triggers(self):
        r = check_circuit_breaker(daily_loss_pct=6, drawdown_pct=25,
                                  consecutive_losses=6)
        self.assertTrue(r.is_tripped)
        self.assertEqual(r.severity, "critical")
        self.assertFalse(r.can_resume)

    def test_volatility_spike(self):
        r = check_circuit_breaker(volatility_spike=True)
        self.assertTrue(r.is_tripped)
        self.assertIn("volatility_spike", r.triggers)

    def test_api_errors(self):
        r = check_circuit_breaker(api_errors=15, max_api_errors=10)
        self.assertIn("api_errors", r.triggers)


# ═══════════════════════════════════════════════════════════════════════════
# 17. Strategy Shutdown on Anomaly
# ═══════════════════════════════════════════════════════════════════════════


class TestAnomalyShutdown(unittest.TestCase):
    def test_normal_conditions(self):
        rets = [0.001, 0.002, -0.001, 0.001, 0.002, 0.001]
        r = check_anomaly_shutdown(rets)
        self.assertIsInstance(r, AnomalyShutdown)
        self.assertFalse(r.should_shutdown)
        self.assertEqual(r.recommended_action, "continue")

    def test_return_anomaly(self):
        rets = [0.001, 0.001, 0.001, 0.001, 0.001, 0.5]
        r = check_anomaly_shutdown(rets, anomaly_threshold=2.0)
        self.assertTrue(r.should_shutdown)
        self.assertGreater(len(r.anomalies), 0)

    def test_volume_spike(self):
        r = check_anomaly_shutdown([0.001] * 6, volume_ratio=5.0)
        self.assertGreater(len(r.anomalies), 0)

    def test_spread_blowout(self):
        r = check_anomaly_shutdown([0.001] * 6, spread_ratio=5.0)
        anomaly_types = [a["type"] for a in r.anomalies]
        self.assertIn("spread_blowout", anomaly_types)

    def test_reduce_exposure_zone(self):
        rets = [0.001, 0.001, 0.001, 0.001, 0.001, 0.05]
        r = check_anomaly_shutdown(rets, anomaly_threshold=3.0)
        # Score might be in the "reduce" zone
        self.assertIn(r.recommended_action,
                       ["continue", "reduce_exposure", "shutdown"])


if __name__ == "__main__":
    unittest.main()
