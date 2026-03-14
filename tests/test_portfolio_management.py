"""Tests for bot/portfolio_management.py – 11 portfolio management categories."""

import unittest

from bot.portfolio_management import (
    # 1. Multi-asset portfolio management
    MultiAssetPortfolio,
    evaluate_multi_asset_portfolio,
    # 2. Portfolio rebalancing
    RebalanceAction,
    RebalancePlan,
    plan_rebalance,
    # 3. Portfolio diversification
    DiversificationScore,
    assess_diversification,
    # 4. Asset allocation engine
    AssetAllocationPlan,
    compute_asset_allocation,
    # 5. Correlation matrix monitoring
    CorrelationMatrix,
    compute_correlation_matrix,
    # 6. Portfolio optimization
    OptimizedPortfolio,
    optimize_portfolio,
    # 7. Risk-adjusted return optimization
    RiskAdjustedMetrics,
    evaluate_risk_adjusted_returns,
    # 8. Capital distribution across strategies
    StrategyCapitalPlan,
    distribute_capital,
    # 9. Sector exposure monitoring
    SectorExposure,
    monitor_sector_exposure,
    # 10. Cross-market hedging
    HedgeRecommendation,
    recommend_hedges,
    # 11. Long/short portfolio balancing
    LongShortBalance,
    analyze_long_short_balance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_returns(n: int = 30, base: float = 0.001, noise: float = 0.01) -> list:
    """Generate synthetic return series."""
    import math
    return [base + noise * math.sin(i * 0.5) for i in range(n)]


def _make_returns_neg(n: int = 30) -> list:
    import math
    return [-0.002 + 0.01 * math.sin(i * 0.5) for i in range(n)]


def _make_prices(n: int = 30, start: float = 100.0, trend: float = 0.5) -> list:
    return [start + trend * i for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════
#  1. Multi-Asset Portfolio Management
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiAssetPortfolio(unittest.TestCase):
    def test_basic_holdings(self):
        r = evaluate_multi_asset_portfolio({"BTC": 1000, "ETH": 500, "SOL": 300})
        self.assertIsInstance(r, MultiAssetPortfolio)
        self.assertEqual(r.num_assets, 3)
        self.assertAlmostEqual(r.total_value, 1800.0, places=1)
        self.assertAlmostEqual(sum(r.weights.values()), 1.0, places=3)

    def test_with_prices(self):
        r = evaluate_multi_asset_portfolio(
            {"BTC": 0.5, "ETH": 10},
            prices={"BTC": 60000, "ETH": 3000},
        )
        self.assertEqual(r.num_assets, 2)
        self.assertGreater(r.total_value, 0)

    def test_single_asset_concentration(self):
        r = evaluate_multi_asset_portfolio({"BTC": 10000})
        self.assertAlmostEqual(r.concentration_score, 1.0, places=4)

    def test_empty_holdings(self):
        r = evaluate_multi_asset_portfolio({})
        self.assertEqual(r.num_assets, 0)

    def test_reason_populated(self):
        r = evaluate_multi_asset_portfolio({"A": 100, "B": 200})
        self.assertIn("portfolio", r.reason)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Portfolio Rebalancing
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioRebalancing(unittest.TestCase):
    def test_no_drift(self):
        r = plan_rebalance(
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "B": 0.5},
            portfolio_value=100000,
        )
        self.assertIsInstance(r, RebalancePlan)
        self.assertFalse(r.needs_rebalance)
        self.assertEqual(len(r.actions), 0)

    def test_drift_triggers_rebalance(self):
        r = plan_rebalance(
            {"A": 0.7, "B": 0.3},
            {"A": 0.5, "B": 0.5},
            portfolio_value=100000,
            threshold_pct=5.0,
        )
        self.assertTrue(r.needs_rebalance)
        self.assertGreater(len(r.actions), 0)
        self.assertGreater(r.total_turnover, 0)

    def test_action_directions(self):
        r = plan_rebalance(
            {"A": 0.7, "B": 0.3},
            {"A": 0.5, "B": 0.5},
            portfolio_value=100000,
        )
        dirs = {a.asset: a.direction for a in r.actions}
        self.assertEqual(dirs.get("A"), "sell")
        self.assertEqual(dirs.get("B"), "buy")

    def test_new_asset_in_target(self):
        r = plan_rebalance(
            {"A": 1.0},
            {"A": 0.5, "B": 0.5},
            portfolio_value=100000,
        )
        self.assertTrue(r.needs_rebalance)

    def test_high_threshold_no_action(self):
        r = plan_rebalance(
            {"A": 0.52, "B": 0.48},
            {"A": 0.5, "B": 0.5},
            portfolio_value=100000,
            threshold_pct=10.0,
        )
        self.assertFalse(r.needs_rebalance)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Portfolio Diversification
# ═══════════════════════════════════════════════════════════════════════════


class TestDiversification(unittest.TestCase):
    def test_empty_portfolio(self):
        r = assess_diversification({})
        self.assertIsInstance(r, DiversificationScore)
        self.assertEqual(r.score, 0.0)

    def test_single_asset(self):
        r = assess_diversification({"BTC": 1.0})
        self.assertEqual(r.num_assets, 1)
        self.assertAlmostEqual(r.hhi, 1.0, places=4)

    def test_equal_weight_diversified(self):
        wts = {f"A{i}": 0.1 for i in range(10)}
        r = assess_diversification(wts)
        self.assertGreater(r.score, 0.5)
        self.assertGreater(r.effective_assets, 5)

    def test_concentrated_low_score(self):
        r = assess_diversification({"A": 0.95, "B": 0.05})
        self.assertLess(r.score, 0.2)

    def test_recommendations(self):
        r = assess_diversification({"A": 1.0})
        self.assertIn("reduce_concentration", r.recommendations)

    def test_with_correlated_returns(self):
        rets = {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05],
            "B": [0.01, 0.02, 0.03, 0.04, 0.05],  # perfectly correlated
        }
        r = assess_diversification({"A": 0.5, "B": 0.5}, returns_map=rets)
        self.assertIn("high_avg_correlation", r.recommendations)


# ═══════════════════════════════════════════════════════════════════════════
#  4. Asset Allocation Engine
# ═══════════════════════════════════════════════════════════════════════════


class TestAssetAllocation(unittest.TestCase):
    def test_equal_weight(self):
        rets = {"A": _make_returns(), "B": _make_returns(), "C": _make_returns()}
        r = compute_asset_allocation(["A", "B", "C"], rets, method="equal_weight")
        self.assertIsInstance(r, AssetAllocationPlan)
        self.assertEqual(r.method, "equal_weight")
        self.assertAlmostEqual(sum(r.target_weights.values()), 1.0, places=3)

    def test_inverse_vol(self):
        rets = {
            "A": _make_returns(noise=0.001),
            "B": _make_returns(noise=0.05),
        }
        r = compute_asset_allocation(["A", "B"], rets, method="inverse_vol")
        # Lower vol asset should get higher weight
        self.assertGreater(r.target_weights["A"], r.target_weights["B"])

    def test_momentum(self):
        rets = {
            "A": _make_returns(base=0.01),
            "B": _make_returns(base=-0.01),
        }
        r = compute_asset_allocation(["A", "B"], rets, method="momentum")
        self.assertGreater(r.target_weights["A"], 0)

    def test_empty_assets(self):
        r = compute_asset_allocation([], {})
        self.assertEqual(len(r.target_weights), 0)

    def test_reason(self):
        r = compute_asset_allocation(["X"], {"X": _make_returns()})
        self.assertIn("allocation", r.reason)


# ═══════════════════════════════════════════════════════════════════════════
#  5. Correlation Matrix Monitoring
# ═══════════════════════════════════════════════════════════════════════════


class TestCorrelationMatrix(unittest.TestCase):
    def test_basic_matrix(self):
        import math
        rets = {
            "A": _make_returns(),
            "B": _make_returns(),
            "C": [r * -1 for r in _make_returns()],
        }
        r = compute_correlation_matrix(["A", "B", "C"], rets)
        self.assertIsInstance(r, CorrelationMatrix)
        self.assertEqual(r.matrix["A"]["A"], 1.0)
        self.assertIn(r.risk_level, ("low", "moderate", "high", "critical"))

    def test_high_correlation_pairs(self):
        rets = {
            "A": [0.01, 0.02, 0.03, 0.04, 0.05],
            "B": [0.01, 0.02, 0.03, 0.04, 0.05],
        }
        r = compute_correlation_matrix(["A", "B"], rets, alert_threshold=0.5)
        self.assertGreater(len(r.high_pairs), 0)

    def test_low_correlation(self):
        import math
        rets_a = [math.sin(i * 0.3) * 0.01 for i in range(30)]
        rets_b = [math.cos(i * 0.7) * 0.01 for i in range(30)]
        r = compute_correlation_matrix(["A", "B"], {"A": rets_a, "B": rets_b})
        self.assertLessEqual(r.max_correlation, 1.0)

    def test_single_asset(self):
        r = compute_correlation_matrix(["A"], {"A": _make_returns()})
        self.assertEqual(r.matrix["A"]["A"], 1.0)
        self.assertEqual(len(r.high_pairs), 0)

    def test_empty(self):
        r = compute_correlation_matrix([], {})
        self.assertEqual(len(r.matrix), 0)


# ═══════════════════════════════════════════════════════════════════════════
#  6. Portfolio Optimization
# ═══════════════════════════════════════════════════════════════════════════


class TestPortfolioOptimization(unittest.TestCase):
    def test_min_variance(self):
        rets = {"A": _make_returns(noise=0.001), "B": _make_returns(noise=0.05)}
        r = optimize_portfolio(["A", "B"], rets, method="min_variance")
        self.assertIsInstance(r, OptimizedPortfolio)
        self.assertEqual(r.method, "min_variance")
        self.assertAlmostEqual(sum(r.weights.values()), 1.0, places=3)

    def test_max_sharpe(self):
        rets = {"A": _make_returns(base=0.005), "B": _make_returns(base=0.001)}
        r = optimize_portfolio(["A", "B"], rets, method="max_sharpe")
        self.assertAlmostEqual(sum(r.weights.values()), 1.0, places=3)

    def test_equal_risk(self):
        rets = {"A": _make_returns(), "B": _make_returns()}
        r = optimize_portfolio(["A", "B"], rets, method="equal_risk")
        self.assertAlmostEqual(sum(r.weights.values()), 1.0, places=3)

    def test_weight_cap(self):
        rets = {"A": _make_returns(noise=0.001), "B": _make_returns(noise=0.05)}
        r = optimize_portfolio(["A", "B"], rets, max_weight=0.6)
        for w in r.weights.values():
            self.assertLessEqual(w, 0.65)  # small tolerance for rounding

    def test_empty_assets(self):
        r = optimize_portfolio([], {})
        self.assertEqual(len(r.weights), 0)

    def test_drawdown_estimate(self):
        rets = {"A": _make_returns()}
        r = optimize_portfolio(["A"], rets)
        self.assertGreaterEqual(r.max_drawdown_est, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  7. Risk-Adjusted Return Optimization
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskAdjustedReturns(unittest.TestCase):
    def test_basic_metrics(self):
        rets = {"A": _make_returns(), "B": _make_returns_neg()}
        r = evaluate_risk_adjusted_returns(["A", "B"], rets)
        self.assertIsInstance(r, RiskAdjustedMetrics)
        self.assertIn("A", r.per_asset)
        self.assertIn("B", r.per_asset)

    def test_best_worst(self):
        rets = {"GOOD": _make_returns(base=0.01), "BAD": _make_returns_neg()}
        r = evaluate_risk_adjusted_returns(["GOOD", "BAD"], rets)
        self.assertEqual(r.best_risk_adjusted, "GOOD")
        self.assertEqual(r.worst_risk_adjusted, "BAD")

    def test_per_asset_keys(self):
        rets = {"X": _make_returns()}
        r = evaluate_risk_adjusted_returns(["X"], rets)
        keys = set(r.per_asset["X"].keys())
        self.assertIn("sharpe", keys)
        self.assertIn("sortino", keys)
        self.assertIn("calmar", keys)

    def test_with_prices(self):
        rets = {"A": _make_returns()}
        prices = {"A": _make_prices()}
        r = evaluate_risk_adjusted_returns(["A"], rets, prices_map=prices)
        self.assertGreaterEqual(r.per_asset["A"]["max_drawdown"], 0)

    def test_reason(self):
        r = evaluate_risk_adjusted_returns(["A"], {"A": _make_returns()})
        self.assertIn("risk_adj", r.reason)


# ═══════════════════════════════════════════════════════════════════════════
#  8. Capital Distribution Across Strategies
# ═══════════════════════════════════════════════════════════════════════════


class TestCapitalDistribution(unittest.TestCase):
    def test_equal_distribution(self):
        r = distribute_capital(100000, ["S1", "S2", "S3"], method="equal")
        self.assertIsInstance(r, StrategyCapitalPlan)
        self.assertEqual(len(r.allocations), 3)
        self.assertGreater(r.reserve_amount, 0)  # default 10% reserve
        self.assertAlmostEqual(
            r.total_deployed + r.reserve_amount,
            r.total_capital,
            delta=100,  # allow rounding
        )

    def test_performance_method(self):
        rets = {"S1": _make_returns(base=0.01), "S2": _make_returns(base=0.001)}
        r = distribute_capital(100000, ["S1", "S2"],
                               strategy_returns=rets, method="performance")
        self.assertGreater(r.allocations.get("S1", 0), 0)

    def test_inverse_vol(self):
        rets = {
            "S1": _make_returns(noise=0.001),
            "S2": _make_returns(noise=0.05),
        }
        r = distribute_capital(100000, ["S1", "S2"],
                               strategy_returns=rets, method="inverse_vol")
        self.assertGreater(r.allocations["S1"], r.allocations["S2"])

    def test_zero_capital(self):
        r = distribute_capital(0, ["S1"])
        self.assertEqual(len(r.allocations), 0)

    def test_reserve(self):
        r = distribute_capital(100000, ["S1"], reserve_pct=20.0)
        self.assertAlmostEqual(r.reserve_amount, 20000, delta=1)


# ═══════════════════════════════════════════════════════════════════════════
#  9. Sector Exposure Monitoring
# ═══════════════════════════════════════════════════════════════════════════


class TestSectorExposure(unittest.TestCase):
    def test_basic_sectors(self):
        positions = {"BTC": 5000, "ETH": 3000, "AAPL": 2000}
        sectors = {"BTC": "crypto", "ETH": "crypto", "AAPL": "tech"}
        r = monitor_sector_exposure(positions, sectors, portfolio_value=10000)
        self.assertIsInstance(r, SectorExposure)
        self.assertIn("crypto", r.sector_weights)

    def test_overexposure(self):
        positions = {"BTC": 9000, "ETH": 500, "SOL": 500}
        sectors = {"BTC": "crypto", "ETH": "crypto", "SOL": "crypto"}
        r = monitor_sector_exposure(positions, sectors, portfolio_value=10000,
                                     max_sector_pct=80.0)
        crypto_wt = r.sector_weights.get("crypto", 0)
        self.assertGreater(crypto_wt, 80)
        self.assertIn("crypto", r.overexposed)

    def test_underexposure(self):
        positions = {"A": 100}
        sectors = {"A": "small_cap"}
        r = monitor_sector_exposure(positions, sectors, portfolio_value=10000,
                                     min_sector_pct=5.0)
        self.assertIn("small_cap", r.underexposed)

    def test_zero_portfolio(self):
        r = monitor_sector_exposure({}, {}, portfolio_value=0)
        self.assertEqual(r.portfolio_value, 0.0)

    def test_reason(self):
        r = monitor_sector_exposure({"A": 100}, {"A": "x"}, portfolio_value=1000)
        self.assertIn("sector", r.reason)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Cross-Market Hedging
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossMarketHedging(unittest.TestCase):
    def test_negative_correlation_hedge(self):
        import math
        rets_a = [math.sin(i * 0.5) * 0.01 for i in range(30)]
        rets_b = [-math.sin(i * 0.5) * 0.01 for i in range(30)]  # inverse
        rets = {"LONG": rets_a, "HEDGE": rets_b}
        r = recommend_hedges(["LONG"], ["HEDGE"], rets)
        self.assertIsInstance(r, HedgeRecommendation)
        self.assertIn("LONG", r.hedge_pairs)
        self.assertEqual(r.hedge_pairs["LONG"], "HEDGE")

    def test_no_candidates(self):
        r = recommend_hedges(["A"], [], {"A": _make_returns()})
        self.assertEqual(len(r.hedge_pairs), 0)

    def test_empty_portfolio(self):
        r = recommend_hedges([], ["H1"], {"H1": _make_returns()})
        self.assertEqual(len(r.hedge_pairs), 0)

    def test_risk_reduction(self):
        import math
        rets_a = [math.sin(i * 0.5) * 0.01 for i in range(30)]
        rets_b = [-math.sin(i * 0.5) * 0.01 for i in range(30)]
        rets = {"A": rets_a, "B": rets_b}
        r = recommend_hedges(["A"], ["B"], rets)
        self.assertGreaterEqual(r.risk_reduction_pct, 0)

    def test_reason(self):
        r = recommend_hedges(["A"], ["B"], {
            "A": _make_returns(),
            "B": [r * -1 for r in _make_returns()],
        })
        self.assertIn("hedge", r.reason)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Long / Short Portfolio Balancing
# ═══════════════════════════════════════════════════════════════════════════


class TestLongShortBalance(unittest.TestCase):
    def test_long_only(self):
        positions = {"A": 5000, "B": 3000, "C": 2000}
        r = analyze_long_short_balance(positions, portfolio_value=10000)
        self.assertIsInstance(r, LongShortBalance)
        self.assertEqual(r.short_exposure, 0)
        self.assertEqual(len(r.short_assets), 0)
        self.assertEqual(len(r.long_assets), 3)

    def test_balanced_portfolio(self):
        positions = {"L1": 5000, "S1": -5000}
        r = analyze_long_short_balance(positions, portfolio_value=10000,
                                        target_net_pct=0.0, tolerance_pct=5.0)
        self.assertTrue(r.is_balanced)
        self.assertAlmostEqual(r.net_pct, 0, places=1)

    def test_unbalanced_long_heavy(self):
        positions = {"L1": 8000, "S1": -1000}
        r = analyze_long_short_balance(positions, portfolio_value=10000,
                                        target_net_pct=0.0, tolerance_pct=10.0)
        self.assertFalse(r.is_balanced)
        self.assertEqual(r.recommendation, "increase_short_or_reduce_long")

    def test_gross_exposure(self):
        positions = {"L": 6000, "S": -4000}
        r = analyze_long_short_balance(positions, portfolio_value=10000)
        self.assertAlmostEqual(r.gross_exposure, 10000, places=1)

    def test_reason(self):
        r = analyze_long_short_balance({"A": 1000}, portfolio_value=5000)
        self.assertIn("ls_balance", r.reason)

    def test_empty_positions(self):
        r = analyze_long_short_balance({}, portfolio_value=10000)
        self.assertEqual(r.long_exposure, 0)
        self.assertEqual(r.short_exposure, 0)


if __name__ == "__main__":
    unittest.main()
