"""Tests for bot/backtesting.py – 13 backtesting & simulation categories."""

import unittest
import math

from bot.backtesting import (
    # 1. Historical backtesting
    BacktestTrade,
    HistoricalBacktestResult,
    run_historical_backtest,
    # 2. Tick-level backtesting
    Tick,
    TickBacktestResult,
    run_tick_backtest,
    # 3. Walk-forward testing
    WalkForwardResult,
    run_walk_forward_test,
    # 4. Monte Carlo simulation
    MonteCarloResult,
    run_monte_carlo,
    # 5. Parameter optimization
    ParamOptResult,
    optimize_parameters,
    # 6. Strategy robustness testing
    RobustnessResult,
    evaluate_strategy_robustness,
    # 7. Stress testing
    StressTestResult,
    run_stress_test,
    # 8. Slippage simulation
    SlippageModel,
    SlippageResult,
    simulate_slippage,
    # 9. Transaction cost modeling
    TransactionCostModel,
    TransactionCostResult,
    model_transaction_costs,
    # 10. Liquidity simulation
    LiquiditySimResult,
    simulate_liquidity,
    # 11. Out-of-sample testing
    OutOfSampleResult,
    run_out_of_sample_test,
    # 12. Forward testing
    ForwardTestResult,
    run_forward_test,
    # 13. Paper trading mode
    PaperOrder,
    PaperTradingState,
    PaperTradingEngine,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=60, start=100.0, trend=0.5):
    return [start + trend * i for i in range(n)]


def _make_returns(n=60, base=0.001, noise=0.01):
    return [base + noise * math.sin(i * 0.5) for i in range(n)]


def _make_signals(n=60):
    """Alternating buy/sell signals."""
    sigs = []
    for i in range(n):
        if i % 10 < 5:
            sigs.append(1)
        else:
            sigs.append(-1)
    return sigs


def _simple_strategy(prices, params=None):
    """Trivial strategy: return total return."""
    if len(prices) < 2:
        return 0.0
    return (prices[-1] - prices[0]) / prices[0]


def _signal_func(tick, recent):
    """Simple tick signal: buy if price up, sell if down."""
    if len(recent) < 2:
        return 0
    if recent[-1].price > recent[0].price:
        return 1
    elif recent[-1].price < recent[0].price:
        return -1
    return 0


def _bar_signal_func(prices):
    """Simple bar signal."""
    if len(prices) < 5:
        return 0
    if prices[-1] > prices[-3]:
        return 1
    elif prices[-1] < prices[-3]:
        return -1
    return 0


def _make_trades(n=5, base_price=100.0):
    trades = []
    for i in range(n):
        ep = base_price + i * 2
        xp = ep + 1.0 + (i % 3)
        qty = 10.0
        pnl = (xp - ep) * qty
        trades.append(BacktestTrade(
            entry_index=i * 10,
            exit_index=i * 10 + 5,
            side="long",
            entry_price=ep,
            exit_price=xp,
            quantity=qty,
            pnl=pnl,
            return_pct=pnl / (ep * qty),
        ))
    return trades


# ═══════════════════════════════════════════════════════════════════════════
#  1. Historical Backtesting
# ═══════════════════════════════════════════════════════════════════════════


class TestHistoricalBacktest(unittest.TestCase):
    def test_basic_backtest(self):
        prices = _make_prices(60)
        signals = _make_signals(60)
        r = run_historical_backtest(prices, signals)
        self.assertIsInstance(r, HistoricalBacktestResult)
        self.assertGreater(len(r.equity_curve), 1)
        self.assertGreater(r.total_trades, 0)

    def test_empty_input(self):
        r = run_historical_backtest([], [])
        self.assertEqual(r.total_trades, 0)
        self.assertEqual(r.total_return, 0.0)

    def test_all_hold(self):
        prices = _make_prices(30)
        signals = [0] * 30
        r = run_historical_backtest(prices, signals)
        self.assertEqual(r.total_trades, 0)
        self.assertAlmostEqual(r.total_return, 0.0, places=5)

    def test_win_rate_bounds(self):
        prices = _make_prices(60)
        signals = _make_signals(60)
        r = run_historical_backtest(prices, signals)
        self.assertGreaterEqual(r.win_rate, 0.0)
        self.assertLessEqual(r.win_rate, 1.0)

    def test_custom_capital(self):
        prices = _make_prices(60)
        signals = _make_signals(60)
        r = run_historical_backtest(prices, signals, initial_capital=50000)
        self.assertAlmostEqual(r.equity_curve[0], 50000.0, places=1)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Tick-Level Backtesting
# ═══════════════════════════════════════════════════════════════════════════


class TestTickBacktest(unittest.TestCase):
    def test_basic_tick_backtest(self):
        ticks = [Tick(timestamp=i, price=100.0 + i * 0.5, volume=1000)
                 for i in range(50)]
        r = run_tick_backtest(ticks, _signal_func)
        self.assertIsInstance(r, TickBacktestResult)
        self.assertGreater(len(r.equity_curve), 1)

    def test_empty_ticks(self):
        r = run_tick_backtest([], _signal_func)
        self.assertEqual(r.total_trades, 0)
        self.assertEqual(r.total_return, 0.0)

    def test_single_tick(self):
        ticks = [Tick(timestamp=0, price=100, volume=100)]
        r = run_tick_backtest(ticks, _signal_func)
        self.assertEqual(r.total_trades, 0)

    def test_flat_prices(self):
        ticks = [Tick(timestamp=i, price=100.0, volume=1000)
                 for i in range(50)]
        r = run_tick_backtest(ticks, _signal_func)
        self.assertIsInstance(r, TickBacktestResult)

    def test_avg_duration(self):
        ticks = [Tick(timestamp=i, price=100.0 + i * 0.5, volume=1000)
                 for i in range(50)]
        r = run_tick_backtest(ticks, _signal_func)
        self.assertGreaterEqual(r.avg_trade_duration, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Walk-Forward Testing
# ═══════════════════════════════════════════════════════════════════════════


class TestWalkForward(unittest.TestCase):
    def test_basic_walk_forward(self):
        prices = _make_prices(100)
        params = [{"window": 5}, {"window": 10}, {"window": 20}]
        r = run_walk_forward_test(prices, _simple_strategy, params, n_folds=3)
        self.assertIsInstance(r, WalkForwardResult)
        self.assertGreater(len(r.folds), 0)

    def test_empty_prices(self):
        r = run_walk_forward_test([], _simple_strategy, [{"w": 5}])
        self.assertEqual(len(r.folds), 0)

    def test_efficiency_ratio(self):
        prices = _make_prices(100, trend=1.0)
        params = [{"w": 5}, {"w": 10}]
        r = run_walk_forward_test(prices, _simple_strategy, params, n_folds=3)
        self.assertIsInstance(r.efficiency_ratio, float)

    def test_fold_data(self):
        prices = _make_prices(100)
        params = [{"w": 5}]
        r = run_walk_forward_test(prices, _simple_strategy, params, n_folds=4)
        for f in r.folds:
            self.assertGreaterEqual(f.test_start, f.train_end)


# ═══════════════════════════════════════════════════════════════════════════
#  4. Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════


class TestMonteCarlo(unittest.TestCase):
    def test_basic_mc(self):
        rets = _make_returns(60)
        r = run_monte_carlo(rets, num_simulations=100, seed=42)
        self.assertIsInstance(r, MonteCarloResult)
        self.assertEqual(r.num_simulations, 100)

    def test_empty_returns(self):
        r = run_monte_carlo([])
        self.assertEqual(r.num_simulations, 0)

    def test_prob_profit(self):
        rets = _make_returns(60, base=0.005)
        r = run_monte_carlo(rets, num_simulations=200, seed=42)
        self.assertGreaterEqual(r.prob_profit, 0.0)
        self.assertLessEqual(r.prob_profit, 1.0)

    def test_percentiles(self):
        rets = _make_returns(60)
        r = run_monte_carlo(rets, num_simulations=500, seed=42)
        self.assertLessEqual(r.p5_return, r.median_return)
        self.assertLessEqual(r.median_return, r.p95_return)

    def test_store_paths(self):
        rets = _make_returns(30)
        r = run_monte_carlo(rets, num_simulations=10, seed=42, store_paths=True)
        self.assertEqual(len(r.paths), 10)


# ═══════════════════════════════════════════════════════════════════════════
#  5. Parameter Optimization
# ═══════════════════════════════════════════════════════════════════════════


class TestParamOptimization(unittest.TestCase):
    def test_basic_optimization(self):
        prices = _make_prices(60)
        params = [{"w": 5}, {"w": 10}, {"w": 20}]
        r = optimize_parameters(prices, _simple_strategy, params)
        self.assertIsInstance(r, ParamOptResult)
        self.assertIn("w", r.best_params)
        self.assertEqual(r.total_combinations, 3)

    def test_empty_grid(self):
        r = optimize_parameters(_make_prices(30), _simple_strategy, [])
        self.assertEqual(r.total_combinations, 0)

    def test_top_n(self):
        prices = _make_prices(60)
        params = [{"w": i} for i in range(1, 11)]
        r = optimize_parameters(prices, _simple_strategy, params, top_n=3)
        self.assertLessEqual(len(r.all_results), 3)

    def test_best_metric(self):
        prices = _make_prices(60)
        params = [{"w": 5}, {"w": 10}]
        r = optimize_parameters(prices, _simple_strategy, params)
        self.assertIsInstance(r.best_metric, float)


# ═══════════════════════════════════════════════════════════════════════════
#  6. Strategy Robustness Testing
# ═══════════════════════════════════════════════════════════════════════════


class TestRobustness(unittest.TestCase):
    def test_basic_robustness(self):
        prices = _make_prices(60)
        r = evaluate_strategy_robustness(
            prices, _simple_strategy, {"w": 10},
            num_perturbations=20, seed=42,
        )
        self.assertIsInstance(r, RobustnessResult)

    def test_empty_prices(self):
        r = evaluate_strategy_robustness([], _simple_strategy, {"w": 10})
        self.assertFalse(r.is_robust)

    def test_robust_strategy(self):
        prices = _make_prices(60, trend=1.0)
        r = evaluate_strategy_robustness(
            prices, _simple_strategy, {"w": 10},
            num_perturbations=20, seed=42, max_degradation=0.9,
        )
        self.assertIsInstance(r.is_robust, bool)

    def test_sensitivity_keys(self):
        prices = _make_prices(60)
        r = evaluate_strategy_robustness(
            prices, _simple_strategy, {"w": 10, "k": 5},
            num_perturbations=10, seed=42,
        )
        self.assertIn("w", r.param_sensitivity)
        self.assertIn("k", r.param_sensitivity)


# ═══════════════════════════════════════════════════════════════════════════
#  7. Stress Testing
# ═══════════════════════════════════════════════════════════════════════════


class TestStressTesting(unittest.TestCase):
    def test_basic_stress(self):
        prices = _make_prices(60)
        r = run_stress_test(prices, _simple_strategy)
        self.assertIsInstance(r, StressTestResult)
        self.assertGreater(r.scenarios_total, 0)

    def test_empty_prices(self):
        r = run_stress_test([], _simple_strategy)
        self.assertEqual(r.scenarios_total, 0)

    def test_crash_scenarios(self):
        prices = _make_prices(60)
        r = run_stress_test(prices, _simple_strategy, crash_pcts=(-0.20, -0.40))
        crash_names = [s.name for s in r.scenarios if "crash" in s.name]
        self.assertEqual(len(crash_names), 2)

    def test_resilience_flag(self):
        prices = _make_prices(60, trend=1.0)
        r = run_stress_test(prices, _simple_strategy, max_acceptable_loss=-0.99)
        self.assertIsInstance(r.is_resilient, bool)

    def test_worst_return(self):
        prices = _make_prices(60)
        r = run_stress_test(prices, _simple_strategy)
        self.assertIsInstance(r.worst_return, float)


# ═══════════════════════════════════════════════════════════════════════════
#  8. Slippage Simulation
# ═══════════════════════════════════════════════════════════════════════════


class TestSlippageSimulation(unittest.TestCase):
    def test_basic_slippage(self):
        trades = _make_trades(5)
        r = simulate_slippage(trades)
        self.assertIsInstance(r, SlippageResult)
        self.assertGreater(r.total_slippage, 0)

    def test_empty_trades(self):
        r = simulate_slippage([])
        self.assertEqual(r.total_slippage, 0.0)

    def test_custom_model(self):
        trades = _make_trades(5)
        model = SlippageModel(fixed_bps=10.0, spread_bps=20.0)
        r = simulate_slippage(trades, model=model)
        self.assertGreater(r.total_slippage, 0)

    def test_volume_impact(self):
        trades = _make_trades(3)
        volumes = [50000.0, 100000.0, 200000.0]
        model = SlippageModel(volume_impact_factor=0.5)
        r = simulate_slippage(trades, volumes=volumes, model=model)
        self.assertGreater(r.total_slippage, 0)

    def test_return_after_slippage(self):
        trades = _make_trades(5)
        r = simulate_slippage(trades)
        self.assertLess(r.net_return_after_slippage, r.net_return_before_slippage)


# ═══════════════════════════════════════════════════════════════════════════
#  9. Transaction Cost Modeling
# ═══════════════════════════════════════════════════════════════════════════


class TestTransactionCosts(unittest.TestCase):
    def test_basic_costs(self):
        trades = _make_trades(5)
        r = model_transaction_costs(trades)
        self.assertIsInstance(r, TransactionCostResult)
        self.assertGreater(r.total_fees, 0)

    def test_empty_trades(self):
        r = model_transaction_costs([])
        self.assertEqual(r.total_cost, 0.0)

    def test_custom_fees(self):
        trades = _make_trades(3)
        model = TransactionCostModel(maker_fee_pct=0.05, taker_fee_pct=0.1)
        r = model_transaction_costs(trades, cost_model=model)
        self.assertGreater(r.total_fees, 0)

    def test_tax(self):
        trades = _make_trades(3)
        model = TransactionCostModel(tax_pct=10.0)
        r = model_transaction_costs(trades, cost_model=model)
        self.assertGreater(r.total_tax, 0)

    def test_fee_breakdown(self):
        trades = _make_trades(3)
        r = model_transaction_costs(trades)
        self.assertIn("maker_fees", r.fee_breakdown)
        self.assertIn("taker_fees", r.fee_breakdown)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Liquidity Simulation
# ═══════════════════════════════════════════════════════════════════════════


class TestLiquiditySimulation(unittest.TestCase):
    def test_basic_liquidity(self):
        trades = _make_trades(5)
        volumes = [100000.0] * 50
        r = simulate_liquidity(trades, volumes)
        self.assertIsInstance(r, LiquiditySimResult)
        self.assertGreater(r.fill_rate, 0)

    def test_empty_input(self):
        r = simulate_liquidity([], [])
        self.assertEqual(r.total_orders, 0)

    def test_low_liquidity(self):
        trades = _make_trades(5)
        volumes = [10.0] * 50  # Very low
        r = simulate_liquidity(trades, volumes, min_volume_threshold=100.0)
        self.assertEqual(r.rejected_orders, 5)

    def test_partial_fills(self):
        trades = _make_trades(3)
        volumes = [500.0] * 50  # Moderate
        r = simulate_liquidity(trades, volumes, max_participation=0.01)
        self.assertIsInstance(r, LiquiditySimResult)

    def test_liquidity_profile(self):
        trades = _make_trades(3)
        volumes = [10000.0 + i * 100 for i in range(50)]
        r = simulate_liquidity(trades, volumes)
        self.assertIsNotNone(r.liquidity_profile)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Out-of-Sample Testing
# ═══════════════════════════════════════════════════════════════════════════


class TestOutOfSample(unittest.TestCase):
    def test_basic_oos(self):
        prices = _make_prices(60)
        r = run_out_of_sample_test(prices, _simple_strategy)
        self.assertIsInstance(r, OutOfSampleResult)
        self.assertGreater(r.split_index, 0)

    def test_short_prices(self):
        r = run_out_of_sample_test([100, 101], _simple_strategy)
        self.assertEqual(r.split_index, 0)

    def test_overfit_detection(self):
        prices = _make_prices(60)
        r = run_out_of_sample_test(prices, _simple_strategy, max_degradation=0.0001)
        self.assertIsInstance(r.is_overfit, bool)

    def test_custom_split(self):
        prices = _make_prices(100)
        r = run_out_of_sample_test(prices, _simple_strategy, split_ratio=0.8)
        self.assertEqual(r.split_index, 80)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Forward Testing
# ═══════════════════════════════════════════════════════════════════════════


class TestForwardTesting(unittest.TestCase):
    def test_basic_forward(self):
        prices = _make_prices(100)
        r = run_forward_test(prices, _bar_signal_func, lookback=10)
        self.assertIsInstance(r, ForwardTestResult)
        self.assertGreater(len(r.equity_curve), 1)

    def test_short_data(self):
        r = run_forward_test([100, 101, 102], _bar_signal_func, lookback=10)
        self.assertEqual(r.total_trades, 0)

    def test_execution_delay(self):
        prices = _make_prices(100)
        r1 = run_forward_test(prices, _bar_signal_func, lookback=10, execution_delay=1)
        r2 = run_forward_test(prices, _bar_signal_func, lookback=10, execution_delay=3)
        self.assertIsInstance(r1, ForwardTestResult)
        self.assertIsInstance(r2, ForwardTestResult)

    def test_execution_rate(self):
        prices = _make_prices(100)
        r = run_forward_test(prices, _bar_signal_func, lookback=10)
        self.assertGreaterEqual(r.execution_rate, 0.0)
        self.assertLessEqual(r.execution_rate, 1.0)


# ═══════════════════════════════════════════════════════════════════════════
# 13. Paper Trading Mode
# ═══════════════════════════════════════════════════════════════════════════


class TestPaperTrading(unittest.TestCase):
    def test_basic_buy(self):
        engine = PaperTradingEngine(initial_capital=10000)
        order = engine.submit_order("BTC", "buy", 50000.0, 0.1)
        self.assertIsInstance(order, PaperOrder)
        self.assertTrue(order.filled)

    def test_insufficient_funds(self):
        engine = PaperTradingEngine(initial_capital=100)
        order = engine.submit_order("BTC", "buy", 50000.0, 1.0)
        self.assertFalse(order.filled)

    def test_sell_position(self):
        engine = PaperTradingEngine(initial_capital=100000)
        engine.submit_order("BTC", "buy", 50000.0, 0.5)
        order = engine.submit_order("BTC", "sell", 51000.0, 0.5)
        self.assertTrue(order.filled)

    def test_sell_without_position(self):
        engine = PaperTradingEngine(initial_capital=10000)
        order = engine.submit_order("BTC", "sell", 50000.0, 1.0)
        self.assertFalse(order.filled)

    def test_portfolio_value(self):
        engine = PaperTradingEngine(initial_capital=100000)
        engine.submit_order("BTC", "buy", 50000.0, 1.0)
        val = engine.get_portfolio_value({"BTC": 55000.0})
        self.assertGreater(val, 0)

    def test_equity_tracking(self):
        engine = PaperTradingEngine(initial_capital=10000)
        engine.submit_order("ETH", "buy", 3000.0, 1.0)
        engine.update_equity({"ETH": 3100.0})
        engine.update_equity({"ETH": 3200.0})
        self.assertGreater(len(engine.state.equity_curve), 1)

    def test_reset(self):
        engine = PaperTradingEngine(initial_capital=10000)
        engine.submit_order("BTC", "buy", 5000.0, 1.0)
        engine.reset()
        self.assertAlmostEqual(engine.state.cash, 10000.0, places=1)
        self.assertEqual(len(engine.state.orders), 0)

    def test_fees_tracked(self):
        engine = PaperTradingEngine(initial_capital=100000, fee_pct=0.1)
        engine.submit_order("BTC", "buy", 50000.0, 1.0)
        self.assertGreater(engine.state.total_fees, 0)

    def test_slippage_applied(self):
        engine = PaperTradingEngine(slippage_bps=10.0)
        order = engine.submit_order("BTC", "buy", 50000.0, 0.01)
        self.assertGreater(order.fill_price, 50000.0)

    def test_get_state(self):
        engine = PaperTradingEngine(initial_capital=10000)
        state = engine.get_state()
        self.assertIsInstance(state, PaperTradingState)
        self.assertAlmostEqual(state.cash, 10000.0)


if __name__ == "__main__":
    unittest.main()
