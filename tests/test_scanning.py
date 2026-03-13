"""Tests for bot/scanning.py – 10 market scanning categories."""

import unittest
import math

from bot.scanning import (
    # 1. Multi-market scanning
    MultiMarketScanResult,
    scan_multiple_markets,
    # 2. Liquidity filtering
    LiquidityFilterResult,
    filter_by_liquidity,
    # 3. Volume filtering
    VolumeFilterResult,
    filter_by_volume,
    # 4. Volatility filtering
    VolatilityFilterResult,
    filter_by_volatility,
    # 5. Spread filtering
    SpreadFilterResult,
    filter_by_spread,
    # 6. Momentum scanning
    MomentumScanResult,
    scan_momentum,
    # 7. Breakout scanning
    BreakoutScanResult,
    scan_breakouts,
    # 8. Arbitrage opportunity scanning
    ArbitrageScanResult,
    scan_arbitrage,
    # 9. Trend scanning
    TrendScanResult,
    scan_trends,
    # 10. Custom signal scanning
    CustomSignalResult,
    scan_custom_signals,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_prices(n=60, start=100.0, trend=0.5):
    """Generate a series of *n* prices with a linear *trend*."""
    return [start + trend * i for i in range(n)]


def _make_depth(bid_prices=None, ask_prices=None, vol=1.0):
    """Generate a depth dict with given bid/ask prices and volume."""
    bids = bid_prices or [100, 99, 98, 97, 96]
    asks = ask_prices or [101, 102, 103, 104, 105]
    return {
        "buy": [[str(p), str(vol)] for p in bids],
        "sell": [[str(p), str(vol)] for p in asks],
    }


def _make_volumes(n=30, base=500.0):
    """Generate a volume series of *n* values around *base*."""
    return [base + 10.0 * math.sin(i * 0.3) for i in range(n)]


# ---------------------------------------------------------------------------
# 1. Multi-market scanning
# ---------------------------------------------------------------------------

class TestMultiMarketScanning(unittest.TestCase):
    def test_basic_scan_returns_results(self):
        data = {"BTC": _make_prices(30, 100.0, 5.0)}
        results = scan_multiple_markets(data, min_score=0.0)
        self.assertIsInstance(results, list)
        self.assertTrue(all(isinstance(r, MultiMarketScanResult) for r in results))

    def test_empty_market_data(self):
        results = scan_multiple_markets({})
        self.assertEqual(results, [])

    def test_single_price_skipped(self):
        results = scan_multiple_markets({"X": [100.0]}, min_score=0.0)
        self.assertEqual(results, [])

    def test_min_score_filters_low(self):
        # Flat prices → small score → filtered by high min_score
        data = {"FLAT": [100.0] * 30}
        results = scan_multiple_markets(data, min_score=10.0)
        self.assertEqual(results, [])

    def test_buy_signal_on_strong_uptrend(self):
        data = {"UP": _make_prices(30, 100.0, 20.0)}
        results = scan_multiple_markets(data, min_score=0.0)
        self.assertTrue(len(results) >= 1)
        up = [r for r in results if r.market == "UP"]
        self.assertTrue(len(up) == 1)
        self.assertEqual(up[0].signal, "buy")
        self.assertIn("trend", up[0].metrics)


# ---------------------------------------------------------------------------
# 2. Liquidity filtering
# ---------------------------------------------------------------------------

class TestLiquidityFiltering(unittest.TestCase):
    def test_basic_pass(self):
        depth = _make_depth(vol=300.0)
        results = filter_by_liquidity({"BTC": depth}, min_liquidity=100.0)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], LiquidityFilterResult)
        self.assertTrue(results[0].passed)

    def test_empty_markets(self):
        results = filter_by_liquidity({})
        self.assertEqual(results, [])

    def test_low_liquidity_fails(self):
        depth = _make_depth(vol=1.0)
        results = filter_by_liquidity({"X": depth}, min_liquidity=99999.0)
        self.assertFalse(results[0].passed)

    def test_wide_spread_fails(self):
        depth = _make_depth(bid_prices=[100], ask_prices=[200], vol=1000.0)
        results = filter_by_liquidity({"X": depth}, min_liquidity=0.0, max_spread_pct=1.0)
        self.assertFalse(results[0].passed)

    def test_total_liquidity_calculation(self):
        depth = _make_depth(bid_prices=[100, 99], ask_prices=[101, 102], vol=10.0)
        results = filter_by_liquidity({"M": depth}, min_liquidity=0.0)
        r = results[0]
        self.assertAlmostEqual(r.bid_volume, 20.0)
        self.assertAlmostEqual(r.ask_volume, 20.0)
        self.assertAlmostEqual(r.total_liquidity, 40.0)


# ---------------------------------------------------------------------------
# 3. Volume filtering
# ---------------------------------------------------------------------------

class TestVolumeFiltering(unittest.TestCase):
    def test_basic_pass(self):
        vols = _make_volumes(30, 500.0)
        results = filter_by_volume({"BTC": vols}, min_avg_volume=100.0)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], VolumeFilterResult)
        self.assertTrue(results[0].passed)

    def test_empty_market_volumes(self):
        results = filter_by_volume({})
        self.assertEqual(results, [])

    def test_empty_volume_list_skipped(self):
        results = filter_by_volume({"X": []})
        self.assertEqual(results, [])

    def test_low_avg_volume_fails(self):
        results = filter_by_volume({"X": [1.0, 2.0, 3.0]}, min_avg_volume=1000.0)
        self.assertFalse(results[0].passed)

    def test_volume_ratio_computation(self):
        vols = [100.0, 100.0, 100.0, 200.0]
        results = filter_by_volume({"M": vols}, min_avg_volume=0.0, min_volume_ratio=0.0)
        r = results[0]
        self.assertAlmostEqual(r.current_volume, 200.0)
        self.assertGreater(r.volume_ratio, 1.0)


# ---------------------------------------------------------------------------
# 4. Volatility filtering
# ---------------------------------------------------------------------------

class TestVolatilityFiltering(unittest.TestCase):
    def test_basic_pass(self):
        prices = _make_prices(30, 100.0, 0.1)
        results = filter_by_volatility({"BTC": prices}, min_vol=0.0, max_vol=1.0)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], VolatilityFilterResult)
        self.assertTrue(results[0].passed)

    def test_empty_data(self):
        results = filter_by_volatility({})
        self.assertEqual(results, [])

    def test_too_few_prices_skipped(self):
        results = filter_by_volatility({"X": [100.0]}, min_vol=0.0, max_vol=1.0)
        self.assertEqual(results, [])

    def test_high_vol_outside_range_fails(self):
        # Wild swings produce high volatility
        prices = [100.0 + 50.0 * ((-1) ** i) for i in range(30)]
        results = filter_by_volatility({"X": prices}, min_vol=0.0, max_vol=0.00001)
        if results:
            self.assertFalse(results[0].passed)

    def test_annualized_vol_positive(self):
        prices = _make_prices(30, 100.0, 0.5)
        results = filter_by_volatility({"M": prices}, min_vol=0.0, max_vol=10.0)
        self.assertGreater(results[0].annualized_vol, 0.0)


# ---------------------------------------------------------------------------
# 5. Spread filtering
# ---------------------------------------------------------------------------

class TestSpreadFiltering(unittest.TestCase):
    def test_basic_tight_spread_passes(self):
        depth = _make_depth(bid_prices=[100], ask_prices=[100.5], vol=10.0)
        results = filter_by_spread({"BTC": depth}, max_spread_pct=1.0)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SpreadFilterResult)
        self.assertTrue(results[0].passed)

    def test_empty_markets(self):
        results = filter_by_spread({})
        self.assertEqual(results, [])

    def test_wide_spread_fails(self):
        depth = _make_depth(bid_prices=[100], ask_prices=[200], vol=10.0)
        results = filter_by_spread({"X": depth}, max_spread_pct=1.0)
        self.assertFalse(results[0].passed)

    def test_spread_calculation(self):
        depth = _make_depth(bid_prices=[100], ask_prices=[102], vol=10.0)
        results = filter_by_spread({"M": depth}, max_spread_pct=100.0)
        r = results[0]
        self.assertAlmostEqual(r.bid_price, 100.0)
        self.assertAlmostEqual(r.ask_price, 102.0)
        self.assertAlmostEqual(r.spread, 2.0)
        self.assertAlmostEqual(r.spread_pct, 2.0)

    def test_empty_depth_produces_result(self):
        depth = {"buy": [], "sell": []}
        results = filter_by_spread({"M": depth}, max_spread_pct=1.0)
        self.assertEqual(len(results), 1)
        self.assertAlmostEqual(results[0].spread_pct, 0.0)


# ---------------------------------------------------------------------------
# 6. Momentum scanning
# ---------------------------------------------------------------------------

class TestMomentumScanning(unittest.TestCase):
    def test_bullish_momentum(self):
        prices = _make_prices(20, 100.0, 5.0)
        results = scan_momentum({"UP": prices}, lookback=10)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], MomentumScanResult)
        self.assertEqual(results[0].signal, "bullish")
        self.assertGreater(results[0].roc, 0)

    def test_bearish_momentum(self):
        prices = _make_prices(20, 200.0, -5.0)
        results = scan_momentum({"DOWN": prices}, lookback=10)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].signal, "bearish")

    def test_empty_data(self):
        results = scan_momentum({})
        self.assertEqual(results, [])

    def test_too_few_prices_skipped(self):
        results = scan_momentum({"X": [100.0, 101.0]}, lookback=10)
        self.assertEqual(results, [])

    def test_strength_is_abs_roc(self):
        prices = _make_prices(20, 100.0, 5.0)
        results = scan_momentum({"M": prices}, lookback=10)
        r = results[0]
        self.assertAlmostEqual(r.strength, abs(r.roc))


# ---------------------------------------------------------------------------
# 7. Breakout scanning
# ---------------------------------------------------------------------------

class TestBreakoutScanning(unittest.TestCase):
    def test_resistance_breakout(self):
        # 20 prices at 100, then current jumps to 200
        prices = [100.0] * 20 + [200.0]
        results = scan_breakouts({"BTC": prices}, lookback=20)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], BreakoutScanResult)
        self.assertEqual(results[0].breakout_type, "resistance_break")

    def test_support_break(self):
        prices = [100.0] * 20 + [50.0]
        results = scan_breakouts({"BTC": prices}, lookback=20)
        self.assertEqual(results[0].breakout_type, "support_break")

    def test_no_breakout(self):
        prices = [100.0] * 21
        results = scan_breakouts({"BTC": prices}, lookback=20)
        self.assertEqual(results[0].breakout_type, "none")

    def test_empty_data(self):
        results = scan_breakouts({})
        self.assertEqual(results, [])

    def test_too_few_prices_skipped(self):
        results = scan_breakouts({"X": [100.0] * 5}, lookback=20)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 8. Arbitrage opportunity scanning
# ---------------------------------------------------------------------------

class TestArbitrageScanning(unittest.TestCase):
    def test_actionable_opportunity(self):
        prices = {"A": 100.0, "B": 110.0}
        results = scan_arbitrage(prices, fee_pct=0.1)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], ArbitrageScanResult)
        self.assertTrue(results[0].actionable)

    def test_no_opportunity_with_high_fees(self):
        prices = {"A": 100.0, "B": 100.1}
        results = scan_arbitrage(prices, fee_pct=5.0)
        self.assertFalse(results[0].actionable)

    def test_empty_prices(self):
        results = scan_arbitrage({})
        self.assertEqual(results, [])

    def test_single_market_no_pairs(self):
        results = scan_arbitrage({"A": 100.0})
        self.assertEqual(results, [])

    def test_spread_pct_and_profit(self):
        prices = {"A": 100.0, "B": 105.0}
        results = scan_arbitrage(prices, fee_pct=0.0)
        r = results[0]
        self.assertAlmostEqual(r.spread_pct, 5.0)
        self.assertAlmostEqual(r.potential_profit, 5.0)


# ---------------------------------------------------------------------------
# 9. Trend scanning
# ---------------------------------------------------------------------------

class TestTrendScanning(unittest.TestCase):
    def test_uptrend_detected(self):
        prices = _make_prices(30, 100.0, 2.0)
        results = scan_trends({"UP": prices}, fast_period=5, slow_period=20)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], TrendScanResult)
        self.assertEqual(results[0].trend, "uptrend")

    def test_downtrend_detected(self):
        prices = _make_prices(30, 200.0, -2.0)
        results = scan_trends({"DOWN": prices}, fast_period=5, slow_period=20)
        self.assertEqual(results[0].trend, "downtrend")

    def test_sideways_market(self):
        prices = [100.0] * 30
        results = scan_trends({"FLAT": prices}, fast_period=5, slow_period=20)
        self.assertEqual(results[0].trend, "sideways")

    def test_empty_data(self):
        results = scan_trends({})
        self.assertEqual(results, [])

    def test_too_few_prices_skipped(self):
        results = scan_trends({"X": [100.0] * 5}, fast_period=5, slow_period=20)
        self.assertEqual(results, [])


# ---------------------------------------------------------------------------
# 10. Custom signal scanning
# ---------------------------------------------------------------------------

class TestCustomSignalScanning(unittest.TestCase):
    def test_basic_triggered_signal(self):
        def above_mean(prices):
            avg = sum(prices) / len(prices)
            return (prices[-1] > avg, prices[-1])

        data = {"BTC": _make_prices(20, 100.0, 1.0)}
        signals = [{"name": "above_mean", "condition": above_mean}]
        results = scan_custom_signals(data, signals)
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], CustomSignalResult)
        self.assertTrue(results[0].triggered)

    def test_signal_not_triggered(self):
        def always_false(prices):
            return (False, 0.0)

        data = {"BTC": [100.0, 101.0]}
        signals = [{"name": "never", "condition": always_false}]
        results = scan_custom_signals(data, signals)
        self.assertFalse(results[0].triggered)

    def test_empty_market_data(self):
        results = scan_custom_signals({}, [{"name": "x", "condition": lambda p: (True, 0)}])
        self.assertEqual(results, [])

    def test_empty_prices_skipped(self):
        results = scan_custom_signals({"X": []}, [{"name": "x", "condition": lambda p: (True, 0)}])
        self.assertEqual(results, [])

    def test_faulty_condition_skipped(self):
        def broken(prices):
            raise ValueError("boom")

        data = {"BTC": [100.0]}
        signals = [{"name": "bad", "condition": broken}]
        results = scan_custom_signals(data, signals)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
