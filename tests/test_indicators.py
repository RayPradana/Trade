"""Tests for bot.indicators — comprehensive technical indicator library."""

import math
import unittest

from bot.analysis import Candle
from bot.indicators import (
    CustomIndicatorRegistry,
    CustomIndicatorResult,
    DonchianChannel,
    FibonacciLevels,
    IchimokuCloud,
    KeltnerChannel,
    MomentumSnapshot,
    PatternResult,
    PivotPoints,
    SRLevel,
    StochasticResult,
    Trendline,
    VolatilitySnapshot,
    VolumeProfile,
    VolumeSnapshot,
    compute_atr,
    compute_donchian,
    compute_fibonacci,
    compute_ichimoku,
    compute_keltner,
    compute_momentum_snapshot,
    compute_pivot_points,
    compute_stochastic,
    compute_volatility_snapshot,
    compute_volume_profile,
    compute_volume_snapshot,
    compute_vwap,
    compute_wma,
    detect_patterns,
    detect_support_resistance,
    detect_trendline,
)


def _make_candles(
    prices: list[float],
    *,
    spread: float = 1.0,
    volume: float = 10.0,
) -> list[Candle]:
    """Helper: build candles from close prices with a fixed high-low spread."""
    return [
        Candle(
            timestamp=i,
            open=p,
            high=p + spread,
            low=p - spread,
            close=p,
            volume=volume,
        )
        for i, p in enumerate(prices)
    ]


# ───────────────────────────────────────────────────────────────────────
#  1. WMA
# ───────────────────────────────────────────────────────────────────────


class WMATests(unittest.TestCase):
    def test_basic_wma(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = compute_wma(values, period=3)
        # First 2 values should be NaN
        self.assertTrue(math.isnan(result[0]))
        self.assertTrue(math.isnan(result[1]))
        # WMA(3) at index 2: (1*1 + 2*2 + 3*3) / 6 = 14/6 ≈ 2.333
        self.assertAlmostEqual(result[2], 14 / 6, places=4)
        self.assertFalse(math.isnan(result[-1]))

    def test_wma_period_1_equals_values(self) -> None:
        values = [10.0, 20.0, 30.0]
        result = compute_wma(values, period=1)
        for v, r in zip(values, result):
            self.assertAlmostEqual(v, r)

    def test_wma_empty(self) -> None:
        self.assertEqual(compute_wma([], period=5), [])


# ───────────────────────────────────────────────────────────────────────
#  2. Stochastic Oscillator
# ───────────────────────────────────────────────────────────────────────


class StochasticTests(unittest.TestCase):
    def test_overbought(self) -> None:
        """Close at period high → %K ≈ 100."""
        candles = [
            Candle(i, 100 + i, 100 + i + 1, 100, 100 + i + 1, 1.0)
            for i in range(20)
        ]
        result = compute_stochastic(candles, k_period=14)
        self.assertGreater(result.k, 80)

    def test_oversold(self) -> None:
        """Close at period low → %K ≈ 0."""
        candles = [
            Candle(i, 200 - i, 200, 200 - i - 1, 200 - i - 1, 1.0)
            for i in range(20)
        ]
        result = compute_stochastic(candles, k_period=14)
        self.assertLess(result.k, 20)

    def test_not_enough_candles(self) -> None:
        candles = _make_candles([100, 101])
        result = compute_stochastic(candles, k_period=14)
        self.assertAlmostEqual(result.k, 50.0)
        self.assertAlmostEqual(result.d, 50.0)


# ───────────────────────────────────────────────────────────────────────
#  3. ATR
# ───────────────────────────────────────────────────────────────────────


class ATRTests(unittest.TestCase):
    def test_atr_positive(self) -> None:
        candles = _make_candles(list(range(100, 120)), spread=5.0)
        atr = compute_atr(candles, period=14)
        self.assertGreater(atr, 0)

    def test_atr_flat_prices(self) -> None:
        candles = _make_candles([100.0] * 20, spread=0.0)
        atr = compute_atr(candles, period=14)
        # All TR should be 0 when H=L=C for every candle
        # But H-L=0, |H-prev_close|=0, |L-prev_close|=0 → TR=0
        self.assertAlmostEqual(atr, 0.0)

    def test_atr_single_candle(self) -> None:
        candles = _make_candles([100.0])
        self.assertAlmostEqual(compute_atr(candles), 0.0)


# ───────────────────────────────────────────────────────────────────────
#  4. VWAP
# ───────────────────────────────────────────────────────────────────────


class VWAPTests(unittest.TestCase):
    def test_uniform_volume(self) -> None:
        candles = _make_candles([100, 110, 120], volume=10.0, spread=0.0)
        vwap = compute_vwap(candles)
        # With equal volume, VWAP = mean of typical prices.
        self.assertAlmostEqual(vwap, (100 + 110 + 120) / 3, places=2)

    def test_empty(self) -> None:
        self.assertAlmostEqual(compute_vwap([]), 0.0)


# ───────────────────────────────────────────────────────────────────────
#  5. Volume Profile
# ───────────────────────────────────────────────────────────────────────


class VolumeProfileTests(unittest.TestCase):
    def test_basic_profile(self) -> None:
        candles = _make_candles([100, 105, 110, 100, 105], spread=2.0, volume=10.0)
        vp = compute_volume_profile(candles, num_bins=5)
        self.assertGreater(len(vp.levels), 0)
        self.assertGreater(vp.poc, 0)
        self.assertGreaterEqual(vp.value_area_high, vp.value_area_low)

    def test_empty(self) -> None:
        vp = compute_volume_profile([])
        self.assertEqual(vp.levels, [])
        self.assertAlmostEqual(vp.poc, 0.0)

    def test_single_price(self) -> None:
        candles = _make_candles([100] * 5, spread=0.0)
        vp = compute_volume_profile(candles)
        self.assertAlmostEqual(vp.poc, 100.0)


# ───────────────────────────────────────────────────────────────────────
#  6. Ichimoku Cloud
# ───────────────────────────────────────────────────────────────────────


class IchimokuTests(unittest.TestCase):
    def test_basic_ichimoku(self) -> None:
        candles = _make_candles(list(range(100, 160)), spread=2.0)
        ic = compute_ichimoku(candles)
        self.assertGreater(ic.tenkan_sen, 0)
        self.assertGreater(ic.kijun_sen, 0)
        self.assertAlmostEqual(ic.senkou_span_a, (ic.tenkan_sen + ic.kijun_sen) / 2)
        self.assertAlmostEqual(ic.chikou_span, candles[-1].close)

    def test_empty(self) -> None:
        ic = compute_ichimoku([])
        self.assertAlmostEqual(ic.tenkan_sen, 0.0)


# ───────────────────────────────────────────────────────────────────────
#  7. Donchian Channel
# ───────────────────────────────────────────────────────────────────────


class DonchianTests(unittest.TestCase):
    def test_basic_donchian(self) -> None:
        candles = _make_candles([100, 110, 90, 105], spread=2.0)
        dc = compute_donchian(candles, period=4)
        self.assertAlmostEqual(dc.upper, 112.0)  # max high
        self.assertAlmostEqual(dc.lower, 88.0)   # min low
        self.assertAlmostEqual(dc.mid, (112.0 + 88.0) / 2)

    def test_empty(self) -> None:
        dc = compute_donchian([])
        self.assertAlmostEqual(dc.upper, 0.0)


# ───────────────────────────────────────────────────────────────────────
#  8. Keltner Channel
# ───────────────────────────────────────────────────────────────────────


class KeltnerTests(unittest.TestCase):
    def test_basic_keltner(self) -> None:
        candles = _make_candles(list(range(100, 130)), spread=3.0)
        kc = compute_keltner(candles)
        self.assertGreater(kc.upper, kc.mid)
        self.assertLess(kc.lower, kc.mid)
        self.assertGreater(kc.mid, 0)

    def test_empty(self) -> None:
        kc = compute_keltner([])
        self.assertAlmostEqual(kc.upper, 0.0)


# ───────────────────────────────────────────────────────────────────────
#  9. Fibonacci Retracement
# ───────────────────────────────────────────────────────────────────────


class FibonacciTests(unittest.TestCase):
    def test_fibonacci_levels_order(self) -> None:
        candles = _make_candles(list(range(100, 200)), spread=0.0)
        fib = compute_fibonacci(candles)
        # Levels should be in descending order from high to low.
        self.assertGreaterEqual(fib.level_0, fib.level_236)
        self.assertGreaterEqual(fib.level_236, fib.level_382)
        self.assertGreaterEqual(fib.level_382, fib.level_500)
        self.assertGreaterEqual(fib.level_500, fib.level_618)
        self.assertGreaterEqual(fib.level_618, fib.level_786)
        self.assertGreaterEqual(fib.level_786, fib.level_1)

    def test_level_0_equals_high(self) -> None:
        candles = _make_candles([50, 100, 75], spread=0.0)
        fib = compute_fibonacci(candles)
        self.assertAlmostEqual(fib.level_0, fib.high)
        self.assertAlmostEqual(fib.level_1, fib.low)

    def test_empty(self) -> None:
        fib = compute_fibonacci([])
        self.assertAlmostEqual(fib.high, 0.0)


# ───────────────────────────────────────────────────────────────────────
# 10. Pivot Points
# ───────────────────────────────────────────────────────────────────────


class PivotPointTests(unittest.TestCase):
    def test_basic_pivot(self) -> None:
        candle = Candle(0, 100, 110, 90, 105, 10.0)
        pp = compute_pivot_points([candle])
        expected_p = (110 + 90 + 105) / 3
        self.assertAlmostEqual(pp.pivot, expected_p)
        self.assertGreater(pp.r1, pp.pivot)
        self.assertLess(pp.s1, pp.pivot)
        self.assertGreater(pp.r2, pp.r1)
        self.assertLess(pp.s2, pp.s1)
        self.assertGreater(pp.r3, pp.r2)
        self.assertLess(pp.s3, pp.s2)

    def test_empty(self) -> None:
        pp = compute_pivot_points([])
        self.assertAlmostEqual(pp.pivot, 0.0)


# ───────────────────────────────────────────────────────────────────────
# 11. Trendline Detection
# ───────────────────────────────────────────────────────────────────────


class TrendlineTests(unittest.TestCase):
    def test_uptrend(self) -> None:
        candles = _make_candles(list(range(100, 130)))
        tl = detect_trendline(candles)
        self.assertEqual(tl.direction, "up")
        self.assertGreater(tl.slope, 0)
        self.assertGreater(tl.strength, 0.9)  # Nearly perfect linear trend

    def test_downtrend(self) -> None:
        candles = _make_candles(list(range(130, 100, -1)))
        tl = detect_trendline(candles)
        self.assertEqual(tl.direction, "down")
        self.assertLess(tl.slope, 0)

    def test_flat(self) -> None:
        candles = _make_candles([100.0] * 10)
        tl = detect_trendline(candles)
        self.assertEqual(tl.direction, "flat")
        self.assertAlmostEqual(tl.slope, 0.0)

    def test_too_few_candles(self) -> None:
        candles = _make_candles([100, 110])
        tl = detect_trendline(candles)
        self.assertEqual(tl.direction, "flat")


# ───────────────────────────────────────────────────────────────────────
# 12. Support / Resistance Detection
# ───────────────────────────────────────────────────────────────────────


class SupportResistanceTests(unittest.TestCase):
    def test_detects_levels(self) -> None:
        # Oscillating prices should create detectable S/R levels.
        prices = []
        for _ in range(5):
            prices.extend([100, 102, 104, 106, 108, 110, 108, 106, 104, 102])
        candles = _make_candles(prices, spread=0.5)
        levels = detect_support_resistance(candles, lookback=50, min_touches=2)
        self.assertGreater(len(levels), 0)
        for level in levels:
            self.assertIn(level.kind, ("support", "resistance"))
            self.assertGreater(level.touches, 0)

    def test_too_few_candles(self) -> None:
        candles = _make_candles([100, 101])
        levels = detect_support_resistance(candles)
        self.assertEqual(levels, [])


# ───────────────────────────────────────────────────────────────────────
# 13. Pattern Recognition
# ───────────────────────────────────────────────────────────────────────


class PatternRecognitionTests(unittest.TestCase):
    def test_head_and_shoulders(self) -> None:
        # Build prices: left shoulder, neckline, head (higher), neckline, right shoulder
        prices = (
            [100] * 5 +
            [105, 108, 110, 108, 105] +  # left shoulder
            [100] * 3 +
            [105, 110, 115, 118, 115, 110, 105] +  # head
            [100] * 3 +
            [105, 108, 110, 108, 105] +  # right shoulder
            [100] * 3
        )
        candles = _make_candles(prices, spread=0.5)
        results = detect_patterns(candles, lookback=len(prices))
        # We should detect head and shoulders.
        h_and_s = [r for r in results if r.pattern == "head_and_shoulders"]
        self.assertGreater(len(h_and_s), 0)
        self.assertEqual(h_and_s[0].direction, "bearish")

    def test_no_pattern_in_flat_prices(self) -> None:
        candles = _make_candles([100.0] * 30, spread=0.0)
        results = detect_patterns(candles)
        self.assertEqual(results, [])


# ───────────────────────────────────────────────────────────────────────
# 14. Momentum Indicators
# ───────────────────────────────────────────────────────────────────────


class MomentumSnapshotTests(unittest.TestCase):
    def test_all_fields_populated(self) -> None:
        candles = _make_candles(list(range(100, 150)), spread=2.0)
        snap = compute_momentum_snapshot(candles)
        self.assertTrue(0 <= snap.rsi <= 100)
        self.assertTrue(0 <= snap.stochastic_k <= 100)
        self.assertTrue(0 <= snap.stochastic_d <= 100)
        self.assertNotAlmostEqual(snap.roc, 0.0)
        self.assertTrue(-100 <= snap.williams_r <= 0)

    def test_empty_candles(self) -> None:
        snap = compute_momentum_snapshot([])
        self.assertAlmostEqual(snap.rsi, 50.0)


# ───────────────────────────────────────────────────────────────────────
# 15. Volatility Indicators
# ───────────────────────────────────────────────────────────────────────


class VolatilitySnapshotTests(unittest.TestCase):
    def test_all_fields_populated(self) -> None:
        candles = _make_candles(list(range(100, 150)), spread=3.0)
        snap = compute_volatility_snapshot(candles)
        self.assertGreater(snap.atr, 0)
        self.assertGreater(snap.bb_upper, snap.bb_mid)
        self.assertLess(snap.bb_lower, snap.bb_mid)
        self.assertGreater(snap.keltner_upper, snap.keltner_mid)
        self.assertGreater(snap.historical_vol, 0)

    def test_empty(self) -> None:
        snap = compute_volatility_snapshot([])
        self.assertAlmostEqual(snap.atr, 0.0)


# ───────────────────────────────────────────────────────────────────────
# 16. Volume Indicators
# ───────────────────────────────────────────────────────────────────────


class VolumeSnapshotTests(unittest.TestCase):
    def test_all_fields_populated(self) -> None:
        candles = _make_candles(list(range(100, 150)), spread=2.0, volume=100.0)
        snap = compute_volume_snapshot(candles)
        self.assertGreater(snap.vwap, 0)
        self.assertGreater(snap.volume_sma, 0)
        self.assertNotEqual(snap.obv, 0.0)
        self.assertTrue(0 <= snap.mfi <= 100)
        self.assertTrue(-1 <= snap.cmf <= 1)

    def test_empty(self) -> None:
        snap = compute_volume_snapshot([])
        self.assertAlmostEqual(snap.vwap, 0.0)


# ───────────────────────────────────────────────────────────────────────
# 17. Custom Indicator Framework
# ───────────────────────────────────────────────────────────────────────


class CustomIndicatorTests(unittest.TestCase):
    def test_register_and_compute(self) -> None:
        registry = CustomIndicatorRegistry()

        @registry.register("avg_close")
        def avg_close(candles, **kwargs):
            return sum(c.close for c in candles) / len(candles)

        candles = _make_candles([100, 110, 120])
        result = registry.compute("avg_close", candles)
        self.assertEqual(result.name, "avg_close")
        self.assertAlmostEqual(result.value, 110.0)

    def test_compute_all(self) -> None:
        registry = CustomIndicatorRegistry()
        registry.add("max_high", lambda candles, **kw: max(c.high for c in candles))
        registry.add("min_low", lambda candles, **kw: min(c.low for c in candles))

        candles = _make_candles([90, 100, 110], spread=5.0)
        results = registry.compute_all(candles)
        self.assertEqual(len(results), 2)
        names = {r.name for r in results}
        self.assertIn("max_high", names)
        self.assertIn("min_low", names)

    def test_remove_indicator(self) -> None:
        registry = CustomIndicatorRegistry()
        registry.add("temp", lambda c, **kw: 0)
        self.assertIn("temp", registry.names)
        registry.remove("temp")
        self.assertNotIn("temp", registry.names)

    def test_compute_unregistered_raises(self) -> None:
        registry = CustomIndicatorRegistry()
        with self.assertRaises(KeyError):
            registry.compute("nonexistent", [])

    def test_names_property(self) -> None:
        registry = CustomIndicatorRegistry()
        registry.add("a", lambda c, **kw: 1)
        registry.add("b", lambda c, **kw: 2)
        self.assertEqual(sorted(registry.names), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
