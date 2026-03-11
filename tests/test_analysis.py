import math
import unittest

from bot.analysis import (
    Candle,
    analyze_orderbook,
    analyze_trend,
    analyze_volatility,
    build_candles,
    derive_indicators,
    moving_average,
    support_resistance,
)


class AnalysisTests(unittest.TestCase):
    def test_build_candles_groups_by_interval(self) -> None:
        trades = [
            {"date": 0, "price": "100", "amount": "1"},
            {"date": 50, "price": "110", "amount": "2"},
            {"date": 60, "price": "120", "amount": "1"},
            {"date": 119, "price": "130", "amount": "1"},
        ]
        candles = build_candles(trades, interval_seconds=60)
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].open, 100.0)
        self.assertEqual(candles[0].close, 110.0)
        self.assertAlmostEqual(candles[1].high, 130.0)
        self.assertAlmostEqual(candles[1].volume, 2.0)

    def test_moving_average_and_trend(self) -> None:
        candles = [Candle(i, i, i, i, i, 1.0) for i in range(1, 11)]
        trend = analyze_trend(candles, fast_window=3, slow_window=5)
        self.assertEqual(trend.direction, "up")
        self.assertFalse(math.isnan(trend.fast_ma))
        self.assertGreater(trend.strength, 0)

    def test_orderbook_analysis(self) -> None:
        depth = {"buy": [["100", "2"], ["99", "1"]], "sell": [["101", "1.5"], ["102", "2"]]}
        insight = analyze_orderbook(depth)
        self.assertGreater(insight.bid_volume, 0)
        self.assertLess(insight.spread_pct, 0.02)
        self.assertTrue(-1 <= insight.imbalance <= 1)

    def test_volatility(self) -> None:
        candles = [
            Candle(0, 100, 100, 100, 100, 1),
            Candle(1, 105, 106, 104, 105, 1),
            Candle(2, 95, 96, 94, 95, 1),
        ]
        vol = analyze_volatility(candles)
        self.assertGreater(vol.volatility, 0)
        self.assertAlmostEqual(vol.avg_volume, 1)

    def test_support_resistance(self) -> None:
        candles = [
            Candle(0, 100, 100, 100, 100, 1),
            Candle(1, 110, 111, 109, 110, 1),
            Candle(2, 90, 91, 89, 90, 1),
        ]
        levels = support_resistance(candles, lookback=3)
        self.assertEqual(levels.support, 90)
        self.assertEqual(levels.resistance, 110)

    def test_indicators_include_rsi_and_macd(self) -> None:
        candles = [
            Candle(i, 100 + i, 101 + i, 99 + i, 100 + i, 1.0) for i in range(40)
        ]
        indicators = derive_indicators(candles)
        self.assertTrue(0 <= indicators.rsi <= 100)
        # MACD histogram should be finite
        self.assertFalse(math.isnan(indicators.macd_hist))


if __name__ == "__main__":
    unittest.main()
