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

    def test_build_candles_ignores_non_dict_entries(self) -> None:
        trades = [
            {"date": 0, "price": "100", "amount": "1"},
            "bad",  # should be ignored
        ]
        candles = build_candles(trades, interval_seconds=60)
        self.assertEqual(len(candles), 1)

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


class OhlcHelpersTests(unittest.TestCase):
    """Tests for candles_from_ohlc and interval_to_ohlc_tf."""

    def test_candles_from_ohlc_parses_rows(self) -> None:
        from bot.analysis import candles_from_ohlc
        ohlc = [
            {"Time": 1000, "Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0, "Volume": "50"},
            {"Time": 2000, "Open": 105.0, "High": 115.0, "Low": 100.0, "Close": 110.0, "Volume": "60.5"},
        ]
        candles = candles_from_ohlc(ohlc)
        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].timestamp, 1000)
        self.assertAlmostEqual(candles[0].close, 105.0)
        self.assertAlmostEqual(candles[0].volume, 50.0)
        self.assertAlmostEqual(candles[1].volume, 60.5)

    def test_candles_from_ohlc_sorted_oldest_first(self) -> None:
        from bot.analysis import candles_from_ohlc
        ohlc = [
            {"Time": 3000, "Open": 1, "High": 2, "Low": 0, "Close": 1, "Volume": "1"},
            {"Time": 1000, "Open": 1, "High": 2, "Low": 0, "Close": 1, "Volume": "1"},
            {"Time": 2000, "Open": 1, "High": 2, "Low": 0, "Close": 1, "Volume": "1"},
        ]
        candles = candles_from_ohlc(ohlc)
        timestamps = [c.timestamp for c in candles]
        self.assertEqual(timestamps, [1000, 2000, 3000])

    def test_candles_from_ohlc_skips_invalid(self) -> None:
        from bot.analysis import candles_from_ohlc
        ohlc = [
            "not_a_dict",
            {"Time": 1000, "Open": "bad", "High": 2, "Low": 0, "Close": 1, "Volume": "1"},
            None,
            {"Time": 2000, "Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0, "Volume": "10"},
        ]
        candles = candles_from_ohlc(ohlc)  # type: ignore[arg-type]
        # Only the last row is valid
        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0].timestamp, 2000)

    def test_candles_from_ohlc_empty(self) -> None:
        from bot.analysis import candles_from_ohlc
        self.assertEqual(candles_from_ohlc([]), [])

    def test_interval_to_ohlc_tf(self) -> None:
        from bot.analysis import interval_to_ohlc_tf
        self.assertEqual(interval_to_ohlc_tf(60), "1")
        self.assertEqual(interval_to_ohlc_tf(300), "15")   # 5-min → 15-min tf
        self.assertEqual(interval_to_ohlc_tf(900), "15")
        self.assertEqual(interval_to_ohlc_tf(1800), "30")
        self.assertEqual(interval_to_ohlc_tf(3600), "60")
        self.assertEqual(interval_to_ohlc_tf(14400), "240")
        self.assertEqual(interval_to_ohlc_tf(86400), "1D")


if __name__ == "__main__":
    unittest.main()
