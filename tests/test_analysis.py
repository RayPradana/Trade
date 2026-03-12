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


class MultiTimeframeTest(unittest.TestCase):
    def _make_candles(self, direction: str, n: int = 60) -> list:
        from bot.analysis import Candle
        candles = []
        price = 1000.0
        for i in range(n):
            if direction == "up":
                price *= 1.001
            elif direction == "down":
                price *= 0.999
            candles.append(Candle(timestamp=i, open=price, high=price * 1.001, low=price * 0.999, close=price, volume=100.0))
        return candles

    def test_aligned_up_returns_up(self):
        from bot.analysis import multi_timeframe_confirm
        candles = self._make_candles("up")
        result = multi_timeframe_confirm({"1m": candles, "15m": candles, "60m": candles})
        self.assertEqual(result.direction, "up")
        self.assertTrue(result.aligned)

    def test_aligned_down_returns_down(self):
        from bot.analysis import multi_timeframe_confirm
        candles = self._make_candles("down")
        result = multi_timeframe_confirm({"1m": candles, "15m": candles})
        self.assertEqual(result.direction, "down")
        self.assertTrue(result.aligned)

    def test_mixed_returns_not_aligned(self):
        from bot.analysis import multi_timeframe_confirm
        up = self._make_candles("up")
        down = self._make_candles("down")
        result = multi_timeframe_confirm({"1m": up, "15m": down})
        self.assertFalse(result.aligned)

    def test_empty_input_returns_flat(self):
        from bot.analysis import multi_timeframe_confirm
        result = multi_timeframe_confirm({})
        self.assertEqual(result.direction, "flat")
        self.assertFalse(result.aligned)

    def test_tf_directions_populated(self):
        from bot.analysis import multi_timeframe_confirm
        candles = self._make_candles("up")
        result = multi_timeframe_confirm({"1m": candles, "15m": candles})
        self.assertIn("1m", result.tf_directions)
        self.assertIn("15m", result.tf_directions)


class WhaleDetectionTest(unittest.TestCase):
    def test_detects_large_bid_wall(self):
        from bot.analysis import detect_whale_activity
        # Create bids where one level has 10x average volume
        levels = [[str(100 - i), "1.0"] for i in range(19)]  # 19 small levels
        levels.insert(0, ["101", "100.0"])  # one huge level
        depth = {"buy": levels, "sell": levels}
        result = detect_whale_activity(depth)
        self.assertTrue(result.detected)
        self.assertEqual(result.side, "bid")
        self.assertGreater(result.ratio, 5.0)

    def test_no_whale_on_flat_book(self):
        from bot.analysis import detect_whale_activity
        levels = [[str(100 - i), "1.0"] for i in range(20)]
        depth = {"buy": levels, "sell": levels}
        result = detect_whale_activity(depth)
        self.assertFalse(result.detected)
        self.assertIsNone(result.side)

    def test_empty_book_no_whale(self):
        from bot.analysis import detect_whale_activity
        result = detect_whale_activity({"buy": [], "sell": []})
        self.assertFalse(result.detected)


class SpoofingDetectionTest(unittest.TestCase):
    def _make_flat_book(self, price_start: float = 100.0, levels: int = 20) -> list:
        """Uniform-volume order book — no spoofing signal."""
        return [[str(price_start - i), "1.0"] for i in range(levels)]

    def test_no_spoof_on_flat_book(self):
        from bot.analysis import detect_spoofing
        levels = self._make_flat_book()
        depth = {"buy": levels, "sell": levels}
        result = detect_spoofing(depth)
        self.assertFalse(result.detected)
        self.assertIsNone(result.side)

    def test_detects_distant_large_bid_wall(self):
        from bot.analysis import detect_spoofing
        # Top bid at 100, but a huge spoof at 90 (10% away, 10× volume)
        levels = self._make_flat_book()
        levels.append(["90", "100.0"])   # far + large
        depth = {"buy": levels, "sell": self._make_flat_book(110.0)}
        result = detect_spoofing(depth, min_distance_pct=0.03)
        self.assertTrue(result.detected)
        self.assertEqual(result.side, "bid")
        self.assertGreater(result.distance_pct, 0.05)

    def test_detects_distant_large_ask_wall(self):
        from bot.analysis import detect_spoofing
        bid_levels = self._make_flat_book(100.0)
        ask_levels = self._make_flat_book(105.0)
        ask_levels.append(["150", "100.0"])   # far ask + huge
        depth = {"buy": bid_levels, "sell": ask_levels}
        result = detect_spoofing(depth, min_distance_pct=0.03)
        self.assertTrue(result.detected)
        self.assertEqual(result.side, "ask")

    def test_nearby_large_wall_not_spoof(self):
        from bot.analysis import detect_spoofing
        # Large wall only 0.5% away → below 3% distance threshold → not a spoof
        levels = self._make_flat_book(100.0)
        levels.insert(2, ["99.5", "100.0"])   # large but near top of book
        depth = {"buy": levels, "sell": self._make_flat_book(105.0)}
        result = detect_spoofing(depth, min_distance_pct=0.03)
        self.assertFalse(result.detected)

    def test_empty_book_no_spoof(self):
        from bot.analysis import detect_spoofing
        result = detect_spoofing({"buy": [], "sell": []})
        self.assertFalse(result.detected)


class SmartEntryEngineTests(unittest.TestCase):
    """Tests for detect_pre_pump_signal, detect_whale_pressure,
    detect_fake_breakout, and smart_entry_filter."""

    def _make_candles(self, volumes, close_start=100.0):
        """Build a minimal Candle list with the given volume series."""
        from bot.analysis import Candle
        return [
            Candle(timestamp=i, open=close_start + i, high=close_start + i + 1,
                   low=close_start + i - 1, close=close_start + i, volume=v)
            for i, v in enumerate(volumes)
        ]

    def _flat_book(self, base_price, n=5, vol=10.0):
        return [[str(base_price - i * 0.1), str(vol)] for i in range(n)]

    # ── Pre-pump detection ──────────────────────────────────────────────────

    def test_pre_pump_detected_on_volume_surge(self):
        from bot.analysis import detect_pre_pump_signal
        # baseline low volume, last 3 candles surge
        vols = [1.0] * 10 + [5.0, 5.0, 5.0]
        candles = self._make_candles(vols)
        result = detect_pre_pump_signal(candles, volume_surge_ratio=2.0)
        self.assertTrue(result.detected)
        self.assertGreater(result.volume_surge_ratio, 2.0)
        self.assertGreater(result.score, 0.0)

    def test_pre_pump_not_detected_on_flat_volume(self):
        from bot.analysis import detect_pre_pump_signal
        vols = [2.0] * 13
        candles = self._make_candles(vols)
        result = detect_pre_pump_signal(candles, volume_surge_ratio=2.0)
        self.assertFalse(result.detected)
        self.assertAlmostEqual(result.volume_surge_ratio, 1.0, places=1)

    def test_pre_pump_insufficient_candles(self):
        from bot.analysis import detect_pre_pump_signal
        # Only 4 candles available — not enough for recent_n=3 + 2 baseline
        candles = self._make_candles([1.0, 2.0, 3.0, 4.0])
        result = detect_pre_pump_signal(candles, volume_surge_ratio=2.0)
        self.assertFalse(result.detected)
        self.assertEqual(result.volume_surge_ratio, 0.0)

    def test_pre_pump_score_saturates_at_one(self):
        from bot.analysis import detect_pre_pump_signal
        # Very large volume surge should not produce score > 1.0
        vols = [1.0] * 10 + [1000.0, 1000.0, 1000.0]
        candles = self._make_candles(vols)
        result = detect_pre_pump_signal(candles, volume_surge_ratio=2.0)
        self.assertLessEqual(result.score, 1.0)

    # ── Whale pressure ─────────────────────────────────────────────────────

    def test_whale_pressure_detected_buy_side(self):
        from bot.analysis import detect_whale_pressure
        # Large bid wall → net buying pressure
        bids = self._flat_book(100.0, n=5)
        bids.append(["99.0", "500.0"])   # massive bid level
        asks = self._flat_book(101.0, n=5)
        depth = {"buy": bids, "sell": asks}
        result = detect_whale_pressure(depth, pressure_threshold=2.0)
        self.assertTrue(result.detected)
        self.assertEqual(result.side, "buy")
        self.assertGreater(result.pressure, 0)

    def test_whale_pressure_detected_sell_side(self):
        from bot.analysis import detect_whale_pressure
        bids = self._flat_book(100.0, n=5)
        asks = self._flat_book(101.0, n=5)
        asks.append(["110.0", "500.0"])   # massive ask wall
        depth = {"buy": bids, "sell": asks}
        result = detect_whale_pressure(depth, pressure_threshold=2.0)
        self.assertTrue(result.detected)
        self.assertEqual(result.side, "sell")
        self.assertLess(result.pressure, 0)

    def test_whale_pressure_not_detected_flat_book(self):
        from bot.analysis import detect_whale_pressure
        bids = self._flat_book(100.0, n=5)
        asks = self._flat_book(101.0, n=5)
        depth = {"buy": bids, "sell": asks}
        result = detect_whale_pressure(depth, pressure_threshold=2.0)
        self.assertFalse(result.detected)

    def test_whale_pressure_empty_book(self):
        from bot.analysis import detect_whale_pressure
        result = detect_whale_pressure({"buy": [], "sell": []})
        self.assertFalse(result.detected)
        self.assertIsNone(result.side)

    # ── Fake breakout detection ─────────────────────────────────────────────

    def test_fake_breakout_detected_thin_volume(self):
        from bot.analysis import detect_fake_breakout, SupportResistance
        # Current price above resistance, but very thin volume
        vols = [10.0] * 9 + [1.0]   # last candle has 10% of average
        candles = self._make_candles(vols, close_start=100.0)
        levels = SupportResistance(support=90.0, resistance=105.0, lookback=10)
        result = detect_fake_breakout(candles, current_price=110.0, levels=levels)
        self.assertTrue(result.breakout_present)
        self.assertTrue(result.detected)
        self.assertLess(result.volume_ratio, 0.7)
        self.assertGreater(result.score, 0.0)

    def test_fake_breakout_not_detected_with_volume(self):
        from bot.analysis import detect_fake_breakout, SupportResistance
        # Breakout present AND confirmed by volume surge
        vols = [10.0] * 9 + [20.0]   # last candle has 2× average
        candles = self._make_candles(vols, close_start=100.0)
        levels = SupportResistance(support=90.0, resistance=105.0, lookback=10)
        result = detect_fake_breakout(candles, current_price=110.0, levels=levels)
        self.assertTrue(result.breakout_present)
        self.assertFalse(result.detected)   # volume confirms → not fake
        self.assertGreater(result.volume_ratio, 1.0)

    def test_no_breakout_below_resistance(self):
        from bot.analysis import detect_fake_breakout, SupportResistance
        vols = [10.0] * 10
        candles = self._make_candles(vols, close_start=100.0)
        levels = SupportResistance(support=90.0, resistance=115.0, lookback=10)
        result = detect_fake_breakout(candles, current_price=110.0, levels=levels)
        self.assertFalse(result.breakout_present)
        self.assertFalse(result.detected)

    def test_fake_breakout_no_levels(self):
        from bot.analysis import detect_fake_breakout
        vols = [10.0] * 10
        candles = self._make_candles(vols, close_start=100.0)
        result = detect_fake_breakout(candles, current_price=110.0, levels=None)
        self.assertFalse(result.breakout_present)
        self.assertFalse(result.detected)

    # ── smart_entry_filter wrapper ──────────────────────────────────────────

    def test_smart_entry_filter_returns_all_components(self):
        from bot.analysis import smart_entry_filter, SupportResistance
        vols = [10.0] * 10 + [30.0, 30.0, 30.0]
        candles = self._make_candles(vols, close_start=100.0)
        bids = self._flat_book(100.0, n=5)
        bids.append(["99.0", "500.0"])
        depth = {"buy": bids, "sell": self._flat_book(101.0, n=5)}
        levels = SupportResistance(support=90.0, resistance=105.0, lookback=10)
        result = smart_entry_filter(candles, depth, current_price=102.0, levels=levels)
        # All three sub-results should be present
        self.assertIsNotNone(result.pre_pump)
        self.assertIsNotNone(result.whale_pressure)
        self.assertIsNotNone(result.fake_breakout)
        # Volume surge in last 3 candles → pre-pump should be detected
        self.assertTrue(result.pre_pump.detected)
