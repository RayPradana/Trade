"""Tests for bot.orderbook — advanced orderbook analysis module."""

import unittest
from typing import Dict, List

from bot.orderbook import (
    DepthModel,
    EnhancedSpoofingResult,
    HeatmapAnalysis,
    HiddenLiquidityResult,
    IcebergResult,
    ImbalanceResult,
    LiquidityGapResult,
    OrderFlowImbalance,
    PressureAnalysis,
    SlippagePrediction,
    SpreadAnalysis,
    WhaleDetectionResult,
    analyze_heatmap,
    analyze_pressure,
    analyze_spread,
    compute_order_flow_imbalance,
    detect_hidden_liquidity,
    detect_iceberg_orders,
    detect_imbalance,
    detect_liquidity_gaps,
    detect_spoofing_enhanced,
    detect_whale_orders,
    model_market_depth,
    predict_slippage,
)


def _simple_depth(
    bid_prices: List[float] | None = None,
    ask_prices: List[float] | None = None,
    bid_vol: float = 1.0,
    ask_vol: float = 1.0,
) -> Dict:
    """Build a simple depth dict for testing."""
    bids = bid_prices or [100, 99, 98, 97, 96]
    asks = ask_prices or [101, 102, 103, 104, 105]
    return {
        "buy": [[str(p), str(bid_vol)] for p in bids],
        "sell": [[str(p), str(ask_vol)] for p in asks],
    }


# ───────────────────────────────────────────────────────────────────────
#  1. Spread Analysis
# ───────────────────────────────────────────────────────────────────────


class SpreadAnalysisTests(unittest.TestCase):
    def test_basic_spread(self) -> None:
        depth = _simple_depth()
        result = analyze_spread(depth)
        self.assertAlmostEqual(result.best_bid, 100.0)
        self.assertAlmostEqual(result.best_ask, 101.0)
        self.assertAlmostEqual(result.spread_abs, 1.0)
        self.assertAlmostEqual(result.mid_price, 100.5)
        self.assertGreater(result.spread_pct, 0)
        self.assertLess(result.spread_pct, 0.02)

    def test_wide_spread(self) -> None:
        depth = _simple_depth(bid_prices=[100], ask_prices=[110])
        result = analyze_spread(depth, wide_threshold_pct=0.05)
        self.assertTrue(result.is_wide)

    def test_tight_spread(self) -> None:
        depth = _simple_depth()
        result = analyze_spread(depth, wide_threshold_pct=0.05)
        self.assertFalse(result.is_wide)

    def test_empty_depth(self) -> None:
        result = analyze_spread({"buy": [], "sell": []})
        self.assertAlmostEqual(result.spread_abs, 0.0)
        self.assertAlmostEqual(result.mid_price, 0.0)


# ───────────────────────────────────────────────────────────────────────
#  2. Imbalance Detection
# ───────────────────────────────────────────────────────────────────────


class ImbalanceTests(unittest.TestCase):
    def test_balanced(self) -> None:
        depth = _simple_depth(bid_vol=1.0, ask_vol=1.0)
        result = detect_imbalance(depth)
        self.assertEqual(result.dominant_side, "balanced")
        self.assertAlmostEqual(result.imbalance, 0.0, places=1)

    def test_bid_dominant(self) -> None:
        depth = _simple_depth(bid_vol=10.0, ask_vol=1.0)
        result = detect_imbalance(depth)
        self.assertEqual(result.dominant_side, "bid")
        self.assertGreater(result.imbalance, 0.3)
        self.assertGreater(result.bid_total, result.ask_total)

    def test_ask_dominant(self) -> None:
        depth = _simple_depth(bid_vol=1.0, ask_vol=10.0)
        result = detect_imbalance(depth)
        self.assertEqual(result.dominant_side, "ask")
        self.assertLess(result.imbalance, -0.3)

    def test_empty_depth(self) -> None:
        result = detect_imbalance({"buy": [], "sell": []})
        self.assertAlmostEqual(result.imbalance, 0.0)
        self.assertEqual(result.dominant_side, "balanced")


# ───────────────────────────────────────────────────────────────────────
#  3. Liquidity Gap Detection
# ───────────────────────────────────────────────────────────────────────


class LiquidityGapTests(unittest.TestCase):
    def test_no_gaps_in_tight_book(self) -> None:
        depth = _simple_depth()
        result = detect_liquidity_gaps(depth, min_gap_pct=0.05)
        self.assertFalse(result.detected)
        self.assertEqual(len(result.gaps), 0)

    def test_detects_gap(self) -> None:
        # 10% gap between 100 and 90 on bid side
        depth = {
            "buy": [["100", "1"], ["90", "1"], ["89", "1"]],
            "sell": [["101", "1"], ["102", "1"]],
        }
        result = detect_liquidity_gaps(depth, min_gap_pct=0.05)
        self.assertTrue(result.detected)
        self.assertGreater(result.worst_gap_pct, 0.05)
        bid_gaps = [g for g in result.gaps if g.side == "bid"]
        self.assertGreater(len(bid_gaps), 0)

    def test_both_sides(self) -> None:
        depth = {
            "buy": [["100", "1"], ["80", "1"]],
            "sell": [["120", "1"], ["140", "1"]],
        }
        result = detect_liquidity_gaps(depth, min_gap_pct=0.10)
        self.assertTrue(result.detected)
        sides = {g.side for g in result.gaps}
        self.assertEqual(sides, {"bid", "ask"})

    def test_empty(self) -> None:
        result = detect_liquidity_gaps({"buy": [], "sell": []})
        self.assertFalse(result.detected)


# ───────────────────────────────────────────────────────────────────────
#  4. Hidden Liquidity Detection
# ───────────────────────────────────────────────────────────────────────


class HiddenLiquidityTests(unittest.TestCase):
    def test_hidden_detected(self) -> None:
        depth = {"buy": [["100", "1.0"]], "sell": [["101", "0.5"]]}
        # Trades at 101 filled 5.0 but only 0.5 visible
        trades = [
            {"price": "101", "amount": "2.0", "type": "buy"},
            {"price": "101", "amount": "3.0", "type": "buy"},
        ]
        result = detect_hidden_liquidity(depth, trades, fill_multiplier=1.5)
        self.assertTrue(result.detected)
        self.assertGreater(result.estimated_hidden_volume, 0)

    def test_no_hidden(self) -> None:
        depth = {"buy": [["100", "10.0"]], "sell": [["101", "10.0"]]}
        trades = [{"price": "101", "amount": "1.0", "type": "buy"}]
        result = detect_hidden_liquidity(depth, trades, fill_multiplier=1.5)
        self.assertFalse(result.detected)

    def test_no_trades(self) -> None:
        depth = _simple_depth()
        result = detect_hidden_liquidity(depth, [])
        self.assertFalse(result.detected)


# ───────────────────────────────────────────────────────────────────────
#  5. Whale Order Detection
# ───────────────────────────────────────────────────────────────────────


class WhaleDetectionTests(unittest.TestCase):
    def test_whale_on_bid(self) -> None:
        depth = {
            "buy": [["100", "1"], ["99", "1"], ["98", "1"], ["97", "50"]],
            "sell": [["101", "1"], ["102", "1"], ["103", "1"]],
        }
        result = detect_whale_orders(depth, multiplier=3.0)
        self.assertTrue(result.detected)
        self.assertEqual(result.dominant_side, "bid")
        self.assertGreater(result.total_whale_notional, 0)
        self.assertGreater(len(result.whales), 0)

    def test_no_whales(self) -> None:
        depth = _simple_depth(bid_vol=1.0, ask_vol=1.0)
        result = detect_whale_orders(depth, multiplier=5.0)
        self.assertFalse(result.detected)
        self.assertEqual(len(result.whales), 0)

    def test_whale_enumeration(self) -> None:
        # 8 small levels + 2 giant whales → mean is pulled up but whales
        # still dominate.
        depth = {
            "buy": [
                ["100", "1"], ["99", "1"], ["98", "1"], ["97", "1"],
                ["96", "1"], ["95", "1"], ["94", "1"], ["93", "1"],
                ["92", "100"], ["91", "100"],
            ],
            "sell": [["101", "1"], ["102", "1"], ["103", "1"]],
        }
        result = detect_whale_orders(depth, multiplier=3.0)
        # avg ≈ 20.8, ratios for 100 ≈ 4.8 → detected
        self.assertGreaterEqual(len(result.whales), 2)
        for w in result.whales:
            self.assertGreater(w.ratio, 3.0)
            self.assertGreater(w.notional, 0)


# ───────────────────────────────────────────────────────────────────────
#  6. Iceberg Order Detection
# ───────────────────────────────────────────────────────────────────────


class IcebergTests(unittest.TestCase):
    def test_iceberg_detected(self) -> None:
        # Same volume at same price across 5 snapshots → iceberg
        snapshots = [
            {"buy": [["100", "5.0"]], "sell": [["101", "5.0"]]}
            for _ in range(5)
        ]
        results = detect_iceberg_orders(snapshots, min_refills=3)
        self.assertGreater(len(results), 0)
        self.assertTrue(results[0].detected)
        self.assertEqual(results[0].refill_count, 5)

    def test_no_iceberg_with_variable_volumes(self) -> None:
        # Volumes vary too much on both sides → no iceberg
        snapshots = [
            {"buy": [["100", str(v)]], "sell": [["101", str(v * 2)]]}
            for v in [1.0, 10.0, 0.5, 20.0, 0.1]
        ]
        results = detect_iceberg_orders(snapshots, min_refills=3, volume_tolerance=0.1)
        self.assertEqual(len(results), 0)

    def test_too_few_snapshots(self) -> None:
        snapshots = [_simple_depth()]
        results = detect_iceberg_orders(snapshots, min_refills=3)
        self.assertEqual(len(results), 0)


# ───────────────────────────────────────────────────────────────────────
#  7. Market Depth Modeling
# ───────────────────────────────────────────────────────────────────────


class DepthModelTests(unittest.TestCase):
    def test_basic_depth_model(self) -> None:
        depth = _simple_depth()
        model = model_market_depth(depth)
        self.assertAlmostEqual(model.mid_price, 100.5)
        self.assertGreater(len(model.levels), 0)
        self.assertGreater(model.bid_depth_total, 0)
        self.assertGreater(model.ask_depth_total, 0)
        self.assertGreater(model.depth_ratio, 0)

    def test_cumulative_increases(self) -> None:
        depth = _simple_depth()
        model = model_market_depth(depth)
        # Wider distances should have >= cumulative volumes
        for i in range(1, len(model.levels)):
            prev = model.levels[i - 1]
            curr = model.levels[i]
            self.assertGreaterEqual(
                curr.cumulative_bid_volume, prev.cumulative_bid_volume,
            )
            self.assertGreaterEqual(
                curr.cumulative_ask_volume, prev.cumulative_ask_volume,
            )

    def test_empty_depth(self) -> None:
        model = model_market_depth({"buy": [], "sell": []})
        self.assertAlmostEqual(model.mid_price, 0.0)


# ───────────────────────────────────────────────────────────────────────
#  8. Order Flow Imbalance
# ───────────────────────────────────────────────────────────────────────


class OrderFlowTests(unittest.TestCase):
    def test_buy_aggressive(self) -> None:
        trades = [
            {"type": "buy", "price": "100", "amount": "10"},
            {"type": "buy", "price": "101", "amount": "10"},
            {"type": "sell", "price": "99", "amount": "1"},
        ]
        result = compute_order_flow_imbalance(trades)
        self.assertGreater(result.imbalance, 0)
        self.assertEqual(result.aggressive_side, "buy")
        self.assertGreater(result.buy_volume, result.sell_volume)

    def test_sell_aggressive(self) -> None:
        trades = [
            {"type": "sell", "price": "100", "amount": "10"},
            {"type": "sell", "price": "99", "amount": "10"},
            {"type": "buy", "price": "101", "amount": "1"},
        ]
        result = compute_order_flow_imbalance(trades)
        self.assertLess(result.imbalance, 0)
        self.assertEqual(result.aggressive_side, "sell")

    def test_neutral(self) -> None:
        trades = [
            {"type": "buy", "price": "100", "amount": "5"},
            {"type": "sell", "price": "100", "amount": "5"},
        ]
        result = compute_order_flow_imbalance(trades)
        self.assertAlmostEqual(result.imbalance, 0.0)
        self.assertIsNone(result.aggressive_side)

    def test_empty_trades(self) -> None:
        result = compute_order_flow_imbalance([])
        self.assertAlmostEqual(result.imbalance, 0.0)
        self.assertEqual(result.trade_count, 0)


# ───────────────────────────────────────────────────────────────────────
#  9. Buy vs Sell Pressure
# ───────────────────────────────────────────────────────────────────────


class PressureTests(unittest.TestCase):
    def test_buy_pressure(self) -> None:
        depth = _simple_depth(bid_vol=10.0, ask_vol=1.0)
        trades = [
            {"type": "buy", "price": "101", "amount": "10"},
        ]
        result = analyze_pressure(depth, trades)
        self.assertEqual(result.signal, "buy_pressure")
        self.assertGreater(result.pressure, 0)

    def test_sell_pressure(self) -> None:
        depth = _simple_depth(bid_vol=1.0, ask_vol=10.0)
        trades = [
            {"type": "sell", "price": "100", "amount": "10"},
        ]
        result = analyze_pressure(depth, trades)
        self.assertEqual(result.signal, "sell_pressure")
        self.assertLess(result.pressure, 0)

    def test_neutral_pressure(self) -> None:
        depth = _simple_depth(bid_vol=1.0, ask_vol=1.0)
        trades = [
            {"type": "buy", "price": "100", "amount": "1"},
            {"type": "sell", "price": "100", "amount": "1"},
        ]
        result = analyze_pressure(depth, trades)
        self.assertEqual(result.signal, "neutral")


# ───────────────────────────────────────────────────────────────────────
# 10. Enhanced Spoofing Detection
# ───────────────────────────────────────────────────────────────────────


class SpoofingTests(unittest.TestCase):
    def test_spoof_wall_detected(self) -> None:
        depth = {
            "buy": [
                ["100", "1"], ["99", "1"], ["98", "1"],
                ["95", "1"], ["90", "50"],  # far + large = spoof
            ],
            "sell": [["101", "1"], ["102", "1"], ["103", "1"]],
        }
        result = detect_spoofing_enhanced(depth, volume_multiplier=3.0, min_distance_pct=0.05)
        self.assertTrue(result.detected)
        self.assertGreater(len(result.walls), 0)
        self.assertGreater(result.risk_score, 0)

    def test_no_spoof(self) -> None:
        depth = _simple_depth()
        result = detect_spoofing_enhanced(depth)
        self.assertFalse(result.detected)
        self.assertAlmostEqual(result.risk_score, 0.0)


# ───────────────────────────────────────────────────────────────────────
# 11. Slippage Prediction
# ───────────────────────────────────────────────────────────────────────


class SlippageTests(unittest.TestCase):
    def test_small_order_low_slippage(self) -> None:
        depth = _simple_depth(ask_vol=10.0, bid_vol=10.0)
        result = predict_slippage(depth, order_size=0.5, side="buy")
        self.assertTrue(result.fully_filled)
        self.assertLess(result.estimated_slippage_pct, 0.01)
        self.assertEqual(result.levels_consumed, 1)

    def test_large_order_higher_slippage(self) -> None:
        depth = _simple_depth(ask_vol=1.0, bid_vol=1.0)
        result = predict_slippage(depth, order_size=4.0, side="buy")
        self.assertTrue(result.fully_filled)
        self.assertGreater(result.estimated_slippage_pct, 0)
        self.assertGreater(result.levels_consumed, 1)

    def test_sell_side(self) -> None:
        depth = _simple_depth(bid_vol=2.0)
        result = predict_slippage(depth, order_size=1.0, side="sell")
        self.assertTrue(result.fully_filled)
        self.assertEqual(result.side, "sell")

    def test_insufficient_liquidity(self) -> None:
        depth = _simple_depth(ask_vol=0.1)
        result = predict_slippage(depth, order_size=100.0, side="buy")
        self.assertFalse(result.fully_filled)

    def test_empty_book(self) -> None:
        result = predict_slippage({"buy": [], "sell": []}, order_size=1.0)
        self.assertFalse(result.fully_filled)


# ───────────────────────────────────────────────────────────────────────
# 12. Orderbook Heatmap
# ───────────────────────────────────────────────────────────────────────


class HeatmapTests(unittest.TestCase):
    def test_basic_heatmap(self) -> None:
        depth = _simple_depth()
        result = analyze_heatmap(depth, num_bins=5)
        self.assertGreater(len(result.bins), 0)
        self.assertGreater(result.concentration_price, 0)
        self.assertIn(result.concentration_side, ("bid", "ask"))

    def test_bin_intensity_normalised(self) -> None:
        depth = _simple_depth()
        result = analyze_heatmap(depth, num_bins=10)
        for b in result.bins:
            self.assertGreaterEqual(b.intensity, 0.0)
            self.assertLessEqual(b.intensity, 1.0)
        # At least one bin should have intensity 1.0
        max_intensity = max(b.intensity for b in result.bins)
        self.assertAlmostEqual(max_intensity, 1.0)

    def test_empty_depth(self) -> None:
        result = analyze_heatmap({"buy": [], "sell": []})
        self.assertEqual(len(result.bins), 0)

    def test_single_level(self) -> None:
        depth = {"buy": [["100", "5"]], "sell": [["100", "3"]]}
        result = analyze_heatmap(depth, num_bins=10)
        self.assertGreater(len(result.bins), 0)


if __name__ == "__main__":
    unittest.main()
