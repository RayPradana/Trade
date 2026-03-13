"""Tests for bot.advanced_strategies — advanced trading strategies module."""

import unittest
from typing import Dict, List

from bot.analysis import Candle, OrderbookInsight
from bot.advanced_strategies import (
    ArbitrageOpportunity,
    BasketSignal,
    BreakoutSignal,
    EnhancedGridPlan,
    HybridSignal,
    MarketMakingSignal,
    MeanReversionSignal,
    MomentumSignal,
    MultiTimeframeSignal,
    PairsTradeSignal,
    PositionSignal,
    ScalpSignal,
    StatArbSignal,
    SwingSignal,
    TrendFollowSignal,
    basket_signal,
    breakout_signal,
    build_enhanced_grid,
    detect_arbitrage,
    hybrid_signal,
    market_making_signal,
    mean_reversion_signal,
    momentum_signal,
    multi_timeframe_signal,
    pairs_trade_signal,
    position_signal,
    scalp_signal,
    stat_arb_signal,
    swing_signal,
    trend_following_signal,
)


def _make_candles(
    prices: List[float],
    volume: float = 100.0,
    spread: float = 0.01,
) -> List[Candle]:
    """Build candles from a list of close prices."""
    candles = []
    for i, close in enumerate(prices):
        candles.append(Candle(
            timestamp=1000 + i,
            open=close * (1 - spread / 2),
            high=close * (1 + spread),
            low=close * (1 - spread),
            close=close,
            volume=volume,
        ))
    return candles


def _uptrend_candles(n: int = 50, start: float = 100.0, step: float = 1.0) -> List[Candle]:
    """Generate candles with a clear uptrend."""
    prices = [start + i * step for i in range(n)]
    return _make_candles(prices)


def _downtrend_candles(n: int = 50, start: float = 200.0, step: float = 1.0) -> List[Candle]:
    """Generate candles with a clear downtrend."""
    prices = [start - i * step for i in range(n)]
    return _make_candles(prices)


def _flat_candles(n: int = 50, price: float = 100.0) -> List[Candle]:
    """Generate flat / sideways candles."""
    prices = [price] * n
    return _make_candles(prices)


# ───────────────────────────────────────────────────────────────────────
#  1. Trend-Following
# ───────────────────────────────────────────────────────────────────────


class TrendFollowingTests(unittest.TestCase):
    def test_uptrend_buy(self) -> None:
        candles = _uptrend_candles(50)
        result = trend_following_signal(candles, fast_period=10, slow_period=30)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.strength, 0)
        self.assertGreater(result.ma_fast, result.ma_slow)

    def test_downtrend_sell(self) -> None:
        candles = _downtrend_candles(50)
        result = trend_following_signal(candles, fast_period=10, slow_period=30)
        self.assertEqual(result.action, "sell")
        self.assertLess(result.ma_fast, result.ma_slow)

    def test_flat_hold(self) -> None:
        candles = _flat_candles(50)
        result = trend_following_signal(candles, fast_period=10, slow_period=30)
        self.assertEqual(result.action, "hold")

    def test_insufficient_data(self) -> None:
        candles = _make_candles([100, 101])
        result = trend_following_signal(candles, slow_period=30)
        self.assertEqual(result.action, "hold")
        self.assertIn("insufficient", result.reason)


# ───────────────────────────────────────────────────────────────────────
#  2. Mean Reversion
# ───────────────────────────────────────────────────────────────────────


class MeanReversionTests(unittest.TestCase):
    def test_oversold_buy(self) -> None:
        # Price crashes well below the mean
        prices = [100.0] * 18 + [80.0, 70.0]
        candles = _make_candles(prices)
        result = mean_reversion_signal(candles, lookback=20)
        self.assertEqual(result.action, "buy")
        self.assertLess(result.z_score, 0)

    def test_overbought_sell(self) -> None:
        prices = [100.0] * 18 + [120.0, 130.0]
        candles = _make_candles(prices)
        result = mean_reversion_signal(candles, lookback=20)
        self.assertEqual(result.action, "sell")
        self.assertGreater(result.z_score, 0)

    def test_normal_hold(self) -> None:
        candles = _flat_candles(20)
        result = mean_reversion_signal(candles, lookback=20)
        self.assertEqual(result.action, "hold")

    def test_insufficient_data(self) -> None:
        candles = _make_candles([100])
        result = mean_reversion_signal(candles, lookback=20)
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
#  3. Momentum Trading
# ───────────────────────────────────────────────────────────────────────


class MomentumTests(unittest.TestCase):
    def test_positive_momentum_buy(self) -> None:
        # Accelerating uptrend: increasing step sizes
        prices = [100 + i * i * 0.5 for i in range(20)]
        candles = _make_candles(prices)
        result = momentum_signal(candles, lookback=14, threshold=0.02)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.roc, 0)

    def test_negative_momentum_sell(self) -> None:
        # Accelerating downtrend: increasing step sizes downward
        prices = [300 - i * i * 0.5 for i in range(20)]
        candles = _make_candles(prices)
        result = momentum_signal(candles, lookback=14, threshold=0.02)
        self.assertEqual(result.action, "sell")
        self.assertLess(result.roc, 0)

    def test_flat_hold(self) -> None:
        candles = _flat_candles(20)
        result = momentum_signal(candles, lookback=14)
        self.assertEqual(result.action, "hold")

    def test_insufficient_data(self) -> None:
        candles = _make_candles([100, 101])
        result = momentum_signal(candles)
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
#  4. Breakout Strategies
# ───────────────────────────────────────────────────────────────────────


class BreakoutTests(unittest.TestCase):
    def test_upside_breakout(self) -> None:
        # 20 candles in range, then breakout above
        prices = [100.0] * 20 + [115.0]
        candles = _make_candles(prices, spread=0.005)
        result = breakout_signal(candles, lookback=20, volume_multiplier=0.5)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.breakout_level, 0)

    def test_downside_breakout(self) -> None:
        prices = [100.0] * 20 + [85.0]
        candles = _make_candles(prices, spread=0.005)
        result = breakout_signal(candles, lookback=20, volume_multiplier=0.5)
        self.assertEqual(result.action, "sell")

    def test_no_breakout(self) -> None:
        candles = _flat_candles(25)
        result = breakout_signal(candles, lookback=20)
        self.assertEqual(result.action, "hold")

    def test_volume_confirmation(self) -> None:
        prices = [100.0] * 20 + [115.0]
        candles = _make_candles(prices, volume=200, spread=0.005)
        result = breakout_signal(candles, lookback=20, volume_multiplier=0.5)
        self.assertTrue(result.volume_confirmation)

    def test_insufficient_data(self) -> None:
        candles = _make_candles([100])
        result = breakout_signal(candles)
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
#  5. Arbitrage Strategies
# ───────────────────────────────────────────────────────────────────────


class ArbitrageTests(unittest.TestCase):
    def test_arbitrage_detected(self) -> None:
        prices = {"exchange_a": 100.0, "exchange_b": 102.0}
        result = detect_arbitrage(prices, fee_pct=0.003, min_spread_pct=0.005)
        self.assertTrue(result.detected)
        self.assertEqual(result.buy_source, "exchange_a")
        self.assertEqual(result.sell_source, "exchange_b")
        self.assertGreater(result.net_profit_pct, 0)

    def test_no_arbitrage_small_spread(self) -> None:
        prices = {"a": 100.0, "b": 100.1}
        result = detect_arbitrage(prices, fee_pct=0.003, min_spread_pct=0.005)
        self.assertFalse(result.detected)

    def test_single_source(self) -> None:
        prices = {"only": 100.0}
        result = detect_arbitrage(prices)
        self.assertFalse(result.detected)
        self.assertIn("2 sources", result.reason)

    def test_empty_prices(self) -> None:
        result = detect_arbitrage({})
        self.assertFalse(result.detected)


# ───────────────────────────────────────────────────────────────────────
#  6. Statistical Arbitrage
# ───────────────────────────────────────────────────────────────────────


class StatArbTests(unittest.TestCase):
    def test_spread_too_wide(self) -> None:
        # Asset A diverges high from B → sell A, buy B
        a = [100.0] * 10 + [120.0]
        b = [100.0] * 11
        result = stat_arb_signal(a, b, entry_z=1.0)
        self.assertEqual(result.action, "sell_a_buy_b")
        self.assertGreater(result.spread_z, 0)

    def test_spread_too_narrow(self) -> None:
        a = [100.0] * 10 + [80.0]
        b = [100.0] * 11
        result = stat_arb_signal(a, b, entry_z=1.0)
        self.assertEqual(result.action, "buy_a_sell_b")
        self.assertLess(result.spread_z, 0)

    def test_spread_normal(self) -> None:
        a = [100.0] * 15
        b = [100.0] * 15
        result = stat_arb_signal(a, b)
        self.assertEqual(result.action, "hold")

    def test_insufficient_data(self) -> None:
        result = stat_arb_signal([100], [100])
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
#  7. Market Making
# ───────────────────────────────────────────────────────────────────────


class MarketMakingTests(unittest.TestCase):
    def test_basic_quotes(self) -> None:
        result = market_making_signal(mid_price=100.0, volatility=0.1)
        self.assertGreater(result.bid.price, 0)
        self.assertGreater(result.ask.price, 0)
        self.assertLess(result.bid.price, result.ask.price)

    def test_wider_spread_in_high_vol(self) -> None:
        low_vol = market_making_signal(mid_price=100.0, volatility=0.01)
        high_vol = market_making_signal(mid_price=100.0, volatility=0.5)
        low_spread = low_vol.ask.price - low_vol.bid.price
        high_spread = high_vol.ask.price - high_vol.bid.price
        self.assertGreater(high_spread, low_spread)

    def test_inventory_skew(self) -> None:
        long_pos = market_making_signal(mid_price=100.0, volatility=0.1, inventory=5.0)
        # Long inventory → lower bid (less eager to buy more)
        neutral = market_making_signal(mid_price=100.0, volatility=0.1, inventory=0.0)
        self.assertGreater(long_pos.inventory_skew, 0)

    def test_zero_price(self) -> None:
        result = market_making_signal(mid_price=0.0, volatility=0.1)
        self.assertAlmostEqual(result.bid.price, 0.0)
        self.assertAlmostEqual(result.ask.price, 0.0)


# ───────────────────────────────────────────────────────────────────────
#  8. Enhanced Grid Trading
# ───────────────────────────────────────────────────────────────────────


class EnhancedGridTests(unittest.TestCase):
    def test_basic_grid(self) -> None:
        result = build_enhanced_grid(
            current_price=100.0, volatility=0.1, capital=10000.0, num_levels=3,
        )
        self.assertGreater(len(result.levels), 0)
        self.assertEqual(len(result.levels), 6)  # 3 buy + 3 sell
        buy_levels = [l for l in result.levels if l.side == "buy"]
        sell_levels = [l for l in result.levels if l.side == "sell"]
        self.assertEqual(len(buy_levels), 3)
        self.assertEqual(len(sell_levels), 3)

    def test_wider_spacing_high_vol(self) -> None:
        low_vol = build_enhanced_grid(
            current_price=100.0, volatility=0.01, capital=10000.0,
        )
        high_vol = build_enhanced_grid(
            current_price=100.0, volatility=0.5, capital=10000.0,
        )
        low_buy = [l for l in low_vol.levels if l.side == "buy"][0]
        high_buy = [l for l in high_vol.levels if l.side == "buy"][0]
        # Higher volatility → wider distance from anchor
        self.assertGreater(high_buy.distance_pct, low_buy.distance_pct)

    def test_zero_price(self) -> None:
        result = build_enhanced_grid(current_price=0.0, volatility=0.1, capital=1000.0)
        self.assertEqual(len(result.levels), 0)

    def test_capital_coverage(self) -> None:
        result = build_enhanced_grid(
            current_price=100.0, volatility=0.1, capital=10000.0, num_levels=5,
        )
        self.assertGreater(result.total_capital_required, 0)


# ───────────────────────────────────────────────────────────────────────
#  9. Scalping
# ───────────────────────────────────────────────────────────────────────


class ScalpingTests(unittest.TestCase):
    def test_scalp_buy_on_imbalance(self) -> None:
        candles = _make_candles([100, 100.1, 100.2, 100.3, 100.4], spread=0.02)
        ob = OrderbookInsight(spread_pct=0.001, bid_volume=1000, ask_volume=500, imbalance=0.5)
        result = scalp_signal(candles, orderbook=ob, min_edge_pct=0.001)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.edge_pct, 0)

    def test_scalp_sell_on_negative_imbalance(self) -> None:
        candles = _make_candles([100, 100.1, 100.2, 100.3, 100.4], spread=0.02)
        ob = OrderbookInsight(spread_pct=0.001, bid_volume=500, ask_volume=1000, imbalance=-0.5)
        result = scalp_signal(candles, orderbook=ob, min_edge_pct=0.001)
        self.assertEqual(result.action, "sell")

    def test_wide_spread_hold(self) -> None:
        candles = _make_candles([100] * 5)
        ob = OrderbookInsight(spread_pct=0.01, bid_volume=500, ask_volume=500, imbalance=0.0)
        result = scalp_signal(candles, orderbook=ob, max_spread_pct=0.003)
        self.assertEqual(result.action, "hold")
        self.assertIn("spread", result.reason)

    def test_insufficient_data(self) -> None:
        candles = _make_candles([100])
        result = scalp_signal(candles)
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
# 10. Swing Trading
# ───────────────────────────────────────────────────────────────────────


class SwingTradingTests(unittest.TestCase):
    def test_buy_near_support(self) -> None:
        # Price at the bottom of range → buy
        prices = [110, 115, 120, 118, 115, 112, 110, 108, 105, 102]
        candles = _make_candles(prices, spread=0.001)
        result = swing_signal(candles, min_risk_reward=1.5)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.risk_reward, 1.5)

    def test_sell_near_resistance(self) -> None:
        prices = [100, 105, 110, 115, 118, 120, 119, 118, 119, 120]
        candles = _make_candles(prices, spread=0.001)
        result = swing_signal(candles, min_risk_reward=1.5)
        # Price near the high → should signal sell if distance to low > distance to high
        if result.action == "sell":
            self.assertGreater(result.risk_reward, 1.5)

    def test_hold_in_middle(self) -> None:
        prices = [100, 110, 100, 110, 100, 110, 100, 110, 105, 105]
        candles = _make_candles(prices, spread=0.001)
        result = swing_signal(candles, min_risk_reward=3.0)
        # In the middle with high R:R requirement → hold
        self.assertEqual(result.action, "hold")

    def test_insufficient_data(self) -> None:
        candles = _make_candles([100])
        result = swing_signal(candles)
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
# 11. Position Trading
# ───────────────────────────────────────────────────────────────────────


class PositionTradingTests(unittest.TestCase):
    def test_bullish_position(self) -> None:
        candles = _uptrend_candles(60, start=100, step=2)
        result = position_signal(candles, long_period=50, threshold=0.1)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.combined_score, 0)

    def test_bearish_position(self) -> None:
        candles = _downtrend_candles(60, start=300, step=3)
        result = position_signal(candles, long_period=50, threshold=0.1)
        self.assertEqual(result.action, "sell")
        self.assertLess(result.combined_score, 0)

    def test_neutral(self) -> None:
        candles = _flat_candles(60)
        result = position_signal(candles, long_period=50)
        self.assertEqual(result.action, "hold")

    def test_insufficient_data(self) -> None:
        candles = _make_candles([100] * 5)
        result = position_signal(candles, long_period=50)
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
# 12. Pairs Trading
# ───────────────────────────────────────────────────────────────────────


class PairsTradingTests(unittest.TestCase):
    def test_ratio_high_sell_a(self) -> None:
        a = [100.0] * 10 + [130.0]
        b = [100.0] * 11
        result = pairs_trade_signal("BTC", "ETH", a, b, entry_z=1.0)
        self.assertEqual(result.action, "sell_a_buy_b")
        self.assertGreater(result.ratio_z, 0)

    def test_ratio_low_buy_a(self) -> None:
        a = [100.0] * 10 + [70.0]
        b = [100.0] * 11
        result = pairs_trade_signal("BTC", "ETH", a, b, entry_z=1.0)
        self.assertEqual(result.action, "buy_a_sell_b")
        self.assertLess(result.ratio_z, 0)

    def test_ratio_normal(self) -> None:
        a = [100.0] * 15
        b = [100.0] * 15
        result = pairs_trade_signal("BTC", "ETH", a, b)
        self.assertEqual(result.action, "hold")

    def test_insufficient_data(self) -> None:
        result = pairs_trade_signal("A", "B", [100], [100])
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
# 13. Basket Trading
# ───────────────────────────────────────────────────────────────────────


class BasketTradingTests(unittest.TestCase):
    def test_bullish_basket(self) -> None:
        components = [
            {"pair": "BTC", "weight": 0.5, "signal": "buy", "strength": 0.8, "current_weight": 0.5},
            {"pair": "ETH", "weight": 0.3, "signal": "buy", "strength": 0.7, "current_weight": 0.3},
            {"pair": "SOL", "weight": 0.2, "signal": "hold", "strength": 0.0, "current_weight": 0.2},
        ]
        result = basket_signal(components)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.aggregate_score, 0)

    def test_bearish_basket(self) -> None:
        components = [
            {"pair": "BTC", "weight": 0.5, "signal": "sell", "strength": 0.8, "current_weight": 0.5},
            {"pair": "ETH", "weight": 0.5, "signal": "sell", "strength": 0.7, "current_weight": 0.5},
        ]
        result = basket_signal(components)
        self.assertEqual(result.action, "sell")
        self.assertLess(result.aggregate_score, 0)

    def test_rebalance_needed(self) -> None:
        components = [
            {"pair": "BTC", "weight": 0.5, "signal": "hold", "strength": 0.0, "current_weight": 0.8},
        ]
        result = basket_signal(components, rebalance_threshold=0.1)
        self.assertTrue(result.rebalance_needed)

    def test_empty_basket(self) -> None:
        result = basket_signal([])
        self.assertEqual(result.action, "hold")
        self.assertIn("empty", result.reason)


# ───────────────────────────────────────────────────────────────────────
# 14. Multi-Timeframe
# ───────────────────────────────────────────────────────────────────────


class MultiTimeframeTests(unittest.TestCase):
    def test_all_up_aligned(self) -> None:
        timeframes = {
            "1h": _uptrend_candles(30),
            "4h": _uptrend_candles(30),
            "1d": _uptrend_candles(30),
        }
        result = multi_timeframe_signal(timeframes, ma_period=20)
        self.assertEqual(result.action, "buy")
        self.assertTrue(result.aligned)
        self.assertEqual(result.dominant_trend, "up")

    def test_all_down_aligned(self) -> None:
        timeframes = {
            "1h": _downtrend_candles(30),
            "4h": _downtrend_candles(30),
            "1d": _downtrend_candles(30),
        }
        result = multi_timeframe_signal(timeframes, ma_period=20)
        self.assertEqual(result.action, "sell")
        self.assertTrue(result.aligned)
        self.assertEqual(result.dominant_trend, "down")

    def test_mixed_hold(self) -> None:
        timeframes = {
            "1h": _uptrend_candles(30),
            "4h": _downtrend_candles(30),
            "1d": _flat_candles(30),
        }
        result = multi_timeframe_signal(timeframes, ma_period=20)
        self.assertEqual(result.action, "hold")
        self.assertFalse(result.aligned)

    def test_empty_timeframes(self) -> None:
        result = multi_timeframe_signal({})
        self.assertEqual(result.action, "hold")


# ───────────────────────────────────────────────────────────────────────
# 15. Hybrid Strategies
# ───────────────────────────────────────────────────────────────────────


class HybridTests(unittest.TestCase):
    def test_consensus_buy(self) -> None:
        votes = [
            {"strategy": "trend", "action": "buy", "weight": 1.0, "confidence": 0.8},
            {"strategy": "momentum", "action": "buy", "weight": 1.0, "confidence": 0.7},
            {"strategy": "mean_rev", "action": "hold", "weight": 0.5, "confidence": 0.3},
        ]
        result = hybrid_signal(votes)
        self.assertEqual(result.action, "buy")
        self.assertGreater(result.consensus_score, 0)
        self.assertGreater(result.agreement_pct, 0.5)

    def test_consensus_sell(self) -> None:
        votes = [
            {"strategy": "trend", "action": "sell", "weight": 1.0, "confidence": 0.9},
            {"strategy": "momentum", "action": "sell", "weight": 1.0, "confidence": 0.8},
        ]
        result = hybrid_signal(votes)
        self.assertEqual(result.action, "sell")
        self.assertLess(result.consensus_score, 0)

    def test_no_consensus(self) -> None:
        votes = [
            {"strategy": "a", "action": "buy", "weight": 1.0, "confidence": 0.5},
            {"strategy": "b", "action": "sell", "weight": 1.0, "confidence": 0.5},
        ]
        result = hybrid_signal(votes, min_agreement=0.7)
        self.assertEqual(result.action, "hold")

    def test_empty_votes(self) -> None:
        result = hybrid_signal([])
        self.assertEqual(result.action, "hold")
        self.assertIn("no strategy", result.reason)

    def test_vote_weights(self) -> None:
        # Heavy buy vote should override light sell vote
        votes = [
            {"strategy": "strong", "action": "buy", "weight": 3.0, "confidence": 0.8},
            {"strategy": "weak", "action": "sell", "weight": 0.5, "confidence": 0.3},
        ]
        result = hybrid_signal(votes, min_agreement=0.3)
        self.assertEqual(result.action, "buy")


if __name__ == "__main__":
    unittest.main()
