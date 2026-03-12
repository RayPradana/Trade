import unittest

from bot.config import BotConfig
from bot.grid import build_grid_plan
from bot.strategies import StrategyDecision, make_trade_decision, select_strategy, _position_size
from bot.analysis import OrderbookInsight, TrendResult, VolatilityStats, SupportResistance


class StrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BotConfig(api_key=None, dry_run=True)

    def test_select_strategy_scalping(self) -> None:
        trend = TrendResult("up", 101, 100, 0.01)
        orderbook = OrderbookInsight(
            spread_pct=0.0005,
            bid_volume=10,
            ask_volume=5,
            imbalance=(10 - 5) / (10 + 5),
        )
        vol = VolatilityStats(volatility=0.005, avg_volume=1)
        mode = select_strategy(trend, orderbook, vol)
        self.assertEqual(mode, "scalping")

    def test_trade_decision_buy(self) -> None:
        trend = TrendResult("up", 102, 100, 0.02)
        orderbook = OrderbookInsight(spread_pct=0.001, bid_volume=10, ask_volume=8, imbalance=0.1)
        vol = VolatilityStats(volatility=0.01, avg_volume=1)
        decision: StrategyDecision = make_trade_decision(
            trend, orderbook, vol, 100.0, self.config, None
        )
        self.assertEqual(decision.action, "buy")
        self.assertGreater(decision.confidence, 0)
        self.assertGreater(decision.take_profit, decision.target_price)

    def test_trade_decision_hold_on_flat_trend(self) -> None:
        trend = TrendResult("flat", 100, 100, 0.0)
        orderbook = OrderbookInsight(spread_pct=0.002, bid_volume=5, ask_volume=5, imbalance=0.0)
        vol = VolatilityStats(volatility=0.02, avg_volume=1)
        decision = make_trade_decision(trend, orderbook, vol, 100.0, self.config, None)
        self.assertEqual(decision.action, "hold")

    def test_confidence_reduced_near_resistance(self) -> None:
        trend = TrendResult("up", 102, 100, 0.02)
        orderbook = OrderbookInsight(spread_pct=0.0005, bid_volume=10, ask_volume=8, imbalance=0.2)
        vol = VolatilityStats(volatility=0.005, avg_volume=1)
        levels = SupportResistance(support=90, resistance=101, lookback=30)
        decision = make_trade_decision(trend, orderbook, vol, 100.5, self.config, levels)
        self.assertLess(decision.confidence, 1.0)

    def test_dynamic_sizing_scales_with_price(self) -> None:
        """Order size must be inversely proportional to coin price so that the
        IDR-equivalent value stays constant regardless of which coin is traded."""
        config = BotConfig(api_key=None, risk_per_trade=0.01, initial_capital=1_000_000)
        vol = VolatilityStats(volatility=0.01, avg_volume=1)

        # BTC-like: high price → small unit count
        btc_price = 1_000_000_000.0
        stop_btc = btc_price * 0.995
        size_btc = _position_size(btc_price, stop_btc, config, btc_price - stop_btc, 0.8, vol)

        # PEPE-like: low price → large unit count
        pepe_price = 10.0
        stop_pepe = pepe_price * 0.995
        size_pepe = _position_size(pepe_price, stop_pepe, config, pepe_price - stop_pepe, 0.8, vol)

        # Both order values in quote currency should be approximately equal
        value_btc = size_btc * btc_price
        value_pepe = size_pepe * pepe_price
        self.assertAlmostEqual(value_btc, value_pepe, delta=value_btc * 0.01)

    def test_dynamic_sizing_zero_price_returns_zero(self) -> None:
        """A zero price must not cause division-by-zero; size should be 0."""
        config = BotConfig(api_key=None)
        vol = VolatilityStats(volatility=0.01, avg_volume=1)
        size = _position_size(0.0, None, config, 0.0, 0.8, vol)
        self.assertEqual(size, 0.0)

    def test_grid_dynamic_amount_scales_with_price(self) -> None:
        """Grid order amount must adapt to price so the IDR value is consistent."""
        config = BotConfig(api_key=None, risk_per_trade=0.01, initial_capital=1_000_000,
                           grid_levels_per_side=1, grid_spacing_pct=0.01)

        plan_btc = build_grid_plan(1_000_000_000.0, config)
        plan_pepe = build_grid_plan(10.0, config)

        # Amount × price should be equal (= risk_per_trade * initial_capital = 10,000 IDR)
        value_btc = plan_btc.buy_orders[0].amount * 1_000_000_000.0
        value_pepe = plan_pepe.buy_orders[0].amount * 10.0
        self.assertAlmostEqual(value_btc, value_pepe, delta=value_btc * 0.01)


if __name__ == "__main__":
    unittest.main()


class SmartEntryStrategyTests(unittest.TestCase):
    """Tests for the SmartEntryResult confidence adjustments in make_trade_decision."""

    def _base_inputs(self):
        from bot.analysis import (
            TrendResult, OrderbookInsight, VolatilityStats, SupportResistance
        )
        trend = TrendResult(direction="up", fast_ma=105.0, slow_ma=100.0, strength=0.05)
        orderbook = OrderbookInsight(spread_pct=0.001, bid_volume=100, ask_volume=80, imbalance=0.1)
        vol = VolatilityStats(volatility=0.01, avg_volume=10)
        levels = SupportResistance(support=90.0, resistance=115.0, lookback=10)
        config = BotConfig(api_key=None)
        return trend, orderbook, vol, levels, config

    def _make_see(self, pre_pump=False, pp_score=0.0, wp_detected=False,
                  wp_side=None, wp_pressure=0.0, fb_detected=False,
                  fb_score=0.0):
        from bot.analysis import (
            PrePumpSignal, WhalePressure, FakeBreakoutRisk, SmartEntryResult
        )
        return SmartEntryResult(
            pre_pump=PrePumpSignal(detected=pre_pump, volume_surge_ratio=2.0 if pre_pump else 1.0, score=pp_score),
            whale_pressure=WhalePressure(detected=wp_detected, side=wp_side, pressure=wp_pressure),
            fake_breakout=FakeBreakoutRisk(breakout_present=fb_detected, detected=fb_detected,
                                           volume_ratio=1.0 - fb_score, score=fb_score),
        )

    def test_pre_pump_boosts_buy_confidence(self):
        trend, orderbook, vol, levels, config = self._base_inputs()
        base_decision = make_trade_decision(trend, orderbook, vol, 100.0, config, levels)
        see = self._make_see(pre_pump=True, pp_score=1.0)
        see_decision = make_trade_decision(trend, orderbook, vol, 100.0, config, levels, smart_entry=see)
        # Pre-pump should boost confidence
        self.assertGreater(see_decision.confidence, base_decision.confidence)
        self.assertIn("see_pre_pump", see_decision.reason)

    def test_pre_pump_ignored_on_sell(self):
        from bot.analysis import TrendResult, OrderbookInsight, VolatilityStats
        # Downtrend → sell action
        trend = TrendResult(direction="down", fast_ma=95.0, slow_ma=100.0, strength=0.05)
        orderbook = OrderbookInsight(spread_pct=0.001, bid_volume=80, ask_volume=100, imbalance=-0.1)
        vol = VolatilityStats(volatility=0.01, avg_volume=10)
        config = BotConfig(api_key=None)
        base = make_trade_decision(trend, orderbook, vol, 100.0, config)
        see = self._make_see(pre_pump=True, pp_score=1.0)
        see_dec = make_trade_decision(trend, orderbook, vol, 100.0, config, smart_entry=see)
        # Pre-pump should NOT affect sell
        self.assertAlmostEqual(base.confidence, see_dec.confidence, places=3)

    def test_whale_confirm_buy_boosts_confidence(self):
        trend, orderbook, vol, levels, config = self._base_inputs()
        base = make_trade_decision(trend, orderbook, vol, 100.0, config, levels)
        see = self._make_see(wp_detected=True, wp_side="buy", wp_pressure=3.0)
        see_dec = make_trade_decision(trend, orderbook, vol, 100.0, config, levels, smart_entry=see)
        self.assertGreater(see_dec.confidence, base.confidence)
        self.assertIn("see_whale_confirm", see_dec.reason)

    def test_whale_oppose_buy_reduces_confidence(self):
        trend, orderbook, vol, levels, config = self._base_inputs()
        base = make_trade_decision(trend, orderbook, vol, 100.0, config, levels)
        see = self._make_see(wp_detected=True, wp_side="sell", wp_pressure=-3.0)
        see_dec = make_trade_decision(trend, orderbook, vol, 100.0, config, levels, smart_entry=see)
        self.assertLess(see_dec.confidence, base.confidence)
        self.assertIn("see_whale_oppose", see_dec.reason)

    def test_fake_breakout_reduces_buy_confidence(self):
        trend, orderbook, vol, levels, config = self._base_inputs()
        base = make_trade_decision(trend, orderbook, vol, 100.0, config, levels)
        see = self._make_see(fb_detected=True, fb_score=0.8)
        see_dec = make_trade_decision(trend, orderbook, vol, 100.0, config, levels, smart_entry=see)
        self.assertLess(see_dec.confidence, base.confidence)
        self.assertIn("see_fake_breakout", see_dec.reason)

    def test_no_smart_entry_unchanged(self):
        """Passing smart_entry=None should produce the same result as omitting it."""
        trend, orderbook, vol, levels, config = self._base_inputs()
        base = make_trade_decision(trend, orderbook, vol, 100.0, config, levels)
        with_none = make_trade_decision(trend, orderbook, vol, 100.0, config, levels, smart_entry=None)
        self.assertEqual(base.confidence, with_none.confidence)
