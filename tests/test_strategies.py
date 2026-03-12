import unittest

from bot.config import BotConfig
from bot.grid import build_grid_plan
from bot.strategies import StrategyDecision, make_trade_decision, select_strategy, _position_size, confidence_position_pct
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


class AdaptiveSizingTests(unittest.TestCase):
    """Tests for adaptive_risk_per_trade() and adaptive_max_positions()."""

    from bot.strategies import adaptive_risk_per_trade, adaptive_max_positions

    def _cfg(self, **kwargs) -> BotConfig:
        defaults = dict(
            api_key=None, dry_run=True,
            adaptive_sizing_enabled=True,
            adaptive_tier1_equity=2_000_000.0,
            adaptive_tier2_equity=5_000_000.0,
            adaptive_tier0_risk=0.10,
            adaptive_tier1_risk=0.07,
            adaptive_tier2_risk=0.03,
            adaptive_tier0_max_pos=3,
            adaptive_tier1_max_pos=4,
            adaptive_tier2_max_pos=5,
        )
        defaults.update(kwargs)
        return BotConfig(**defaults)

    def test_small_cap_risk(self):
        """Equity below tier1 threshold → tier0 risk."""
        from bot.strategies import adaptive_risk_per_trade
        cfg = self._cfg()
        self.assertAlmostEqual(adaptive_risk_per_trade(500_000.0, cfg), 0.10)

    def test_medium_cap_risk(self):
        """Equity between tier1 and tier2 → tier1 risk."""
        from bot.strategies import adaptive_risk_per_trade
        cfg = self._cfg()
        self.assertAlmostEqual(adaptive_risk_per_trade(3_000_000.0, cfg), 0.07)

    def test_large_cap_risk(self):
        """Equity above tier2 → tier2 risk."""
        from bot.strategies import adaptive_risk_per_trade
        cfg = self._cfg()
        self.assertAlmostEqual(adaptive_risk_per_trade(10_000_000.0, cfg), 0.03)

    def test_boundary_tier1(self):
        """Equity exactly at tier1 threshold should use tier1 risk (medium cap)."""
        from bot.strategies import adaptive_risk_per_trade
        cfg = self._cfg()
        self.assertAlmostEqual(adaptive_risk_per_trade(2_000_000.0, cfg), 0.07)

    def test_boundary_tier2(self):
        """Equity exactly at tier2 threshold should use tier2 risk (large cap)."""
        from bot.strategies import adaptive_risk_per_trade
        cfg = self._cfg()
        self.assertAlmostEqual(adaptive_risk_per_trade(5_000_000.0, cfg), 0.03)

    def test_disabled_returns_static_risk(self):
        """When adaptive_sizing_enabled=False, static risk_per_trade is returned."""
        from bot.strategies import adaptive_risk_per_trade
        cfg = self._cfg(adaptive_sizing_enabled=False, risk_per_trade=0.02)
        self.assertAlmostEqual(adaptive_risk_per_trade(500_000.0, cfg), 0.02)

    def test_small_cap_max_pos(self):
        """Equity < tier1 → tier0 max positions."""
        from bot.strategies import adaptive_max_positions
        cfg = self._cfg()
        self.assertEqual(adaptive_max_positions(500_000.0, cfg), 3)

    def test_medium_cap_max_pos(self):
        """Equity in tier1 → tier1 max positions."""
        from bot.strategies import adaptive_max_positions
        cfg = self._cfg()
        self.assertEqual(adaptive_max_positions(3_000_000.0, cfg), 4)

    def test_large_cap_max_pos(self):
        """Equity >= tier2 → tier2 max positions."""
        from bot.strategies import adaptive_max_positions
        cfg = self._cfg()
        self.assertEqual(adaptive_max_positions(10_000_000.0, cfg), 5)

    def test_disabled_returns_static_max_pos(self):
        """When adaptive_sizing_enabled=False, static max_open_positions is returned."""
        from bot.strategies import adaptive_max_positions
        cfg = self._cfg(adaptive_sizing_enabled=False, max_open_positions=7)
        self.assertEqual(adaptive_max_positions(500_000.0, cfg), 7)

    def test_position_size_larger_for_small_cap(self):
        """With adaptive sizing, small capital should produce larger relative position than without."""
        from bot.analysis import VolatilityStats
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        cfg_adaptive = self._cfg(initial_capital=500_000.0)
        cfg_static = self._cfg(adaptive_sizing_enabled=False, risk_per_trade=0.01, initial_capital=500_000.0)
        # With adaptive: risk = 10% of 500k
        # With static: risk = 1% of 500k → much smaller
        size_adaptive = _position_size(50_000.0, 49_000.0, cfg_adaptive, 1000.0, 0.7, vol, 500_000.0)
        size_static = _position_size(50_000.0, 49_000.0, cfg_static, 1000.0, 0.7, vol, 500_000.0)
        self.assertGreater(size_adaptive, size_static)

    def test_adaptive_note_in_reason(self):
        """Decision reason should include adaptive_risk when adaptive sizing is enabled."""
        from bot.analysis import OrderbookInsight, TrendResult, VolatilityStats
        trend = TrendResult(direction="up", fast_ma=102.0, slow_ma=100.0, strength=0.02)
        ob = OrderbookInsight(spread_pct=0.001, bid_volume=100.0, ask_volume=70.0, imbalance=0.3)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        cfg = self._cfg(initial_capital=500_000.0)
        dec = make_trade_decision(trend, ob, vol, 50_000.0, cfg, effective_capital=500_000.0)
        self.assertIn("adaptive_risk=", dec.reason)


class BuyFilterTests(unittest.TestCase):
    """Tests for hard-skip buy filters: RSI overbought and resistance proximity."""

    def _base_inputs(self):
        trend = TrendResult(direction="up", fast_ma=102.0, slow_ma=100.0, strength=0.02)
        ob = OrderbookInsight(spread_pct=0.001, bid_volume=100.0, ask_volume=70.0, imbalance=0.3)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        return trend, ob, vol

    def test_rsi_overbought_forces_hold(self):
        """When RSI >= buy_max_rsi and buy_max_rsi > 0, action must be hold."""
        from bot.analysis import MomentumIndicators
        trend, ob, vol = self._base_inputs()
        cfg = BotConfig(api_key=None, dry_run=True, buy_max_rsi=85.0)
        ind = MomentumIndicators(rsi=90.0, macd=0.0, macd_signal=0.0, macd_hist=0.0, bb_upper=None, bb_mid=None, bb_lower=None)
        dec = make_trade_decision(trend, ob, vol, 100.0, cfg, indicators=ind)
        self.assertEqual(dec.action, "hold")
        self.assertIn("rsi_overbought", dec.reason)

    def test_rsi_below_threshold_allows_buy(self):
        """When RSI < buy_max_rsi, buy should not be blocked."""
        from bot.analysis import MomentumIndicators
        trend, ob, vol = self._base_inputs()
        cfg = BotConfig(api_key=None, dry_run=True, buy_max_rsi=85.0, min_confidence=0.0)
        ind = MomentumIndicators(rsi=60.0, macd=0.0, macd_signal=0.0, macd_hist=0.0, bb_upper=None, bb_mid=None, bb_lower=None)
        dec = make_trade_decision(trend, ob, vol, 100.0, cfg, indicators=ind)
        self.assertEqual(dec.action, "buy")

    def test_rsi_filter_disabled_when_zero(self):
        """buy_max_rsi=0 disables the filter even when RSI is 100."""
        from bot.analysis import MomentumIndicators
        trend, ob, vol = self._base_inputs()
        cfg = BotConfig(api_key=None, dry_run=True, buy_max_rsi=0.0, min_confidence=0.0)
        ind = MomentumIndicators(rsi=100.0, macd=0.0, macd_signal=0.0, macd_hist=0.0, bb_upper=None, bb_mid=None, bb_lower=None)
        dec = make_trade_decision(trend, ob, vol, 100.0, cfg, indicators=ind)
        self.assertEqual(dec.action, "buy")

    def test_resistance_proximity_forces_hold(self):
        """When price is within buy_max_resistance_proximity_pct of resistance, force hold."""
        from bot.analysis import SupportResistance
        trend, ob, vol = self._base_inputs()
        # price=99, resistance=100 → distance = 1/100 = 1% → equals threshold → blocked
        cfg = BotConfig(api_key=None, dry_run=True, buy_max_resistance_proximity_pct=0.01)
        levels = SupportResistance(support=90.0, resistance=100.0, lookback=20)
        dec = make_trade_decision(trend, ob, vol, 99.0, cfg, levels=levels)
        self.assertEqual(dec.action, "hold")
        self.assertIn("too_close_to_resistance", dec.reason)

    def test_resistance_proximity_far_allows_buy(self):
        """When price is far from resistance, buy should not be blocked."""
        from bot.analysis import SupportResistance
        trend, ob, vol = self._base_inputs()
        # price=95, resistance=100 → distance = 5% > threshold 1%
        cfg = BotConfig(api_key=None, dry_run=True, buy_max_resistance_proximity_pct=0.01, min_confidence=0.0)
        levels = SupportResistance(support=85.0, resistance=100.0, lookback=20)
        dec = make_trade_decision(trend, ob, vol, 95.0, cfg, levels=levels)
        self.assertEqual(dec.action, "buy")

    def test_resistance_proximity_disabled_when_zero(self):
        """buy_max_resistance_proximity_pct=0 disables filter at any distance."""
        from bot.analysis import SupportResistance
        trend, ob, vol = self._base_inputs()
        cfg = BotConfig(api_key=None, dry_run=True, buy_max_resistance_proximity_pct=0.0, min_confidence=0.0)
        levels = SupportResistance(support=90.0, resistance=100.0, lookback=20)
        dec = make_trade_decision(trend, ob, vol, 99.9, cfg, levels=levels)
        self.assertEqual(dec.action, "buy")


class TestConfidencePositionSizing(unittest.TestCase):
    """Tests for confidence_position_pct() and _position_size() with confidence-tier mode."""

    def _cfg(self, **kw) -> BotConfig:
        defaults = dict(
            api_key=None, dry_run=True,
            confidence_position_sizing_enabled=True,
            confidence_tier_skip=0.40,
            confidence_tier_low=0.50,
            confidence_tier_mid=0.65,
            confidence_tier_high=0.80,
            confidence_tier_low_pct=0.10,
            confidence_tier_mid_pct=0.15,
            confidence_tier_high_pct=0.20,
            confidence_tier_max_pct=0.25,
            initial_capital=500_000.0,
        )
        defaults.update(kw)
        return BotConfig(**defaults)

    # ── confidence_position_pct() ────────────────────────────────────────────

    def test_below_skip_returns_zero(self):
        cfg = self._cfg()
        self.assertEqual(confidence_position_pct(0.30, cfg), 0.0)
        self.assertEqual(confidence_position_pct(0.39, cfg), 0.0)

    def test_at_skip_boundary_returns_low_pct(self):
        """Exactly at tier_skip → enters the low tier (not skip)."""
        cfg = self._cfg()
        self.assertEqual(confidence_position_pct(0.40, cfg), 0.10)

    def test_low_tier_range(self):
        cfg = self._cfg()
        self.assertEqual(confidence_position_pct(0.45, cfg), 0.10)
        self.assertEqual(confidence_position_pct(0.499, cfg), 0.10)

    def test_mid_tier_range(self):
        cfg = self._cfg()
        self.assertEqual(confidence_position_pct(0.50, cfg), 0.15)
        self.assertEqual(confidence_position_pct(0.60, cfg), 0.15)
        self.assertEqual(confidence_position_pct(0.649, cfg), 0.15)

    def test_high_tier_range(self):
        cfg = self._cfg()
        self.assertEqual(confidence_position_pct(0.65, cfg), 0.20)
        self.assertEqual(confidence_position_pct(0.79, cfg), 0.20)

    def test_max_tier_at_and_above_high(self):
        cfg = self._cfg()
        self.assertEqual(confidence_position_pct(0.80, cfg), 0.25)
        self.assertEqual(confidence_position_pct(1.00, cfg), 0.25)

    # ── _position_size() with confidence_position_sizing_enabled ─────────────

    def test_position_size_tier_mid_at_price(self):
        """Mid tier (conf=0.55, 15 %) on 500k capital at price 100k = 0.75 units."""
        cfg = self._cfg()
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price = 100_000.0
        size = _position_size(price, price * 0.99, cfg, price * 0.01, 0.55, vol, 500_000.0)
        expected = (0.15 * 500_000.0) / price
        self.assertAlmostEqual(size, expected, places=8)

    def test_position_size_high_tier(self):
        """High tier (conf=0.70, 20 %) on 500k capital at price 91_496."""
        cfg = self._cfg()
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price = 91_496.0
        size = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0)
        expected = (0.20 * 500_000.0) / price
        self.assertAlmostEqual(size, expected, places=8)

    def test_position_size_below_skip_returns_zero(self):
        """Confidence below tier_skip → position size = 0."""
        cfg = self._cfg()
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        size = _position_size(50_000.0, 49_000.0, cfg, 1000.0, 0.35, vol, 500_000.0)
        self.assertEqual(size, 0.0)

    def test_position_size_zero_price_returns_zero(self):
        """Zero price should not cause division by zero regardless of mode."""
        cfg = self._cfg()
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        size = _position_size(0.0, 0.0, cfg, 0.0, 0.80, vol)
        self.assertEqual(size, 0.0)

    def test_feature_disabled_uses_original_formula(self):
        """When confidence_position_sizing_enabled=False, original formula is used."""
        from bot.analysis import VolatilityStats
        cfg_off = BotConfig(api_key=None, confidence_position_sizing_enabled=False,
                            initial_capital=500_000.0)
        cfg_on = self._cfg(initial_capital=500_000.0)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price, stop = 50_000.0, 49_000.0
        size_off = _position_size(price, stop, cfg_off, price - stop, 0.70, vol, 500_000.0)
        size_on = _position_size(price, stop, cfg_on, price - stop, 0.70, vol, 500_000.0)
        # Both should be positive but they use different formulas → values differ
        self.assertGreater(size_off, 0.0)
        self.assertGreater(size_on, 0.0)
        self.assertNotAlmostEqual(size_off, size_on, places=4)

    def test_idr_value_matches_expected_pct_of_capital(self):
        """IDR value of position should equal tier_pct * capital."""
        cfg = self._cfg()
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        capital = 500_000.0
        price = 91_496.0
        # confidence=0.82 → max tier → 25 %
        size = _position_size(price, price * 0.99, cfg, price * 0.01, 0.82, vol, capital)
        idr_value = size * price
        self.assertAlmostEqual(idr_value, 0.25 * capital, delta=1.0)


class TestObImbalanceSizeBoost(unittest.TestCase):
    """Tests for the OB imbalance position-size boost in _position_size()."""

    def _cfg(self, threshold: float = 0.5, multiplier: float = 2.0, **kwargs) -> BotConfig:
        return BotConfig(
            api_key=None,
            confidence_position_sizing_enabled=True,
            confidence_tier_skip=0.40,
            confidence_tier_low=0.50,
            confidence_tier_mid=0.65,
            confidence_tier_high=0.80,
            confidence_tier_low_pct=0.10,
            confidence_tier_mid_pct=0.15,
            confidence_tier_high_pct=0.20,
            confidence_tier_max_pct=0.25,
            ob_imbalance_boost_threshold=threshold,
            ob_imbalance_size_multiplier=multiplier,
            **kwargs,
        )

    def test_boost_applied_when_imbalance_meets_threshold(self):
        """When imbalance >= threshold, size should be multiplied."""
        cfg = self._cfg(threshold=0.5, multiplier=2.0, initial_capital=500_000.0)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price = 100_000.0
        # confidence=0.70 → high tier → 20% of 500k = 100k IDR
        base_size = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.0)
        boosted_size = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.5)
        self.assertAlmostEqual(boosted_size, base_size * 2.0, places=8)

    def test_boost_not_applied_below_threshold(self):
        """When imbalance < threshold, size should not be multiplied."""
        cfg = self._cfg(threshold=0.5, multiplier=2.0, initial_capital=500_000.0)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price = 100_000.0
        base_size = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.0)
        below_thresh = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.49)
        self.assertAlmostEqual(below_thresh, base_size, places=8)

    def test_boost_disabled_when_threshold_zero(self):
        """When ob_imbalance_boost_threshold=0 (disabled), size is unaffected."""
        cfg = self._cfg(threshold=0.0, multiplier=2.0, initial_capital=500_000.0)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price = 100_000.0
        base = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.0)
        high_imbalance = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.99)
        self.assertAlmostEqual(base, high_imbalance, places=8)

    def test_boost_idr_value_doubles(self):
        """When boosted 2×, IDR value of position should double vs. non-boosted."""
        cfg = self._cfg(threshold=0.5, multiplier=2.0, initial_capital=500_000.0)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price, capital = 91_496.0, 500_000.0
        # confidence=0.70 → 20% of 500k = 100k IDR without boost
        size_no_boost = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, capital, ob_imbalance=0.0)
        size_boosted = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, capital, ob_imbalance=0.5)
        self.assertAlmostEqual(size_boosted * price, size_no_boost * price * 2.0, delta=1.0)

    def test_boost_exact_threshold_triggers(self):
        """Imbalance exactly equal to the threshold should trigger the boost."""
        cfg = self._cfg(threshold=0.5, multiplier=3.0, initial_capital=500_000.0)
        vol = VolatilityStats(volatility=0.01, avg_volume=1000.0)
        price = 100_000.0
        base = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.0)
        boosted = _position_size(price, price * 0.99, cfg, price * 0.01, 0.70, vol, 500_000.0, ob_imbalance=0.5)
        self.assertAlmostEqual(boosted, base * 3.0, places=8)
