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
