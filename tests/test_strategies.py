import unittest

from bot.config import BotConfig
from bot.strategies import StrategyDecision, make_trade_decision, select_strategy
from bot.analysis import OrderbookInsight, TrendResult, VolatilityStats, SupportResistance


class StrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = BotConfig(api_key=None, api_secret=None, dry_run=True)

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


if __name__ == "__main__":
    unittest.main()
