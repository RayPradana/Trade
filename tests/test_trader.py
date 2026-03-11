import unittest
from typing import Dict, Any

from bot.config import BotConfig
from bot.strategies import StrategyDecision
from bot.trader import Trader


class StubTrader(Trader):
    def __init__(self, config: BotConfig, snapshots: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(config, client=None)  # client not used due to override
        self._snapshots = snapshots

    def analyze_market(self) -> Dict[str, Any]:
        pair = self.config.pair
        return self._snapshots[pair]


class TraderSelectionTests(unittest.TestCase):
    def test_scan_and_choose_picks_best_confidence(self) -> None:
        config = BotConfig(api_key=None, api_secret=None, scan_pairs=["a_idr", "b_idr"], pair="a_idr")
        snapshots = {
            "a_idr": {
                "pair": "a_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="buy",
                    confidence=0.4,
                    reason="low",
                    target_price=100,
                    amount=0.1,
                    stop_loss=99,
                    take_profit=101,
                ),
            },
            "b_idr": {
                "pair": "b_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="buy",
                    confidence=0.8,
                    reason="high",
                    target_price=100,
                    amount=0.1,
                    stop_loss=99,
                    take_profit=101,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "b_idr")
        self.assertEqual(snapshot["decision"].confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
