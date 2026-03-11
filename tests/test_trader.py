import unittest
from typing import Dict, Any

from bot.config import BotConfig
from bot.strategies import StrategyDecision
from bot.trader import Trader
from bot.tracking import PortfolioTracker


class StubTrader(Trader):
    def __init__(self, config: BotConfig, snapshots: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(config, client=None)  # client not used due to override
        self._snapshots = snapshots

    def analyze_market(self, pair: str | None = None) -> Dict[str, Any]:
        key = pair or self.config.pair
        return self._snapshots[key]


class GuardedTrader(Trader):
    """Trader with stubbed client to test balance guards."""

    class _Client:
        def get_depth(self, pair: str, count: int = 5) -> Dict[str, Any]:
            return {"buy": [["100", "1"]], "sell": [["100.05", "1"]]}

    def __init__(self, config: BotConfig) -> None:
        super().__init__(config, client=self._Client())


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

    def test_maybe_execute_limits_buy_amount_by_available_cash(self) -> None:
        config = BotConfig(
            api_key=None,
            api_secret=None,
            dry_run=True,
            initial_capital=50.0,
            max_loss_pct=0.9,
            target_profit_pct=1.0,
        )
        trader = GuardedTrader(config)
        trader.tracker.cash = 50.0  # very small cash
        decision = StrategyDecision(
            mode="day_trading",
            action="buy",
            confidence=1.0,
            reason="test",
            target_price=100,
            amount=10.0,
            stop_loss=95.0,
            take_profit=105.0,
        )
        snapshot = {
            "pair": "btc_idr",
            "price": 100.0,
            "decision": decision,
        }
        outcome = trader.maybe_execute(snapshot)
        self.assertEqual(outcome["status"], "simulated")
        self.assertLessEqual(outcome["amount"], 0.5)  # 50 cash / 100 price


if __name__ == "__main__":
    unittest.main()
