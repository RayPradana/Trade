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


class AutoPairsTrader(Trader):
    """Trader that auto-loads pairs from stubbed client and analyzes provided snapshots."""

    class _Client:
        def __init__(self, pairs: list[str]) -> None:
            self._pairs = pairs

        def get_pairs(self) -> list[Dict[str, Any]]:
            return [{"name": p} for p in self._pairs]

    def __init__(self, config: BotConfig, snapshots: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(config, client=self._Client(list(snapshots.keys())))
        self._snapshots = snapshots

    def analyze_market(self, pair: str | None = None) -> Dict[str, Any]:
        key = pair or self.config.pair
        return self._snapshots[key]


class TraderSelectionTests(unittest.TestCase):
    def test_scan_and_choose_picks_best_confidence(self) -> None:
        config = BotConfig(api_key=None, scan_pairs=["a_idr", "b_idr"], pair="a_idr", auto_resume=False)
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

    def test_scan_and_choose_without_manual_input_uses_auto_pairs(self) -> None:
        config = BotConfig(api_key=None, scan_pairs=None, pair="manual_idr", auto_resume=False)
        snapshots = {
            "auto_a": {
                "pair": "auto_a",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="swing_trading",
                    action="buy",
                    confidence=0.6,
                    reason="ok",
                    target_price=100,
                    amount=0.1,
                    stop_loss=95,
                    take_profit=110,
                ),
            },
            "auto_b": {
                "pair": "auto_b",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="buy",
                    confidence=0.9,
                    reason="better",
                    target_price=100,
                    amount=0.1,
                    stop_loss=95,
                    take_profit=110,
                ),
            },
        }
        trader = AutoPairsTrader(config, snapshots)
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "auto_b")
        self.assertEqual(snapshot["decision"].confidence, 0.9)
        self.assertEqual(trader.config.pair, "manual_idr")  # config stays as fallback

    def test_scan_and_choose_falls_back_when_all_hold(self) -> None:
        config = BotConfig(api_key=None, scan_pairs=["a_idr", "b_idr"], pair="fallback_idr", auto_resume=False)
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
                    action="hold",
                    confidence=0.4,
                    reason="hold",
                    target_price=100,
                    amount=0.0,
                    stop_loss=None,
                    take_profit=None,
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
                    action="hold",
                    confidence=0.3,
                    reason="hold",
                    target_price=100,
                    amount=0.0,
                    stop_loss=None,
                    take_profit=None,
                ),
            },
            "fallback_idr": {
                "pair": "fallback_idr",
                "price": 120.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="position_trading",
                    action="buy",
                    confidence=0.6,
                    reason="fallback",
                    target_price=120,
                    amount=0.2,
                    stop_loss=110,
                    take_profit=130,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "fallback_idr")
        self.assertEqual(snapshot["decision"].action, "buy")

    def test_maybe_execute_limits_buy_amount_by_available_cash(self) -> None:
        config = BotConfig(
            api_key=None,
            dry_run=True,
            initial_capital=50.0,
            max_loss_pct=0.9,
            target_profit_pct=1.0,
            auto_resume=False,
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
