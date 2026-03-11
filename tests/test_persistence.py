import tempfile
import unittest
from pathlib import Path

from bot.config import BotConfig
from bot.persistence import StatePersistence
from bot.trader import Trader
from bot.tracking import PortfolioTracker


class DummyClient:
    def __init__(self) -> None:
        pass


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_portfolio_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.2, max_loss_pct=0.1)
            tracker.cash = 900
            tracker.base_position = 2
            tracker.avg_cost = 50
            tracker.realized_pnl = 20
            persistence = StatePersistence(state_path)
            persistence.save({"portfolio": tracker.to_state(), "pair": "btc_idr"})

            loaded = persistence.load()
            self.assertIsNotNone(loaded)
            restored_tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.2, max_loss_pct=0.1)
            restored_tracker.load_state(loaded["portfolio"])
            self.assertEqual(restored_tracker.cash, 900)
            self.assertEqual(restored_tracker.base_position, 2)
            self.assertEqual(restored_tracker.avg_cost, 50)

    def test_trader_auto_resume_applies_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            persistence = StatePersistence(state_path)
            persistence.save(
                {
                    "pair": "eth_idr",
                    "portfolio": {"cash": 750.0, "base_position": 1.5, "avg_cost": 12000.0, "realized_pnl": 15.0},
                }
            )
            config = BotConfig(api_key=None, state_file=str(state_path), auto_resume=True)
            trader = Trader(config, client=DummyClient())  # type: ignore[arg-type]
            self.assertEqual(trader.tracker.cash, 750.0)
            self.assertEqual(trader.config.pair, "eth_idr")


if __name__ == "__main__":
    unittest.main()
