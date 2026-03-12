import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from bot.config import BotConfig
from bot.persistence import StatePersistence
from bot.tracking import PortfolioTracker
from bot.trader import Trader


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

    def test_backup_creates_copy(self) -> None:
        """backup() must create a copy of the state file at the given path."""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            backup_path = Path(tmp) / "state_backup.json"
            persistence = StatePersistence(state_path)
            persistence.save({"portfolio": {}, "pair": "btc_idr"})
            self.assertTrue(state_path.exists())

            persistence.backup(backup_path)
            self.assertTrue(backup_path.exists())
            # Backup content must match the original
            import json
            orig = json.loads(state_path.read_text())
            bak = json.loads(backup_path.read_text())
            self.assertEqual(orig["pair"], bak["pair"])

    def test_backup_does_nothing_when_no_state_file(self) -> None:
        """backup() must be a no-op when the state file does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            persistence = StatePersistence(Path(tmp) / "missing.json")
            backup_path = Path(tmp) / "backup.json"
            persistence.backup(backup_path)  # must not raise
            self.assertFalse(backup_path.exists())

    def test_none_path_is_noop(self) -> None:
        """StatePersistence(None) must silently no-op for all operations."""
        p = StatePersistence(None)
        p.save({"key": "value"})   # must not raise
        p.clear()                  # must not raise
        p.backup(Path("/tmp/x"))   # must not raise
        self.assertIsNone(p.load())


# ---------------------------------------------------------------------------
# Trader auto-resume integration tests
# ---------------------------------------------------------------------------

class _FakeClient:
    """Minimal client stub for auto-resume tests."""

    def get_pairs(self) -> list:
        return [{"name": "btc_idr"}]

    def get_summaries(self) -> dict:
        return {}

    def get_depth(self, pair: str, count: int = 5) -> Dict[str, Any]:
        return {"buy": [["100", "1"]], "sell": [["100.05", "1"]]}


class _BuyTrader(Trader):
    """Trader that always returns a buy signal for maybe_execute tests."""

    def analyze_market(self, pair=None, prefetched_ticker=None) -> Dict[str, Any]:
        from bot.strategies import StrategyDecision
        return {
            "pair": pair or "btc_idr",
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "decision": StrategyDecision(
                mode="scalping", action="buy", confidence=0.8, reason="test",
                target_price=100.0, amount=1.0, stop_loss=99.0, take_profit=101.0,
            ),
        }


class TraderAutoResumeTests(unittest.TestCase):
    """Tests for Trader auto-resume / state-persistence behaviour."""

    def test_trader_saves_state_after_buy(self) -> None:
        """State file must exist after a successful dry-run buy."""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "bot_state.json"
            config = BotConfig(api_key=None, dry_run=True, state_path=state_path,
                               staged_entry_steps=1, min_order_idr=1.0, multi_position_enabled=False)  # disable min-order guard
            trader = _BuyTrader(config, client=_FakeClient())
            snapshot = trader.analyze_market("btc_idr")
            trader.maybe_execute(snapshot)
            self.assertTrue(state_path.exists(), "State file must be written after a buy")
            persistence = StatePersistence(state_path)
            state = persistence.load()
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.get("pair"), "btc_idr")
            self.assertGreater(state["portfolio"]["base_position"], 0)

    def test_trader_clears_state_after_full_sell(self) -> None:
        """State file must be removed after the entire position is sold."""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "bot_state.json"
            config = BotConfig(api_key=None, dry_run=True, state_path=state_path, staged_entry_steps=1, multi_position_enabled=False)
            trader = _BuyTrader(config, client=_FakeClient())
            # Manually open a position and save state
            trader.tracker.record_trade("buy", 100.0, 2.0)
            trader._save_state("btc_idr")
            self.assertTrue(state_path.exists())
            # Now force-sell the entire position
            snapshot = {"pair": "btc_idr", "price": 100.0}
            trader.force_sell(snapshot)
            self.assertFalse(state_path.exists(), "State file must be removed after full sell")

    def test_trader_restores_pair_on_init(self) -> None:
        """Trader must restore the active pair and position from a saved state file."""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "bot_state.json"
            # Write a state file with an open position
            persistence = StatePersistence(state_path)
            tracker = PortfolioTracker(initial_capital=1_000_000, target_profit_pct=0.2, max_loss_pct=0.1)
            tracker.record_trade("buy", 1_000_000.0, 0.5)
            persistence.save({"portfolio": tracker.to_state(), "pair": "eth_idr", "dry_run": True})
            # Create a fresh Trader — must load the saved state
            config = BotConfig(api_key=None, dry_run=True, state_path=state_path)
            fresh_trader = Trader(config, client=_FakeClient())
            self.assertEqual(fresh_trader.restored_pair, "eth_idr")
            self.assertAlmostEqual(fresh_trader.tracker.base_position, 0.5, places=6)

    def test_trader_no_restore_when_dry_run_mismatch(self) -> None:
        """State saved under dry_run=True must NOT be loaded when running live (dry_run=False)."""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "bot_state.json"
            # Save a dry-run state
            persistence = StatePersistence(state_path)
            tracker = PortfolioTracker(initial_capital=1_000_000, target_profit_pct=0.2, max_loss_pct=0.1)
            tracker.record_trade("buy", 1_000_000.0, 0.5)
            persistence.save({"portfolio": tracker.to_state(), "pair": "btc_idr", "dry_run": True})
            # Create a live Trader (dry_run=False) — must NOT restore the virtual state
            config = BotConfig(api_key="fake_key", api_secret="fake_secret",
                               dry_run=False, state_path=state_path)
            fresh_trader = Trader(config, client=_FakeClient())
            self.assertIsNone(fresh_trader.restored_pair)
            self.assertEqual(fresh_trader.tracker.base_position, 0.0)

    def test_stale_state_cleared_when_no_position(self) -> None:
        """A saved state with base_position=0 must be cleared on startup."""
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "bot_state.json"
            persistence = StatePersistence(state_path)
            tracker = PortfolioTracker(initial_capital=1_000_000, target_profit_pct=0.2, max_loss_pct=0.1)
            # Save state with no open position
            persistence.save({"portfolio": tracker.to_state(), "pair": "btc_idr", "dry_run": True})
            self.assertTrue(state_path.exists())
            config = BotConfig(api_key=None, dry_run=True, state_path=state_path)
            Trader(config, client=_FakeClient())
            self.assertFalse(state_path.exists(), "Stale zero-position state file must be deleted on init")


if __name__ == "__main__":
    unittest.main()
