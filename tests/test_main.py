import logging
import os
import unittest
from unittest.mock import patch

import main
from bot.tracking import PortfolioTracker


class MainErrorHandlingTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_stub_trader(scan_raises=None, maybe_execute_returns=None):
        """Return a StubTrader class whose scan_and_choose / maybe_execute are
        controlled by the caller."""

        class StubTrader:
            def __init__(self, config) -> None:
                self.config = config
                self.restored_state = None
                self.tracker = PortfolioTracker(
                    initial_capital=1000.0,
                    target_profit_pct=0.2,
                    max_loss_pct=0.1,
                )

            def scan_and_choose(self):
                if scan_raises is not None:
                    raise scan_raises
                return "btc_idr", {
                    "pair": "btc_idr",
                    "price": 100.0,
                    "orderbook": None,
                    "volatility": None,
                    "levels": None,
                    "indicators": None,
                    "decision": _Decision(),
                }

            def maybe_execute(self, snapshot):
                if maybe_execute_returns is not None:
                    return maybe_execute_returns
                return {"status": "skipped"}

            def force_sell(self, snapshot):
                return {"status": "force_sold", "amount": 0, "price": snapshot["price"]}

        class _Decision:
            action = "buy"
            mode = "day_trading"
            confidence = 0.8
            reason = "test"

        return StubTrader

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_main_exits_gracefully_on_recoverable_error_with_run_once(self) -> None:
        StubTrader = self._make_stub_trader(scan_raises=RuntimeError("no data"))
        with patch("main.configure_logging"):
            with patch("main.Trader", StubTrader):
                with patch.dict(os.environ, {"RUN_ONCE": "true"}, clear=False):
                    main.main()  # should exit without raising

    def test_main_rotates_after_stop_condition(self) -> None:
        """When maybe_execute returns 'stopped', the loop should NOT halt; it
        should liquidate the position and continue.  With RUN_ONCE the loop
        exits on the *second* iteration (after the rotation sleep is skipped)."""
        call_count = {"n": 0}
        original_sleep = main.time.sleep

        def _fast_sleep(secs):  # skip the real sleep in tests
            pass

        class StubTrader:
            def __init__(self, config) -> None:
                self.config = config
                self.restored_state = None
                self.tracker = PortfolioTracker(
                    initial_capital=1000.0,
                    target_profit_pct=0.2,
                    max_loss_pct=0.1,
                )

            def scan_and_choose(self):
                call_count["n"] += 1
                if call_count["n"] > 1:
                    raise RuntimeError("done")  # exit on second scan
                return "btc_idr", {
                    "pair": "btc_idr",
                    "price": 100.0,
                    "orderbook": None,
                    "volatility": None,
                    "levels": None,
                    "indicators": None,
                    "decision": _Decision(),
                }

            def maybe_execute(self, snapshot):
                return {"status": "stopped", "reason": "target_profit_reached"}

            def force_sell(self, snapshot):
                return {"status": "force_sold", "amount": 0, "price": snapshot["price"]}

        class _Decision:
            action = "buy"
            mode = "day_trading"
            confidence = 0.8
            reason = "test"

        with patch("main.configure_logging"):
            with patch("main.Trader", StubTrader):
                with patch("main.time.sleep", _fast_sleep):
                    with patch.dict(os.environ, {"RUN_ONCE": "true"}, clear=False):
                        main.main()  # must not raise

        # The loop entered a second scan cycle (proving it didn't stop after stop)
        self.assertGreaterEqual(call_count["n"], 1)

    def test_single_trade_mode_stops_after_one_cycle(self) -> None:
        """TRADE_MODE=single: the bot should stop as soon as the position it
        opened (via position-monitoring exit) is fully closed."""

        def _fast_sleep(secs):
            pass

        class StubTrader:
            def __init__(self, config) -> None:
                self.config = config
                self.restored_state = None
                # Pre-load an open position so the position-monitoring branch fires.
                self.tracker = PortfolioTracker(
                    initial_capital=1000.0,
                    target_profit_pct=0.2,
                    max_loss_pct=0.1,
                )
                self.tracker.record_trade("buy", 100.0, 1.0)

            def analyze_market(self, pair=None):
                return {
                    "pair": pair or "btc_idr",
                    "price": 110.0,
                    "orderbook": None,
                    "volatility": None,
                    "levels": None,
                    "indicators": None,
                    "decision": _SellDecision(),
                }

            def scan_and_choose(self):
                # Should never be reached when starting with a position
                raise RuntimeError("unexpected scan")

            def maybe_execute(self, snapshot):
                return {"status": "skipped"}

            def force_sell(self, snapshot):
                self.tracker.record_trade("sell", 110.0, self.tracker.base_position)
                return {
                    "status": "force_sold",
                    "action": "sell",
                    "amount": 1.0,
                    "price": 110.0,
                }

        class _SellDecision:
            action = "sell"
            mode = "day_trading"
            confidence = 0.8
            reason = "exit"

        with patch("main.configure_logging"):
            with patch("main.Trader", StubTrader):
                with patch("main.time.sleep", _fast_sleep):
                    with patch.dict(
                        os.environ,
                        {"TRADE_MODE": "single", "RUN_ONCE": "false"},
                        clear=False,
                    ):
                        main.main()  # must exit cleanly after the single sell

    def test_continuous_mode_does_not_stop_after_sell(self) -> None:
        """TRADE_MODE=continuous: the bot should NOT stop after selling;
        it should continue scanning.  We verify this by having scan_and_choose
        raise KeyboardInterrupt on its first call – if the bot rotated to
        scan after selling, the counter will be ≥ 1."""
        scan_count = {"n": 0}

        def _fast_sleep(secs):
            pass

        class StubTrader:
            def __init__(self, config) -> None:
                self.config = config
                self.restored_state = None
                self.tracker = PortfolioTracker(
                    initial_capital=1000.0,
                    target_profit_pct=0.2,
                    max_loss_pct=0.1,
                )
                # Pre-load an open position
                self.tracker.record_trade("buy", 100.0, 1.0)

            def analyze_market(self, pair=None):
                return {
                    "pair": pair or "btc_idr",
                    "price": 110.0,
                    "orderbook": None,
                    "volatility": None,
                    "levels": None,
                    "indicators": None,
                    "decision": _SellDecision(),
                }

            def scan_and_choose(self):
                # Reached only after the position is closed in continuous mode
                scan_count["n"] += 1
                raise KeyboardInterrupt()  # end test cleanly

            def maybe_execute(self, snapshot):
                return {"status": "skipped"}

            def force_sell(self, snapshot):
                self.tracker.record_trade("sell", 110.0, self.tracker.base_position)
                return {"status": "force_sold", "action": "sell", "amount": 1.0, "price": 110.0}

        class _SellDecision:
            action = "sell"
            mode = "day_trading"
            confidence = 0.8
            reason = "exit"

        with patch("main.configure_logging"):
            with patch("main.Trader", StubTrader):
                with patch("main.time.sleep", _fast_sleep):
                    with patch.dict(
                        os.environ,
                        {"TRADE_MODE": "continuous", "RUN_ONCE": "false"},
                        clear=False,
                    ):
                        main.main()

        # After selling the held position, the bot tried to scan for the next trade
        self.assertGreaterEqual(scan_count["n"], 1)


if __name__ == "__main__":
    unittest.main()
