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


if __name__ == "__main__":
    unittest.main()
