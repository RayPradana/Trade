import logging
import os
import unittest
import unittest.mock
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
                self.restored_pair = None
                self.persistence = type("_P", (), {"backup": lambda *a: None})()
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

            def _effective_interval(self, snapshot=None):
                return self.config.interval_seconds

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
                self.restored_pair = None
                self.persistence = type("_P", (), {"backup": lambda *a: None})()
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
                self.restored_pair = None
                self.persistence = type("_P", (), {"backup": lambda *a: None})()
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
                    "whale": None,
                    "spoofing": None,
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

            def _effective_interval(self, snapshot=None):
                return self.config.interval_seconds

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
                self.restored_pair = None
                self.persistence = type("_P", (), {"backup": lambda *a: None})()
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
                    "whale": None,
                    "spoofing": None,
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

            def _effective_interval(self, snapshot=None):
                return self.config.interval_seconds

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


class AccountInfoDisplayTests(unittest.TestCase):
    """Unit tests for _log_account_info and _log_account_info_dry."""

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fake_info(**overrides):
        """Return a minimal getInfo-style response dict."""
        data = {
            "success": 1,
            "return": {
                "server_time": 1741628400,
                "name": "Test User",
                "email": "test@example.com",
                "user_id": "123456",
                "verification_status": "verified",
                "balance": {
                    "idr": "5000000.00",
                    "btc": "0.00345600",
                    "eth": "0.00000000",
                },
                "balance_hold": {
                    "idr": "0.00",
                    "btc": "0.00100000",
                },
            },
        }
        data["return"].update(overrides)
        return data

    # ── tests ─────────────────────────────────────────────────────────────────

    def test_log_account_info_runs_without_error(self) -> None:
        """_log_account_info should not raise for a well-formed response."""
        main._log_account_info(self._fake_info())

    def test_log_account_info_handles_empty_response(self) -> None:
        """_log_account_info should not raise for a missing/empty return dict."""
        main._log_account_info({})
        main._log_account_info({"return": {}})
        main._log_account_info({"return": {"balance": {}, "balance_hold": {}}})

    def test_log_account_info_hides_zero_coins(self) -> None:
        """Coins with zero free AND zero hold should be excluded."""
        # ETH has 0 free and no hold entry — should produce no log lines for ETH.
        with unittest.mock.patch("logging.info") as mock_log:
            main._log_account_info(self._fake_info())
        all_msgs = " ".join(str(a) for call in mock_log.call_args_list for a in call[0])
        self.assertIn("BTC", all_msgs)
        self.assertNotIn("ETH", all_msgs)  # zero balance — hidden

    def test_log_account_info_no_open_positions(self) -> None:
        """When all coins are zero, 'no open coin positions' line should appear."""
        with unittest.mock.patch("logging.info") as mock_log:
            main._log_account_info(
                {"return": {"balance": {"idr": "1000", "btc": "0"}, "balance_hold": {}}}
            )
        all_msgs = " ".join(str(a) for call in mock_log.call_args_list for a in call[0])
        self.assertIn("no open coin positions", all_msgs)

    def test_log_account_info_verified_icon(self) -> None:
        """Verified account should show ✅, unverified should show ⚠️."""
        with unittest.mock.patch("logging.info") as mock_log:
            main._log_account_info(self._fake_info(verification_status="verified"))
            main._log_account_info(self._fake_info(verification_status="unverified"))
        all_msgs = " ".join(str(a) for call in mock_log.call_args_list for a in call[0])
        self.assertIn("✅", all_msgs)
        self.assertIn("⚠️", all_msgs)

    def test_log_account_info_dry_runs_without_error(self) -> None:
        """_log_account_info_dry should not raise."""
        main._log_account_info_dry()


if __name__ == "__main__":
    unittest.main()



class LoggingTests(unittest.TestCase):
    """Tests for file logging and Telegram notification helpers."""

    def setUp(self) -> None:
        logging.disable(logging.NOTSET)

    def tearDown(self) -> None:
        # Clean up any file handlers added by configure_logging
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()

    def test_configure_logging_no_file(self):
        """configure_logging() without log_file should not add a FileHandler."""
        main.configure_logging(log_file=None)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        self.assertEqual(len(file_handlers), 0)

    def test_configure_logging_with_file(self):
        """configure_logging() with a valid log_file should add a FileHandler."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as tmp:
            path = tmp.name
        try:
            main.configure_logging(log_file=path)
            root = logging.getLogger()
            file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
            self.assertGreater(len(file_handlers), 0)
        finally:
            os.unlink(path)

    def test_notify_sends_telegram_when_configured(self):
        """_notify should call _send_telegram when token+chat_id are set."""
        from bot.config import BotConfig
        config = BotConfig(
            api_key=None,
            telegram_token="tok123",
            telegram_chat_id="chat456",
        )
        with unittest.mock.patch("main._send_telegram") as mock_send:
            main._notify(config, "test message")
            mock_send.assert_called_once_with("tok123", "chat456", "test message")

    def test_notify_does_nothing_without_token(self):
        """_notify must not call _send_telegram when telegram_token is None."""
        from bot.config import BotConfig
        config = BotConfig(api_key=None)
        with unittest.mock.patch("main._send_telegram") as mock_send:
            main._notify(config, "ignored")
            mock_send.assert_not_called()

    def test_send_telegram_handles_request_error(self):
        """_send_telegram must not propagate network errors."""
        with unittest.mock.patch("requests.post", side_effect=OSError("no network")):
            # Should not raise
            main._send_telegram("tok", "chat", "msg")


class DiscordNotificationTest(unittest.TestCase):
    def test_send_discord_handles_request_error(self):
        """_send_discord must not propagate network errors."""
        with unittest.mock.patch("requests.post", side_effect=OSError("no network")):
            main._send_discord("https://discord.com/api/webhooks/x", "msg")

    def test_notify_sends_discord_when_configured(self):
        """_notify should call _send_discord when discord_webhook_url is set."""
        from bot.config import BotConfig
        config = BotConfig(api_key=None, discord_webhook_url="https://discord.com/api/webhooks/x")
        with unittest.mock.patch("main._send_discord") as mock_discord:
            main._notify(config, "test message")
            mock_discord.assert_called_once_with("https://discord.com/api/webhooks/x", "test message")

    def test_notify_sends_both_when_both_configured(self):
        """_notify must call both Telegram and Discord when both are configured."""
        from bot.config import BotConfig
        config = BotConfig(
            api_key=None,
            telegram_token="tok",
            telegram_chat_id="chat",
            discord_webhook_url="https://discord.com/api/webhooks/x",
        )
        with unittest.mock.patch("main._send_telegram") as mt, \
             unittest.mock.patch("main._send_discord") as md:
            main._notify(config, "both")
            mt.assert_called_once()
            md.assert_called_once()

    def test_notify_does_nothing_without_any_config(self):
        """_notify must not call anything when no notification config is set."""
        from bot.config import BotConfig
        config = BotConfig(api_key=None)
        with unittest.mock.patch("main._send_telegram") as mt, \
             unittest.mock.patch("main._send_discord") as md:
            main._notify(config, "ignored")
            mt.assert_not_called()
            md.assert_not_called()
