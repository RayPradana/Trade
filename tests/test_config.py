import os
from unittest import TestCase
from unittest.mock import patch

from bot.config import BotConfig


class ConfigRealTimeTest(TestCase):
    def test_realtime_sets_default_interval_to_one_second(self):
        with patch.dict(os.environ, {"REALTIME_MODE": "true"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertTrue(cfg.real_time)
            self.assertEqual(cfg.interval_seconds, 1)

    def test_realtime_respects_custom_interval(self):
        with patch.dict(os.environ, {"REALTIME_MODE": "true", "INTERVAL_SECONDS": "2"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertTrue(cfg.real_time)
            self.assertEqual(cfg.interval_seconds, 2)

    def test_api_secret_loaded_from_env(self):
        with patch.dict(os.environ, {"INDODAX_SECRET": "mysecret"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.api_secret, "mysecret")

    def test_trailing_stop_defaults_to_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.trailing_stop_pct, 0.0)

    def test_trailing_stop_loaded_from_env(self):
        with patch.dict(os.environ, {"TRAILING_STOP_PCT": "0.02"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.trailing_stop_pct, 0.02)

    def test_require_auth_raises_without_secret(self):
        cfg = BotConfig(api_key="key", api_secret=None)
        with self.assertRaises(ValueError):
            cfg.require_auth()

    def test_require_auth_passes_with_key_and_secret(self):
        cfg = BotConfig(api_key="key", api_secret="secret")
        cfg.require_auth()  # should not raise

    def test_trade_mode_defaults_to_continuous(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.trade_mode, "continuous")

    def test_trade_mode_single_loaded_from_env(self):
        with patch.dict(os.environ, {"TRADE_MODE": "single"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.trade_mode, "single")

    def test_trade_mode_invalid_raises(self):
        with patch.dict(os.environ, {"TRADE_MODE": "invalid"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()


class NewConfigFieldsTest(TestCase):
    def test_min_volume_idr_default(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.min_volume_idr, 0.0)

    def test_min_volume_idr_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"MIN_VOLUME_IDR": "500000000"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.min_volume_idr, 500_000_000.0)

    def test_min_volume_idr_negative_raises(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"MIN_VOLUME_IDR": "-1"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_log_file_default_none(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertIsNone(cfg.log_file)

    def test_log_file_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"LOG_FILE": "/tmp/bot.log"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.log_file, "/tmp/bot.log")

    def test_telegram_fields_default_none(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertIsNone(cfg.telegram_token)
            self.assertIsNone(cfg.telegram_chat_id)

    def test_telegram_fields_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ,
                        {"TELEGRAM_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "456"},
                        clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.telegram_token, "123:abc")
            self.assertEqual(cfg.telegram_chat_id, "456")

    def test_ws_stale_threshold_default(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.ws_stale_threshold, 120.0)

    def test_ws_stale_threshold_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"WS_STALE_THRESHOLD": "60"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.ws_stale_threshold, 60.0)

    def test_ws_stale_threshold_zero_raises(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"WS_STALE_THRESHOLD": "0"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()


class AdditionalFeaturesConfigTest(TestCase):
    def test_mtf_timeframes_default_empty(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.mtf_timeframes, [])

    def test_mtf_timeframes_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"MTF_TIMEFRAMES": "1,15,60"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.mtf_timeframes, ["1", "15", "60"])

    def test_mtf_timeframes_invalid_raises(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"MTF_TIMEFRAMES": "999"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_max_exposure_default_zero(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.max_exposure_per_coin_pct, 0.0)

    def test_max_exposure_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"MAX_EXPOSURE_PER_COIN_PCT": "0.3"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.max_exposure_per_coin_pct, 0.3)

    def test_max_daily_loss_default_zero(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.max_daily_loss_pct, 0.0)

    def test_discord_webhook_default_none(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertIsNone(cfg.discord_webhook_url)

    def test_discord_webhook_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/x"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.discord_webhook_url, "https://discord.com/api/webhooks/x")

    def test_dynamic_pairs_defaults(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.dynamic_pairs_refresh_cycles, 0)
            self.assertEqual(cfg.dynamic_pairs_top_n, 50)

    def test_dynamic_pairs_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"DYNAMIC_PAIRS_REFRESH_CYCLES": "10", "DYNAMIC_PAIRS_TOP_N": "20"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.dynamic_pairs_refresh_cycles, 10)
            self.assertEqual(cfg.dynamic_pairs_top_n, 20)
