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
