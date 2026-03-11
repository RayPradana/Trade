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
