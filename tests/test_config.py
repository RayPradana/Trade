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

    def test_trailing_stop_default(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.trailing_stop_pct, 0.03)

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
            self.assertEqual(cfg.min_volume_idr, 500_000.0)

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

    def test_small_coin_volume_and_trades_defaults(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.min_coin_price_idr, 50.0)
            self.assertEqual(cfg.small_coin_min_volume_24h_idr, 1_000_000.0)
            self.assertEqual(cfg.small_coin_min_trades_24h, 50)

    def test_small_coin_volume_and_trades_from_env(self):
        from unittest.mock import patch
        with patch.dict(
            __import__("os").environ,
            {"MIN_COIN_PRICE_IDR": "25", "SMALL_COIN_MIN_VOLUME_24H_IDR": "2500000", "SMALL_COIN_MIN_TRADES_24H": "75"},
            clear=True,
        ):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.min_coin_price_idr, 25.0)
            self.assertEqual(cfg.small_coin_min_volume_24h_idr, 2_500_000.0)
            self.assertEqual(cfg.small_coin_min_trades_24h, 75)

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
            self.assertEqual(cfg.dynamic_pairs_refresh_cycles, 5)
            self.assertEqual(cfg.dynamic_pairs_top_n, 20)

    def test_dynamic_pairs_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"DYNAMIC_PAIRS_REFRESH_CYCLES": "10", "DYNAMIC_PAIRS_TOP_N": "20"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.dynamic_pairs_refresh_cycles, 10)
            self.assertEqual(cfg.dynamic_pairs_top_n, 20)


class NewFeaturesConfigTest(TestCase):
    def test_partial_tp_default(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.partial_tp_fraction, 0.0)

    def test_partial_tp_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"PARTIAL_TP_FRACTION": "0.5"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.partial_tp_fraction, 0.5)

    def test_partial_tp_invalid(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"PARTIAL_TP_FRACTION": "1.5"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_re_entry_cooldown_default(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.re_entry_cooldown_seconds, 0.0)
            self.assertEqual(cfg.re_entry_dip_pct, 0.0)

    def test_re_entry_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"RE_ENTRY_COOLDOWN_SECONDS": "120", "RE_ENTRY_DIP_PCT": "0.03"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.re_entry_cooldown_seconds, 120.0)
            self.assertAlmostEqual(cfg.re_entry_dip_pct, 0.03)

    def test_adaptive_interval_defaults(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertFalse(cfg.adaptive_interval_enabled)
            self.assertEqual(cfg.adaptive_interval_min_seconds, 30)

    def test_adaptive_interval_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"ADAPTIVE_INTERVAL_ENABLED": "true", "ADAPTIVE_INTERVAL_MIN_SECONDS": "15"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertTrue(cfg.adaptive_interval_enabled)
            self.assertEqual(cfg.adaptive_interval_min_seconds, 15)

    def test_portfolio_risk_default(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.max_portfolio_risk_pct, 0.0)

    def test_liquidity_depth_default(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.min_liquidity_depth_idr, 100_000.0)

    def test_liquidity_depth_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"MIN_LIQUIDITY_DEPTH_IDR": "50000000"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.min_liquidity_depth_idr, 50_000_000.0)


class DynamicTpConfigTest(TestCase):
    def test_trailing_tp_defaults(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.trailing_tp_pct, 0.02)
            self.assertEqual(cfg.conditional_tp_min_trend_strength, 0.0)
            self.assertEqual(cfg.conditional_tp_min_ob_imbalance, 0.0)
            self.assertEqual(cfg.conditional_tp_max_rsi, 0.0)

    def test_trailing_tp_from_env(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"TRAILING_TP_PCT": "0.01"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.trailing_tp_pct, 0.01)

    def test_conditional_tp_from_env(self):
        from unittest.mock import patch
        env = {
            "CONDITIONAL_TP_MIN_TREND_STRENGTH": "0.4",
            "CONDITIONAL_TP_MIN_OB_IMBALANCE": "0.1",
            "CONDITIONAL_TP_MAX_RSI": "70",
        }
        with patch.dict(__import__("os").environ, env, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.conditional_tp_min_trend_strength, 0.4)
            self.assertAlmostEqual(cfg.conditional_tp_min_ob_imbalance, 0.1)
            self.assertAlmostEqual(cfg.conditional_tp_max_rsi, 70.0)

    def test_trailing_tp_negative_invalid(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"TRAILING_TP_PCT": "-0.01"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_profit_buffer_drawdown_pct_boundary(self):
        from unittest.mock import patch
        with patch.dict(__import__("os").environ, {"PROFIT_BUFFER_DRAWDOWN_PCT": "1.0"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_orderbook_wall_defaults_to_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.orderbook_wall_threshold, 0.0)

    def test_orderbook_wall_from_env(self):
        with patch.dict(os.environ, {"ORDERBOOK_WALL_THRESHOLD": "5"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.orderbook_wall_threshold, 5.0)

    def test_orderbook_wall_negative_invalid(self):
        with patch.dict(os.environ, {"ORDERBOOK_WALL_THRESHOLD": "-1"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_pump_protection_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.pump_protection_pct, 0.0)
            self.assertAlmostEqual(cfg.pump_lookback_seconds, 60.0)

    def test_pump_protection_from_env(self):
        env = {"PUMP_PROTECTION_PCT": "0.05", "PUMP_LOOKBACK_SECONDS": "120"}
        with patch.dict(os.environ, env, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.pump_protection_pct, 0.05)
            self.assertAlmostEqual(cfg.pump_lookback_seconds, 120.0)

    def test_pump_protection_negative_pct_invalid(self):
        with patch.dict(os.environ, {"PUMP_PROTECTION_PCT": "-0.01"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_pump_lookback_zero_invalid(self):
        with patch.dict(os.environ, {"PUMP_LOOKBACK_SECONDS": "0"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()

    def test_max_spread_pct_defaults_to_zero(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.max_spread_pct, 0.0)

    def test_max_spread_pct_from_env(self):
        with patch.dict(os.environ, {"MAX_SPREAD_PCT": "0.002"}, clear=True):
            cfg = BotConfig.from_env()
            self.assertAlmostEqual(cfg.max_spread_pct, 0.002)

    def test_max_spread_pct_negative_invalid(self):
        with patch.dict(os.environ, {"MAX_SPREAD_PCT": "-0.001"}, clear=True):
            with self.assertRaises(ValueError):
                BotConfig.from_env()


class ConfigNewFieldsTest(TestCase):
    def test_new_fields_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.max_consecutive_losses, 0)
            self.assertEqual(cfg.volatility_cooldown_pct, 0.0)
            self.assertEqual(cfg.volatility_cooldown_seconds, 0.0)
            self.assertEqual(cfg.circuit_breaker_max_errors, 0)
            self.assertAlmostEqual(cfg.circuit_breaker_pause_seconds, 300.0)
            self.assertFalse(cfg.balance_check_enabled)
            self.assertEqual(cfg.stale_order_seconds, 0.0)
            self.assertEqual(cfg.strategy_auto_disable_losses, 0)
            self.assertEqual(cfg.partial_tp2_fraction, 0.0)
            self.assertEqual(cfg.partial_tp2_target_pct, 0.0)
            self.assertIsNone(cfg.journal_path)
            self.assertEqual(cfg.max_open_positions, 0)
            self.assertEqual(cfg.spread_anomaly_multiplier, 0.0)
            self.assertEqual(cfg.orderbook_absorption_threshold, 0.0)
            self.assertEqual(cfg.flash_dump_pct, 0.0)
            self.assertAlmostEqual(cfg.flash_dump_lookback_seconds, 60.0)

    def test_new_fields_from_env(self):
        env = {
            "MAX_CONSECUTIVE_LOSSES": "3",
            "VOLATILITY_COOLDOWN_PCT": "0.05",
            "VOLATILITY_COOLDOWN_SECONDS": "300",
            "CIRCUIT_BREAKER_MAX_ERRORS": "5",
            "CIRCUIT_BREAKER_PAUSE_SECONDS": "600",
            "BALANCE_CHECK_ENABLED": "true",
            "STALE_ORDER_SECONDS": "120",
            "STRATEGY_AUTO_DISABLE_LOSSES": "3",
            "PARTIAL_TP2_FRACTION": "0.3",
            "PARTIAL_TP2_TARGET_PCT": "0.05",
            "JOURNAL_PATH": "/tmp/journal.csv",
            "MAX_OPEN_POSITIONS": "5",
            "SPREAD_ANOMALY_MULTIPLIER": "3.0",
            "ORDERBOOK_ABSORPTION_THRESHOLD": "0.5",
            "FLASH_DUMP_PCT": "0.05",
            "FLASH_DUMP_LOOKBACK_SECONDS": "120",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = BotConfig.from_env()
            self.assertEqual(cfg.max_consecutive_losses, 3)
            self.assertAlmostEqual(cfg.volatility_cooldown_pct, 0.05)
            self.assertAlmostEqual(cfg.volatility_cooldown_seconds, 300.0)
            self.assertEqual(cfg.circuit_breaker_max_errors, 5)
            self.assertAlmostEqual(cfg.circuit_breaker_pause_seconds, 600.0)
            self.assertTrue(cfg.balance_check_enabled)
            self.assertAlmostEqual(cfg.stale_order_seconds, 120.0)
            self.assertEqual(cfg.strategy_auto_disable_losses, 3)
            self.assertAlmostEqual(cfg.partial_tp2_fraction, 0.3)
            self.assertAlmostEqual(cfg.partial_tp2_target_pct, 0.05)
            self.assertEqual(cfg.journal_path, "/tmp/journal.csv")
            self.assertEqual(cfg.max_open_positions, 5)
            self.assertAlmostEqual(cfg.spread_anomaly_multiplier, 3.0)
            self.assertAlmostEqual(cfg.orderbook_absorption_threshold, 0.5)
            self.assertAlmostEqual(cfg.flash_dump_pct, 0.05)
            self.assertAlmostEqual(cfg.flash_dump_lookback_seconds, 120.0)

    def test_validation_negative_max_consecutive_losses(self):
        cfg = BotConfig(api_key=None, max_consecutive_losses=-1)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_validation_invalid_partial_tp2_fraction(self):
        cfg = BotConfig(api_key=None, partial_tp2_fraction=1.5)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_validation_flash_dump_lookback_zero(self):
        cfg = BotConfig(api_key=None, flash_dump_lookback_seconds=0.0)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_new_adaptive_fields_defaults(self):
        """New adaptive/trade-flow fields default to 0 (disabled)."""
        cfg = BotConfig(api_key=None)
        self.assertEqual(cfg.ob_imbalance_min_entry, 0.0)
        self.assertEqual(cfg.trade_flow_min_buy_ratio, 0.0)
        self.assertEqual(cfg.momentum_exit_ob_threshold, 0.0)
        self.assertEqual(cfg.momentum_exit_min_profit_pct, 0.0)
        self.assertEqual(cfg.partial_tp3_fraction, 0.0)
        self.assertEqual(cfg.partial_tp3_target_pct, 0.0)

    def test_validation_ob_imbalance_min_entry_out_of_range(self):
        cfg = BotConfig(api_key=None, ob_imbalance_min_entry=1.5)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_validation_trade_flow_min_buy_ratio_out_of_range(self):
        cfg = BotConfig(api_key=None, trade_flow_min_buy_ratio=1.2)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_validation_partial_tp3_fraction_out_of_range(self):
        cfg = BotConfig(api_key=None, partial_tp3_fraction=1.0)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_validation_momentum_exit_min_profit_negative(self):
        cfg = BotConfig(api_key=None, momentum_exit_min_profit_pct=-0.01)
        with self.assertRaises(ValueError):
            cfg._validate()


class MultiPositionDefaultTest(TestCase):
    def test_multi_position_enabled_default_true(self):
        """multi_position_enabled should default to True so the bot scans while holding."""
        cfg = BotConfig(api_key=None)
        self.assertTrue(cfg.multi_position_enabled)

    def test_multi_position_enabled_can_be_disabled(self):
        """Users can explicitly opt-in to classic single-position mode."""
        cfg = BotConfig(api_key=None, multi_position_enabled=False)
        self.assertFalse(cfg.multi_position_enabled)

    def test_multi_position_enabled_from_env_default_true(self):
        """MULTI_POSITION_ENABLED env default should be 'true'."""
        with patch.dict(os.environ, {"INDODAX_API_KEY": "k", "INDODAX_API_SECRET": "s", "INITIAL_CAPITAL": "100000"}):
            os.environ.pop("MULTI_POSITION_ENABLED", None)
            cfg = BotConfig.from_env()
            self.assertTrue(cfg.multi_position_enabled)

    def test_multi_position_disabled_via_env(self):
        """Setting MULTI_POSITION_ENABLED=false should disable multi-position mode."""
        with patch.dict(os.environ, {"INDODAX_API_KEY": "k", "INDODAX_API_SECRET": "s", "INITIAL_CAPITAL": "100000", "MULTI_POSITION_ENABLED": "false"}):
            cfg = BotConfig.from_env()
            self.assertFalse(cfg.multi_position_enabled)


class EnvParsingRobustnessTest(TestCase):
    """Tests for _env_float/_env_int error handling in from_env()."""

    def test_invalid_float_env_falls_back_to_default(self):
        """A non-numeric INITIAL_CAPITAL env var must use the default value."""
        with patch.dict(os.environ, {
            "INDODAX_API_KEY": "k",
            "INITIAL_CAPITAL": "not_a_number",
        }):
            cfg = BotConfig.from_env()
            # Must fall back to the default "1000000"
            self.assertAlmostEqual(cfg.initial_capital, 1_000_000.0)

    def test_invalid_int_env_falls_back_to_default(self):
        """A non-numeric FAST_WINDOW env var must use the default value."""
        with patch.dict(os.environ, {
            "INDODAX_API_KEY": "k",
            "INITIAL_CAPITAL": "1000000",
            "FAST_WINDOW": "abc",
        }):
            cfg = BotConfig.from_env()
            # Must fall back to the default "12"
            self.assertEqual(cfg.fast_window, 12)

    def test_zero_initial_capital_is_invalid(self):
        """INITIAL_CAPITAL=0 must raise ValueError."""
        cfg = BotConfig(api_key=None, initial_capital=0.0)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_negative_initial_capital_is_invalid(self):
        """Negative INITIAL_CAPITAL must raise ValueError."""
        cfg = BotConfig(api_key=None, initial_capital=-1.0)
        with self.assertRaises(ValueError):
            cfg._validate()
