import time
import unittest

from bot.tracking import PortfolioTracker


class TrackingTests(unittest.TestCase):
    def test_target_and_loss_stop(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.1, max_loss_pct=0.05)
        tracker.record_trade("buy", 100, 5)  # spend 500
        self.assertIsNone(tracker.stop_reason(100))
        tracker.record_trade("sell", 120, 5)  # gain 100
        self.assertEqual(tracker.stop_reason(120), "target_profit_reached")

        tracker2 = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.05)
        tracker2.record_trade("buy", 100, 5)  # spend 500
        self.assertEqual(tracker2.stop_reason(50), "max_loss_reached")

    def test_trailing_stop_triggers(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        tracker.record_trade("buy", 100, 1)  # hold 1 unit
        # No trailing stop yet
        self.assertIsNone(tracker.stop_reason(100))
        # Price rises to 120; trailing stop should follow at 120 * 0.9 = 108
        tracker.update_trailing_stop(120, 0.10)
        self.assertIsNone(tracker.stop_reason(110))  # 110 > 108, no trigger
        # Price falls below trailing stop
        self.assertEqual(tracker.stop_reason(105), "trailing_stop_triggered")

    def test_trailing_stop_rises_with_price(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        tracker.record_trade("buy", 100, 1)
        tracker.update_trailing_stop(110, 0.10)  # stop = 99
        tracker.update_trailing_stop(120, 0.10)  # stop rises to 108
        tracker.update_trailing_stop(115, 0.10)  # price dips but stop stays at 108
        self.assertEqual(tracker.trailing_stop, 120 * 0.90)

    def test_trailing_stop_disabled_when_zero(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        tracker.record_trade("buy", 100, 1)
        tracker.update_trailing_stop(100, 0.0)  # trailing_pct=0 → no-op
        self.assertIsNone(tracker.trailing_stop)
        self.assertIsNone(tracker.stop_reason(1))  # should not trigger

    def test_trailing_stop_resets_after_full_sell(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        tracker.record_trade("buy", 100, 1)
        tracker.update_trailing_stop(120, 0.10)
        self.assertIsNotNone(tracker.trailing_stop)
        tracker.record_trade("sell", 120, 1)  # full close
        self.assertIsNone(tracker.trailing_stop)
        self.assertIsNone(tracker.peak_price)

    def test_trade_statistics(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        tracker.record_trade("buy", 100, 1)
        tracker.record_trade("sell", 120, 1)  # profitable sell
        tracker.record_trade("buy", 100, 1)
        tracker.record_trade("sell", 90, 1)   # losing sell
        self.assertEqual(tracker.trade_count, 4)
        self.assertEqual(tracker.total_sell_count, 2)
        self.assertEqual(tracker.profitable_sells, 1)
        self.assertAlmostEqual(tracker.win_rate, 0.5)

    def test_win_rate_zero_with_no_sells(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        self.assertEqual(tracker.win_rate, 0.0)

    def test_state_serialization_roundtrip(self) -> None:
        tracker = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        tracker.record_trade("buy", 100, 2)
        tracker.record_trade("sell", 115, 1)
        tracker.update_trailing_stop(115, 0.05)

        state = tracker.to_state()

        tracker2 = PortfolioTracker(initial_capital=1000, target_profit_pct=0.5, max_loss_pct=0.5)
        tracker2.load_state(state)

        self.assertAlmostEqual(tracker2.cash, tracker.cash)
        self.assertAlmostEqual(tracker2.base_position, tracker.base_position)
        self.assertAlmostEqual(tracker2.realized_pnl, tracker.realized_pnl)
        self.assertEqual(tracker2.trade_count, tracker.trade_count)
        self.assertEqual(tracker2.profitable_sells, tracker.profitable_sells)
        self.assertEqual(tracker2.total_sell_count, tracker.total_sell_count)
        self.assertAlmostEqual(tracker2._trailing_stop, tracker.trailing_stop)


if __name__ == "__main__":
    unittest.main()


class DailyLossCapTest(unittest.TestCase):
    def test_no_loss_returns_zero(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        self.assertAlmostEqual(tracker.daily_loss(100_000.0), 0.0)
        self.assertAlmostEqual(tracker.daily_loss_pct(100_000.0), 0.0)

    def test_loss_detected(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        # Simulate buying coin then price drops
        tracker.record_trade("buy", 100.0, 100.0)   # spend 10_000 IDR on 100 coins
        loss = tracker.daily_loss(80.0)   # coin price drops to 80 — equity drops
        self.assertGreater(loss, 0.0)

    def test_daily_loss_pct_proportion(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 100.0)
        pct = tracker.daily_loss_pct(90.0)
        self.assertGreater(pct, 0.0)
        self.assertLess(pct, 1.0)

    def test_day_reset_changes_baseline(self):
        import datetime
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        # Simulate day change by faking the stamp
        tracker._day_stamp = 20000101  # a day in the past
        tracker._day_open_equity = 50_000.0  # old baseline
        # After reset, day_open_equity should be current equity
        _ = tracker.daily_loss(100_000.0)  # triggers reset
        self.assertAlmostEqual(tracker._day_open_equity, 100_000.0)

    def test_to_state_includes_daily_fields(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        state = tracker.to_state()
        self.assertIn("day_open_equity", state)
        self.assertIn("day_stamp", state)

    def test_load_state_restores_daily_fields(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.load_state({"day_open_equity": 95_000.0, "day_stamp": 20250101})
        self.assertAlmostEqual(tracker._day_open_equity, 95_000.0)
        self.assertEqual(tracker._day_stamp, 20250101)


class ReEntryAllowedTest(unittest.TestCase):
    def test_no_config_always_allowed(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        self.assertTrue(tracker.re_entry_allowed(90_000.0, 0, 0))

    def test_cooldown_blocks_immediate_re_entry(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.last_sell_price = 1000.0
        tracker.last_sell_time = time.time()  # just sold
        self.assertFalse(tracker.re_entry_allowed(990.0, cooldown_seconds=300))

    def test_cooldown_passes_after_enough_time(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.last_sell_price = 1000.0
        tracker.last_sell_time = time.time() - 400  # 400s ago
        self.assertTrue(tracker.re_entry_allowed(990.0, cooldown_seconds=300))

    def test_dip_pct_blocks_when_no_dip(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.last_sell_price = 1000.0
        tracker.last_sell_time = time.time() - 100  # sold 100s ago, no cooldown restriction
        # Price at 995, but we need 2% dip → required ≤ 980
        self.assertFalse(tracker.re_entry_allowed(995.0, dip_pct=0.02))

    def test_dip_pct_allows_when_price_dipped(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.last_sell_price = 1000.0
        tracker.last_sell_time = time.time() - 100  # sold 100s ago, no cooldown restriction
        # Price at 970 is 3% below last sell → 2% requirement met
        self.assertTrue(tracker.re_entry_allowed(970.0, dip_pct=0.02))

    def test_no_sell_always_allowed(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        self.assertTrue(tracker.re_entry_allowed(1000.0, cooldown_seconds=300, dip_pct=0.05))

    def test_partial_tp_taken_resets_on_buy(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.partial_tp_taken = True
        tracker.record_trade("buy", 1000.0, 1.0)
        self.assertFalse(tracker.partial_tp_taken)

    def test_to_state_includes_re_entry_fields(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.last_sell_price = 999.0
        tracker.last_sell_time = 12345.0
        tracker.partial_tp_taken = True
        state = tracker.to_state()
        self.assertIn("last_sell_price", state)
        self.assertIn("last_sell_time", state)
        self.assertIn("partial_tp_taken", state)
        self.assertEqual(state["last_sell_price"], 999.0)
        self.assertTrue(state["partial_tp_taken"])

    def test_load_state_restores_re_entry_fields(self):
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.load_state({"last_sell_price": 888.0, "last_sell_time": 9999.0, "partial_tp_taken": True})
        self.assertAlmostEqual(tracker.last_sell_price, 888.0)
        self.assertAlmostEqual(tracker.last_sell_time, 9999.0)
        self.assertTrue(tracker.partial_tp_taken)


class ProfitBufferCapitalTest(unittest.TestCase):
    """Tests for principal / profit-buffer capital management."""

    def _tracker(self, capital: float = 100_000.0) -> PortfolioTracker:
        return PortfolioTracker(capital, 0.2, 0.1)

    def test_principal_equals_initial_capital(self):
        t = self._tracker(100_000.0)
        self.assertEqual(t.principal, 100_000.0)

    def test_profit_buffer_zero_before_any_trade(self):
        t = self._tracker()
        self.assertEqual(t.profit_buffer, 0.0)

    def test_effective_capital_equals_principal_before_trades(self):
        t = self._tracker(100_000.0)
        self.assertAlmostEqual(t.effective_capital(), 100_000.0)

    def test_profit_buffer_grows_after_profitable_sell(self):
        t = self._tracker(100_000.0)
        t.record_trade("buy", 100.0, 10.0)   # buy 10 coins at 100 = 1000 cost
        t.record_trade("sell", 120.0, 10.0)  # sell at 120 = +200 profit
        self.assertAlmostEqual(t.profit_buffer, 200.0)
        self.assertAlmostEqual(t.effective_capital(), 100_200.0)

    def test_profit_buffer_not_negative_after_loss(self):
        t = self._tracker(100_000.0)
        t.record_trade("buy", 100.0, 10.0)
        t.record_trade("sell", 80.0, 10.0)   # sell at 80 = -200 loss
        self.assertEqual(t.profit_buffer, 0.0)  # never goes negative
        self.assertAlmostEqual(t.effective_capital(), 100_000.0)  # principal unchanged

    def test_peak_profit_buffer_tracks_maximum(self):
        t = self._tracker(100_000.0)
        # First profitable trade
        t.record_trade("buy", 100.0, 5.0)
        t.record_trade("sell", 110.0, 5.0)  # +50
        self.assertAlmostEqual(t._peak_profit_buffer, 50.0)
        # Second profitable trade
        t.record_trade("buy", 100.0, 5.0)
        t.record_trade("sell", 120.0, 5.0)  # +100 more → total realized = 150
        self.assertAlmostEqual(t._peak_profit_buffer, 150.0)

    def test_profit_buffer_drawdown_pct_at_zero_when_no_profit(self):
        t = self._tracker()
        self.assertEqual(t.profit_buffer_drawdown_pct(), 0.0)

    def test_profit_buffer_drawdown_pct_at_zero_when_at_peak(self):
        t = self._tracker(100_000.0)
        t.record_trade("buy", 100.0, 5.0)
        t.record_trade("sell", 110.0, 5.0)  # realized_pnl = 50 = peak
        self.assertAlmostEqual(t.profit_buffer_drawdown_pct(), 0.0)

    def test_profit_buffer_drawdown_pct_after_partial_loss(self):
        t = self._tracker(100_000.0)
        # Gain 100, then lose 50 (net realized = 50, peak = 100)
        t.record_trade("buy", 100.0, 10.0)
        t.record_trade("sell", 110.0, 10.0)  # pnl = +100, peak = 100
        t.record_trade("buy", 110.0, 10.0)
        t.record_trade("sell", 105.0, 10.0)  # pnl = -50, net = 50
        expected_dd = (100.0 - 50.0) / 100.0  # 50%
        self.assertAlmostEqual(t.profit_buffer_drawdown_pct(), expected_dd, places=4)

    def test_peak_profit_buffer_persisted_in_state(self):
        t = self._tracker(100_000.0)
        t.record_trade("buy", 100.0, 5.0)
        t.record_trade("sell", 110.0, 5.0)
        state = t.to_state()
        self.assertIn("peak_profit_buffer", state)
        t2 = self._tracker(100_000.0)
        t2.load_state(state)
        self.assertAlmostEqual(t2._peak_profit_buffer, t._peak_profit_buffer)


class TrailingTpTest(unittest.TestCase):
    """Tests for trailing take-profit mechanics."""

    def _tracker(self, capital: float = 100_000.0) -> PortfolioTracker:
        return PortfolioTracker(capital, 0.2, 0.1)

    def test_activate_trailing_tp_sets_floor(self):
        t = self._tracker()
        t.record_trade("buy", 100.0, 5.0)
        t.activate_trailing_tp(110.0, trailing_tp_pct=0.02)  # 2% below 110 = 107.8
        self.assertIsNotNone(t.trailing_tp_stop)
        self.assertAlmostEqual(t.trailing_tp_stop, 107.8)
        self.assertTrue(t.tp_activated)

    def test_trailing_tp_rises_with_price(self):
        t = self._tracker()
        t.record_trade("buy", 100.0, 5.0)
        t.activate_trailing_tp(110.0, 0.02)  # floor = 107.8
        t.activate_trailing_tp(120.0, 0.02)  # floor should rise to 117.6
        self.assertAlmostEqual(t.trailing_tp_stop, 117.6)

    def test_trailing_tp_does_not_fall(self):
        t = self._tracker()
        t.record_trade("buy", 100.0, 5.0)
        t.activate_trailing_tp(120.0, 0.02)  # floor = 117.6
        t.activate_trailing_tp(115.0, 0.02)  # price dipped; floor must NOT fall
        self.assertAlmostEqual(t.trailing_tp_stop, 117.6)

    def test_stop_reason_returns_trailing_tp_triggered_when_price_below_floor(self):
        t = self._tracker()
        t.record_trade("buy", 100.0, 5.0)
        # Manually set up trailing TP state
        t.activate_trailing_tp(130.0, 0.02)  # floor = 127.4
        # Equity is above target (profit), but price fell below TP floor
        reason = t.stop_reason(126.0)  # below floor
        self.assertEqual(reason, "trailing_tp_triggered")

    def test_stop_reason_returns_none_when_trailing_tp_active_and_price_above_floor(self):
        t = self._tracker(100_000.0)
        t.record_trade("buy", 100.0, 1000.0)  # costs 100k
        # After a big gain equity far exceeds target
        t.activate_trailing_tp(200.0, 0.02)  # floor = 196
        # Price at 198, still above floor — should hold
        reason = t.stop_reason(198.0)
        self.assertIsNone(reason)

    def test_trailing_tp_state_resets_on_full_close(self):
        t = self._tracker(100_000.0)
        t.record_trade("buy", 100.0, 5.0)
        t.activate_trailing_tp(110.0, 0.02)
        self.assertTrue(t.tp_activated)
        t.record_trade("sell", 110.0, 5.0)  # full close
        self.assertFalse(t.tp_activated)
        self.assertIsNone(t.trailing_tp_stop)

    def test_trailing_tp_state_persisted_in_state(self):
        t = self._tracker(100_000.0)
        t.record_trade("buy", 100.0, 5.0)
        t.activate_trailing_tp(110.0, 0.02)
        state = t.to_state()
        self.assertIn("trailing_tp_stop", state)
        self.assertIn("trailing_tp_peak", state)
        self.assertIn("tp_activated", state)
        t2 = self._tracker(100_000.0)
        t2.load_state(state)
        self.assertTrue(t2.tp_activated)
        self.assertAlmostEqual(t2.trailing_tp_stop, t.trailing_tp_stop)
