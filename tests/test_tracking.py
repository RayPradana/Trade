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
