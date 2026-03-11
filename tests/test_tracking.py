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


if __name__ == "__main__":
    unittest.main()
