from __future__ import annotations

import os
import tempfile
import unittest

from bot.journal import TradeJournal, TradeRecord, PerformanceMetrics


class TestTradeJournal(unittest.TestCase):
    def _make_journal(self) -> TradeJournal:
        return TradeJournal()

    def test_empty_journal_metrics(self) -> None:
        j = self._make_journal()
        m = j.metrics()
        self.assertEqual(m.win_rate, 0.0)
        self.assertEqual(m.total_pnl, 0.0)
        self.assertEqual(m.profit_factor, 0.0)

    def test_log_trade_in_memory(self) -> None:
        j = self._make_journal()
        j.log_trade(
            timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
            pair="btc_idr", action="buy", price=100.0, amount=1.0,
            idr_value=100.0, pnl=0.0, strategy="scalping", confidence=0.8,
            reason="test", avg_cost=100.0, equity=1100.0,
        )
        self.assertEqual(len(j._records), 1)

    def test_metrics_win_rate(self) -> None:
        j = self._make_journal()
        # 2 wins, 1 loss
        for pnl in [10.0, 20.0, -5.0]:
            j.log_trade(
                timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
                pair="btc_idr", action="sell", price=110.0, amount=1.0,
                idr_value=110.0, pnl=pnl, strategy="scalping", confidence=0.8,
                reason="test", avg_cost=100.0, equity=1100.0,
            )
        m = j.metrics()
        self.assertAlmostEqual(m.win_rate, 2/3)
        self.assertEqual(m.total_pnl, 25.0)

    def test_metrics_profit_factor(self) -> None:
        j = self._make_journal()
        for pnl in [10.0, -5.0]:
            j.log_trade(
                timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
                pair="btc_idr", action="sell", price=110.0, amount=1.0,
                idr_value=110.0, pnl=pnl, strategy="scalping", confidence=0.8,
                reason="test", avg_cost=100.0, equity=1100.0,
            )
        m = j.metrics()
        self.assertAlmostEqual(m.profit_factor, 2.0)

    def test_metrics_consecutive(self) -> None:
        j = self._make_journal()
        for pnl in [10.0, 10.0, 10.0, -5.0, -5.0]:
            j.log_trade(
                timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
                pair="btc_idr", action="sell", price=110.0, amount=1.0,
                idr_value=110.0, pnl=pnl, strategy="scalping", confidence=0.8,
                reason="test", avg_cost=100.0, equity=1100.0,
            )
        m = j.metrics()
        self.assertEqual(m.consecutive_wins, 3)
        self.assertEqual(m.consecutive_losses, 2)

    def test_csv_write_and_load(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            j1 = TradeJournal(path)
            j1.log_trade(
                timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
                pair="btc_idr", action="sell", price=110.0, amount=1.0,
                idr_value=110.0, pnl=10.0, strategy="scalping", confidence=0.8,
                reason="test", avg_cost=100.0, equity=1100.0,
            )
            # Load in new journal
            j2 = TradeJournal(path)
            self.assertEqual(len(j2._records), 1)
            self.assertEqual(j2._records[0].pair, "btc_idr")
        finally:
            os.unlink(path)

    def test_summary_str(self) -> None:
        j = self._make_journal()
        j.log_trade(
            timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
            pair="btc_idr", action="sell", price=110.0, amount=1.0,
            idr_value=110.0, pnl=10.0, strategy="scalping", confidence=0.8,
            reason="test", avg_cost=100.0, equity=1100.0,
        )
        s = j.summary_str()
        self.assertIn("sells", s)
        self.assertIn("WinRate", s)

    def test_strategy_stats_in_metrics(self) -> None:
        j = self._make_journal()
        for strategy, pnl in [("scalping", 10.0), ("scalping", -5.0), ("day_trading", 20.0)]:
            j.log_trade(
                timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
                pair="btc_idr", action="sell", price=110.0, amount=1.0,
                idr_value=110.0, pnl=pnl, strategy=strategy, confidence=0.8,
                reason="test", avg_cost=100.0, equity=1100.0,
            )
        m = j.metrics()
        self.assertIn("scalping", m.strategy_stats)
        self.assertIn("day_trading", m.strategy_stats)
        self.assertEqual(m.strategy_stats["scalping"]["wins"], 1)
        self.assertEqual(m.strategy_stats["scalping"]["losses"], 1)

    def test_no_path_no_file_created(self) -> None:
        j = TradeJournal()
        j.log_trade(
            timestamp=1000.0, datetime_str="2024-01-01 00:00:00",
            pair="btc_idr", action="buy", price=100.0, amount=1.0,
            idr_value=100.0, pnl=0.0, strategy="scalping", confidence=0.8,
            reason="test", avg_cost=100.0, equity=1100.0,
        )
        self.assertIsNone(j.path)
        self.assertEqual(len(j._records), 1)
