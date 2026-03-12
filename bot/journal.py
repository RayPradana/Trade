from __future__ import annotations

import csv
import logging
import threading
from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


def _file_empty(path: str) -> bool:
    """Return True when *path* is empty or cannot be stat-ed (race-safe)."""
    try:
        return os.path.getsize(path) == 0
    except OSError:
        return True


@dataclass
class TradeRecord:
    timestamp: float
    datetime: str
    pair: str
    action: str
    price: float
    amount: float
    idr_value: float
    pnl: float
    strategy: str
    confidence: float
    reason: str
    avg_cost: float
    equity: float


@dataclass
class PerformanceMetrics:
    win_rate: float
    profit_factor: float
    expectancy: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    consecutive_wins: int
    consecutive_losses: int
    strategy_stats: Dict[str, Dict]


_CSV_FIELDS = [
    "timestamp", "datetime", "pair", "action", "price", "amount",
    "idr_value", "pnl", "strategy", "confidence", "reason", "avg_cost", "equity",
]


class TradeJournal:
    """Structured CSV trade journal for logging and performance analysis."""

    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._records: List[TradeRecord] = []
        self._lock = threading.Lock()
        if path:
            self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        self._records.append(TradeRecord(
                            timestamp=float(row.get("timestamp", 0)),
                            datetime=row.get("datetime", ""),
                            pair=row.get("pair", ""),
                            action=row.get("action", ""),
                            price=float(row.get("price", 0)),
                            amount=float(row.get("amount", 0)),
                            idr_value=float(row.get("idr_value", 0)),
                            pnl=float(row.get("pnl", 0)),
                            strategy=row.get("strategy", ""),
                            confidence=float(row.get("confidence", 0)),
                            reason=row.get("reason", ""),
                            avg_cost=float(row.get("avg_cost", 0)),
                            equity=float(row.get("equity", 0)),
                        ))
                    except (ValueError, KeyError):
                        pass
        except Exception as exc:
            logger.warning("Failed to load journal from %s: %s", self.path, exc)

    def log_trade(
        self,
        timestamp: float,
        datetime_str: str,
        pair: str,
        action: str,
        price: float,
        amount: float,
        idr_value: float,
        pnl: float,
        strategy: str,
        confidence: float,
        reason: str,
        avg_cost: float,
        equity: float,
    ) -> None:
        record = TradeRecord(
            timestamp=timestamp,
            datetime=datetime_str,
            pair=pair,
            action=action,
            price=price,
            amount=amount,
            idr_value=idr_value,
            pnl=pnl,
            strategy=strategy,
            confidence=confidence,
            reason=reason,
            avg_cost=avg_cost,
            equity=equity,
        )
        with self._lock:
            self._records.append(record)
            if self.path:
                self._append_to_csv(record)

    def _append_to_csv(self, record: TradeRecord) -> None:
        # Caller must hold self._lock before calling this method.
        write_header = not os.path.exists(self.path) or _file_empty(self.path)
        try:
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    "timestamp": record.timestamp,
                    "datetime": record.datetime,
                    "pair": record.pair,
                    "action": record.action,
                    "price": record.price,
                    "amount": record.amount,
                    "idr_value": record.idr_value,
                    "pnl": record.pnl,
                    "strategy": record.strategy,
                    "confidence": record.confidence,
                    "reason": record.reason,
                    "avg_cost": record.avg_cost,
                    "equity": record.equity,
                })
        except Exception as exc:
            logger.warning("Failed to write journal entry: %s", exc)

    def metrics(self) -> PerformanceMetrics:
        with self._lock:
            sell_records = [r for r in self._records if r.action == "sell"]
        if not sell_records:
            return PerformanceMetrics(
                win_rate=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                total_pnl=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                max_win=0.0,
                max_loss=0.0,
                consecutive_wins=0,
                consecutive_losses=0,
                strategy_stats={},
            )
        pnls = [r.pnl for r in sell_records]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls) if pnls else 0.0
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")  # all wins, no losses
        else:
            profit_factor = 0.0
        expectancy = mean(pnls) if pnls else 0.0
        avg_win = mean(wins) if wins else 0.0
        avg_loss = mean(losses) if losses else 0.0
        max_win = max(wins) if wins else 0.0
        max_loss = min(losses) if losses else 0.0

        # consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        cur_w = 0
        cur_l = 0
        for p in pnls:
            if p > 0:
                cur_w += 1
                cur_l = 0
            else:
                cur_l += 1
                cur_w = 0
            max_consec_wins = max(max_consec_wins, cur_w)
            max_consec_losses = max(max_consec_losses, cur_l)

        # strategy stats
        strategy_stats: Dict[str, Dict] = {}
        for r in sell_records:
            s = r.strategy
            if s not in strategy_stats:
                strategy_stats[s] = {"wins": 0, "losses": 0, "total_pnl": 0.0}
            strategy_stats[s]["total_pnl"] += r.pnl
            if r.pnl > 0:
                strategy_stats[s]["wins"] += 1
            else:
                strategy_stats[s]["losses"] += 1
        for s, st in strategy_stats.items():
            total = st["wins"] + st["losses"]
            st["win_rate"] = st["wins"] / total if total > 0 else 0.0

        return PerformanceMetrics(
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_win=max_win,
            max_loss=max_loss,
            consecutive_wins=max_consec_wins,
            consecutive_losses=max_consec_losses,
            strategy_stats=strategy_stats,
        )

    def summary_str(self) -> str:
        m = self.metrics()
        sell_count = len([r for r in self._records if r.action == "sell"])
        return (
            f"Journal: {sell_count} sells | "
            f"PnL={m.total_pnl:+.2f} | "
            f"WinRate={m.win_rate:.1%} | "
            f"PF={m.profit_factor:.2f} | "
            f"Expect={m.expectancy:+.2f} | "
            f"MaxWin={m.max_win:+.2f} MaxLoss={m.max_loss:+.2f}"
        )
