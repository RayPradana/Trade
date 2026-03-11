from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    base_position: float
    realized_pnl: float
    target_equity: float
    min_equity: float


class PortfolioTracker:
    """Simple cash/position tracker to manage profit target and max loss stops."""

    def __init__(self, initial_capital: float, target_profit_pct: float, max_loss_pct: float) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.base_position = 0.0
        self.realized_pnl = 0.0
        self.avg_cost = 0.0
        self.target_equity = initial_capital * (1 + target_profit_pct)
        self.min_equity = initial_capital * (1 - max_loss_pct)

    def record_trade(self, action: str, price: float, amount: float) -> None:
        notional = price * amount
        if action == "buy":
            self.cash -= notional
            total_cost = self.avg_cost * self.base_position + price * amount
            self.base_position += amount
            self.avg_cost = total_cost / self.base_position if self.base_position > 0 else 0.0
        elif action == "sell":
            sell_qty = min(amount, self.base_position)
            self.cash += price * sell_qty
            self.realized_pnl += (price - self.avg_cost) * sell_qty
            self.base_position -= sell_qty
            if self.base_position <= 0:
                self.avg_cost = 0.0

    def snapshot(self, mark_price: float) -> PortfolioSnapshot:
        equity = self.cash + self.base_position * mark_price
        return PortfolioSnapshot(
            equity=equity,
            cash=self.cash,
            base_position=self.base_position,
            realized_pnl=self.realized_pnl,
            target_equity=self.target_equity,
            min_equity=self.min_equity,
        )

    def stop_reason(self, mark_price: float) -> Optional[str]:
        snap = self.snapshot(mark_price)
        if snap.equity >= self.target_equity:
            return "target_profit_reached"
        if snap.equity <= snap.min_equity:
            return "max_loss_reached"
        return None

    def as_dict(self, mark_price: float) -> Dict[str, float]:
        snap = self.snapshot(mark_price)
        return {
            "equity": snap.equity,
            "cash": snap.cash,
            "base_position": snap.base_position,
            "realized_pnl": snap.realized_pnl,
            "target_equity": snap.target_equity,
            "min_equity": snap.min_equity,
        }
