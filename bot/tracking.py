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
    """Cash/position tracker with profit target, max-loss stops, trailing stop, and trade stats."""

    def __init__(self, initial_capital: float, target_profit_pct: float, max_loss_pct: float) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.base_position = 0.0
        self.realized_pnl = 0.0
        self.avg_cost = 0.0
        self.target_equity = initial_capital * (1 + target_profit_pct)
        self.min_equity = initial_capital * (1 - max_loss_pct)
        # Trailing stop
        self._trailing_stop: Optional[float] = None
        self._peak_price: Optional[float] = None
        # Trade statistics
        self.trade_count: int = 0
        self.profitable_sells: int = 0
        self.total_sell_count: int = 0

    def record_trade(self, action: str, price: float, amount: float) -> None:
        notional = price * amount
        if action == "buy":
            self.cash -= notional
            total_cost = self.avg_cost * self.base_position + price * amount
            self.base_position += amount
            self.avg_cost = total_cost / self.base_position if self.base_position > 0 else 0.0
            self.trade_count += 1
        elif action == "sell":
            sell_qty = min(amount, self.base_position)
            self.cash += price * sell_qty
            self.realized_pnl += (price - self.avg_cost) * sell_qty
            self.trade_count += 1
            self.total_sell_count += 1
            if price > self.avg_cost:
                self.profitable_sells += 1
            self.base_position -= sell_qty
            if self.base_position <= 0:
                self.avg_cost = 0.0
                # Reset trailing stop when position is fully closed
                self._trailing_stop = None
                self._peak_price = None

    def update_trailing_stop(self, mark_price: float, trailing_pct: float) -> None:
        """Update the trailing stop based on the current market price.

        The trailing stop rises with the price but never falls.  A ``trailing_pct``
        of ``0`` disables the feature.
        """
        if trailing_pct <= 0 or self.base_position <= 0:
            return
        if self._peak_price is None or mark_price > self._peak_price:
            self._peak_price = mark_price
        new_stop = self._peak_price * (1 - trailing_pct)
        if self._trailing_stop is None or new_stop > self._trailing_stop:
            self._trailing_stop = new_stop

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

    def load_state(self, state: Dict[str, float]) -> None:
        """Restore tracker state from persisted portfolio dictionary."""
        self.cash = float(state.get("cash", self.cash))
        self.base_position = float(state.get("base_position", self.base_position))
        self.realized_pnl = float(state.get("realized_pnl", self.realized_pnl))
        self.avg_cost = float(state.get("avg_cost", self.avg_cost))
        self.trade_count = int(state.get("trade_count", self.trade_count))
        self.profitable_sells = int(state.get("profitable_sells", self.profitable_sells))
        self.total_sell_count = int(state.get("total_sell_count", self.total_sell_count))
        trailing_stop = state.get("trailing_stop")
        self._trailing_stop = float(trailing_stop) if trailing_stop is not None else None
        peak_price = state.get("peak_price")
        self._peak_price = float(peak_price) if peak_price is not None else None
        # target/min equity recomputed from initial_capital to keep guard consistent

    def stop_reason(self, mark_price: float) -> Optional[str]:
        snap = self.snapshot(mark_price)
        if snap.equity >= self.target_equity:
            return "target_profit_reached"
        if snap.equity <= snap.min_equity:
            return "max_loss_reached"
        if self._trailing_stop is not None and mark_price <= self._trailing_stop:
            return "trailing_stop_triggered"
        return None

    @property
    def win_rate(self) -> float:
        """Fraction of sell trades that were profitable (0.0–1.0)."""
        return self.profitable_sells / self.total_sell_count if self.total_sell_count > 0 else 0.0

    @property
    def trailing_stop(self) -> Optional[float]:
        """Current trailing stop price, or ``None`` when not active."""
        return self._trailing_stop

    @property
    def peak_price(self) -> Optional[float]:
        """Highest price seen since the current position was opened."""
        return self._peak_price

    def as_dict(self, mark_price: float) -> Dict[str, object]:
        snap = self.snapshot(mark_price)
        return {
            "equity": snap.equity,
            "cash": snap.cash,
            "base_position": snap.base_position,
            "realized_pnl": snap.realized_pnl,
            "target_equity": snap.target_equity,
            "min_equity": snap.min_equity,
            "avg_cost": self.avg_cost,
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 3),
            "trailing_stop": self._trailing_stop,
        }

    def to_state(self) -> Dict[str, object]:
        """Serialize state without requiring a mark price."""
        return {
            "cash": self.cash,
            "base_position": self.base_position,
            "realized_pnl": self.realized_pnl,
            "avg_cost": self.avg_cost,
            "trade_count": self.trade_count,
            "profitable_sells": self.profitable_sells,
            "total_sell_count": self.total_sell_count,
            "trailing_stop": self._trailing_stop,
            "peak_price": self._peak_price,
        }
