from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional, Set


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    base_position: float
    realized_pnl: float
    target_equity: float
    min_equity: float


class PortfolioTracker:
    """Cash/position tracker with profit target, max-loss stops, trailing stop, and trade stats.

    Capital model
    -------------
    The tracker separates capital into two pools:

    * **Principal** (``self.principal``) – the fixed initial capital.  This is
      the "anchor" that is never deliberately eroded; drawdown protection
      targets are expressed relative to it.
    * **Profit buffer** (``self.profit_buffer``) – the cumulative realised
      profit above the principal (i.e. ``max(0, realized_pnl)``).  As trades
      generate returns this pool grows; it is used to compound position sizes
      via ``effective_capital()``.

    ``effective_capital()`` = principal + profit_buffer.  Passing this value
    to strategy position-sizing gives automatic compounding: each profitable
    trade cycle makes the next trade size slightly larger.

    An optional *profit-buffer drawdown* guard (configured via
    :attr:`~BotConfig.profit_buffer_drawdown_pct`) stops new buys when the
    profit buffer has fallen more than the allowed fraction from its all-time
    peak.  This protects realised profits from being eroded by a losing streak.
    """

    def __init__(self, initial_capital: float, target_profit_pct: float, max_loss_pct: float) -> None:
        self.initial_capital = initial_capital
        # Principal: the immutable starting capital used as an anchor.
        # Exposed as a named attribute for clarity; always equals initial_capital.
        self.principal: float = initial_capital
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
        # Daily loss tracking — reset to equity-at-open at the start of each UTC day
        self._day_open_equity: float = initial_capital
        self._day_stamp: int = self._today_stamp()
        # Re-entry tracking
        self.last_sell_price: float = 0.0
        self.last_sell_time: float = 0.0
        # Partial TP state: True when the first partial TP has been taken on the
        # current position.  Reset to False on each new buy or full close.
        self.partial_tp_taken: bool = False
        # Profit-buffer compounding: track the peak profit buffer for drawdown guard.
        self._peak_profit_buffer: float = 0.0
        # Trailing take-profit: activates once price crosses the initial TP level and
        # trailing_tp_pct > 0.  Tracks a floor price that rises with the market.
        self._tp_activated: bool = False
        self._trailing_tp_stop: Optional[float] = None
        self._trailing_tp_peak: Optional[float] = None
        # Loss streak and strategy stats
        self.loss_streak: int = 0
        self._all_sell_pnls: List[float] = []
        self._strategy_stats: Dict[str, Dict] = {}
        self._disabled_strategies: Set[str] = set()

    @property
    def profit_buffer(self) -> float:
        """Accumulated realised profit above the principal (never negative).

        This is the compounding pool: as the bot earns money the profit buffer
        grows, and ``effective_capital()`` grows with it so each subsequent
        trade is sized slightly larger.
        """
        return max(0.0, self.realized_pnl)

    def effective_capital(self) -> float:
        """Total capital available for position sizing = principal + profit buffer.

        This is what should be passed to strategy position-sizing logic instead
        of the bare ``initial_capital``.  When there is no profit yet it equals
        ``initial_capital``; as profits accumulate the value grows, enabling
        automatic compounding.
        """
        return self.principal + self.profit_buffer

    def profit_buffer_drawdown_pct(self) -> float:
        """Return how far the profit buffer has dropped from its all-time peak (0 → 1).

        Used to implement the profit-buffer drawdown guard: if this value
        exceeds the configured threshold, new buys are blocked to protect
        realised profits.

        Returns ``0.0`` when there has never been a profit buffer (no gains
        yet) or when the buffer is at its all-time peak.
        """
        if self._peak_profit_buffer <= 0:
            return 0.0
        current_pb = self.profit_buffer
        return max(0.0, (self._peak_profit_buffer - current_pb) / self._peak_profit_buffer)

    def record_trade(self, action: str, price: float, amount: float) -> None:
        notional = price * amount
        if action == "buy":
            self.cash -= notional
            total_cost = self.avg_cost * self.base_position + price * amount
            self.base_position += amount
            self.avg_cost = total_cost / self.base_position if self.base_position > 0 else 0.0
            self.trade_count += 1
            # A new buy resets the partial-TP flag for the fresh position
            self.partial_tp_taken = False
        elif action == "sell":
            sell_qty = min(amount, self.base_position)
            self.cash += price * sell_qty
            self.realized_pnl += (price - self.avg_cost) * sell_qty
            self.trade_count += 1
            self.total_sell_count += 1
            if price > self.avg_cost:
                self.profitable_sells += 1
            # Track pnl for loss streak and profit metrics
            _pnl = (price - self.avg_cost) * sell_qty
            self._all_sell_pnls.append(_pnl)
            if _pnl > 0:
                self.loss_streak = 0
            else:
                self.loss_streak += 1
            self.base_position -= sell_qty
            self.last_sell_price = price
            self.last_sell_time = time.time()
            # Update peak profit buffer so drawdown guard always has a reference
            pb = self.profit_buffer
            if pb > self._peak_profit_buffer:
                self._peak_profit_buffer = pb
            if self.base_position <= 0:
                self.avg_cost = 0.0
                self.partial_tp_taken = False
                # Reset trailing stop when position is fully closed
                self._trailing_stop = None
                self._peak_price = None
                # Reset trailing TP state for the next trade
                self._tp_activated = False
                self._trailing_tp_stop = None
                self._trailing_tp_peak = None

    def activate_trailing_tp(self, mark_price: float, trailing_tp_pct: float) -> None:
        """Activate or tighten the trailing take-profit floor.

        Call this every tick once the position's equity has crossed the initial
        TP target.  The trailing floor rises with the price but never falls, so
        profits are locked in progressively.

        :param mark_price: Current market price.
        :param trailing_tp_pct: Distance below the peak price to place the
            floor, expressed as a fraction (e.g. ``0.01`` = 1% below peak).
        """
        if trailing_tp_pct <= 0 or self.base_position <= 0:
            return
        self._tp_activated = True
        if self._trailing_tp_peak is None or mark_price > self._trailing_tp_peak:
            self._trailing_tp_peak = mark_price
        new_stop = self._trailing_tp_peak * (1 - trailing_tp_pct)
        if self._trailing_tp_stop is None or new_stop > self._trailing_tp_stop:
            self._trailing_tp_stop = new_stop

    @property
    def trailing_tp_stop(self) -> Optional[float]:
        """Current trailing take-profit floor price, or ``None`` when not active."""
        return self._trailing_tp_stop

    @property
    def tp_activated(self) -> bool:
        """``True`` once the initial TP level has been reached in the current position."""
        return self._tp_activated

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
        # Restore daily tracking if persisted
        day_open = state.get("day_open_equity")
        if day_open is not None:
            self._day_open_equity = float(day_open)
        day_stamp = state.get("day_stamp")
        if day_stamp is not None:
            self._day_stamp = int(day_stamp)
        # Restore re-entry and partial-TP state
        lsp = state.get("last_sell_price")
        if lsp is not None:
            self.last_sell_price = float(lsp)
        lst = state.get("last_sell_time")
        if lst is not None:
            self.last_sell_time = float(lst)
        ptt = state.get("partial_tp_taken")
        if ptt is not None:
            self.partial_tp_taken = bool(ptt)
        # Restore profit-buffer peak for drawdown guard
        ppb = state.get("peak_profit_buffer")
        if ppb is not None:
            self._peak_profit_buffer = float(ppb)
        # Restore trailing TP state
        tp_act = state.get("tp_activated")
        if tp_act is not None:
            self._tp_activated = bool(tp_act)
        ttp_stop = state.get("trailing_tp_stop")
        if ttp_stop is not None:
            self._trailing_tp_stop = float(ttp_stop)
        ttp_peak = state.get("trailing_tp_peak")
        if ttp_peak is not None:
            self._trailing_tp_peak = float(ttp_peak)
        # Restore loss streak and strategy stats
        self.loss_streak = int(state.get("loss_streak", 0))
        self._all_sell_pnls = list(state.get("all_sell_pnls", []))
        self._strategy_stats = dict(state.get("strategy_stats", {}))
        disabled = state.get("disabled_strategies", [])
        self._disabled_strategies = set(disabled) if disabled else set()
        # target/min equity recomputed from initial_capital to keep guard consistent

    def stop_reason(self, mark_price: float) -> Optional[str]:
        snap = self.snapshot(mark_price)
        # Trailing TP: if floor is active and price fell below it, take profit
        if self._trailing_tp_stop is not None and mark_price <= self._trailing_tp_stop:
            return "trailing_tp_triggered"
        if snap.equity >= self.target_equity:
            # When the trailing TP is already active, hold — the trailing
            # floor protects profits while letting the position run.
            if self._trailing_tp_stop is not None:
                return None
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
            # Capital management
            "principal": self.principal,
            "profit_buffer": self.profit_buffer,
            "effective_capital": self.effective_capital(),
            "peak_profit_buffer": self._peak_profit_buffer,
            "profit_buffer_drawdown": round(self.profit_buffer_drawdown_pct(), 4),
            # Trailing TP
            "tp_activated": self._tp_activated,
            "trailing_tp_stop": self._trailing_tp_stop,
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
            "day_open_equity": self._day_open_equity,
            "day_stamp": self._day_stamp,
            "last_sell_price": self.last_sell_price,
            "last_sell_time": self.last_sell_time,
            "partial_tp_taken": self.partial_tp_taken,
            "peak_profit_buffer": self._peak_profit_buffer,
            # Trailing TP
            "tp_activated": self._tp_activated,
            "trailing_tp_stop": self._trailing_tp_stop,
            "trailing_tp_peak": self._trailing_tp_peak,
            "loss_streak": self.loss_streak,
            "all_sell_pnls": self._all_sell_pnls,
            "strategy_stats": self._strategy_stats,
            "disabled_strategies": list(self._disabled_strategies),
        }

    # ── Daily loss cap helpers ────────────────────────────────────────────

    @staticmethod
    def _today_stamp() -> int:
        """Return today's UTC date as an integer YYYYMMDD."""
        import datetime
        d = datetime.datetime.now(datetime.timezone.utc).date()
        return d.year * 10000 + d.month * 100 + d.day

    def _maybe_reset_day(self, mark_price: float) -> None:
        """Reset the daily equity baseline when the UTC date has changed."""
        today = self._today_stamp()
        if today != self._day_stamp:
            self._day_open_equity = self.snapshot(mark_price).equity
            self._day_stamp = today

    def daily_loss(self, mark_price: float) -> float:
        """Return today's realised-plus-unrealised loss as a positive number.

        Returns 0.0 when the portfolio is flat or up on the day.
        """
        self._maybe_reset_day(mark_price)
        current_equity = self.snapshot(mark_price).equity
        return max(0.0, self._day_open_equity - current_equity)

    def daily_loss_pct(self, mark_price: float) -> float:
        """Return today's loss as a fraction of the day-open equity (0 → 1).

        Returns 0.0 when the portfolio is flat or up on the day, or when
        day-open equity is zero.
        """
        if self._day_open_equity <= 0:
            return 0.0
        return self.daily_loss(mark_price) / self._day_open_equity

    # ── Re-entry helpers ──────────────────────────────────────────────────

    def re_entry_allowed(
        self,
        current_price: float,
        cooldown_seconds: float = 0.0,
        dip_pct: float = 0.0,
    ) -> bool:
        """Return ``True`` when re-entry conditions are met after a sell.

        :param current_price: Latest trade price.
        :param cooldown_seconds: Minimum seconds that must have elapsed since
            the last sell.  ``0`` means no cooldown.
        :param dip_pct: Price must have dropped at least this fraction below
            the last sell price.  ``0`` means any price is acceptable.
        :returns: ``True`` when both conditions pass, or when neither condition
            is configured (i.e., both parameters are 0).
        """
        # If no cooldown and no dip requirement, always allow
        if cooldown_seconds <= 0 and dip_pct <= 0:
            return True
        # If we've never sold, allow freely
        if self.last_sell_time == 0.0:
            return True
        # Check time cooldown
        if cooldown_seconds > 0:
            elapsed = time.time() - self.last_sell_time
            if elapsed < cooldown_seconds:
                return False
        # Check price dip requirement
        if dip_pct > 0 and self.last_sell_price > 0:
            required_price = self.last_sell_price * (1 - dip_pct)
            if current_price > required_price:
                return False
        return True

    def record_trade_with_strategy(
        self,
        action: str,
        price: float,
        amount: float,
        strategy: str,
        pnl_override: Optional[float] = None,
    ) -> None:
        """Record a trade and update per-strategy stats."""
        saved_avg_cost = self.avg_cost
        saved_position = self.base_position
        self.record_trade(action, price, amount)
        if action == "sell":
            sell_qty = min(amount, saved_position)
            pnl = pnl_override if pnl_override is not None else (price - saved_avg_cost) * sell_qty
            if strategy not in self._strategy_stats:
                self._strategy_stats[strategy] = {
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "consecutive_losses": 0,
                }
            st = self._strategy_stats[strategy]
            st["total_pnl"] += pnl
            if pnl > 0:
                st["wins"] += 1
                st["consecutive_losses"] = 0
            else:
                st["losses"] += 1
                st["consecutive_losses"] = st.get("consecutive_losses", 0) + 1

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(p for p in self._all_sell_pnls if p > 0)
        gross_loss = abs(sum(p for p in self._all_sell_pnls if p < 0))
        return gross_profit / gross_loss if gross_loss > 0 else 0.0

    @property
    def expectancy(self) -> float:
        return mean(self._all_sell_pnls) if self._all_sell_pnls else 0.0

    def strategy_stats(self, strategy: str) -> Dict:
        st = self._strategy_stats.get(strategy, {"wins": 0, "losses": 0, "total_pnl": 0.0})
        total = st["wins"] + st["losses"]
        return {**st, "win_rate": st["wins"] / total if total > 0 else 0.0}

    def disable_strategy(self, strategy: str) -> None:
        self._disabled_strategies.add(strategy)

    def is_strategy_disabled(self, strategy: str) -> bool:
        return strategy in self._disabled_strategies
