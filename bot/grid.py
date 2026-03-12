from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .config import BotConfig


@dataclass
class GridOrder:
    side: str  # "buy" or "sell"
    price: float
    amount: float


@dataclass
class GridPlan:
    anchor_price: float
    buy_orders: List[GridOrder]
    sell_orders: List[GridOrder]

    @property
    def orders(self) -> List[GridOrder]:
        return self.buy_orders + self.sell_orders


def build_grid_plan(current_price: float, config: BotConfig) -> GridPlan:
    """Build symmetric grid orders above and below the anchor price."""
    if not current_price or current_price <= 0:
        return GridPlan(anchor_price=0.0, buy_orders=[], sell_orders=[])
    levels = max(1, config.grid_levels_per_side)
    spacing = max(1e-6, config.grid_spacing_pct)
    # Use explicit override, otherwise derive dynamically from risk budget and current price
    amount = config.grid_order_size or (config.risk_per_trade * config.initial_capital) / current_price

    buy_orders: List[GridOrder] = []
    sell_orders: List[GridOrder] = []

    for level in range(1, levels + 1):
        offset = spacing * level
        buy_price = round(current_price * (1 - offset), 8)
        sell_price = round(current_price * (1 + offset), 8)
        buy_orders.append(GridOrder("buy", buy_price, amount))
        sell_orders.append(GridOrder("sell", sell_price, amount))

    return GridPlan(anchor_price=current_price, buy_orders=buy_orders, sell_orders=sell_orders)
