from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .analysis import OrderbookInsight, SupportResistance, TrendResult, VolatilityStats
from .config import BotConfig

SCALP_SPREAD_THRESHOLD = 0.0015
ORDERBOOK_SPREAD_BONUS = 0.002
ORDERBOOK_IMBALANCE_WEIGHT = 50
VOLATILITY_PENALTY_CAP = 0.8
# Minimum stop distance expressed in absolute price units to avoid zero division and unrealistic sizing
MIN_STOP_DISTANCE = 1e-6
LEVEL_PROXIMITY = 0.02  # 2% proximity to support/resistance levels


@dataclass
class StrategyDecision:
    mode: str
    action: str
    confidence: float
    reason: str
    target_price: float
    amount: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    support: Optional[float] = None
    resistance: Optional[float] = None


def select_strategy(trend: TrendResult, orderbook: OrderbookInsight, vol: VolatilityStats) -> str:
    if (
        orderbook.spread_pct < SCALP_SPREAD_THRESHOLD
        and abs(orderbook.imbalance) > 0.25
        and vol.volatility < 0.01
    ):
        return "scalping"
    if trend.direction != "flat" and vol.volatility >= 0.01 and vol.volatility <= 0.03:
        return "day_trading"
    if trend.direction != "flat" and trend.strength > 0.01 and vol.volatility > 0.015:
        return "swing_trading"
    return "position_trading"


def _confidence(trend: TrendResult, orderbook: OrderbookInsight, vol: VolatilityStats) -> float:
    trend_score = min(trend.strength * 10, 1.0)
    spread_bonus = max(0, ORDERBOOK_SPREAD_BONUS - orderbook.spread_pct) * ORDERBOOK_IMBALANCE_WEIGHT
    orderbook_score = min(abs(orderbook.imbalance) + spread_bonus, 1.0)
    vol_score = 1 - min(vol.volatility * 10, VOLATILITY_PENALTY_CAP)
    return round(max(0.0, min(1.0, (trend_score * 0.45 + orderbook_score * 0.35 + vol_score * 0.2))), 3)


def _position_size(current_price: float, stop_loss: Optional[float], config: BotConfig, risk_per_unit: float) -> float:
    """Risk-based position sizing capped by configured risk_per_trade."""
    if stop_loss is None or risk_per_unit < MIN_STOP_DISTANCE:
        return 0.0
    desired_risk_value = config.initial_capital * config.risk_per_trade
    base_order_risk = risk_per_unit * config.base_order_size
    dynamic_min_stop = max(MIN_STOP_DISTANCE, current_price * 1e-6)
    scale = min(2.0, desired_risk_value / max(dynamic_min_stop, base_order_risk))
    return max(config.base_order_size * scale, config.base_order_size * 0.25)


def make_trade_decision(
    trend: TrendResult,
    orderbook: OrderbookInsight,
    vol: VolatilityStats,
    current_price: float,
    config: BotConfig,
    levels: Optional[SupportResistance] = None,
) -> StrategyDecision:
    mode = select_strategy(trend, orderbook, vol)
    conf = _confidence(trend, orderbook, vol)

    if trend.direction == "up":
        action = "buy"
        stop_loss = current_price * (1 - max(0.0025, vol.volatility * 2))
        take_profit = current_price * (1 + max(0.005, vol.volatility * 3))
    elif trend.direction == "down":
        action = "sell"
        stop_loss = current_price * (1 + max(0.0025, vol.volatility * 2))
        take_profit = current_price * (1 - max(0.005, vol.volatility * 3))
    else:
        action = "hold"
        stop_loss = None
        take_profit = None

    sr_note = ""
    if levels:
        if action == "buy" and levels.resistance:
            distance = (levels.resistance - current_price) / levels.resistance
            if distance <= LEVEL_PROXIMITY:
                conf *= 0.7
                sr_note = "near_resistance"
        if action == "sell" and levels.support:
            distance = (current_price - levels.support) / levels.support
            if distance <= LEVEL_PROXIMITY:
                conf *= 0.7
                sr_note = "near_support"

    reason = (
        f"{mode} | trend={trend.direction} strength={trend.strength:.4f} "
        f"vol={vol.volatility:.4f} ob_imbalance={orderbook.imbalance:.2f} {sr_note}"
    ).strip()

    # size based on risk per trade relative to stop distance
    risk_per_unit = abs(current_price - stop_loss) if stop_loss else 0.0
    amount = _position_size(current_price, stop_loss, config, risk_per_unit)

    return StrategyDecision(
        mode=mode,
        action=action,
        confidence=conf,
        reason=reason,
        target_price=current_price,
        amount=amount,
        stop_loss=stop_loss,
        take_profit=take_profit,
        support=levels.support if levels else None,
        resistance=levels.resistance if levels else None,
    )
