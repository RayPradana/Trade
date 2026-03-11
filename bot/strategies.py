from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .analysis import (
    MomentumIndicators,
    MultiTimeframeResult,
    OrderbookInsight,
    SpoofingResult,
    SupportResistance,
    TrendResult,
    VolatilityStats,
    WhaleActivity,
)
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
    # Fraction of position to sell at first TP (0 = sell all, disabled by default)
    partial_tp_fraction: float = 0.0


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


def _confidence_with_indicators(
    trend: TrendResult,
    orderbook: OrderbookInsight,
    vol: VolatilityStats,
    price: float,
    indicators: Optional[MomentumIndicators],
) -> float:
    base_conf = _confidence(trend, orderbook, vol)
    if not indicators:
        return base_conf

    bonus = 0.0
    # RSI: favour oversold in uptrend / overbought in downtrend
    if trend.direction == "up" and indicators.rsi < 35:
        bonus += 0.08
    if trend.direction == "down" and indicators.rsi > 65:
        bonus += 0.08

    # MACD histogram trend strength
    macd_bias = max(-0.1, min(0.1, indicators.macd_hist))
    bonus += macd_bias

    # Bollinger band mean reversion confidence
    if indicators.bb_mid:
        distance_mid = abs(price - indicators.bb_mid) / indicators.bb_mid
        bonus += max(0.0, 0.05 - distance_mid * 0.1)

    conf = base_conf + bonus
    return round(max(0.0, min(1.0, conf)), 3)


def _position_size(
    current_price: float,
    stop_loss: Optional[float],
    config: BotConfig,
    risk_per_unit: float,
    confidence: float,
    vol: VolatilityStats,
    effective_capital: Optional[float] = None,
) -> float:
    """Dynamic risk-based position sizing: base size = risk_per_trade * capital / price.

    :param effective_capital: When provided, this is used as the capital base
        for sizing instead of ``config.initial_capital``.  Pass
        ``tracker.effective_capital()`` to enable automatic compounding: as
        profits accumulate, position sizes grow proportionally.
    """
    if stop_loss is None or risk_per_unit < MIN_STOP_DISTANCE or current_price <= 0:
        return 0.0
    # Use compounding capital when available, otherwise fall back to initial_capital
    capital = effective_capital if (effective_capital is not None and effective_capital > 0) else config.initial_capital
    # Compute how many units of the base asset represent one "risk unit" of capital
    dynamic_base = (config.risk_per_trade * capital) / current_price
    desired_risk_value = capital * config.risk_per_trade
    base_order_risk = risk_per_unit * dynamic_base
    dynamic_min_stop = max(MIN_STOP_DISTANCE, current_price * 1e-6)
    scale = min(2.0, desired_risk_value / max(dynamic_min_stop, base_order_risk))
    confidence_multiplier = max(0.5, min(1.5, confidence + 0.5))
    volatility_multiplier = max(0.4, 1 - min(vol.volatility, 0.05) * 5)
    size = dynamic_base * scale * confidence_multiplier * volatility_multiplier
    return max(size, dynamic_base * 0.25)


def make_trade_decision(
    trend: TrendResult,
    orderbook: OrderbookInsight,
    vol: VolatilityStats,
    current_price: float,
    config: BotConfig,
    levels: Optional[SupportResistance] = None,
    indicators: Optional[MomentumIndicators] = None,
    mtf: Optional[MultiTimeframeResult] = None,
    whale: Optional[WhaleActivity] = None,
    spoofing: Optional[SpoofingResult] = None,
    effective_capital: Optional[float] = None,
) -> StrategyDecision:
    """Produce a :class:`StrategyDecision` incorporating all available signals.

    Parameters
    ----------
    trend:
        Short-timeframe trend derived from the primary candle series.
    orderbook:
        Current order-book analysis.
    vol:
        Volatility metrics from the primary candle series.
    current_price:
        Latest trade price.
    config:
        Bot configuration (risk, capital, etc.).
    levels:
        Support/resistance levels from the primary candle series.
    indicators:
        RSI, MACD, Bollinger Band values.
    mtf:
        Multi-timeframe consensus.  When ``mtf.aligned`` is ``True`` and the
        direction matches the primary trend, confidence receives a small bonus.
        When aligned but *opposite* to the primary trend, confidence is reduced
        (noise filter).
    whale:
        Smart-money / large-order detection result.  A bullish whale wall
        (``side="bid"``) adds a confidence bonus on buy signals; a bearish wall
        (``side="ask"``) adds a bonus on sell signals.  Opposing walls reduce
        confidence.
    spoofing:
        Order-book spoofing / manipulation detection result.  When spoofing is
        detected, confidence is penalised to avoid entering during manipulation.
    effective_capital:
        When provided, position sizing uses this value instead of
        ``config.initial_capital`` as the capital base.  Pass
        ``tracker.effective_capital()`` to enable compounding: each profitable
        trade cycle increases the size of the next position proportionally.
    """
    mode = select_strategy(trend, orderbook, vol)
    conf = _confidence_with_indicators(trend, orderbook, vol, current_price, indicators)

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

    if indicators:
        if action == "buy" and indicators.bb_upper:
            take_profit = min(take_profit, indicators.bb_upper)
        if action == "sell" and indicators.bb_lower:
            take_profit = max(take_profit, indicators.bb_lower)
        if indicators.rsi > 70 and action == "buy":
            conf *= 0.8  # avoid buying overbought
        if indicators.rsi < 30 and action == "sell":
            conf *= 0.8  # avoid shorting oversold

    # ── Multi-timeframe noise filter ─────────────────────────────────────────
    mtf_note = ""
    if mtf is not None and action != "hold":
        if mtf.aligned and mtf.direction == trend.direction:
            # All timeframes agree → boost confidence
            conf = min(1.0, conf + 0.07)
            mtf_note = "mtf_aligned"
        elif mtf.aligned and mtf.direction != trend.direction:
            # Timeframes aligned but *against* the primary signal → noise filter
            conf *= 0.6
            mtf_note = "mtf_opposing"
        elif not mtf.aligned:
            # Mixed signals → mild penalty
            conf *= 0.9
            mtf_note = "mtf_mixed"

    # ── Whale / smart money confirmation ─────────────────────────────────────
    whale_note = ""
    if whale is not None and whale.detected:
        if action == "buy" and whale.side == "bid":
            # Large bid wall → buy pressure confirmed
            conf = min(1.0, conf + 0.05)
            whale_note = "whale_bid"
        elif action == "sell" and whale.side == "ask":
            # Large ask wall → sell pressure confirmed
            conf = min(1.0, conf + 0.05)
            whale_note = "whale_ask"
        elif action == "buy" and whale.side == "ask":
            # Large ask wall resists buy → reduce confidence
            conf *= 0.85
            whale_note = "whale_ask_resist"
        elif action == "sell" and whale.side == "bid":
            # Large bid wall supports price → reduce sell confidence
            conf *= 0.85
            whale_note = "whale_bid_support"

    # ── Spoofing / manipulation penalty ──────────────────────────────────────
    spoof_note = ""
    if spoofing is not None and spoofing.detected:
        # Spoofed walls are a red flag regardless of direction → reduce conf
        conf *= 0.75
        spoof_note = f"spoofing_{spoofing.side}_{spoofing.distance_pct:.2%}"

    conf = round(max(0.0, min(1.0, conf)), 3)

    _notes = " ".join(n for n in (sr_note, mtf_note, whale_note, spoof_note) if n)
    reason = (
        f"{mode} | trend={trend.direction} strength={trend.strength:.4f} "
        f"vol={vol.volatility:.4f} ob_imbalance={orderbook.imbalance:.2f} "
        f"rsi={indicators.rsi if indicators else float('nan'):.2f}"
        + (f" {_notes}" if _notes else "")
    )

    # size based on risk per trade relative to stop distance
    risk_per_unit = abs(current_price - stop_loss) if stop_loss else 0.0
    amount = _position_size(current_price, stop_loss, config, risk_per_unit, conf, vol, effective_capital)

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
        partial_tp_fraction=config.partial_tp_fraction,
    )
