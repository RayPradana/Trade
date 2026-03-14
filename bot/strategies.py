from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from .analysis import MarketRegime

from .analysis import (
    MomentumIndicators,
    MultiTimeframeResult,
    OrderbookInsight,
    SmartEntryResult,
    SpoofingResult,
    SupportResistance,
    TradeFlowResult,
    TrendResult,
    VolatilityStats,
    WhaleActivity,
    LiquiditySweep,
    LiquidityTrap,
    LiquidityVacuum,
    MicroTrend,
    SmartMoneyFootprint,
    VolumeAcceleration,
)
from .config import BotConfig

SCALP_SPREAD_THRESHOLD = 0.0015
ORDERBOOK_SPREAD_BONUS = 0.002
ORDERBOOK_IMBALANCE_WEIGHT = 50
VOLATILITY_PENALTY_CAP = 0.8
# Minimum stop distance expressed in absolute price units to avoid zero division and unrealistic sizing
MIN_STOP_DISTANCE = 1e-6
LEVEL_PROXIMITY = 0.02  # 2% proximity to support/resistance levels
# AI-style entry scoring weights/scales (heuristic, tuned for 0..1 range)
AI_TREND_SCALE = 12.0
AI_VOL_SCALE = 50.0
AI_TREND_WEIGHT = 0.4
AI_OB_WEIGHT = 0.35
AI_VOL_WEIGHT = 0.25
AI_PRE_PUMP_BONUS = 0.1
AI_PUMP_SNIPER_BONUS = 0.1
AI_WHALE_BONUS = 0.1
AI_EB_BONUS = 0.05
AI_FAKE_BREAKOUT_PENALTY = 0.15
AI_TREND_MISALIGNMENT_PENALTY = 0.4


def _ai_entry_score(
    trend: TrendResult,
    orderbook: OrderbookInsight,
    vol: VolatilityStats,
    action: str,
    smart_entry: Optional[SmartEntryResult],
) -> float:
    """Heuristic machine-style score synthesising multiple signals (0..1)."""
    trend_score = min(1.0, max(0.0, abs(trend.strength) * AI_TREND_SCALE))
    if action == "buy" and trend.direction != "up":
        trend_score *= AI_TREND_MISALIGNMENT_PENALTY
    if action == "sell" and trend.direction != "down":
        trend_score *= AI_TREND_MISALIGNMENT_PENALTY

    ob_bias = orderbook.imbalance
    ob_score = (
        max(0.0, min(1.0, (ob_bias + 0.5))) if action == "buy" else max(0.0, min(1.0, (-ob_bias + 0.5)))
    )

    vol_score = max(0.0, min(1.0, 1.0 - vol.volatility * AI_VOL_SCALE))

    see_bonus = 0.0
    if smart_entry:
        if smart_entry.pre_pump.detected and action == "buy":
            see_bonus += AI_PRE_PUMP_BONUS * smart_entry.pre_pump.score
        if getattr(smart_entry, "pump_sniper", None) and smart_entry.pump_sniper.detected and action == "buy":
            see_bonus += AI_PUMP_SNIPER_BONUS * smart_entry.pump_sniper.score
        if smart_entry.whale_pressure.detected:
            aligned = (action == "buy" and smart_entry.whale_pressure.side == "buy") or (
                action == "sell" and smart_entry.whale_pressure.side == "sell"
            )
            if aligned:
                see_bonus += AI_WHALE_BONUS
            else:
                see_bonus -= AI_WHALE_BONUS
        if getattr(smart_entry, "early_breakout", None) and smart_entry.early_breakout.detected and action == "buy":
            see_bonus += AI_EB_BONUS * smart_entry.early_breakout.score
        if smart_entry.fake_breakout.detected and action == "buy":
            see_bonus -= AI_FAKE_BREAKOUT_PENALTY * smart_entry.fake_breakout.score

    base = (trend_score * AI_TREND_WEIGHT) + (ob_score * AI_OB_WEIGHT) + (vol_score * AI_VOL_WEIGHT)
    score = max(0.0, min(1.0, base + see_bonus))
    return round(score, 3)


def adaptive_risk_per_trade(equity: float, config: BotConfig) -> float:
    """Return the effective risk-per-trade fraction for the given equity.

    When ``config.adaptive_sizing_enabled`` is ``False`` the static
    ``config.risk_per_trade`` is returned unchanged.

    When adaptive sizing is enabled, equity is mapped to one of three tiers:

    * **Tier 0** (small cap, ``equity < adaptive_tier1_equity``):
      uses ``adaptive_tier0_risk`` (default 10%).
    * **Tier 1** (medium cap, ``adaptive_tier1_equity ≤ equity < adaptive_tier2_equity``):
      uses ``adaptive_tier1_risk`` (default 7%).
    * **Tier 2** (large cap, ``equity ≥ adaptive_tier2_equity``):
      uses ``adaptive_tier2_risk`` (default 3%).
    """
    if not config.adaptive_sizing_enabled or equity <= 0:
        return config.risk_per_trade
    if equity < config.adaptive_tier1_equity:
        return config.adaptive_tier0_risk
    if equity < config.adaptive_tier2_equity:
        return config.adaptive_tier1_risk
    return config.adaptive_tier2_risk


def adaptive_max_positions(equity: float, config: BotConfig) -> int:
    """Return the effective max-open-positions limit for the given equity.

    When ``config.adaptive_sizing_enabled`` is ``False``, returns
    ``config.max_open_positions`` (0 = no limit).

    When adaptive sizing is enabled, returns the tier-specific value
    (``adaptive_tier0_max_pos`` / ``adaptive_tier1_max_pos`` / ``adaptive_tier2_max_pos``),
    overriding the static ``max_open_positions``.
    """
    if not config.adaptive_sizing_enabled or equity <= 0:
        return config.max_open_positions
    if equity < config.adaptive_tier1_equity:
        return config.adaptive_tier0_max_pos
    if equity < config.adaptive_tier2_equity:
        return config.adaptive_tier1_max_pos
    return config.adaptive_tier2_max_pos


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


@dataclass
class StrategyScore:
    strategy: str
    score: float
    enabled: bool


def score_strategies(
    trend: TrendResult,
    orderbook: OrderbookInsight,
    vol: VolatilityStats,
    regime: Optional["MarketRegime"] = None,
) -> Dict[str, float]:
    """Score each strategy 0..1 based on current market conditions."""
    scores: Dict[str, float] = {
        "scalping": 0.0,
        "day_trading": 0.0,
        "swing_trading": 0.0,
        "position_trading": 0.0,
    }
    # Scalping: tight spread, strong imbalance, low volatility
    if orderbook.spread_pct < SCALP_SPREAD_THRESHOLD:
        scores["scalping"] += 0.5
    if abs(orderbook.imbalance) > 0.25:
        scores["scalping"] += 0.3
    if vol.volatility < 0.01:
        scores["scalping"] += 0.2

    # Day trading: trending with moderate volatility
    if trend.direction != "flat":
        scores["day_trading"] += 0.4
    if 0.01 <= vol.volatility <= 0.03:
        scores["day_trading"] += 0.4
    if trend.strength > 0.01:
        scores["day_trading"] += 0.2

    # Swing trading: strong trend with higher volatility
    if trend.direction != "flat" and trend.strength > 0.01:
        scores["swing_trading"] += 0.5
    if vol.volatility > 0.015:
        scores["swing_trading"] += 0.3
    if trend.strength > 0.02:
        scores["swing_trading"] += 0.2

    # Position trading: always gets a baseline
    scores["position_trading"] = 0.3

    # Regime adjustments
    if regime is not None:
        if regime.regime == "volatile":
            scores["position_trading"] += 0.4
            scores["scalping"] *= 0.3
            scores["day_trading"] *= 0.5
        elif regime.regime == "ranging":
            scores["scalping"] += 0.3
            scores["position_trading"] *= 0.7

    # Clamp all scores
    for k in scores:
        scores[k] = round(min(1.0, max(0.0, scores[k])), 4)
    return scores


def select_strategy(
    trend: TrendResult,
    orderbook: OrderbookInsight,
    vol: VolatilityStats,
    regime: Optional["MarketRegime"] = None,
) -> str:
    # regime-based override
    if regime is not None:
        if regime.regime == "volatile":
            return "position_trading"
        if regime.regime == "ranging":
            return "scalping"
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


def confidence_position_pct(confidence: float, config: BotConfig) -> float:
    """Return the position size as a fraction of capital for the given confidence level.

    Used when ``config.confidence_position_sizing_enabled`` is True.
    Returns 0.0 when confidence is below the effective skip threshold.

    The effective skip threshold is ``min(confidence_tier_skip, min_confidence)``
    so that signals which already passed the bot's confidence gate are never
    silently zeroed out by a higher tier-skip value.

    Tier mapping (defaults):
      < effective_skip → 0.0   (skip)
      effective_skip–0.50 → 10 %  (low tier)
      0.50–0.65 → 15 %
      0.65–0.80 → 20 %
      > 0.80  → 25 %
    """
    effective_skip = min(config.confidence_tier_skip, config.min_confidence)
    if confidence < effective_skip:
        return 0.0
    if confidence < config.confidence_tier_low:
        return config.confidence_tier_low_pct
    if confidence < config.confidence_tier_mid:
        return config.confidence_tier_mid_pct
    if confidence < config.confidence_tier_high:
        return config.confidence_tier_high_pct
    return config.confidence_tier_max_pct


def _position_size(
    current_price: float,
    stop_loss: Optional[float],
    config: BotConfig,
    risk_per_unit: float,
    confidence: float,
    vol: VolatilityStats,
    effective_capital: Optional[float] = None,
    ob_imbalance: float = 0.0,
) -> float:
    """Dynamic position sizing: base size = risk_per_trade * capital / price.

    When ``config.confidence_position_sizing_enabled`` is True the size is instead
    computed as a direct percentage of available capital determined by the
    confidence tier (see :func:`confidence_position_pct`).

    When ``config.ob_imbalance_boost_threshold > 0`` and ``ob_imbalance`` meets
    or exceeds that threshold the computed size is multiplied by
    ``config.ob_imbalance_size_multiplier``.  This allows the bot to enter with
    a larger stake when strong buy pressure (bid >> ask) is detected — a common
    signal before a pump on illiquid pairs.

    :param effective_capital: When provided, this is used as the capital base
        for sizing instead of ``config.initial_capital``.  Pass
        ``tracker.effective_capital()`` to enable automatic compounding: as
        profits accumulate, position sizes grow proportionally.
    :param ob_imbalance: Current order-book imbalance
        ``(bid_vol − ask_vol) / (bid_vol + ask_vol)``, range ``−1 … +1``.
        Used for the OB imbalance size boost when the feature is enabled.
    """
    if current_price <= 0:
        return 0.0
    # Use compounding capital when available, otherwise fall back to initial_capital
    capital = effective_capital if (effective_capital is not None and effective_capital > 0) else config.initial_capital

    # ── Confidence-tier-based sizing ─────────────────────────────────────────
    if config.confidence_position_sizing_enabled:
        pct = confidence_position_pct(confidence, config)
        if pct <= 0.0:
            # Fallback: when primary confidence sizing yields 0 but the signal
            # already passed the bot's confidence gate (confidence ≥ min_confidence)
            # and capital can cover the exchange minimum, use the minimum order
            # value so that valid signals are not silently discarded.
            min_order_idr = getattr(config, "min_order_idr", 0.0)
            if (
                confidence >= config.min_confidence
                and min_order_idr > 0
                and capital >= min_order_idr * 1.003
            ):
                size = min_order_idr / current_price * (1 + 1e-9)
            else:
                return 0.0
        else:
            size = (pct * capital) / current_price
    else:
        # ── Original risk/stop-distance-based sizing ──────────────────────────
        if stop_loss is None or risk_per_unit < MIN_STOP_DISTANCE:
            # Fallback: when stop-loss data is unavailable but the signal
            # passed the confidence gate and capital can cover the exchange
            # minimum, use the minimum order value.
            min_order_idr = getattr(config, "min_order_idr", 0.0)
            if (
                confidence >= config.min_confidence
                and min_order_idr > 0
                and capital >= min_order_idr * 1.003
            ):
                size = min_order_idr / current_price * (1 + 1e-9)
            else:
                return 0.0
        else:
            # Adaptive risk: pick tier-based risk_per_trade when adaptive sizing is on
            risk = adaptive_risk_per_trade(capital, config)
            # Compute how many units of the base asset represent one "risk unit" of capital
            dynamic_base = (risk * capital) / current_price
            desired_risk_value = capital * risk
            base_order_risk = risk_per_unit * dynamic_base
            dynamic_min_stop = max(MIN_STOP_DISTANCE, current_price * 1e-6)
            scale = min(2.0, desired_risk_value / max(dynamic_min_stop, base_order_risk))
            confidence_multiplier = max(0.5, min(1.5, confidence + 0.5))
            volatility_multiplier = max(0.4, 1 - min(vol.volatility, 0.05) * 5)
            size = dynamic_base * scale * confidence_multiplier * volatility_multiplier
            size = max(size, dynamic_base * 0.25)

    # ── Orderbook imbalance size boost ───────────────────────────────────────
    if (
        config.ob_imbalance_boost_threshold > 0
        and ob_imbalance >= config.ob_imbalance_boost_threshold
    ):
        size *= config.ob_imbalance_size_multiplier

    # ── Enforce minimum order value ──────────────────────────────────────────
    # Ensure the computed size meets the exchange minimum order value so that
    # orders are not skipped downstream.  When the available capital can cover
    # the minimum, bump size up; otherwise leave it as-is (the downstream
    # guard will skip gracefully).
    min_order_idr = getattr(config, "min_order_idr", 0.0)
    if min_order_idr > 0 and current_price > 0:
        order_value = size * current_price
        if 0 < order_value < min_order_idr:
            # Tiny 1e-9 buffer prevents floating-point rounding from landing
            # exactly at the threshold where (size * price) < min_order_idr.
            min_size = min_order_idr / current_price * (1 + 1e-9)
            # Only bump up if the capital can afford it
            max_size_from_capital = capital / current_price
            if min_size <= max_size_from_capital:
                size = min_size

    return size


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
    smart_entry: Optional[SmartEntryResult] = None,
    trade_flow: Optional[TradeFlowResult] = None,
    liquidity_sweep: Optional[LiquiditySweep] = None,
    liquidity_trap: Optional[LiquidityTrap] = None,
    liquidity_vacuum: Optional[LiquidityVacuum] = None,
    smart_money: Optional[SmartMoneyFootprint] = None,
    volume_accel: Optional[VolumeAcceleration] = None,
    micro_trend: Optional[MicroTrend] = None,
    regime: Optional[MarketRegime] = None,
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
    smart_entry:
        Result from the Smart Entry Engine.  When provided, confidence is
        adjusted by pre-pump signals (+), whale pressure (+/-), and fake
        breakout risk (-).
    trade_flow:
        Result from recent-trade buy/sell flow analysis.  When
        ``config.trade_flow_min_buy_ratio > 0`` and the buy ratio is below
        the threshold, buy signals are converted to ``"hold"`` (entry
        blocked — aggressive sellers dominate recent trades).
    """
    mode = select_strategy(trend, orderbook, vol, regime=regime)
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

    # ── Hard-skip filters (configurable) ─────────────────────────────────────
    # These override action to "hold" immediately without running further
    # scoring, so no fee-wasting order is ever placed.

    # 1. RSI overbought buy filter
    if action == "buy" and config.buy_max_rsi > 0 and indicators:
        if indicators.rsi >= config.buy_max_rsi:
            action = "hold"
            stop_loss = None
            take_profit = None
            sr_note = f"rsi_overbought(rsi={indicators.rsi:.1f}>={config.buy_max_rsi:.0f})"

    # 2. Distance-to-resistance buy filter
    if action == "buy" and config.buy_max_resistance_proximity_pct > 0 and levels and levels.resistance:
        distance = (levels.resistance - current_price) / levels.resistance
        if 0 <= distance <= config.buy_max_resistance_proximity_pct:
            action = "hold"
            stop_loss = None
            take_profit = None
            sr_note = (
                f"too_close_to_resistance(dist={distance:.2%}"
                f"<={config.buy_max_resistance_proximity_pct:.2%})"
            )

    # 3. Order-book imbalance entry guard (seller dominance filter)
    # Block buy when sellers significantly outnumber buyers in the book.
    if action == "buy" and config.ob_imbalance_min_entry != 0:
        if orderbook.imbalance < config.ob_imbalance_min_entry:
            action = "hold"
            stop_loss = None
            take_profit = None
            sr_note = (
                f"seller_dominant(imbalance={orderbook.imbalance:.3f}"
                f"<{config.ob_imbalance_min_entry:.3f})"
            )

    # 4. Trade flow entry guard (aggressive seller filter)
    # Block buy when the majority of recent trades were sell-initiated.
    if action == "buy" and config.trade_flow_min_buy_ratio > 0 and trade_flow is not None:
        if trade_flow.buy_ratio < config.trade_flow_min_buy_ratio:
            action = "hold"
            stop_loss = None
            take_profit = None
            sr_note = (
                f"sell_flow_dominant(buy_ratio={trade_flow.buy_ratio:.2f}"
                f"<{config.trade_flow_min_buy_ratio:.2f})"
            )

    # 5. Sell-wall TP adjustment
    # When a large ask wall is detected near or above the computed TP, lower
    # the TP to just below the wall so the bot takes profit before hitting
    # strong resistance (sell wall).
    if (
        action == "buy"
        and take_profit is not None
        and whale is not None
        and whale.detected
        and whale.side == "ask"
    ):
        # Find the wall price (largest ask level price) from the orderbook.
        # We approximate it as the top_ask price adjusted for the whale ratio.
        # A simpler, robust approach: if TP is above current price, cap it
        # conservatively to avoid the wall. We reduce TP by 1% as a safety margin.
        adjusted_tp = current_price * (1 + (take_profit / current_price - 1) * 0.7)
        if adjusted_tp < take_profit:
            take_profit = adjusted_tp
            sr_note = (sr_note + " " if sr_note else "") + "sell_wall_tp_adjusted"

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

    # ── Smart Entry Engine confidence adjustments ─────────────────────────────
    see_note = ""
    if smart_entry is not None:
        # 1. Pre-pump signal: early volume accumulation → bonus on buy
        if smart_entry.pre_pump.detected and action == "buy":
            boost = round(0.06 * smart_entry.pre_pump.score, 4)
            conf = min(1.0, conf + boost)
            see_note = "see_pre_pump"
        # 1b. Pump-sniper signal: momentum + volume surge combo
        if getattr(smart_entry, "pump_sniper", None) and smart_entry.pump_sniper.detected and action == "buy":
            conf = min(1.0, conf + 0.05 * smart_entry.pump_sniper.score)
            see_note = (see_note + "_" if see_note else "") + "see_pump_sniper"
        # 2. Whale pressure: net bid/ask imbalance confirms or opposes action
        if smart_entry.whale_pressure.detected:
            if (action == "buy" and smart_entry.whale_pressure.side == "buy") or (
                action == "sell" and smart_entry.whale_pressure.side == "sell"
            ):
                conf = min(1.0, conf + 0.05)
                see_note += ("_" if see_note else "") + "see_whale_confirm"
            elif (action == "buy" and smart_entry.whale_pressure.side == "sell") or (
                action == "sell" and smart_entry.whale_pressure.side == "buy"
            ):
                conf *= 0.85
                see_note += ("_" if see_note else "") + "see_whale_oppose"
        # 3. Fake breakout: thin volume above resistance → penalise buy
        if smart_entry.fake_breakout.detected and action == "buy":
            conf *= 1.0 - 0.4 * smart_entry.fake_breakout.score
            see_note += ("_" if see_note else "") + "see_fake_breakout"
        # 4. Early breakout: price pressing resistance with strong volume → boost buy
        if getattr(smart_entry, "early_breakout", None) and smart_entry.early_breakout.detected and action == "buy":
            conf = min(1.0, conf + 0.04 * smart_entry.early_breakout.score)
            see_note += ("_" if see_note else "") + "see_early_breakout"

    # ── Liquidity Sweep signal ────────────────────────────────────────────────
    sweep_note = ""
    if liquidity_sweep is not None and liquidity_sweep.detected:
        if action == "buy" and liquidity_sweep.direction == "down":
            # Down sweep = stop hunted lows → potential reversal buy
            conf = min(1.0, conf + 0.04)
            sweep_note = f"liq_sweep_down(+)"
        elif action == "buy" and liquidity_sweep.direction == "up":
            # Up sweep = exhaustion top → avoid buying
            conf *= 0.80
            sweep_note = f"liq_sweep_up(-)"

    # ── Liquidity Trap signal ─────────────────────────────────────────────────
    trap_note = ""
    if liquidity_trap is not None and liquidity_trap.detected:
        if action == "buy" and liquidity_trap.direction == "up":
            # Upward trap → buyers just got trapped → avoid
            conf *= 0.70
            trap_note = f"liq_trap_up(-)"
        elif action == "sell" and liquidity_trap.direction == "down":
            # Downward trap → sellers just got trapped → reversal sell avoided
            conf *= 0.70
            trap_note = f"liq_trap_down(-)"

    # ── Liquidity Vacuum (hard skip) ─────────────────────────────────────────
    vacuum_note = ""
    if liquidity_vacuum is not None and liquidity_vacuum.detected and action == "buy":
        # Large gap above means price could gap up, but it's also risky — reduce conf
        conf *= 0.85
        vacuum_note = f"liq_vacuum(gap={liquidity_vacuum.gap_pct:.2%})"

    # ── Smart Money Footprint ─────────────────────────────────────────────────
    smart_money_note = ""
    if smart_money is not None and smart_money.detected:
        if action == "buy" and smart_money.bias == "accumulation":
            conf = min(1.0, conf + 0.06)
            smart_money_note = f"smart_money_accum(x{smart_money.volume_ratio:.1f})"
        elif action == "buy" and smart_money.bias == "distribution":
            conf *= 0.80
            smart_money_note = f"smart_money_dist(-)"
        elif action == "sell" and smart_money.bias == "distribution":
            conf = min(1.0, conf + 0.04)
            smart_money_note = f"smart_money_dist_sell(+)"

    # ── Volume Acceleration ───────────────────────────────────────────────────
    vaccel_note = ""
    if volume_accel is not None and volume_accel.detected:
        if action == "buy":
            boost = min(0.08, 0.02 * volume_accel.acceleration_ratio)
            conf = min(1.0, conf + boost)
            vaccel_note = f"vol_accel(x{volume_accel.acceleration_ratio:.1f})"

    # ── Micro Trend ───────────────────────────────────────────────────────────
    micro_note = ""
    if micro_trend is not None:
        if action == "buy" and micro_trend.direction == "up":
            conf = min(1.0, conf + 0.03 * micro_trend.strength)
            micro_note = f"micro_up"
        elif action == "buy" and micro_trend.direction == "down":
            conf *= 0.85
            micro_note = f"micro_down(-)"
        elif action == "sell" and micro_trend.direction == "down":
            conf = min(1.0, conf + 0.03 * micro_trend.strength)
            micro_note = f"micro_down_sell"

    ai_note = ""
    if config.ai_scoring_enabled:
        ai_score = _ai_entry_score(trend, orderbook, vol, action, smart_entry)
        weight = config.ai_scoring_weight
        conf = (conf * (1 - weight)) + (ai_score * weight)
        ai_note = f"ai={ai_score:.2f}"

    conf = round(max(0.0, min(1.0, conf)), 3)

    # size based on risk per trade relative to stop distance
    risk_per_unit = abs(current_price - stop_loss) if stop_loss else 0.0
    amount = _position_size(
        current_price, stop_loss, config, risk_per_unit, conf, vol,
        effective_capital, ob_imbalance=orderbook.imbalance,
    )

    pump_size_note = ""
    # Pump-sniper sizing boost: scale entry size when pump is confirmed
    if (
        action == "buy"
        and smart_entry is not None
        and getattr(smart_entry, "pump_sniper", None)
        and smart_entry.pump_sniper.detected
        and smart_entry.pump_sniper.score >= config.pump_sniper_size_min_score
    ):
        amount *= config.pump_sniper_size_multiplier
        pump_size_note = f"pump_size_x{config.pump_sniper_size_multiplier:g}"

    _notes = " ".join(
        n
        for n in (
            sr_note,
            mtf_note,
            whale_note,
            spoof_note,
            see_note,
            sweep_note,
            trap_note,
            vacuum_note,
            smart_money_note,
            vaccel_note,
            micro_note,
            ai_note,
            pump_size_note,
        )
        if n
    )

    # Adaptive sizing note: show effective risk when adaptive mode is on
    capital = effective_capital if (effective_capital is not None and effective_capital > 0) else config.initial_capital
    eff_risk = adaptive_risk_per_trade(capital, config)
    adaptive_note = (
        f"adaptive_risk={eff_risk:.0%}" if config.adaptive_sizing_enabled else ""
    )
    _all_notes = " ".join(n for n in (_notes, adaptive_note) if n)

    reason = (
        f"{mode} | trend={trend.direction} strength={trend.strength:.4f} "
        f"vol={vol.volatility:.4f} ob_imbalance={orderbook.imbalance:.2f} "
        f"rsi={indicators.rsi if indicators else float('nan'):.2f}"
        + (f" {_all_notes}" if _all_notes else "")
    )

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
