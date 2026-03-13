"""Advanced trading strategies module.

Provides 15 strategy categories for the Indodax trading bot:

 1. Trend-following strategies
 2. Mean reversion strategies
 3. Momentum trading
 4. Breakout strategies
 5. Arbitrage strategies
 6. Statistical arbitrage
 7. Market making strategies
 8. Grid trading (enhanced)
 9. Scalping strategies
10. Swing trading strategies
11. Position trading strategies
12. Pairs trading
13. Basket trading
14. Multi-timeframe strategies
15. Hybrid strategies

Each strategy is implemented as a pure function operating on standard market
data (candles, orderbook depth, indicators) and returns typed dataclasses.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis import Candle, OrderbookInsight, TrendResult, VolatilityStats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  1. Trend-Following Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TrendFollowSignal:
    """Signal from trend-following strategy analysis.

    ``action`` is ``"buy"``, ``"sell"``, or ``"hold"``.
    ``strength`` is 0.0–1.0 indicating how strong the trend alignment is.
    ``ma_fast`` / ``ma_slow`` are the moving averages used for crossover.
    """

    action: str
    strength: float
    ma_fast: float
    ma_slow: float
    adx_proxy: float
    reason: str


def trend_following_signal(
    candles: Sequence[Candle],
    fast_period: int = 10,
    slow_period: int = 30,
    strength_threshold: float = 0.3,
) -> TrendFollowSignal:
    """Generate trend-following signal using dual-MA crossover + strength filter.

    :param candles: Recent OHLCV candles.
    :param fast_period: Fast moving average period.
    :param slow_period: Slow moving average period.
    :param strength_threshold: Minimum strength to act on trend.
    :returns: :class:`TrendFollowSignal`.
    """
    no_signal = TrendFollowSignal(
        action="hold", strength=0.0, ma_fast=0.0, ma_slow=0.0,
        adx_proxy=0.0, reason="insufficient data",
    )
    if len(candles) < slow_period:
        return no_signal

    closes = [c.close for c in candles]
    ma_fast = mean(closes[-fast_period:])
    ma_slow = mean(closes[-slow_period:])

    if ma_slow == 0:
        return no_signal

    diff_pct = (ma_fast - ma_slow) / ma_slow

    # ADX proxy: average absolute change over recent candles
    recent = closes[-fast_period:]
    changes = [abs(recent[i] - recent[i - 1]) / recent[i - 1]
               for i in range(1, len(recent)) if recent[i - 1] > 0]
    adx_proxy = mean(changes) if changes else 0.0

    strength = min(1.0, abs(diff_pct) * 10)

    if strength < strength_threshold:
        return TrendFollowSignal(
            action="hold", strength=strength, ma_fast=ma_fast,
            ma_slow=ma_slow, adx_proxy=adx_proxy,
            reason="trend too weak",
        )

    if diff_pct > 0:
        action = "buy"
        reason = f"uptrend: fast_ma({ma_fast:.2f}) > slow_ma({ma_slow:.2f})"
    else:
        action = "sell"
        reason = f"downtrend: fast_ma({ma_fast:.2f}) < slow_ma({ma_slow:.2f})"

    return TrendFollowSignal(
        action=action, strength=strength, ma_fast=ma_fast,
        ma_slow=ma_slow, adx_proxy=adx_proxy, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Mean Reversion Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MeanReversionSignal:
    """Signal from mean-reversion analysis.

    ``z_score`` measures how many standard deviations price is from the mean.
    ``expected_reversion`` is the target price to revert toward.
    """

    action: str
    z_score: float
    current_price: float
    mean_price: float
    expected_reversion: float
    reason: str


def mean_reversion_signal(
    candles: Sequence[Candle],
    lookback: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> MeanReversionSignal:
    """Generate mean-reversion signal using z-score of price deviation.

    When price deviates more than *entry_z* standard deviations from the
    rolling mean, a counter-trend signal fires.

    :param candles: Recent OHLCV candles.
    :param lookback: Period for computing mean and stddev.
    :param entry_z: Z-score threshold for entry.
    :param exit_z: Z-score threshold for taking profit / exiting.
    :returns: :class:`MeanReversionSignal`.
    """
    no_signal = MeanReversionSignal(
        action="hold", z_score=0.0, current_price=0.0,
        mean_price=0.0, expected_reversion=0.0, reason="insufficient data",
    )
    if len(candles) < lookback:
        return no_signal

    closes = [c.close for c in candles[-lookback:]]
    current = candles[-1].close
    avg = mean(closes)
    std = pstdev(closes)

    if std == 0:
        return MeanReversionSignal(
            action="hold", z_score=0.0, current_price=current,
            mean_price=avg, expected_reversion=avg, reason="zero volatility",
        )

    z = (current - avg) / std

    if z <= -entry_z:
        action = "buy"
        reason = f"oversold: z={z:.2f} below -{entry_z}"
    elif z >= entry_z:
        action = "sell"
        reason = f"overbought: z={z:.2f} above +{entry_z}"
    else:
        action = "hold"
        reason = f"z={z:.2f} within range"

    return MeanReversionSignal(
        action=action, z_score=z, current_price=current,
        mean_price=avg, expected_reversion=avg, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Momentum Trading
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MomentumSignal:
    """Signal from momentum-based strategy.

    ``roc`` is the rate-of-change over the lookback period.
    ``acceleration`` indicates whether momentum is increasing or decreasing.
    """

    action: str
    roc: float
    acceleration: float
    strength: float
    reason: str


def momentum_signal(
    candles: Sequence[Candle],
    lookback: int = 14,
    threshold: float = 0.02,
) -> MomentumSignal:
    """Generate momentum signal based on rate-of-change.

    :param candles: Recent OHLCV candles.
    :param lookback: Lookback period for ROC calculation.
    :param threshold: Minimum ROC magnitude to trigger a signal.
    :returns: :class:`MomentumSignal`.
    """
    no_signal = MomentumSignal(
        action="hold", roc=0.0, acceleration=0.0,
        strength=0.0, reason="insufficient data",
    )
    if len(candles) < lookback + 2:
        return no_signal

    current = candles[-1].close
    past = candles[-(lookback + 1)].close
    prev_roc_current = candles[-2].close
    prev_roc_past = candles[-(lookback + 2)].close

    if past == 0 or prev_roc_past == 0:
        return no_signal

    roc = (current - past) / past
    prev_roc = (prev_roc_current - prev_roc_past) / prev_roc_past
    acceleration = roc - prev_roc
    strength = min(1.0, abs(roc) / (threshold * 5))

    if roc > threshold and acceleration >= 0:
        action = "buy"
        reason = f"positive momentum: roc={roc:.4f}, accel={acceleration:.4f}"
    elif roc < -threshold and acceleration <= 0:
        action = "sell"
        reason = f"negative momentum: roc={roc:.4f}, accel={acceleration:.4f}"
    else:
        action = "hold"
        reason = f"roc={roc:.4f}, accel={acceleration:.4f} within threshold"

    return MomentumSignal(
        action=action, roc=roc, acceleration=acceleration,
        strength=strength, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Breakout Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BreakoutSignal:
    """Signal from breakout detection strategy.

    ``breakout_level`` is the price level that was broken.
    ``volume_confirmation`` indicates whether volume supports the breakout.
    """

    action: str
    breakout_level: float
    current_price: float
    volume_confirmation: bool
    strength: float
    reason: str


def breakout_signal(
    candles: Sequence[Candle],
    lookback: int = 20,
    volume_multiplier: float = 1.5,
) -> BreakoutSignal:
    """Detect price breakout from recent range with volume confirmation.

    :param candles: Recent OHLCV candles.
    :param lookback: Period to determine the range.
    :param volume_multiplier: Volume must exceed avg × this for confirmation.
    :returns: :class:`BreakoutSignal`.
    """
    no_signal = BreakoutSignal(
        action="hold", breakout_level=0.0, current_price=0.0,
        volume_confirmation=False, strength=0.0, reason="insufficient data",
    )
    if len(candles) < lookback + 1:
        return no_signal

    range_candles = candles[-(lookback + 1):-1]
    current = candles[-1]

    highs = [c.high for c in range_candles]
    lows = [c.low for c in range_candles]
    volumes = [c.volume for c in range_candles]

    resistance = max(highs)
    support = min(lows)
    avg_vol = mean(volumes) if volumes else 0.0

    vol_confirm = current.volume > avg_vol * volume_multiplier if avg_vol > 0 else False
    price_range = resistance - support
    if price_range <= 0:
        return no_signal

    if current.close > resistance:
        distance = (current.close - resistance) / price_range
        strength = min(1.0, distance * 2) * (1.2 if vol_confirm else 0.7)
        strength = min(1.0, strength)
        return BreakoutSignal(
            action="buy", breakout_level=resistance,
            current_price=current.close,
            volume_confirmation=vol_confirm,
            strength=strength,
            reason=f"upside breakout above {resistance:.2f}"
            + (" (vol confirmed)" if vol_confirm else " (low vol)"),
        )

    if current.close < support:
        distance = (support - current.close) / price_range
        strength = min(1.0, distance * 2) * (1.2 if vol_confirm else 0.7)
        strength = min(1.0, strength)
        return BreakoutSignal(
            action="sell", breakout_level=support,
            current_price=current.close,
            volume_confirmation=vol_confirm,
            strength=strength,
            reason=f"downside breakout below {support:.2f}"
            + (" (vol confirmed)" if vol_confirm else " (low vol)"),
        )

    return BreakoutSignal(
        action="hold", breakout_level=0.0, current_price=current.close,
        volume_confirmation=False, strength=0.0,
        reason=f"price within range [{support:.2f}, {resistance:.2f}]",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Arbitrage Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ArbitrageOpportunity:
    """Detected arbitrage opportunity across price feeds.

    ``spread_pct`` is the price difference as a fraction.
    ``net_profit_pct`` accounts for estimated fees.
    """

    detected: bool
    buy_source: str
    sell_source: str
    buy_price: float
    sell_price: float
    spread_pct: float
    net_profit_pct: float
    reason: str


def detect_arbitrage(
    prices: Dict[str, float],
    fee_pct: float = 0.003,
    min_spread_pct: float = 0.005,
) -> ArbitrageOpportunity:
    """Detect simple arbitrage opportunity across multiple price sources.

    :param prices: Dict mapping source name to price (e.g. ``{"binance": 50000, "indodax": 50200}``).
    :param fee_pct: Combined fee percentage for both legs (default 0.3%).
    :param min_spread_pct: Minimum net spread to qualify.
    :returns: :class:`ArbitrageOpportunity`.
    """
    no_arb = ArbitrageOpportunity(
        detected=False, buy_source="", sell_source="",
        buy_price=0.0, sell_price=0.0, spread_pct=0.0,
        net_profit_pct=0.0, reason="no opportunity",
    )
    if len(prices) < 2:
        return ArbitrageOpportunity(
            detected=False, buy_source="", sell_source="",
            buy_price=0.0, sell_price=0.0, spread_pct=0.0,
            net_profit_pct=0.0, reason="need at least 2 sources",
        )

    entries = [(name, p) for name, p in prices.items() if p > 0]
    if len(entries) < 2:
        return no_arb

    entries.sort(key=lambda x: x[1])
    buy_name, buy_price = entries[0]
    sell_name, sell_price = entries[-1]

    spread = (sell_price - buy_price) / buy_price if buy_price > 0 else 0.0
    net = spread - fee_pct

    if net >= min_spread_pct:
        return ArbitrageOpportunity(
            detected=True, buy_source=buy_name, sell_source=sell_name,
            buy_price=buy_price, sell_price=sell_price,
            spread_pct=spread, net_profit_pct=net,
            reason=f"buy@{buy_name}({buy_price:.2f}) sell@{sell_name}({sell_price:.2f}) net={net:.4f}",
        )

    return no_arb


# ═══════════════════════════════════════════════════════════════════════════
#  6. Statistical Arbitrage
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StatArbSignal:
    """Signal from statistical arbitrage (pairs spread analysis).

    ``spread_z`` is the z-score of the current spread between two assets.
    ``half_life`` estimates mean-reversion speed in candle periods.
    """

    action: str
    spread_z: float
    spread_mean: float
    spread_current: float
    half_life: float
    reason: str


def stat_arb_signal(
    prices_a: Sequence[float],
    prices_b: Sequence[float],
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> StatArbSignal:
    """Generate statistical arbitrage signal from two correlated price series.

    Computes the spread (A − B) z-score and signals when it deviates.

    :param prices_a: Price series for asset A.
    :param prices_b: Price series for asset B.
    :param entry_z: Z-score entry threshold.
    :param exit_z: Z-score exit threshold.
    :returns: :class:`StatArbSignal`.
    """
    no_signal = StatArbSignal(
        action="hold", spread_z=0.0, spread_mean=0.0,
        spread_current=0.0, half_life=0.0, reason="insufficient data",
    )
    min_len = min(len(prices_a), len(prices_b))
    if min_len < 10:
        return no_signal

    spreads = [a - b for a, b in zip(prices_a[-min_len:], prices_b[-min_len:])]
    avg = mean(spreads)
    std = pstdev(spreads)
    if std == 0:
        return StatArbSignal(
            action="hold", spread_z=0.0, spread_mean=avg,
            spread_current=spreads[-1], half_life=0.0,
            reason="zero spread volatility",
        )

    current = spreads[-1]
    z = (current - avg) / std

    # Estimate half-life via simple lag-1 autocorrelation
    diffs = [spreads[i] - spreads[i - 1] for i in range(1, len(spreads))]
    lagged = spreads[:-1]
    if len(lagged) > 1:
        mean_d = mean(diffs)
        mean_l = mean(lagged)
        num = sum((d - mean_d) * (l - mean_l) for d, l in zip(diffs, lagged))
        den = sum((l - mean_l) ** 2 for l in lagged)
        beta = num / den if den != 0 else 0.0
        half_life = -math.log(2) / math.log(abs(beta)) if 0 < abs(beta) < 1 else 0.0
    else:
        half_life = 0.0

    if z >= entry_z:
        action = "sell_a_buy_b"
        reason = f"spread too wide: z={z:.2f}, expect convergence"
    elif z <= -entry_z:
        action = "buy_a_sell_b"
        reason = f"spread too narrow: z={z:.2f}, expect divergence"
    else:
        action = "hold"
        reason = f"spread z={z:.2f} within range"

    return StatArbSignal(
        action=action, spread_z=z, spread_mean=avg,
        spread_current=current, half_life=half_life, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7. Market Making Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketMakingQuote:
    """A single side of a market-making quote."""

    price: float
    size: float


@dataclass
class MarketMakingSignal:
    """Market making strategy output.

    ``bid`` and ``ask`` are the proposed resting orders.
    ``spread_target_pct`` is the desired spread width.
    ``inventory_skew`` adjusts quotes based on current position.
    """

    bid: MarketMakingQuote
    ask: MarketMakingQuote
    spread_target_pct: float
    inventory_skew: float
    reason: str


def market_making_signal(
    mid_price: float,
    volatility: float,
    inventory: float = 0.0,
    base_spread_pct: float = 0.002,
    order_size: float = 1.0,
    max_skew: float = 0.5,
) -> MarketMakingSignal:
    """Generate market-making quotes around mid-price.

    Wider spread in high volatility; inventory skew shifts quotes to
    reduce directional exposure.

    :param mid_price: Current mid-price.
    :param volatility: Recent price volatility (0–1 scale).
    :param inventory: Current inventory (positive = long, negative = short).
    :param base_spread_pct: Base half-spread as fraction.
    :param order_size: Size for each quote.
    :param max_skew: Maximum inventory skew fraction.
    :returns: :class:`MarketMakingSignal`.
    """
    if mid_price <= 0:
        empty_quote = MarketMakingQuote(price=0.0, size=0.0)
        return MarketMakingSignal(
            bid=empty_quote, ask=empty_quote,
            spread_target_pct=0.0, inventory_skew=0.0,
            reason="invalid mid price",
        )

    vol_adj = 1.0 + volatility * 2.0
    half_spread = base_spread_pct * vol_adj

    skew = max(-max_skew, min(max_skew, inventory * 0.1))

    bid_price = mid_price * (1 - half_spread - skew)
    ask_price = mid_price * (1 + half_spread - skew)

    return MarketMakingSignal(
        bid=MarketMakingQuote(price=round(bid_price, 8), size=order_size),
        ask=MarketMakingQuote(price=round(ask_price, 8), size=order_size),
        spread_target_pct=half_spread * 2,
        inventory_skew=skew,
        reason=f"mm quotes: bid={bid_price:.2f} ask={ask_price:.2f} skew={skew:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. Grid Trading (Enhanced)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EnhancedGridLevel:
    """A single level in an enhanced grid strategy."""

    side: str
    price: float
    amount: float
    distance_pct: float
    priority: int


@dataclass
class EnhancedGridPlan:
    """Enhanced grid trading plan with dynamic spacing and sizing.

    ``levels`` are the individual grid orders.
    ``total_capital_required`` is the sum of all order notionals.
    """

    anchor_price: float
    levels: List[EnhancedGridLevel]
    total_capital_required: float
    reason: str


def build_enhanced_grid(
    current_price: float,
    volatility: float,
    capital: float,
    num_levels: int = 5,
    base_spacing_pct: float = 0.01,
    size_per_level: float = 0.0,
) -> EnhancedGridPlan:
    """Build an enhanced grid with volatility-adaptive spacing.

    Wider spacing in high-volatility markets, tighter in low-volatility.

    :param current_price: Current market price.
    :param volatility: Recent price volatility (0–1).
    :param capital: Available capital for the grid.
    :param num_levels: Number of levels per side.
    :param base_spacing_pct: Base spacing between levels.
    :param size_per_level: Fixed size per level; 0 = auto-derive from capital.
    :returns: :class:`EnhancedGridPlan`.
    """
    if current_price <= 0:
        return EnhancedGridPlan(
            anchor_price=0.0, levels=[], total_capital_required=0.0,
            reason="invalid price",
        )

    vol_factor = 1.0 + volatility * 3.0
    spacing = base_spacing_pct * vol_factor

    if size_per_level <= 0 and capital > 0:
        size_per_level = capital / (num_levels * 2 * current_price)

    levels: List[EnhancedGridLevel] = []
    total_cap = 0.0

    for i in range(1, num_levels + 1):
        buy_price = round(current_price * (1 - spacing * i), 8)
        sell_price = round(current_price * (1 + spacing * i), 8)

        levels.append(EnhancedGridLevel(
            side="buy", price=buy_price, amount=size_per_level,
            distance_pct=spacing * i, priority=num_levels - i + 1,
        ))
        levels.append(EnhancedGridLevel(
            side="sell", price=sell_price, amount=size_per_level,
            distance_pct=spacing * i, priority=num_levels - i + 1,
        ))
        total_cap += buy_price * size_per_level + sell_price * size_per_level

    return EnhancedGridPlan(
        anchor_price=current_price, levels=levels,
        total_capital_required=total_cap,
        reason=f"grid: {num_levels}×2 levels, spacing={spacing:.4f}, vol_adj={vol_factor:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Scalping Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScalpSignal:
    """Signal from scalping strategy.

    ``edge_pct`` is the expected profit percentage per trade.
    ``hold_time_estimate`` is estimated hold duration in candle periods.
    """

    action: str
    edge_pct: float
    hold_time_estimate: int
    confidence: float
    reason: str


def scalp_signal(
    candles: Sequence[Candle],
    orderbook: Optional[OrderbookInsight] = None,
    min_edge_pct: float = 0.001,
    max_spread_pct: float = 0.003,
) -> ScalpSignal:
    """Generate scalping signal from micro price action and orderbook state.

    :param candles: Recent short-timeframe candles.
    :param orderbook: Current orderbook insight (spread, imbalance).
    :param min_edge_pct: Minimum expected edge to trade.
    :param max_spread_pct: Maximum acceptable spread.
    :returns: :class:`ScalpSignal`.
    """
    no_signal = ScalpSignal(
        action="hold", edge_pct=0.0, hold_time_estimate=0,
        confidence=0.0, reason="insufficient data",
    )
    if len(candles) < 5:
        return no_signal

    if orderbook and orderbook.spread_pct > max_spread_pct:
        return ScalpSignal(
            action="hold", edge_pct=0.0, hold_time_estimate=0,
            confidence=0.0, reason=f"spread too wide: {orderbook.spread_pct:.4f}",
        )

    recent = candles[-5:]
    avg_range = mean([(c.high - c.low) / c.close for c in recent if c.close > 0])
    spread_cost = orderbook.spread_pct if orderbook else 0.001
    edge = avg_range - spread_cost

    imbalance = orderbook.imbalance if orderbook else 0.0

    if edge >= min_edge_pct and imbalance > 0.1:
        action = "buy"
        confidence = min(1.0, edge / min_edge_pct * 0.3 + imbalance * 0.5)
        reason = f"scalp buy: edge={edge:.4f}, imbalance={imbalance:.2f}"
    elif edge >= min_edge_pct and imbalance < -0.1:
        action = "sell"
        confidence = min(1.0, edge / min_edge_pct * 0.3 + abs(imbalance) * 0.5)
        reason = f"scalp sell: edge={edge:.4f}, imbalance={imbalance:.2f}"
    else:
        return ScalpSignal(
            action="hold", edge_pct=edge, hold_time_estimate=0,
            confidence=0.0, reason=f"no scalp edge: edge={edge:.4f}",
        )

    return ScalpSignal(
        action=action, edge_pct=edge, hold_time_estimate=2,
        confidence=confidence, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Swing Trading Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SwingSignal:
    """Signal from swing trading strategy.

    ``swing_high`` / ``swing_low`` are the detected pivot levels.
    ``target`` is the expected price target.
    """

    action: str
    swing_high: float
    swing_low: float
    target: float
    stop_loss: float
    risk_reward: float
    reason: str


def swing_signal(
    candles: Sequence[Candle],
    min_risk_reward: float = 2.0,
) -> SwingSignal:
    """Generate swing trading signal from pivot point analysis.

    :param candles: Daily or 4H candles.
    :param min_risk_reward: Minimum risk:reward ratio to act.
    :returns: :class:`SwingSignal`.
    """
    no_signal = SwingSignal(
        action="hold", swing_high=0.0, swing_low=0.0,
        target=0.0, stop_loss=0.0, risk_reward=0.0,
        reason="insufficient data",
    )
    if len(candles) < 10:
        return no_signal

    highs = [c.high for c in candles[-10:]]
    lows = [c.low for c in candles[-10:]]
    current = candles[-1].close

    swing_high = max(highs)
    swing_low = min(lows)
    price_range = swing_high - swing_low

    if price_range <= 0 or current <= 0:
        return no_signal

    dist_to_high = swing_high - current
    dist_to_low = current - swing_low

    if dist_to_low > 0 and dist_to_high / dist_to_low >= min_risk_reward:
        return SwingSignal(
            action="buy", swing_high=swing_high, swing_low=swing_low,
            target=swing_high, stop_loss=swing_low,
            risk_reward=dist_to_high / dist_to_low,
            reason=f"swing buy near support, R:R={dist_to_high / dist_to_low:.2f}",
        )

    if dist_to_high > 0 and dist_to_low / dist_to_high >= min_risk_reward:
        return SwingSignal(
            action="sell", swing_high=swing_high, swing_low=swing_low,
            target=swing_low, stop_loss=swing_high,
            risk_reward=dist_to_low / dist_to_high,
            reason=f"swing sell near resistance, R:R={dist_to_low / dist_to_high:.2f}",
        )

    return SwingSignal(
        action="hold", swing_high=swing_high, swing_low=swing_low,
        target=0.0, stop_loss=0.0, risk_reward=0.0,
        reason="no swing setup with sufficient R:R",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Position Trading Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PositionSignal:
    """Signal from position (long-term) trading strategy.

    Uses macro trend + momentum to determine long-horizon entries.
    ``trend_score`` aggregates multiple timeframe trends.
    """

    action: str
    trend_score: float
    momentum_score: float
    combined_score: float
    reason: str


def position_signal(
    candles: Sequence[Candle],
    long_period: int = 50,
    threshold: float = 0.4,
) -> PositionSignal:
    """Generate position trading signal from long-term trend assessment.

    :param candles: Long-timeframe candles (daily or weekly).
    :param long_period: Moving average period for trend.
    :param threshold: Minimum combined score to act.
    :returns: :class:`PositionSignal`.
    """
    no_signal = PositionSignal(
        action="hold", trend_score=0.0, momentum_score=0.0,
        combined_score=0.0, reason="insufficient data",
    )
    if len(candles) < long_period:
        return no_signal

    closes = [c.close for c in candles]
    ma_long = mean(closes[-long_period:])
    current = closes[-1]

    if ma_long == 0:
        return no_signal

    trend_score = (current - ma_long) / ma_long

    short_period = max(5, long_period // 10)
    if len(closes) >= short_period + 1:
        recent = closes[-short_period:]
        past_price = closes[-(short_period + 1)]
        momentum_score = (recent[-1] - past_price) / past_price if past_price > 0 else 0.0
    else:
        momentum_score = 0.0

    combined = trend_score * 0.6 + momentum_score * 0.4

    if combined >= threshold:
        action = "buy"
        reason = f"bullish position: trend={trend_score:.4f}, momentum={momentum_score:.4f}"
    elif combined <= -threshold:
        action = "sell"
        reason = f"bearish position: trend={trend_score:.4f}, momentum={momentum_score:.4f}"
    else:
        action = "hold"
        reason = f"neutral: combined={combined:.4f}"

    return PositionSignal(
        action=action, trend_score=trend_score,
        momentum_score=momentum_score, combined_score=combined,
        reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Pairs Trading
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PairsTradeSignal:
    """Signal for pairs trading between two assets.

    ``ratio_z`` is the z-score of the price ratio.
    ``hedge_ratio`` is the beta-adjusted hedge ratio.
    """

    action: str
    pair_a: str
    pair_b: str
    ratio_current: float
    ratio_mean: float
    ratio_z: float
    hedge_ratio: float
    reason: str


def pairs_trade_signal(
    pair_a: str,
    pair_b: str,
    prices_a: Sequence[float],
    prices_b: Sequence[float],
    entry_z: float = 2.0,
) -> PairsTradeSignal:
    """Generate pairs-trading signal based on price ratio z-score.

    :param pair_a: Name of first asset.
    :param pair_b: Name of second asset.
    :param prices_a: Price series for asset A.
    :param prices_b: Price series for asset B.
    :param entry_z: Z-score threshold for entry.
    :returns: :class:`PairsTradeSignal`.
    """
    no_signal = PairsTradeSignal(
        action="hold", pair_a=pair_a, pair_b=pair_b,
        ratio_current=0.0, ratio_mean=0.0, ratio_z=0.0,
        hedge_ratio=1.0, reason="insufficient data",
    )
    min_len = min(len(prices_a), len(prices_b))
    if min_len < 10:
        return no_signal

    ratios = [a / b for a, b in zip(prices_a[-min_len:], prices_b[-min_len:]) if b > 0]
    if len(ratios) < 10:
        return no_signal

    avg = mean(ratios)
    std = pstdev(ratios)
    if std == 0 or avg == 0:
        return PairsTradeSignal(
            action="hold", pair_a=pair_a, pair_b=pair_b,
            ratio_current=ratios[-1], ratio_mean=avg, ratio_z=0.0,
            hedge_ratio=1.0, reason="zero ratio volatility",
        )

    current_ratio = ratios[-1]
    z = (current_ratio - avg) / std
    hedge_ratio = avg

    if z >= entry_z:
        action = "sell_a_buy_b"
        reason = f"ratio high: z={z:.2f}, sell {pair_a} buy {pair_b}"
    elif z <= -entry_z:
        action = "buy_a_sell_b"
        reason = f"ratio low: z={z:.2f}, buy {pair_a} sell {pair_b}"
    else:
        action = "hold"
        reason = f"ratio z={z:.2f} within range"

    return PairsTradeSignal(
        action=action, pair_a=pair_a, pair_b=pair_b,
        ratio_current=current_ratio, ratio_mean=avg,
        ratio_z=z, hedge_ratio=hedge_ratio, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Basket Trading
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BasketComponent:
    """A single component (asset) within a basket."""

    pair: str
    weight: float
    signal: str       # "buy", "sell", "hold"
    strength: float


@dataclass
class BasketSignal:
    """Signal for basket (portfolio) trading.

    ``aggregate_signal`` is the net weighted sentiment across all components.
    ``rebalance_needed`` indicates if portfolio weights have drifted.
    """

    action: str
    components: List[BasketComponent]
    aggregate_score: float
    rebalance_needed: bool
    reason: str


def basket_signal(
    components: Sequence[Dict[str, Any]],
    rebalance_threshold: float = 0.1,
) -> BasketSignal:
    """Generate basket trading signal from multiple asset signals.

    Each component should have keys: ``"pair"``, ``"weight"``, ``"signal"``
    (buy/sell/hold), ``"strength"`` (0–1), ``"current_weight"`` (actual).

    :param components: List of component dicts.
    :param rebalance_threshold: Weight drift threshold for rebalancing.
    :returns: :class:`BasketSignal`.
    """
    if not components:
        return BasketSignal(
            action="hold", components=[], aggregate_score=0.0,
            rebalance_needed=False, reason="empty basket",
        )

    basket_components: List[BasketComponent] = []
    weighted_score = 0.0
    rebalance = False

    for c in components:
        pair = str(c.get("pair", ""))
        weight = _safe_float(c.get("weight", 0))
        signal = str(c.get("signal", "hold")).lower()
        strength = _safe_float(c.get("strength", 0))
        current_w = _safe_float(c.get("current_weight", weight))

        if signal == "buy":
            score = strength
        elif signal == "sell":
            score = -strength
        else:
            score = 0.0

        weighted_score += score * weight

        if abs(current_w - weight) > rebalance_threshold:
            rebalance = True

        basket_components.append(BasketComponent(
            pair=pair, weight=weight, signal=signal, strength=strength,
        ))

    if weighted_score > 0.2:
        action = "buy"
        reason = f"basket bullish: score={weighted_score:.4f}"
    elif weighted_score < -0.2:
        action = "sell"
        reason = f"basket bearish: score={weighted_score:.4f}"
    else:
        action = "hold"
        reason = f"basket neutral: score={weighted_score:.4f}"

    return BasketSignal(
        action=action, components=basket_components,
        aggregate_score=weighted_score,
        rebalance_needed=rebalance, reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 14. Multi-Timeframe Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TimeframeView:
    """Analysis on a single timeframe."""

    timeframe: str
    trend: str        # "up", "down", "neutral"
    strength: float
    ma_value: float


@dataclass
class MultiTimeframeSignal:
    """Signal from multi-timeframe analysis.

    ``aligned`` is True when all timeframes agree on direction.
    ``dominant_trend`` is the consensus direction.
    """

    action: str
    views: List[TimeframeView]
    aligned: bool
    dominant_trend: str
    alignment_score: float
    reason: str


def multi_timeframe_signal(
    timeframes: Dict[str, Sequence[Candle]],
    ma_period: int = 20,
    alignment_threshold: float = 0.6,
) -> MultiTimeframeSignal:
    """Generate signal from multi-timeframe trend alignment.

    :param timeframes: Dict mapping timeframe label to candle series
        (e.g. ``{"1h": candles_1h, "4h": candles_4h, "1d": candles_1d}``).
    :param ma_period: Moving average period for trend on each TF.
    :param alignment_threshold: Fraction of TFs that must agree.
    :returns: :class:`MultiTimeframeSignal`.
    """
    if not timeframes:
        return MultiTimeframeSignal(
            action="hold", views=[], aligned=False,
            dominant_trend="neutral", alignment_score=0.0,
            reason="no timeframes provided",
        )

    views: List[TimeframeView] = []

    for tf_label, candles in timeframes.items():
        if len(candles) < ma_period:
            views.append(TimeframeView(
                timeframe=tf_label, trend="neutral",
                strength=0.0, ma_value=0.0,
            ))
            continue

        closes = [c.close for c in candles]
        ma = mean(closes[-ma_period:])
        current = closes[-1]

        if ma == 0:
            views.append(TimeframeView(
                timeframe=tf_label, trend="neutral",
                strength=0.0, ma_value=ma,
            ))
            continue

        diff = (current - ma) / ma
        if diff > 0.01:
            trend = "up"
        elif diff < -0.01:
            trend = "down"
        else:
            trend = "neutral"

        views.append(TimeframeView(
            timeframe=tf_label, trend=trend,
            strength=min(1.0, abs(diff) * 10),
            ma_value=ma,
        ))

    up_count = sum(1 for v in views if v.trend == "up")
    down_count = sum(1 for v in views if v.trend == "down")
    total = len(views) or 1

    up_ratio = up_count / total
    down_ratio = down_count / total

    if up_ratio >= alignment_threshold:
        aligned = True
        dominant = "up"
        action = "buy"
        reason = f"multi-tf aligned UP ({up_count}/{total})"
    elif down_ratio >= alignment_threshold:
        aligned = True
        dominant = "down"
        action = "sell"
        reason = f"multi-tf aligned DOWN ({down_count}/{total})"
    else:
        aligned = False
        dominant = "neutral"
        action = "hold"
        reason = f"multi-tf mixed: up={up_count}, down={down_count}"

    alignment_score = max(up_ratio, down_ratio)

    return MultiTimeframeSignal(
        action=action, views=views, aligned=aligned,
        dominant_trend=dominant, alignment_score=alignment_score,
        reason=reason,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 15. Hybrid Strategies
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyVote:
    """A single strategy's vote in the hybrid consensus."""

    strategy_name: str
    action: str
    weight: float
    confidence: float


@dataclass
class HybridSignal:
    """Signal from hybrid (ensemble) strategy combining multiple sub-strategies.

    ``consensus_score`` ranges from −1 (strong sell) to +1 (strong buy).
    ``agreement_pct`` is the fraction of strategies that agree with the
    chosen action.
    """

    action: str
    votes: List[StrategyVote]
    consensus_score: float
    agreement_pct: float
    reason: str


def hybrid_signal(
    votes: Sequence[Dict[str, Any]],
    min_agreement: float = 0.5,
    min_score: float = 0.2,
) -> HybridSignal:
    """Combine multiple strategy signals into a consensus hybrid signal.

    Each vote should have keys: ``"strategy"``, ``"action"`` (buy/sell/hold),
    ``"weight"`` (0–1), ``"confidence"`` (0–1).

    :param votes: List of strategy vote dicts.
    :param min_agreement: Minimum agreement fraction to act.
    :param min_score: Minimum absolute consensus score to act.
    :returns: :class:`HybridSignal`.
    """
    if not votes:
        return HybridSignal(
            action="hold", votes=[], consensus_score=0.0,
            agreement_pct=0.0, reason="no strategy votes",
        )

    parsed: List[StrategyVote] = []
    total_weight = 0.0
    weighted_score = 0.0

    for v in votes:
        name = str(v.get("strategy", "unknown"))
        action = str(v.get("action", "hold")).lower()
        weight = _safe_float(v.get("weight", 1.0))
        confidence = _safe_float(v.get("confidence", 0.5))

        if action == "buy":
            score = confidence
        elif action == "sell":
            score = -confidence
        else:
            score = 0.0

        weighted_score += score * weight
        total_weight += weight

        parsed.append(StrategyVote(
            strategy_name=name, action=action,
            weight=weight, confidence=confidence,
        ))

    consensus = weighted_score / total_weight if total_weight > 0 else 0.0

    if consensus > 0:
        proposed = "buy"
    elif consensus < 0:
        proposed = "sell"
    else:
        proposed = "hold"

    agree_count = sum(1 for v in parsed if v.action == proposed)
    agreement = agree_count / len(parsed) if parsed else 0.0

    if abs(consensus) >= min_score and agreement >= min_agreement:
        action = proposed
        reason = f"hybrid consensus: score={consensus:.4f}, agree={agreement:.0%}"
    else:
        action = "hold"
        reason = f"no consensus: score={consensus:.4f}, agree={agreement:.0%}"

    return HybridSignal(
        action=action, votes=parsed, consensus_score=consensus,
        agreement_pct=agreement, reason=reason,
    )
