"""Comprehensive technical indicators for the Indodax trading bot.

This module provides 20 categories of technical analysis indicators that
operate on :class:`~bot.analysis.Candle` sequences.  Every public function
accepts a list of :class:`Candle` objects and returns a typed dataclass or
plain numeric result so that downstream strategy code can consume the data
without guessing field names.

Existing indicators (RSI, MACD, Bollinger Bands, SMA, EMA) live in
``bot.analysis``.  This module adds the remaining indicators requested:

 1. WMA (Weighted Moving Average)
 2. Stochastic Oscillator
 3. ATR (Average True Range)
 4. VWAP (Volume Weighted Average Price)
 5. Volume Profile
 6. Ichimoku Cloud
 7. Donchian Channel
 8. Keltner Channel
 9. Fibonacci Retracement
10. Pivot Points
11. Trendline Detection
12. Support / Resistance Detection
13. Pattern Recognition (triangles, flags, head & shoulders)
14. Momentum Indicators
15. Volatility Indicators
16. Volume Indicators
17. Custom Indicator Framework
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from bot.analysis import Candle, _ema_series, compute_rsi, compute_macd, bollinger_bands

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
#  Result dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StochasticResult:
    """Stochastic Oscillator (%K and %D)."""
    k: float
    d: float


@dataclass
class IchimokuCloud:
    """Ichimoku Kinko Hyo components."""
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float


@dataclass
class DonchianChannel:
    """Donchian Channel (upper / mid / lower)."""
    upper: float
    mid: float
    lower: float


@dataclass
class KeltnerChannel:
    """Keltner Channel (upper / mid / lower)."""
    upper: float
    mid: float
    lower: float


@dataclass
class FibonacciLevels:
    """Fibonacci retracement levels from a detected swing high/low."""
    high: float
    low: float
    level_0: float
    level_236: float
    level_382: float
    level_500: float
    level_618: float
    level_786: float
    level_1: float


@dataclass
class PivotPoints:
    """Standard / classic pivot point levels."""
    pivot: float
    r1: float
    r2: float
    r3: float
    s1: float
    s2: float
    s3: float


@dataclass
class Trendline:
    """A detected trendline (slope + intercept in index-space)."""
    slope: float
    intercept: float
    direction: str  # "up", "down", "flat"
    strength: float  # R² of the regression


@dataclass
class SRLevel:
    """A support or resistance level with a strength score."""
    price: float
    kind: str  # "support" or "resistance"
    touches: int
    strength: float


@dataclass
class PatternResult:
    """Result from pattern recognition."""
    detected: bool
    pattern: str  # e.g. "head_and_shoulders", "triangle", "flag"
    direction: str  # "bullish" or "bearish"
    confidence: float


@dataclass
class VolumeProfileLevel:
    """A single price level in a volume profile."""
    price_low: float
    price_high: float
    volume: float


@dataclass
class VolumeProfile:
    """Volume distribution across price levels."""
    levels: List[VolumeProfileLevel]
    poc: float  # point-of-control (price level with highest volume)
    value_area_high: float
    value_area_low: float


@dataclass
class MomentumSnapshot:
    """Combined momentum indicator snapshot."""
    rsi: float
    stochastic_k: float
    stochastic_d: float
    macd: float
    macd_signal: float
    macd_hist: float
    roc: float  # rate of change
    williams_r: float


@dataclass
class VolatilitySnapshot:
    """Combined volatility indicator snapshot."""
    atr: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_width: float
    keltner_upper: float
    keltner_mid: float
    keltner_lower: float
    historical_vol: float


@dataclass
class VolumeSnapshot:
    """Combined volume indicator snapshot."""
    obv: float  # on-balance volume
    vwap: float
    mfi: float  # money flow index
    cmf: float  # Chaikin money flow
    volume_sma: float
    volume_ratio: float  # current vol / sma


# ═══════════════════════════════════════════════════════════════════════════
#  1. Weighted Moving Average (WMA)
# ═══════════════════════════════════════════════════════════════════════════

def compute_wma(values: Sequence[float], period: int) -> List[float]:
    """Return a Weighted Moving Average series.

    Each output element is ``NaN`` when fewer than *period* values are
    available.  Weights increase linearly: the most recent observation
    carries weight *period*, the one before that *period − 1*, etc.
    """
    n = len(values)
    result: List[float] = []
    denom = period * (period + 1) / 2
    for i in range(n):
        if i + 1 < period:
            result.append(math.nan)
        else:
            window = values[i + 1 - period: i + 1]
            wma = sum(w * v for w, v in zip(range(1, period + 1), window)) / denom
            result.append(wma)
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  2. Stochastic Oscillator
# ═══════════════════════════════════════════════════════════════════════════

def compute_stochastic(
    candles: Sequence[Candle],
    k_period: int = 14,
    d_period: int = 3,
) -> StochasticResult:
    """Compute the Stochastic Oscillator (%K, %D).

    %K measures where the close is relative to the high-low range over the
    last *k_period* candles.  %D is a *d_period* SMA of %K values.
    """
    if len(candles) < k_period:
        return StochasticResult(k=50.0, d=50.0)

    k_values: List[float] = []
    for i in range(k_period - 1, len(candles)):
        window = candles[i + 1 - k_period: i + 1]
        highest = max(c.high for c in window)
        lowest = min(c.low for c in window)
        if highest == lowest:
            k_values.append(50.0)
        else:
            k_values.append(100 * (candles[i].close - lowest) / (highest - lowest))

    k = k_values[-1]
    d = mean(k_values[-d_period:]) if len(k_values) >= d_period else k
    return StochasticResult(k=k, d=d)


# ═══════════════════════════════════════════════════════════════════════════
#  3. ATR (Average True Range)
# ═══════════════════════════════════════════════════════════════════════════

def compute_atr(candles: Sequence[Candle], period: int = 14) -> float:
    """Compute the Average True Range over the last *period* candles.

    True Range is ``max(high − low, |high − prev_close|, |low − prev_close|)``.
    ATR is the simple mean of the last *period* true-range values.
    """
    if len(candles) < 2:
        return 0.0

    true_ranges: List[float] = []
    for i in range(1, len(candles)):
        h = candles[i].high
        l = candles[i].low
        pc = candles[i - 1].close
        tr = max(h - l, abs(h - pc), abs(l - pc))
        true_ranges.append(tr)

    if not true_ranges:
        return 0.0
    window = true_ranges[-period:]
    return mean(window)


# ═══════════════════════════════════════════════════════════════════════════
#  4. VWAP (Volume Weighted Average Price)
# ═══════════════════════════════════════════════════════════════════════════

def compute_vwap(candles: Sequence[Candle]) -> float:
    """Compute the Volume Weighted Average Price for a sequence of candles.

    VWAP = Σ(typical_price × volume) / Σ(volume)
    where typical_price = (high + low + close) / 3.
    """
    if not candles:
        return 0.0
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for c in candles:
        tp = (c.high + c.low + c.close) / 3
        cum_tp_vol += tp * c.volume
        cum_vol += c.volume
    return cum_tp_vol / cum_vol if cum_vol > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  5. Volume Profile
# ═══════════════════════════════════════════════════════════════════════════

def compute_volume_profile(
    candles: Sequence[Candle],
    num_bins: int = 20,
    value_area_pct: float = 0.70,
) -> VolumeProfile:
    """Build a volume-at-price profile over the candle set.

    Divides the price range into *num_bins* equal-width bins and sums the
    volume of every candle that has its typical price in that bin.

    The *point of control* (POC) is the bin with the most volume.
    The *value area* contains the bins surrounding the POC that together
    account for *value_area_pct* of total volume.
    """
    empty = VolumeProfile(
        levels=[], poc=0.0, value_area_high=0.0, value_area_low=0.0,
    )
    if not candles:
        return empty

    low_price = min(c.low for c in candles)
    high_price = max(c.high for c in candles)
    if high_price == low_price:
        return VolumeProfile(
            levels=[VolumeProfileLevel(low_price, high_price, sum(c.volume for c in candles))],
            poc=low_price,
            value_area_high=high_price,
            value_area_low=low_price,
        )

    bin_width = (high_price - low_price) / num_bins
    bins: List[VolumeProfileLevel] = []
    bin_volumes: List[float] = []
    for i in range(num_bins):
        bl = low_price + i * bin_width
        bh = bl + bin_width
        bins.append(VolumeProfileLevel(bl, bh, 0.0))
        bin_volumes.append(0.0)

    for c in candles:
        tp = (c.high + c.low + c.close) / 3
        idx = min(int((tp - low_price) / bin_width), num_bins - 1)
        bin_volumes[idx] += c.volume
        bins[idx] = VolumeProfileLevel(bins[idx].price_low, bins[idx].price_high, bin_volumes[idx])

    poc_idx = max(range(num_bins), key=lambda i: bin_volumes[i])
    poc_price = (bins[poc_idx].price_low + bins[poc_idx].price_high) / 2

    total_vol = sum(bin_volumes)
    if total_vol == 0:
        return VolumeProfile(levels=bins, poc=poc_price, value_area_high=high_price, value_area_low=low_price)

    # Expand from POC outward until value_area_pct is covered.
    va_vol = bin_volumes[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while va_vol / total_vol < value_area_pct and (lo_idx > 0 or hi_idx < num_bins - 1):
        lo_vol = bin_volumes[lo_idx - 1] if lo_idx > 0 else -1.0
        hi_vol = bin_volumes[hi_idx + 1] if hi_idx < num_bins - 1 else -1.0
        if lo_vol >= hi_vol and lo_idx > 0:
            lo_idx -= 1
            va_vol += bin_volumes[lo_idx]
        elif hi_idx < num_bins - 1:
            hi_idx += 1
            va_vol += bin_volumes[hi_idx]
        else:
            break

    return VolumeProfile(
        levels=bins,
        poc=poc_price,
        value_area_high=bins[hi_idx].price_high,
        value_area_low=bins[lo_idx].price_low,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6. Ichimoku Cloud
# ═══════════════════════════════════════════════════════════════════════════

def _period_midpoint(candles: Sequence[Candle], period: int) -> float:
    """Return the midpoint of the high-low range over the last *period* candles."""
    window = candles[-period:] if len(candles) >= period else candles
    if not window:
        return 0.0
    return (max(c.high for c in window) + min(c.low for c in window)) / 2


def compute_ichimoku(
    candles: Sequence[Candle],
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
) -> IchimokuCloud:
    """Compute the five Ichimoku Cloud components.

    * Tenkan-sen (conversion line): midpoint of last *tenkan_period*.
    * Kijun-sen (base line): midpoint of last *kijun_period*.
    * Senkou Span A: average of Tenkan-sen and Kijun-sen.
    * Senkou Span B: midpoint of last *senkou_b_period*.
    * Chikou Span: current close (normally plotted 26 periods back).
    """
    if not candles:
        return IchimokuCloud(0.0, 0.0, 0.0, 0.0, 0.0)

    tenkan = _period_midpoint(candles, tenkan_period)
    kijun = _period_midpoint(candles, kijun_period)
    span_a = (tenkan + kijun) / 2
    span_b = _period_midpoint(candles, senkou_b_period)
    chikou = candles[-1].close

    return IchimokuCloud(
        tenkan_sen=tenkan,
        kijun_sen=kijun,
        senkou_span_a=span_a,
        senkou_span_b=span_b,
        chikou_span=chikou,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7. Donchian Channel
# ═══════════════════════════════════════════════════════════════════════════

def compute_donchian(candles: Sequence[Candle], period: int = 20) -> DonchianChannel:
    """Compute the Donchian Channel (highest high / lowest low over *period*)."""
    if not candles:
        return DonchianChannel(0.0, 0.0, 0.0)

    window = candles[-period:] if len(candles) >= period else candles
    upper = max(c.high for c in window)
    lower = min(c.low for c in window)
    mid = (upper + lower) / 2
    return DonchianChannel(upper=upper, mid=mid, lower=lower)


# ═══════════════════════════════════════════════════════════════════════════
#  8. Keltner Channel
# ═══════════════════════════════════════════════════════════════════════════

def compute_keltner(
    candles: Sequence[Candle],
    ema_period: int = 20,
    atr_period: int = 14,
    multiplier: float = 2.0,
) -> KeltnerChannel:
    """Compute the Keltner Channel (EMA ± multiplier × ATR)."""
    if not candles:
        return KeltnerChannel(0.0, 0.0, 0.0)

    closes = [c.close for c in candles]
    ema_vals = _ema_series(closes, ema_period)
    ema_mid = ema_vals[-1] if ema_vals else 0.0
    atr = compute_atr(candles, atr_period)
    return KeltnerChannel(
        upper=ema_mid + multiplier * atr,
        mid=ema_mid,
        lower=ema_mid - multiplier * atr,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Fibonacci Retracement
# ═══════════════════════════════════════════════════════════════════════════

def compute_fibonacci(
    candles: Sequence[Candle],
    lookback: int = 50,
) -> FibonacciLevels:
    """Compute Fibonacci retracement levels from the swing high/low over *lookback*.

    The swing high and swing low are the max high and min low over the
    last *lookback* candles.  The seven standard retracement levels
    (0 %, 23.6 %, 38.2 %, 50 %, 61.8 %, 78.6 %, 100 %) are returned.
    """
    if not candles:
        return FibonacciLevels(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    window = candles[-lookback:] if len(candles) >= lookback else candles
    high = max(c.high for c in window)
    low = min(c.low for c in window)
    diff = high - low

    return FibonacciLevels(
        high=high,
        low=low,
        level_0=high,
        level_236=high - 0.236 * diff,
        level_382=high - 0.382 * diff,
        level_500=high - 0.500 * diff,
        level_618=high - 0.618 * diff,
        level_786=high - 0.786 * diff,
        level_1=low,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Pivot Points
# ═══════════════════════════════════════════════════════════════════════════

def compute_pivot_points(candles: Sequence[Candle]) -> PivotPoints:
    """Compute classic pivot points from the most recent candle (or last candle).

    Uses the standard formula:
        P  = (H + L + C) / 3
        R1 = 2P − L,  S1 = 2P − H
        R2 = P + (H − L),  S2 = P − (H − L)
        R3 = H + 2(P − L),  S3 = L − 2(H − P)
    """
    if not candles:
        return PivotPoints(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    c = candles[-1]
    p = (c.high + c.low + c.close) / 3
    r1 = 2 * p - c.low
    s1 = 2 * p - c.high
    r2 = p + (c.high - c.low)
    s2 = p - (c.high - c.low)
    r3 = c.high + 2 * (p - c.low)
    s3 = c.low - 2 * (c.high - p)
    return PivotPoints(pivot=p, r1=r1, r2=r2, r3=r3, s1=s1, s2=s2, s3=s3)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Trendline Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_trendline(
    candles: Sequence[Candle],
    lookback: int = 30,
) -> Trendline:
    """Fit a simple linear regression to closing prices over the last *lookback* candles.

    The slope and intercept are expressed in *price-per-index* space.
    ``strength`` is the R² value of the fit.
    """
    neutral = Trendline(slope=0.0, intercept=0.0, direction="flat", strength=0.0)
    if len(candles) < 3:
        return neutral

    window = candles[-lookback:] if len(candles) >= lookback else candles
    n = len(window)
    closes = [c.close for c in window]

    x_mean = (n - 1) / 2.0
    y_mean = mean(closes)

    ss_xy = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(closes))
    ss_xx = sum((i - x_mean) ** 2 for i in range(n))

    if ss_xx == 0:
        return neutral

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    # R²
    ss_tot = sum((y - y_mean) ** 2 for y in closes)
    if ss_tot == 0:
        r_squared = 0.0
    else:
        ss_res = sum((y - (slope * i + intercept)) ** 2 for i, y in enumerate(closes))
        r_squared = max(0.0, 1 - ss_res / ss_tot)

    if slope > 0:
        direction = "up"
    elif slope < 0:
        direction = "down"
    else:
        direction = "flat"

    return Trendline(slope=slope, intercept=intercept, direction=direction, strength=r_squared)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Support / Resistance Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_support_resistance(
    candles: Sequence[Candle],
    lookback: int = 50,
    tolerance_pct: float = 0.005,
    min_touches: int = 2,
) -> List[SRLevel]:
    """Identify support and resistance levels based on local extremes.

    Scans for local highs/lows in the price data and clusters nearby
    extremes within *tolerance_pct* of each other.  Only levels with at
    least *min_touches* are returned, sorted by strength (descending).
    """
    if len(candles) < 5:
        return []

    window = candles[-lookback:] if len(candles) >= lookback else candles

    extremes: List[Tuple[float, str]] = []
    for i in range(1, len(window) - 1):
        if window[i].high >= window[i - 1].high and window[i].high >= window[i + 1].high:
            extremes.append((window[i].high, "resistance"))
        if window[i].low <= window[i - 1].low and window[i].low <= window[i + 1].low:
            extremes.append((window[i].low, "support"))

    if not extremes:
        return []

    # Cluster nearby extremes.
    extremes.sort(key=lambda x: x[0])
    clusters: List[List[Tuple[float, str]]] = [[extremes[0]]]
    for price, kind in extremes[1:]:
        cluster_mean = mean(p for p, _ in clusters[-1])
        if abs(price - cluster_mean) / cluster_mean <= tolerance_pct:
            clusters[-1].append((price, kind))
        else:
            clusters.append([(price, kind)])

    levels: List[SRLevel] = []
    for cluster in clusters:
        touches = len(cluster)
        if touches < min_touches:
            continue
        avg_price = mean(p for p, _ in cluster)
        # Majority vote for kind.
        support_count = sum(1 for _, k in cluster if k == "support")
        kind = "support" if support_count > len(cluster) / 2 else "resistance"
        strength = touches / len(window)
        levels.append(SRLevel(price=avg_price, kind=kind, touches=touches, strength=strength))

    levels.sort(key=lambda l: l.strength, reverse=True)
    return levels


# ═══════════════════════════════════════════════════════════════════════════
# 13. Pattern Recognition
# ═══════════════════════════════════════════════════════════════════════════

def _detect_head_and_shoulders(candles: Sequence[Candle], lookback: int = 30) -> Optional[PatternResult]:
    """Detect head-and-shoulders (bearish) or inverse (bullish) patterns."""
    if len(candles) < lookback:
        return None

    window = candles[-lookback:]
    highs = [c.high for c in window]
    n = len(highs)

    # Find the three highest peaks with spacing constraints.
    peaks: List[Tuple[int, float]] = []
    for i in range(1, n - 1):
        if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            peaks.append((i, highs[i]))

    if len(peaks) < 3:
        return None

    # Sort by height descending, take top-3 and sort by position.
    top3 = sorted(peaks, key=lambda p: p[1], reverse=True)[:3]
    top3.sort(key=lambda p: p[0])

    left, head, right = top3
    # Head should be the highest and in the middle.
    if head[1] <= left[1] or head[1] <= right[1]:
        return None
    if not (left[0] < head[0] < right[0]):
        return None

    # Shoulders should be roughly equal (within 15%).
    shoulder_diff = abs(left[1] - right[1])
    shoulder_avg = (left[1] + right[1]) / 2
    if shoulder_avg == 0 or shoulder_diff / shoulder_avg > 0.15:
        return None

    confidence = min(1.0, (head[1] - shoulder_avg) / shoulder_avg * 10)

    return PatternResult(
        detected=True,
        pattern="head_and_shoulders",
        direction="bearish",
        confidence=confidence,
    )


def _detect_triangle(candles: Sequence[Candle], lookback: int = 20) -> Optional[PatternResult]:
    """Detect converging triangle patterns (symmetric/ascending/descending)."""
    if len(candles) < lookback:
        return None

    window = candles[-lookback:]
    n = len(window)

    # Gather local highs and lows.
    local_highs: List[Tuple[int, float]] = []
    local_lows: List[Tuple[int, float]] = []
    for i in range(1, n - 1):
        if window[i].high >= window[i - 1].high and window[i].high >= window[i + 1].high:
            local_highs.append((i, window[i].high))
        if window[i].low <= window[i - 1].low and window[i].low <= window[i + 1].low:
            local_lows.append((i, window[i].low))

    if len(local_highs) < 2 or len(local_lows) < 2:
        return None

    # Slope of upper and lower bounds.
    h_first, h_last = local_highs[0], local_highs[-1]
    l_first, l_last = local_lows[0], local_lows[-1]

    h_span = h_last[0] - h_first[0]
    l_span = l_last[0] - l_first[0]
    if h_span == 0 or l_span == 0:
        return None

    upper_slope = (h_last[1] - h_first[1]) / h_span
    lower_slope = (l_last[1] - l_first[1]) / l_span

    # Converging = slopes with opposite signs (or one near zero).
    if upper_slope < 0 and lower_slope > 0:
        pattern = "symmetric_triangle"
        direction = "bullish"  # Typically breaks up, but not guaranteed.
    elif upper_slope < 0 and abs(lower_slope) < abs(upper_slope) * 0.3:
        pattern = "descending_triangle"
        direction = "bearish"
    elif lower_slope > 0 and abs(upper_slope) < abs(lower_slope) * 0.3:
        pattern = "ascending_triangle"
        direction = "bullish"
    else:
        return None

    confidence = min(1.0, abs(upper_slope - lower_slope) / max(abs(upper_slope), abs(lower_slope), 1e-9))

    return PatternResult(detected=True, pattern=pattern, direction=direction, confidence=confidence)


def _detect_flag(candles: Sequence[Candle], lookback: int = 20) -> Optional[PatternResult]:
    """Detect flag / pennant continuation patterns.

    A flag is a sharp move (pole) followed by a counter-trend consolidation
    channel.
    """
    if len(candles) < lookback:
        return None

    window = candles[-lookback:]
    pole_len = lookback // 3
    flag_len = lookback - pole_len

    pole = window[:pole_len]
    flag_candles = window[pole_len:]

    # Pole: strong directional move.
    pole_return = (pole[-1].close - pole[0].close) / pole[0].close if pole[0].close else 0.0
    if abs(pole_return) < 0.03:  # Need at least 3% move for a pole.
        return None

    # Flag: low volatility consolidation.
    flag_closes = [c.close for c in flag_candles]
    if len(flag_closes) < 3:
        return None
    flag_vol = pstdev(flag_closes) / mean(flag_closes) if mean(flag_closes) else 0.0
    pole_closes = [c.close for c in pole]
    pole_vol = pstdev(pole_closes) / mean(pole_closes) if mean(pole_closes) else 0.0

    if pole_vol == 0 or flag_vol >= pole_vol:
        return None

    direction = "bullish" if pole_return > 0 else "bearish"
    confidence = min(1.0, abs(pole_return) * 10)

    return PatternResult(detected=True, pattern="flag", direction=direction, confidence=confidence)


def detect_patterns(candles: Sequence[Candle], lookback: int = 30) -> List[PatternResult]:
    """Run all pattern detectors and return any detected patterns."""
    results: List[PatternResult] = []
    for detector in (_detect_head_and_shoulders, _detect_triangle, _detect_flag):
        r = detector(candles, lookback)
        if r is not None and r.detected:
            results.append(r)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 14. Momentum Indicators (combined snapshot)
# ═══════════════════════════════════════════════════════════════════════════

def _rate_of_change(closes: Sequence[float], period: int = 12) -> float:
    """Rate of change: (close - close_n_ago) / close_n_ago × 100."""
    if len(closes) <= period:
        return 0.0
    prev = closes[-period - 1]
    if prev == 0:
        return 0.0
    return ((closes[-1] - prev) / prev) * 100


def _williams_r(candles: Sequence[Candle], period: int = 14) -> float:
    """Williams %R: position of close within the period's high-low range."""
    if len(candles) < period:
        return -50.0
    window = candles[-period:]
    highest = max(c.high for c in window)
    lowest = min(c.low for c in window)
    if highest == lowest:
        return -50.0
    return -100 * (highest - candles[-1].close) / (highest - lowest)


def compute_momentum_snapshot(candles: Sequence[Candle]) -> MomentumSnapshot:
    """Compute a combined momentum indicator snapshot."""
    closes = [c.close for c in candles] if candles else []
    rsi = compute_rsi(closes) if closes else 50.0
    stoch = compute_stochastic(candles)
    macd_line, macd_signal, macd_hist = compute_macd(closes) if closes else (0.0, 0.0, 0.0)
    roc = _rate_of_change(closes)
    wr = _williams_r(candles)
    return MomentumSnapshot(
        rsi=rsi,
        stochastic_k=stoch.k,
        stochastic_d=stoch.d,
        macd=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        roc=roc,
        williams_r=wr,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 15. Volatility Indicators (combined snapshot)
# ═══════════════════════════════════════════════════════════════════════════

def compute_volatility_snapshot(candles: Sequence[Candle]) -> VolatilitySnapshot:
    """Compute a combined volatility indicator snapshot."""
    closes = [c.close for c in candles] if candles else []
    atr = compute_atr(candles)
    bb_upper, bb_mid, bb_lower = bollinger_bands(closes) if closes else (0.0, 0.0, 0.0)
    bb_width = (bb_upper - bb_lower) / bb_mid if bb_mid else 0.0
    kc = compute_keltner(candles)

    # Historical (realised) volatility: std-dev of log returns.
    if len(closes) > 1:
        log_returns = []
        for prev, curr in zip(closes, closes[1:]):
            if prev > 0 and curr > 0:
                log_returns.append(math.log(curr / prev))
        hist_vol = pstdev(log_returns) if len(log_returns) > 1 else 0.0
    else:
        hist_vol = 0.0

    return VolatilitySnapshot(
        atr=atr,
        bb_upper=bb_upper,
        bb_mid=bb_mid,
        bb_lower=bb_lower,
        bb_width=bb_width,
        keltner_upper=kc.upper,
        keltner_mid=kc.mid,
        keltner_lower=kc.lower,
        historical_vol=hist_vol,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 16. Volume Indicators (OBV, MFI, CMF)
# ═══════════════════════════════════════════════════════════════════════════

def _on_balance_volume(candles: Sequence[Candle]) -> float:
    """Compute On-Balance Volume (OBV)."""
    if len(candles) < 2:
        return 0.0
    obv = 0.0
    for i in range(1, len(candles)):
        if candles[i].close > candles[i - 1].close:
            obv += candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            obv -= candles[i].volume
    return obv


def _money_flow_index(candles: Sequence[Candle], period: int = 14) -> float:
    """Compute the Money Flow Index (MFI) — volume-weighted RSI."""
    if len(candles) < period + 1:
        return 50.0

    pos_flow = 0.0
    neg_flow = 0.0
    for i in range(-period, 0):
        tp_curr = (candles[i].high + candles[i].low + candles[i].close) / 3
        tp_prev = (candles[i - 1].high + candles[i - 1].low + candles[i - 1].close) / 3
        mf = tp_curr * candles[i].volume
        if tp_curr > tp_prev:
            pos_flow += mf
        elif tp_curr < tp_prev:
            neg_flow += mf

    if neg_flow == 0:
        return 100.0
    mfr = pos_flow / neg_flow
    return 100 - (100 / (1 + mfr))


def _chaikin_money_flow(candles: Sequence[Candle], period: int = 20) -> float:
    """Compute Chaikin Money Flow (CMF)."""
    if len(candles) < period:
        return 0.0

    window = candles[-period:]
    ad_sum = 0.0
    vol_sum = 0.0
    for c in window:
        hl = c.high - c.low
        if hl > 0:
            clv = ((c.close - c.low) - (c.high - c.close)) / hl
        else:
            clv = 0.0
        ad_sum += clv * c.volume
        vol_sum += c.volume

    return ad_sum / vol_sum if vol_sum else 0.0


def compute_volume_snapshot(candles: Sequence[Candle]) -> VolumeSnapshot:
    """Compute a combined volume indicator snapshot."""
    obv = _on_balance_volume(candles)
    vwap = compute_vwap(candles)
    mfi = _money_flow_index(candles)
    cmf = _chaikin_money_flow(candles)

    volumes = [c.volume for c in candles] if candles else []
    vol_sma = mean(volumes[-20:]) if len(volumes) >= 20 else (mean(volumes) if volumes else 0.0)
    curr_vol = candles[-1].volume if candles else 0.0
    vol_ratio = curr_vol / vol_sma if vol_sma else 0.0

    return VolumeSnapshot(
        obv=obv,
        vwap=vwap,
        mfi=mfi,
        cmf=cmf,
        volume_sma=vol_sma,
        volume_ratio=vol_ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 17. Custom Indicator Framework
# ═══════════════════════════════════════════════════════════════════════════

class IndicatorFunc(Protocol):
    """Protocol for custom indicator functions."""
    def __call__(self, candles: Sequence[Candle], **kwargs: Any) -> Any: ...


@dataclass
class CustomIndicatorResult:
    """Wrapper for a custom indicator computation result."""
    name: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class CustomIndicatorRegistry:
    """Registry for user-defined indicator functions.

    Usage::

        registry = CustomIndicatorRegistry()

        @registry.register("my_indicator")
        def my_indicator(candles, **kwargs):
            closes = [c.close for c in candles]
            return sum(closes) / len(closes)

        results = registry.compute_all(candles)
        # results == [CustomIndicatorResult(name="my_indicator", value=...)]
    """

    def __init__(self) -> None:
        self._indicators: Dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable:
        """Decorator to register a custom indicator function."""
        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._indicators[name] = func
            return func
        return decorator

    def add(self, name: str, func: Callable[..., Any]) -> None:
        """Programmatically register a custom indicator function."""
        self._indicators[name] = func

    def remove(self, name: str) -> None:
        """Remove a registered custom indicator."""
        self._indicators.pop(name, None)

    @property
    def names(self) -> List[str]:
        """List of registered indicator names."""
        return list(self._indicators.keys())

    def compute(self, name: str, candles: Sequence[Candle], **kwargs: Any) -> CustomIndicatorResult:
        """Compute a single custom indicator by name."""
        func = self._indicators.get(name)
        if func is None:
            raise KeyError(f"Indicator '{name}' is not registered")
        value = func(candles, **kwargs)
        return CustomIndicatorResult(name=name, value=value)

    def compute_all(self, candles: Sequence[Candle], **kwargs: Any) -> List[CustomIndicatorResult]:
        """Compute all registered custom indicators."""
        results: List[CustomIndicatorResult] = []
        for name, func in self._indicators.items():
            try:
                value = func(candles, **kwargs)
                results.append(CustomIndicatorResult(name=name, value=value))
            except Exception as exc:  # pragma: no cover
                logger.warning("Custom indicator '%s' failed: %s", name, exc)
        return results
