from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class TrendResult:
    direction: str
    fast_ma: float
    slow_ma: float
    strength: float


@dataclass
class OrderbookInsight:
    spread_pct: float
    bid_volume: float
    ask_volume: float
    imbalance: float


@dataclass
class VolatilityStats:
    volatility: float
    avg_volume: float


@dataclass
class SupportResistance:
    support: float
    resistance: float
    lookback: int


@dataclass
class MomentumIndicators:
    rsi: float
    macd: float
    macd_signal: float
    macd_hist: float
    bb_upper: float
    bb_mid: float
    bb_lower: float


@dataclass
class MultiTimeframeResult:
    """Aggregated directional signal across multiple timeframes.

    ``aligned`` is ``True`` when all sampled timeframes agree on the same
    direction (all up or all down), giving higher confidence.

    ``direction`` is the majority-vote direction: ``"up"``, ``"down"``, or
    ``"flat"`` (tie).

    ``strength`` is the average trend strength across timeframes.

    ``tf_directions`` maps each timeframe label to its individual direction
    string so callers can log the per-TF breakdown.
    """

    direction: str
    aligned: bool
    strength: float
    tf_directions: Dict[str, str] = field(default_factory=dict)


@dataclass
class WhaleActivity:
    """Result of large-order / smart-money detection in the order book.

    ``detected`` is ``True`` when at least one side has an anomalously large
    wall relative to the average level size.

    ``side`` is ``"bid"`` (large buy wall — bullish) or ``"ask"`` (large sell
    wall — bearish), or ``None`` when nothing significant is detected.

    ``ratio`` is the largest-level-volume / average-level-volume multiple.
    """

    detected: bool
    side: Optional[str]
    ratio: float


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Failed to parse float from value=%s", value)
        return 0.0


def build_candles(
    trades: Sequence[Dict[str, object]],
    interval_seconds: int,
    limit: int = 96,
) -> List[Candle]:
    if not trades:
        return []

    valid_trades = [t for t in trades if isinstance(t, dict)]
    if not valid_trades:
        return []

    sorted_trades = sorted(valid_trades, key=lambda t: int(t.get("date", 0)))
    first_ts = int(sorted_trades[0].get("date", 0))
    buckets: Dict[int, List[Dict[str, object]]] = {}
    for trade in sorted_trades:
        ts = int(trade.get("date", 0))
        bucket = first_ts + ((ts - first_ts) // interval_seconds) * interval_seconds
        buckets.setdefault(bucket, []).append(trade)

    candles: List[Candle] = []
    for bucket_ts in sorted(buckets.keys())[-limit:]:
        bucket_trades = buckets[bucket_ts]
        trade_prices = [_safe_float(t.get("price", "")) for t in bucket_trades]
        amounts = [_safe_float(t.get("amount", "0")) for t in bucket_trades]
        if not trade_prices:
            continue
        candles.append(
            Candle(
                timestamp=bucket_ts,
                open=trade_prices[0],
                high=max(trade_prices),
                low=min(trade_prices),
                close=trade_prices[-1],
                volume=sum(amounts),
            )
        )
    return candles


def moving_average(values: Iterable[float], window: int) -> List[float]:
    values_list = list(values)
    if window <= 0:
        raise ValueError("window must be positive")
    result = []
    for i in range(len(values_list)):
        if i + 1 < window:
            result.append(math.nan)
        else:
            result.append(mean(values_list[i + 1 - window : i + 1]))
    return result


def analyze_trend(
    candles: Sequence[Candle], fast_window: int = 12, slow_window: int = 48
) -> TrendResult:
    if not candles:
        return TrendResult("flat", math.nan, math.nan, 0.0)

    closes = [c.close for c in candles]
    fast_ma_series = moving_average(closes, fast_window)
    slow_ma_series = moving_average(closes, slow_window)
    fast_ma = fast_ma_series[-1]
    slow_ma = slow_ma_series[-1]

    if math.isnan(fast_ma) or math.isnan(slow_ma):
        return TrendResult("flat", fast_ma, slow_ma, 0.0)

    if fast_ma > slow_ma:
        direction = "up"
    elif fast_ma < slow_ma:
        direction = "down"
    else:
        direction = "flat"
    strength = abs(fast_ma - slow_ma) / slow_ma if slow_ma else 0.0
    return TrendResult(direction, fast_ma, slow_ma, strength)


def analyze_orderbook(depth: Dict[str, object]) -> OrderbookInsight:
    bids = depth.get("buy") or []
    asks = depth.get("sell") or []
    top_bid = _safe_float(bids[0][0]) if bids else 0.0
    top_ask = _safe_float(asks[0][0]) if asks else 0.0
    spread_pct = (top_ask - top_bid) / top_bid if top_bid else 0.0
    bid_volume = sum(_safe_float(b[1]) for b in bids[:20])
    ask_volume = sum(_safe_float(a[1]) for a in asks[:20])
    total = bid_volume + ask_volume
    imbalance = (bid_volume - ask_volume) / total if total else 0.0
    return OrderbookInsight(
        spread_pct=spread_pct,
        bid_volume=bid_volume,
        ask_volume=ask_volume,
        imbalance=imbalance,
    )


def analyze_volatility(candles: Sequence[Candle]) -> VolatilityStats:
    if not candles or len(candles) < 2:
        return VolatilityStats(volatility=0.0, avg_volume=0.0)
    closes = [c.close for c in candles]
    returns = []
    for prev, curr in zip(closes, closes[1:]):
        if prev == 0:
            returns.append(0.0)
        else:
            returns.append((curr - prev) / prev)
    vol = pstdev(returns) if len(returns) > 1 else 0.0
    avg_volume = mean(c.volume for c in candles)
    return VolatilityStats(volatility=vol, avg_volume=avg_volume)


def support_resistance(candles: Sequence[Candle], lookback: int = 30) -> SupportResistance:
    if not candles:
        return SupportResistance(0.0, 0.0, lookback)
    closes = [c.close for c in candles[-lookback:]]
    return SupportResistance(support=min(closes), resistance=max(closes), lookback=lookback)


def _ema_series(values: Sequence[float], span: int) -> List[float]:
    if not values:
        return []
    k = 2 / (span + 1)
    series: List[float] = []
    ema = values[0]
    for idx, value in enumerate(values):
        if idx == 0:
            ema = value
        else:
            ema = value * k + ema * (1 - k)
        series.append(ema)
    return series


def compute_rsi(closes: Sequence[float], period: int = 14) -> float:
    if len(closes) < 2:
        return 50.0
    deltas = [curr - prev for prev, curr in zip(closes, closes[1:])]
    gains = [max(d, 0.0) for d in deltas]
    losses = [abs(min(d, 0.0)) for d in deltas]
    if not any(gains) and not any(losses):
        return 50.0
    avg_gain = mean(gains[-period:]) if gains else 0.0
    avg_loss = mean(losses[-period:]) if losses else 0.0
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(closes: Sequence[float]) -> tuple[float, float, float]:
    ema12_series = _ema_series(closes, 12)
    ema26_series = _ema_series(closes, 26)
    if not ema12_series or not ema26_series:
        return (0.0, 0.0, 0.0)
    macd_series = [a - b for a, b in zip(ema12_series[-len(ema26_series) :], ema26_series)]
    macd_line = macd_series[-1]
    signal_series = _ema_series(macd_series, 9)
    signal_line = signal_series[-1] if signal_series else 0.0
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(closes: Sequence[float], period: int = 20, std_dev: float = 2) -> tuple[float, float, float]:
    if len(closes) < period:
        window = closes or [0.0]
    else:
        window = closes[-period:]
    mid = mean(window)
    dev = pstdev(window) if len(window) > 1 else 0.0
    upper = mid + std_dev * dev
    lower = mid - std_dev * dev
    return upper, mid, lower


def derive_indicators(candles: Sequence[Candle]) -> MomentumIndicators:
    if not candles:
        return MomentumIndicators(50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    closes = [c.close for c in candles]
    rsi_val = compute_rsi(closes)
    macd_line, macd_signal, macd_hist = compute_macd(closes)
    bb_upper, bb_mid, bb_lower = bollinger_bands(closes)
    return MomentumIndicators(
        rsi=rsi_val,
        macd=macd_line,
        macd_signal=macd_signal,
        macd_hist=macd_hist,
        bb_upper=bb_upper,
        bb_mid=bb_mid,
        bb_lower=bb_lower,
    )


# ---------------------------------------------------------------------------
# Indodax OHLCV history helpers
# ---------------------------------------------------------------------------

# Maps Indodax timeframe strings to their duration in seconds.
_TF_SECONDS: Dict[str, int] = {
    "1": 60,
    "15": 900,
    "30": 1800,
    "60": 3600,
    "240": 14400,
    "1D": 86400,
    "3D": 259200,
    "1W": 604800,
}


def interval_to_ohlc_tf(interval_seconds: int) -> str:
    """Return the Indodax OHLCV timeframe string closest to *interval_seconds*.

    The mapping always picks the smallest available timeframe that is
    **≥** the requested interval so that each candle covers at least one
    full bot cycle.

    Available timeframes: ``"1"`` (1 min), ``"15"``, ``"30"``, ``"60"``,
    ``"240"`` (4 h), ``"1D"``.
    """
    for tf, seconds in [("1", 60), ("15", 900), ("30", 1800), ("60", 3600), ("240", 14400)]:
        if interval_seconds <= seconds:
            return tf
    return "1D"


def candles_from_ohlc(ohlc_data: List[Dict[str, object]]) -> List[Candle]:
    """Convert Indodax ``/tradingview/history_v2`` response to :class:`Candle` objects.

    The API returns a list of dicts with keys ``Time``, ``Open``, ``High``,
    ``Low``, ``Close``, ``Volume`` (capital first letter).  Invalid or
    incomplete rows are silently skipped.  Returns list sorted oldest → newest.
    """
    result: List[Candle] = []
    for row in ohlc_data:
        if not isinstance(row, dict):
            continue
        try:
            result.append(
                Candle(
                    timestamp=int(row["Time"]),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row.get("Volume") or 0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(result, key=lambda c: c.timestamp)



# ---------------------------------------------------------------------------
# Multi-timeframe analysis
# ---------------------------------------------------------------------------

def multi_timeframe_confirm(
    candles_by_tf: Dict[str, Sequence[Candle]],
    fast_window: int = 12,
    slow_window: int = 48,
) -> MultiTimeframeResult:
    """Compute a consensus directional signal across multiple candle timeframes.

    :param candles_by_tf: Mapping of timeframe label (e.g. ``"1m"``, ``"15m"``)
        to the corresponding :class:`Candle` sequence for that timeframe.  Each
        sequence is analysed independently with :func:`analyze_trend`.
    :param fast_window: Fast EMA window forwarded to :func:`analyze_trend`.
    :param slow_window: Slow EMA window forwarded to :func:`analyze_trend`.
    :returns: :class:`MultiTimeframeResult` with aggregated signal.

    Empty or missing timeframe sequences are silently skipped.
    """
    if not candles_by_tf:
        return MultiTimeframeResult(
            direction="flat", aligned=False, strength=0.0, tf_directions={}
        )

    tf_directions: Dict[str, str] = {}
    strengths: List[float] = []

    for tf_label, candles in candles_by_tf.items():
        if not candles:
            continue
        trend = analyze_trend(list(candles), fast_window, slow_window)
        tf_directions[tf_label] = trend.direction
        strengths.append(trend.strength)

    if not tf_directions:
        return MultiTimeframeResult(
            direction="flat", aligned=False, strength=0.0, tf_directions={}
        )

    up_count = sum(1 for d in tf_directions.values() if d == "up")
    down_count = sum(1 for d in tf_directions.values() if d == "down")
    total = len(tf_directions)

    if up_count > down_count:
        direction = "up"
    elif down_count > up_count:
        direction = "down"
    else:
        direction = "flat"

    aligned = (up_count == total) or (down_count == total)
    avg_strength = mean(strengths) if strengths else 0.0

    return MultiTimeframeResult(
        direction=direction,
        aligned=aligned,
        strength=avg_strength,
        tf_directions=tf_directions,
    )


# ---------------------------------------------------------------------------
# Smart money / whale detection
# ---------------------------------------------------------------------------

# A level is classified as a "whale wall" when its volume is at least this
# many times larger than the average level volume on that side.
_WHALE_MULTIPLIER_THRESHOLD = 5.0
# Minimum number of levels required to compute a meaningful average.
_WHALE_MIN_LEVELS = 3


def detect_whale_activity(
    depth: Dict[str, object],
    multiplier_threshold: float = _WHALE_MULTIPLIER_THRESHOLD,
    top_n: int = 20,
) -> WhaleActivity:
    """Detect abnormally large orders (whale walls) in the order book.

    Iterates over the top *top_n* bid and ask levels and flags the side with
    the largest single-level volume when that volume exceeds
    *multiplier_threshold* × average-level-volume on the same side.

    :param depth:  Raw depth dict with ``"buy"`` and ``"sell"`` keys, each a
        list of ``[price, volume]`` pairs (strings or numbers).
    :param multiplier_threshold: How many times above the mean a level must be
        to qualify as a whale wall.  Default is 5×.
    :param top_n: Number of levels to inspect on each side.
    :returns: :class:`WhaleActivity` describing the strongest anomaly found.
    """
    bids = (depth.get("buy") or [])[:top_n]
    asks = (depth.get("sell") or [])[:top_n]

    best_side: Optional[str] = None
    best_ratio = 0.0

    for side_label, levels in (("bid", bids), ("ask", asks)):
        if len(levels) < _WHALE_MIN_LEVELS:
            continue
        try:
            volumes = [float(level[1]) for level in levels]
        except (IndexError, TypeError, ValueError):
            continue
        avg_vol = mean(volumes)
        if avg_vol <= 0:
            continue
        max_vol = max(volumes)
        ratio = max_vol / avg_vol
        if ratio > best_ratio:
            best_ratio = ratio
            best_side = side_label

    detected = best_ratio >= multiplier_threshold
    return WhaleActivity(
        detected=detected,
        side=best_side if detected else None,
        ratio=best_ratio,
    )
