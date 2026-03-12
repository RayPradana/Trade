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
class TradeFlowResult:
    """Result of recent-trade buy/sell flow analysis.

    ``buy_ratio`` is the fraction of recent trades that were buyer-initiated
    (``side="buy"``). Range: ``0.0``–``1.0``.

    ``buy_volume`` and ``sell_volume`` are the aggregated trade volumes on
    each side.

    ``aggressive_buyers`` is ``True`` when the buy ratio exceeds
    ``0.65`` (i.e., more than 65 % of recent trades were market buys),
    indicating aggressive buyer participation.
    """

    buy_ratio: float
    buy_volume: float
    sell_volume: float
    aggressive_buyers: bool


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


@dataclass
class SpoofingResult:
    """Result of order-book spoofing / manipulation detection.

    Spoofing is characterised by large-volume levels placed far from the
    current mid-price — orders that are unlikely to be filled but create the
    illusion of strong support or resistance.

    ``detected`` is ``True`` when at least one large distant wall is found.

    ``side`` is ``"bid"`` (fake buy wall) or ``"ask"`` (fake sell wall), or
    ``None`` when nothing suspicious is found.

    ``distance_pct`` is how far (as a fraction of mid-price) the suspicious
    level is from the top of book on the same side.
    """

    detected: bool
    side: Optional[str]
    distance_pct: float


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


def analyze_trade_flow(
    trades: Sequence[Dict[str, Any]],
    aggressive_buyer_threshold: float = 0.65,
) -> TradeFlowResult:
    """Analyse recent trades to determine buy/sell flow.

    Examines the ``type`` field of each trade (``"buy"`` or ``"sell"``) and
    the trade ``price``×``amount`` to compute the fraction of volume that was
    buyer-initiated.  A high ``buy_ratio`` indicates aggressive buyer
    participation (market orders hitting the ask), which is a bullish signal.

    Parameters
    ----------
    trades:
        List of recent trade dicts as returned by the exchange API.  Each dict
        should contain ``"type"`` (``"buy"`` or ``"sell"``) and ``"amount"``
        (or ``"vol"``) and ``"price"`` fields.
    aggressive_buyer_threshold:
        ``buy_ratio`` above which :attr:`TradeFlowResult.aggressive_buyers` is
        set to ``True``.  Default: ``0.65`` (65 %).

    Returns
    -------
    :class:`TradeFlowResult`
        Always returns a result; if ``trades`` is empty all fields default to
        neutral values (``buy_ratio=0.5``, ``aggressive_buyers=False``).
    """
    if not trades:
        return TradeFlowResult(
            buy_ratio=0.5,
            buy_volume=0.0,
            sell_volume=0.0,
            aggressive_buyers=False,
        )

    buy_vol = 0.0
    sell_vol = 0.0
    for t in trades:
        trade_type = str(t.get("type", "")).lower()
        # Support both amount and vol field names used by different API
        # versions.
        amount = _safe_float(t.get("amount") or t.get("vol", 0))
        price = _safe_float(t.get("price", 0))
        notional = amount * price if price else amount
        if trade_type in ("buy", "bid"):
            buy_vol += notional
        elif trade_type in ("sell", "ask"):
            sell_vol += notional

    total = buy_vol + sell_vol
    buy_ratio = buy_vol / total if total else 0.5
    return TradeFlowResult(
        buy_ratio=round(buy_ratio, 4),
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        aggressive_buyers=buy_ratio >= aggressive_buyer_threshold,
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
    recent = candles[-lookback:]
    support = min(c.low for c in recent)
    resistance = max(c.high for c in recent)
    return SupportResistance(support=support, resistance=resistance, lookback=lookback)


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


# ---------------------------------------------------------------------------
# Spoofing / order-book manipulation detection
# ---------------------------------------------------------------------------

# A level qualifies as a potential spoof wall when:
#  1. Its volume is at least this many times larger than the average level vol.
#  2. Its price is at least this far (as fraction) from the current top of book.
_SPOOF_VOLUME_MULTIPLIER = 5.0
_SPOOF_MIN_DISTANCE_PCT = 0.03   # 3% away from top of book
_SPOOF_MIN_LEVELS = 3


# ---------------------------------------------------------------------------
# Smart Entry Engine — dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PrePumpSignal:
    """Early volume-accumulation signal that may precede a pump.

    ``detected`` is ``True`` when recent candle volume surges above the
    baseline by at least the configured ratio.

    ``volume_surge_ratio`` is recent_avg_volume / baseline_avg_volume.

    ``score`` is a 0..1 normalised signal strength (1 = very strong surge).
    """

    detected: bool
    volume_surge_ratio: float
    score: float


@dataclass
class WhalePressure:
    """Net directional pressure from whale-sized orders in the order book.

    Unlike :class:`WhaleActivity` (which finds the largest individual wall),
    this dataclass captures the *net difference* between the bid-side and
    ask-side whale ratio, indicating whether smart money is predominantly
    buying or selling.

    ``detected`` is ``True`` when ``abs(pressure) >= threshold``.

    ``side`` is ``"buy"`` (net bid dominance) or ``"sell"`` (net ask
    dominance), or ``None`` when nothing significant is detected.

    ``pressure`` is bid_ratio − ask_ratio (positive = buy pressure).
    """

    detected: bool
    side: Optional[str]
    pressure: float


@dataclass
class FakeBreakoutRisk:
    """Volume-confirmation check for price breakouts above resistance.

    A genuine breakout should be accompanied by above-average volume.
    When price is above resistance but volume is thin, the move is likely
    to retrace (fake breakout).

    ``breakout_present`` is ``True`` when current price > resistance.

    ``detected`` is ``True`` when a breakout is present but volume is
    below the configured minimum ratio.

    ``volume_ratio`` is recent_volume / average_volume.

    ``score`` is the 0..1 risk level (1 = very high fake-breakout risk).
    """

    breakout_present: bool
    detected: bool
    volume_ratio: float
    score: float


@dataclass
class SmartEntryResult:
    """Combined result of all Smart Entry Engine checks.

    Aggregates :class:`PrePumpSignal`, :class:`WhalePressure`, and
    :class:`FakeBreakoutRisk` into a single object passed to
    ``make_trade_decision`` for confidence adjustment.
    """

    pre_pump: PrePumpSignal
    whale_pressure: WhalePressure
    fake_breakout: FakeBreakoutRisk


def detect_spoofing(
    depth: Dict[str, object],
    volume_multiplier: float = _SPOOF_VOLUME_MULTIPLIER,
    min_distance_pct: float = _SPOOF_MIN_DISTANCE_PCT,
    top_n: int = 30,
) -> SpoofingResult:
    """Detect potential spoof / wash-trading walls in the order book.

    A "spoof wall" is a large-volume level positioned significantly far from
    the best price on its side.  Spoofed orders are meant to mislead market
    participants but are rarely filled, so they tend to appear deep in the
    order book.

    :param depth: Raw depth dict with ``"buy"`` and ``"sell"`` key lists.
    :param volume_multiplier: Minimum ratio (level vol / avg vol) for the
        level to be flagged.  Default 5×.
    :param min_distance_pct: Minimum fractional price distance from the top
        of book for the suspicious level.  Default 3%.
    :param top_n: Number of levels to inspect per side.
    :returns: :class:`SpoofingResult` describing the most suspicious anomaly.

    Returns ``SpoofingResult(detected=False, ...)`` when no anomaly is found
    or when there is insufficient data.
    """
    bids = (depth.get("buy") or [])[:top_n]
    asks = (depth.get("sell") or [])[:top_n]

    best_side: Optional[str] = None
    best_distance = 0.0

    for side_label, levels in (("bid", bids), ("ask", asks)):
        if len(levels) < _SPOOF_MIN_LEVELS:
            continue
        try:
            prices = [float(level[0]) for level in levels]
            volumes = [float(level[1]) for level in levels]
        except (IndexError, TypeError, ValueError):
            continue

        avg_vol = mean(volumes)
        if avg_vol <= 0:
            continue

        top_price = prices[0]  # best price on this side (nearest to mid)
        if top_price <= 0:
            continue

        for price, vol in zip(prices[1:], volumes[1:]):
            ratio = vol / avg_vol
            distance = abs(price - top_price) / top_price
            if ratio >= volume_multiplier and distance >= min_distance_pct:
                if distance > best_distance:
                    best_distance = distance
                    best_side = side_label

    detected = best_distance >= min_distance_pct
    return SpoofingResult(
        detected=detected,
        side=best_side if detected else None,
        distance_pct=best_distance,
    )


# ---------------------------------------------------------------------------
# Smart Entry Engine — detection functions
# ---------------------------------------------------------------------------

# Number of recent candles used by default when checking for a volume surge.
_SEE_RECENT_CANDLES = 3


def detect_pre_pump_signal(
    candles: Sequence[Candle],
    volume_surge_ratio: float = 2.0,
    recent_n: int = _SEE_RECENT_CANDLES,
) -> PrePumpSignal:
    """Detect early volume accumulation that may precede a pump.

    Compares the average volume of the most recent *recent_n* candles against
    the baseline (all older candles).  A ratio >= *volume_surge_ratio* flags a
    potential pre-pump accumulation phase.

    :param candles: Ordered candle sequence (oldest → newest).
    :param volume_surge_ratio: Threshold ratio (recent avg / baseline avg).
        Default 2.0 — recent candles must carry 2× the baseline volume.
    :param recent_n: Number of trailing candles treated as "recent".
    :returns: :class:`PrePumpSignal` with detection result and score.
    """
    if len(candles) < recent_n + 2:
        return PrePumpSignal(detected=False, volume_surge_ratio=0.0, score=0.0)

    baseline_candles = list(candles)[:-recent_n]
    recent_candles_list = list(candles)[-recent_n:]

    baseline_vols = [c.volume for c in baseline_candles if c.volume > 0]
    recent_vols = [c.volume for c in recent_candles_list if c.volume > 0]

    if not baseline_vols or not recent_vols:
        return PrePumpSignal(detected=False, volume_surge_ratio=0.0, score=0.0)

    baseline_avg = mean(baseline_vols)
    recent_avg = mean(recent_vols)

    if baseline_avg <= 0:
        return PrePumpSignal(detected=False, volume_surge_ratio=0.0, score=0.0)

    surge = recent_avg / baseline_avg
    detected = surge >= volume_surge_ratio
    # Normalise: 0 at ratio=1, saturates at 1 when ratio = 2 × threshold
    score = min(1.0, max(0.0, (surge - 1.0) / max(volume_surge_ratio - 1.0, 1e-6)))
    return PrePumpSignal(
        detected=detected,
        volume_surge_ratio=round(surge, 4),
        score=round(score, 4),
    )


def detect_whale_pressure(
    depth: Dict[str, object],
    pressure_threshold: float = 2.0,
    top_n: int = 20,
) -> WhalePressure:
    """Detect net directional pressure from whale-sized orders.

    Unlike :func:`detect_whale_activity` (which finds the single largest
    wall), this function computes the *difference* between the bid-side and
    ask-side whale ratios to determine whether smart money is net-buying or
    net-selling.

    A positive ``pressure`` means the bid side has a comparatively larger
    anomaly (net buying pressure); a negative value means the opposite.
    When ``abs(pressure) >= pressure_threshold`` the signal is detected.

    :param depth: Raw depth dict with ``"buy"`` / ``"sell"`` lists of
        ``[price, volume]`` pairs.
    :param pressure_threshold: Minimum ``|bid_ratio - ask_ratio|`` to flag.
    :param top_n: Number of order-book levels to inspect per side.
    :returns: :class:`WhalePressure` result.
    """
    bids = (depth.get("buy") or [])[:top_n]
    asks = (depth.get("sell") or [])[:top_n]

    def _max_ratio(levels: list) -> float:
        if len(levels) < _WHALE_MIN_LEVELS:
            return 1.0
        try:
            volumes = [float(lvl[1]) for lvl in levels]
        except (IndexError, TypeError, ValueError):
            return 1.0
        avg = mean(volumes)
        if avg <= 0:
            return 1.0
        return max(volumes) / avg

    bid_ratio = _max_ratio(list(bids))
    ask_ratio = _max_ratio(list(asks))
    pressure = bid_ratio - ask_ratio
    detected = abs(pressure) >= pressure_threshold
    side: Optional[str] = None
    if detected:
        side = "buy" if pressure > 0 else "sell"
    return WhalePressure(
        detected=detected,
        side=side,
        pressure=round(pressure, 4),
    )


def detect_fake_breakout(
    candles: Sequence[Candle],
    current_price: float,
    levels: Optional[SupportResistance],
    min_volume_ratio: float = 0.7,
) -> FakeBreakoutRisk:
    """Assess the risk that a breakout above resistance is not volume-confirmed.

    A genuine breakout should be accompanied by above-average volume.  When
    price is above the resistance level but the most recent candle's volume is
    below *min_volume_ratio* × the historical average, the breakout is flagged
    as potentially fake (likely to retrace).

    :param candles: Candle sequence (oldest → newest).
    :param current_price: Latest trade price.
    :param levels: Support / resistance derived from the candle series.
    :param min_volume_ratio: Minimum (recent_vol / avg_vol) to confirm a real
        breakout.  Default 0.7 (at least 70 % of average volume required).
    :returns: :class:`FakeBreakoutRisk` result.
    """
    no_risk = FakeBreakoutRisk(
        breakout_present=False, detected=False, volume_ratio=1.0, score=0.0
    )

    if not candles or levels is None:
        return no_risk

    resistance = levels.resistance
    if not resistance or current_price <= resistance:
        return no_risk

    # Price is above resistance — check whether volume confirms the move.
    volumes = [c.volume for c in candles if c.volume > 0]
    if len(volumes) < 2:
        return FakeBreakoutRisk(
            breakout_present=True, detected=False, volume_ratio=1.0, score=0.0
        )

    avg_vol = mean(volumes[:-1])  # exclude last candle from baseline
    recent_vol = volumes[-1]

    if avg_vol <= 0:
        return FakeBreakoutRisk(
            breakout_present=True, detected=False, volume_ratio=1.0, score=0.0
        )

    volume_ratio = recent_vol / avg_vol
    detected = volume_ratio < min_volume_ratio
    # Risk score: 1 when volume_ratio ≈ 0, 0 when volume_ratio ≥ 1.
    score = min(1.0, max(0.0, 1.0 - volume_ratio))
    return FakeBreakoutRisk(
        breakout_present=True,
        detected=detected,
        volume_ratio=round(volume_ratio, 4),
        score=round(score, 4),
    )


def smart_entry_filter(
    candles: Sequence[Candle],
    depth: Dict[str, object],
    current_price: float,
    levels: Optional[SupportResistance],
    volume_surge_ratio: float = 2.0,
    whale_pressure_min: float = 2.0,
    breakout_volume_min: float = 0.7,
) -> SmartEntryResult:
    """Run all Smart Entry Engine checks and return a combined result.

    Convenience wrapper that calls :func:`detect_pre_pump_signal`,
    :func:`detect_whale_pressure`, and :func:`detect_fake_breakout` with the
    provided parameters and bundles the results into a
    :class:`SmartEntryResult`.

    :param candles: Primary candle series (oldest → newest).
    :param depth: Current order-book depth dict.
    :param current_price: Latest trade price.
    :param levels: Support / resistance levels.
    :param volume_surge_ratio: Passed to :func:`detect_pre_pump_signal`.
    :param whale_pressure_min: Passed to :func:`detect_whale_pressure`.
    :param breakout_volume_min: Passed to :func:`detect_fake_breakout`.
    :returns: :class:`SmartEntryResult`.
    """
    return SmartEntryResult(
        pre_pump=detect_pre_pump_signal(candles, volume_surge_ratio),
        whale_pressure=detect_whale_pressure(depth, whale_pressure_min),
        fake_breakout=detect_fake_breakout(candles, current_price, levels, breakout_volume_min),
    )


# ---------------------------------------------------------------------------
# Market Regime Detection
# ---------------------------------------------------------------------------

@dataclass
class MarketRegime:
    regime: str  # "trending_up", "trending_down", "ranging", "volatile"
    strength: float  # 0..1
    description: str


def detect_market_regime(
    candles: Sequence[Candle],
    trend: TrendResult,
    vol: VolatilityStats,
) -> MarketRegime:
    """Classify the current market regime."""
    if vol.volatility > 0.04:
        strength = min(1.0, vol.volatility / 0.1)
        return MarketRegime(
            regime="volatile",
            strength=round(strength, 4),
            description=f"High volatility {vol.volatility:.4f}",
        )
    if trend.direction == "up" and trend.strength > 0.01 and vol.volatility < 0.03:
        strength = min(1.0, trend.strength * 10)
        return MarketRegime(
            regime="trending_up",
            strength=round(strength, 4),
            description=f"Uptrend strength={trend.strength:.4f}",
        )
    if trend.direction == "down" and trend.strength > 0.01 and vol.volatility < 0.03:
        strength = min(1.0, trend.strength * 10)
        return MarketRegime(
            regime="trending_down",
            strength=round(strength, 4),
            description=f"Downtrend strength={trend.strength:.4f}",
        )
    return MarketRegime(
        regime="ranging",
        strength=round(max(0.0, 1.0 - trend.strength * 10), 4),
        description=f"Ranging/flat trend={trend.direction} vol={vol.volatility:.4f}",
    )


# ---------------------------------------------------------------------------
# Spread Anomaly Detection
# ---------------------------------------------------------------------------

@dataclass
class SpreadAnomaly:
    detected: bool
    current_spread_pct: float
    avg_spread_pct: float
    ratio: float  # current/avg


def detect_spread_anomaly(
    current_spread_pct: float,
    recent_spreads: Sequence[float],
    multiplier: float = 3.0,
) -> SpreadAnomaly:
    """Detect abnormal spread widening vs recent history."""
    if not recent_spreads:
        return SpreadAnomaly(
            detected=False,
            current_spread_pct=current_spread_pct,
            avg_spread_pct=0.0,
            ratio=0.0,
        )
    avg = mean(list(recent_spreads))
    ratio = current_spread_pct / avg if avg > 0 else 0.0
    detected = avg > 0 and ratio >= multiplier
    return SpreadAnomaly(
        detected=detected,
        current_spread_pct=current_spread_pct,
        avg_spread_pct=avg,
        ratio=round(ratio, 4),
    )


# ---------------------------------------------------------------------------
# Orderbook Absorption Detection
# ---------------------------------------------------------------------------

@dataclass
class OrderbookAbsorption:
    detected: bool
    side: Optional[str]  # "bid" or "ask"
    absorption_ratio: float  # how much of the wall has been absorbed


def detect_orderbook_absorption(
    depth_before: Dict[str, object],
    depth_after: Dict[str, object],
    threshold: float = 0.5,
) -> OrderbookAbsorption:
    """Detect when a large orderbook wall is being consumed by market orders."""
    no_detection = OrderbookAbsorption(detected=False, side=None, absorption_ratio=0.0)

    def _top_volume(depth: Dict[str, object], key: str) -> float:
        levels = depth.get(key) or []
        if not levels:
            return 0.0
        try:
            return float(levels[0][1])
        except (IndexError, TypeError, ValueError):
            return 0.0

    bid_before = _top_volume(depth_before, "buy")
    bid_after = _top_volume(depth_after, "buy")
    ask_before = _top_volume(depth_before, "sell")
    ask_after = _top_volume(depth_after, "sell")

    best_ratio = 0.0
    best_side = None

    if bid_before > 0:
        bid_absorbed = (bid_before - bid_after) / bid_before
        if bid_absorbed > best_ratio:
            best_ratio = bid_absorbed
            best_side = "bid"

    if ask_before > 0:
        ask_absorbed = (ask_before - ask_after) / ask_before
        if ask_absorbed > best_ratio:
            best_ratio = ask_absorbed
            best_side = "ask"

    detected = best_ratio >= threshold
    return OrderbookAbsorption(
        detected=detected,
        side=best_side if detected else None,
        absorption_ratio=round(best_ratio, 4),
    )


# ---------------------------------------------------------------------------
# Flash Dump Detection
# ---------------------------------------------------------------------------

@dataclass
class FlashDumpSignal:
    detected: bool
    drop_pct: float  # price drop fraction in lookback window
    duration_seconds: float


def detect_flash_dump(
    price_history: Sequence[tuple],  # (timestamp, price) pairs
    lookback_seconds: float = 60.0,
    min_drop_pct: float = 0.05,
) -> FlashDumpSignal:
    """Detect a sudden large price drop (flash dump / crash)."""
    no_signal = FlashDumpSignal(detected=False, drop_pct=0.0, duration_seconds=0.0)
    if not price_history:
        return no_signal

    price_list = list(price_history)
    now = price_list[-1][0]
    cutoff = now - lookback_seconds
    window = [(ts, p) for ts, p in price_list if ts >= cutoff]

    if len(window) < 2:
        return no_signal

    peak_price = max(p for _, p in window)
    current_price = window[-1][1]
    if peak_price <= 0:
        return no_signal

    drop = (peak_price - current_price) / peak_price
    duration = window[-1][0] - window[0][0]
    detected = drop >= min_drop_pct
    return FlashDumpSignal(
        detected=detected,
        drop_pct=round(drop, 4),
        duration_seconds=round(duration, 2),
    )


# ---------------------------------------------------------------------------
# Rug-Pull / Dead Coin Detection
# ---------------------------------------------------------------------------

@dataclass
class RugPullRisk:
    """Result of rug-pull / dead-coin risk evaluation for a trading pair.

    A pair is flagged when it exhibits characteristics typical of scam tokens
    or abandoned coins:
    - Extreme 24-h price drop (the coin has "rug-pulled" or crashed)
    - Near-zero or no trading volume (dead / illiquid coin)
    - Extremely low price with no recent trade activity

    ``detected`` is ``True`` when any risk criterion is met.

    ``reason`` describes which criterion triggered the flag
    (e.g. ``"24h_drop=45%"`` or ``"dead_coin_volume=0"``).

    ``drop_24h_pct`` is the absolute fraction of 24-h price decline
    (0.0 when not computed).

    ``volume_24h_idr`` is the 24-h IDR trading volume (0.0 when not available).
    """

    detected: bool
    reason: str
    drop_24h_pct: float
    volume_24h_idr: float


def detect_rug_pull_risk(
    ticker: Dict[str, Any],
    max_drop_24h_pct: float = 0.50,
    min_volume_24h_idr: float = 0.0,
    min_trades_24h: int = 0,
) -> RugPullRisk:
    """Detect rug-pull or dead-coin risk from the 24-h ticker snapshot.

    :param ticker:
        Indodax ticker dict (either ``{"ticker": {...}}`` wrapper or the inner
        dict directly).  Expected keys: ``"high"``, ``"low"``, ``"last"``
        (or ``"last_price"``), ``"vol_idr"`` (or ``"volume"``) and optionally
        ``"trade_count"`` / ``"count"`` for the number of trades in 24 h.
    :param max_drop_24h_pct:
        Flag when ``(high − last) / high ≥ max_drop_24h_pct``.
        Default 0.50 = 50% drop flags the pair as rug-pull risk.
        Set to 0 to disable this check.
    :param min_volume_24h_idr:
        Minimum 24-h IDR volume required.  When volume is below this value the
        pair is flagged as a dead coin.  0 = disabled (default).
    :param min_trades_24h:
        Minimum number of trades in the past 24 h.  0 = disabled (default).
    :returns: :class:`RugPullRisk` with ``detected=False`` when all checks pass.
    """
    no_risk = RugPullRisk(detected=False, reason="", drop_24h_pct=0.0, volume_24h_idr=0.0)

    # Unwrap {"ticker": {...}} wrapper if present
    inner = ticker.get("ticker", ticker) if isinstance(ticker, dict) else {}
    if not isinstance(inner, dict):
        inner = {}

    def _f(key: str, *aliases: str) -> float:
        for k in (key,) + aliases:
            v = inner.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return 0.0

    high = _f("high")
    last = _f("last", "last_price")
    volume_idr = _f("vol_idr", "volume", "vol")

    # ── 24-h price drop check ────────────────────────────────────────────────
    drop_24h = 0.0
    if max_drop_24h_pct > 0 and high > 0 and last >= 0:
        drop_24h = (high - last) / high
        if drop_24h >= max_drop_24h_pct:
            return RugPullRisk(
                detected=True,
                reason=f"24h_drop={drop_24h:.1%}",
                drop_24h_pct=round(drop_24h, 4),
                volume_24h_idr=volume_idr,
            )

    # ── Volume / dead coin check ─────────────────────────────────────────────
    if min_volume_24h_idr > 0 and volume_idr < min_volume_24h_idr:
        return RugPullRisk(
            detected=True,
            reason=f"dead_coin_volume={volume_idr:.0f}_idr",
            drop_24h_pct=round(drop_24h, 4),
            volume_24h_idr=volume_idr,
        )

    # ── Trade count check ────────────────────────────────────────────────────
    if min_trades_24h > 0:
        trade_count = int(_f("trade_count", "count") or 0)
        if trade_count < min_trades_24h:
            return RugPullRisk(
                detected=True,
                reason=f"dead_coin_trades={trade_count}",
                drop_24h_pct=round(drop_24h, 4),
                volume_24h_idr=volume_idr,
            )

    return no_risk
