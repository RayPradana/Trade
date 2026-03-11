from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Sequence


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


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_candles(
    trades: Sequence[Dict[str, object]],
    interval_seconds: int,
    limit: int = 96,
) -> List[Candle]:
    if not trades:
        return []

    sorted_trades = sorted(trades, key=lambda t: int(t.get("date", 0)))
    first_ts = int(sorted_trades[0].get("date", 0))
    buckets: Dict[int, List[Dict[str, object]]] = {}
    for trade in sorted_trades:
        ts = int(trade.get("date", 0))
        bucket = first_ts + ((ts - first_ts) // interval_seconds) * interval_seconds
        buckets.setdefault(bucket, []).append(trade)

    candles: List[Candle] = []
    for bucket_ts in sorted(buckets.keys())[-limit:]:
        bucket_trades = buckets[bucket_ts]
        trade_prices = [_safe_float(t.get("price")) for t in bucket_trades]
        amounts = [_safe_float(t.get("amount", 0)) for t in bucket_trades]
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
