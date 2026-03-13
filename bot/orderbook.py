"""Advanced orderbook analysis module.

Provides 12 orderbook analysis features for the Indodax trading bot:

 1. Bid / Ask spread analysis
 2. Orderbook imbalance detection
 3. Liquidity gap detection
 4. Hidden liquidity detection
 5. Whale order detection
 6. Iceberg order detection
 7. Market depth modeling
 8. Order flow imbalance
 9. Buy vs Sell pressure analysis
10. Spoofing detection
11. Slippage prediction
12. Orderbook heatmap analysis

Each feature is implemented as a pure function operating on the standard
depth dict format (``{"buy": [[price, vol], ...], "sell": [[price, vol], ...]}``).
Results are typed dataclasses for easy downstream consumption.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _parse_levels(
    raw_levels: Sequence[Sequence[Any]],
    top_n: int = 50,
) -> List[Tuple[float, float]]:
    """Parse raw orderbook levels into (price, volume) tuples."""
    result: List[Tuple[float, float]] = []
    for level in raw_levels[:top_n]:
        try:
            price = float(level[0])
            volume = float(level[1])
            if price > 0 and volume >= 0:
                result.append((price, volume))
        except (IndexError, TypeError, ValueError):
            continue
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  1. Bid / Ask Spread Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpreadAnalysis:
    """Detailed bid/ask spread analysis.

    ``spread_abs`` is the absolute spread in price units.
    ``spread_pct`` is the spread as a fraction of mid-price.
    ``mid_price`` is the average of best bid and best ask.
    ``is_wide`` flags spreads above *wide_threshold_pct*.
    """

    best_bid: float
    best_ask: float
    spread_abs: float
    spread_pct: float
    mid_price: float
    is_wide: bool


def analyze_spread(
    depth: Dict[str, object],
    wide_threshold_pct: float = 0.005,
) -> SpreadAnalysis:
    """Compute detailed spread metrics from the orderbook.

    :param depth: Depth dict with ``"buy"`` and ``"sell"`` level lists.
    :param wide_threshold_pct: Spread percentage above which ``is_wide``
        is set to ``True``.  Default 0.5%.
    :returns: :class:`SpreadAnalysis`.
    """
    bids = _parse_levels(depth.get("buy") or [])
    asks = _parse_levels(depth.get("sell") or [])

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    spread_abs = best_ask - best_bid if best_bid and best_ask else 0.0
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0
    spread_pct = spread_abs / mid if mid > 0 else 0.0

    return SpreadAnalysis(
        best_bid=best_bid,
        best_ask=best_ask,
        spread_abs=spread_abs,
        spread_pct=spread_pct,
        mid_price=mid,
        is_wide=spread_pct >= wide_threshold_pct,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Orderbook Imbalance Detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImbalanceResult:
    """Orderbook volume imbalance across configurable depth.

    ``imbalance`` ranges from −1 (all ask) to +1 (all bid).
    ``bid_total`` and ``ask_total`` are notional IDR volumes.
    ``dominant_side`` is ``"bid"``, ``"ask"``, or ``"balanced"``.
    """

    imbalance: float
    bid_total: float
    ask_total: float
    dominant_side: str
    levels_analyzed: int


def detect_imbalance(
    depth: Dict[str, object],
    top_n: int = 20,
    threshold: float = 0.3,
) -> ImbalanceResult:
    """Detect orderbook volume imbalance.

    :param depth: Depth dict.
    :param top_n: Number of levels per side to aggregate.
    :param threshold: Absolute imbalance value above which a side is
        considered dominant.  Default 0.3 (30 %).
    :returns: :class:`ImbalanceResult`.
    """
    bids = _parse_levels(depth.get("buy") or [], top_n)
    asks = _parse_levels(depth.get("sell") or [], top_n)

    bid_total = sum(p * v for p, v in bids)
    ask_total = sum(p * v for p, v in asks)
    total = bid_total + ask_total

    imbalance = (bid_total - ask_total) / total if total > 0 else 0.0

    if imbalance >= threshold:
        dominant = "bid"
    elif imbalance <= -threshold:
        dominant = "ask"
    else:
        dominant = "balanced"

    return ImbalanceResult(
        imbalance=imbalance,
        bid_total=bid_total,
        ask_total=ask_total,
        dominant_side=dominant,
        levels_analyzed=min(len(bids), len(asks)),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Liquidity Gap Detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiquidityGap:
    """A detected gap in the orderbook where price jumps between levels."""

    side: str           # "bid" or "ask"
    price_from: float
    price_to: float
    gap_pct: float      # gap size as fraction of price_from


@dataclass
class LiquidityGapResult:
    """Result of liquidity gap detection."""

    detected: bool
    gaps: List[LiquidityGap]
    worst_gap_pct: float


def detect_liquidity_gaps(
    depth: Dict[str, object],
    min_gap_pct: float = 0.01,
    top_n: int = 30,
) -> LiquidityGapResult:
    """Detect significant price gaps in the orderbook.

    Scans both bid and ask sides for consecutive levels where the price
    difference exceeds *min_gap_pct* of the nearer-to-market price.

    :param depth: Depth dict.
    :param min_gap_pct: Minimum gap as fraction of price (default 1 %).
    :param top_n: Levels per side to scan.
    :returns: :class:`LiquidityGapResult`.
    """
    gaps: List[LiquidityGap] = []

    for side_key, side_label in (("buy", "bid"), ("sell", "ask")):
        levels = _parse_levels(depth.get(side_key) or [], top_n)
        for i in range(len(levels) - 1):
            p1 = levels[i][0]
            p2 = levels[i + 1][0]
            ref = p1 if p1 > 0 else p2
            if ref <= 0:
                continue
            gap = abs(p1 - p2) / ref
            if gap >= min_gap_pct:
                gaps.append(LiquidityGap(
                    side=side_label,
                    price_from=p1,
                    price_to=p2,
                    gap_pct=gap,
                ))

    worst = max((g.gap_pct for g in gaps), default=0.0)
    return LiquidityGapResult(
        detected=len(gaps) > 0,
        gaps=gaps,
        worst_gap_pct=worst,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Hidden Liquidity Detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HiddenLiquidityResult:
    """Result of hidden liquidity detection.

    Detects price levels where repeated fills occur at volumes exceeding the
    visible resting quantity — suggesting hidden / reserve orders.
    """

    detected: bool
    side: Optional[str]
    estimated_hidden_volume: float
    confidence: float


def detect_hidden_liquidity(
    depth: Dict[str, object],
    recent_trades: Sequence[Dict[str, Any]],
    top_n: int = 10,
    fill_multiplier: float = 1.5,
) -> HiddenLiquidityResult:
    """Detect hidden (iceberg-like) liquidity by comparing fill volume to visible depth.

    If the total filled volume at a price level within the top-of-book
    exceeds *fill_multiplier × visible_volume*, hidden liquidity is likely.

    :param depth: Current orderbook snapshot.
    :param recent_trades: Recent trades with ``"price"`` and ``"amount"`` keys.
    :param top_n: Levels to inspect per side.
    :param fill_multiplier: Ratio threshold.
    :returns: :class:`HiddenLiquidityResult`.
    """
    no_hidden = HiddenLiquidityResult(
        detected=False, side=None, estimated_hidden_volume=0.0, confidence=0.0,
    )
    if not recent_trades:
        return no_hidden

    # Build visible volume map from bid/ask levels.
    visible: Dict[float, float] = {}
    for side_key in ("buy", "sell"):
        for price, vol in _parse_levels(depth.get(side_key) or [], top_n):
            visible[round(price, 8)] = vol

    # Aggregate filled volume per price from recent trades.
    filled: Dict[float, float] = {}
    for t in recent_trades:
        if not isinstance(t, dict):
            continue
        tp = _safe_float(t.get("price", 0))
        ta = _safe_float(t.get("amount", 0))
        if tp > 0 and ta > 0:
            key = round(tp, 8)
            filled[key] = filled.get(key, 0.0) + ta

    best_excess = 0.0
    best_side: Optional[str] = None

    bids = _parse_levels(depth.get("buy") or [], top_n)
    asks = _parse_levels(depth.get("sell") or [], top_n)

    for side_label, levels in (("bid", bids), ("ask", asks)):
        for price, vis_vol in levels:
            key = round(price, 8)
            fill_vol = filled.get(key, 0.0)
            if vis_vol > 0 and fill_vol > vis_vol * fill_multiplier:
                excess = fill_vol - vis_vol
                if excess > best_excess:
                    best_excess = excess
                    best_side = side_label

    if best_excess <= 0:
        return no_hidden

    confidence = min(1.0, best_excess / (best_excess + 1.0))
    return HiddenLiquidityResult(
        detected=True,
        side=best_side,
        estimated_hidden_volume=best_excess,
        confidence=confidence,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Whale Order Detection (enhanced)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class WhaleOrder:
    """A single detected whale-sized order."""

    side: str       # "bid" or "ask"
    price: float
    volume: float
    notional: float  # price × volume (IDR equivalent)
    ratio: float     # volume / avg_volume


@dataclass
class WhaleDetectionResult:
    """Enhanced whale detection with individual whale orders enumerated."""

    detected: bool
    whales: List[WhaleOrder]
    total_whale_notional: float
    dominant_side: Optional[str]


def detect_whale_orders(
    depth: Dict[str, object],
    multiplier: float = 5.0,
    top_n: int = 30,
) -> WhaleDetectionResult:
    """Detect and enumerate all whale-sized orders in the orderbook.

    A whale order is one whose volume exceeds *multiplier × mean_volume*
    on its side.  Unlike the simple :func:`~bot.analysis.detect_whale_activity`
    which only returns the largest anomaly, this function returns **every**
    qualifying order.

    :param depth: Depth dict.
    :param multiplier: Volume-to-mean ratio threshold.  Default 5×.
    :param top_n: Levels per side.
    :returns: :class:`WhaleDetectionResult`.
    """
    whales: List[WhaleOrder] = []

    for side_key, side_label in (("buy", "bid"), ("sell", "ask")):
        levels = _parse_levels(depth.get(side_key) or [], top_n)
        if len(levels) < 3:
            continue
        volumes = [v for _, v in levels]
        avg_vol = mean(volumes) if volumes else 0.0
        if avg_vol <= 0:
            continue
        for price, vol in levels:
            ratio = vol / avg_vol
            if ratio >= multiplier:
                whales.append(WhaleOrder(
                    side=side_label,
                    price=price,
                    volume=vol,
                    notional=price * vol,
                    ratio=ratio,
                ))

    total_notional = sum(w.notional for w in whales)
    bid_notional = sum(w.notional for w in whales if w.side == "bid")
    ask_notional = sum(w.notional for w in whales if w.side == "ask")

    if not whales:
        dominant = None
    elif bid_notional > ask_notional:
        dominant = "bid"
    elif ask_notional > bid_notional:
        dominant = "ask"
    else:
        dominant = None

    return WhaleDetectionResult(
        detected=len(whales) > 0,
        whales=whales,
        total_whale_notional=total_notional,
        dominant_side=dominant,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6. Iceberg Order Detection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IcebergResult:
    """Result of iceberg order detection.

    Iceberg orders are large orders split into small visible slices that
    continuously refill.  Detected by comparing stable resting volume at a
    price level across successive snapshots.
    """

    detected: bool
    side: Optional[str]
    price: float
    visible_volume: float
    estimated_total_volume: float
    refill_count: int


def detect_iceberg_orders(
    snapshots: Sequence[Dict[str, object]],
    min_refills: int = 3,
    volume_tolerance: float = 0.2,
    top_n: int = 10,
) -> List[IcebergResult]:
    """Detect iceberg orders by tracking volume refills across snapshots.

    An iceberg is suspected when a price level appears in multiple snapshots
    with roughly the same visible volume (within *volume_tolerance* fraction),
    suggesting automated refilling.

    :param snapshots: Chronological list of depth dicts.
    :param min_refills: Minimum number of snapshots a level must appear in.
    :param volume_tolerance: Max fractional deviation to count as same size.
    :param top_n: Levels per side to inspect.
    :returns: List of :class:`IcebergResult` (empty if none detected).
    """
    if len(snapshots) < min_refills:
        return []

    # Track (side, price) → list of volumes across snapshots.
    tracker: Dict[Tuple[str, float], List[float]] = {}

    for snap in snapshots:
        for side_key, side_label in (("buy", "bid"), ("sell", "ask")):
            levels = _parse_levels(snap.get(side_key) or [], top_n)
            for price, vol in levels:
                key = (side_label, round(price, 8))
                tracker.setdefault(key, []).append(vol)

    results: List[IcebergResult] = []
    for (side, price), vols in tracker.items():
        if len(vols) < min_refills:
            continue
        avg_vol = mean(vols)
        if avg_vol <= 0:
            continue
        # Check stability: all volumes close to the average.
        stable = all(
            abs(v - avg_vol) / avg_vol <= volume_tolerance
            for v in vols
        )
        if stable:
            results.append(IcebergResult(
                detected=True,
                side=side,
                price=price,
                visible_volume=avg_vol,
                estimated_total_volume=avg_vol * len(vols),
                refill_count=len(vols),
            ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
#  7. Market Depth Modeling
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DepthLevel:
    """Cumulative depth at a given price distance from mid."""

    distance_pct: float
    cumulative_bid_volume: float
    cumulative_ask_volume: float
    cumulative_bid_notional: float
    cumulative_ask_notional: float


@dataclass
class DepthModel:
    """Model of orderbook depth at various distances from mid-price.

    ``levels`` are cumulative snapshots at 0.1%, 0.5%, 1%, 2%, 5% from mid.
    ``bid_depth_total`` / ``ask_depth_total`` are full-book notional sums.
    ``depth_ratio`` = bid_depth_total / ask_depth_total.
    """

    mid_price: float
    levels: List[DepthLevel]
    bid_depth_total: float
    ask_depth_total: float
    depth_ratio: float


def model_market_depth(
    depth: Dict[str, object],
    distances: Sequence[float] = (0.001, 0.005, 0.01, 0.02, 0.05),
    top_n: int = 50,
) -> DepthModel:
    """Build a cumulative depth model at various price distances from mid.

    :param depth: Depth dict.
    :param distances: Fractional distances from mid-price to sample.
    :param top_n: Max levels per side.
    :returns: :class:`DepthModel`.
    """
    bids = _parse_levels(depth.get("buy") or [], top_n)
    asks = _parse_levels(depth.get("sell") or [], top_n)

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0.0

    levels: List[DepthLevel] = []
    for dist in distances:
        bid_vol = 0.0
        bid_not = 0.0
        ask_vol = 0.0
        ask_not = 0.0

        if mid > 0:
            lower = mid * (1 - dist)
            upper = mid * (1 + dist)

            for p, v in bids:
                if p >= lower:
                    bid_vol += v
                    bid_not += p * v
            for p, v in asks:
                if p <= upper:
                    ask_vol += v
                    ask_not += p * v

        levels.append(DepthLevel(
            distance_pct=dist,
            cumulative_bid_volume=bid_vol,
            cumulative_ask_volume=ask_vol,
            cumulative_bid_notional=bid_not,
            cumulative_ask_notional=ask_not,
        ))

    bid_total = sum(p * v for p, v in bids)
    ask_total = sum(p * v for p, v in asks)
    ratio = bid_total / ask_total if ask_total > 0 else 0.0

    return DepthModel(
        mid_price=mid,
        levels=levels,
        bid_depth_total=bid_total,
        ask_depth_total=ask_total,
        depth_ratio=ratio,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. Order Flow Imbalance
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OrderFlowImbalance:
    """Order flow imbalance computed from recent trades.

    ``imbalance`` ranges from −1 (pure selling) to +1 (pure buying).
    ``net_flow`` is buy_volume − sell_volume in notional terms.
    ``trade_count`` is total trades analysed.
    """

    imbalance: float
    buy_volume: float
    sell_volume: float
    net_flow: float
    trade_count: int
    aggressive_side: Optional[str]


def compute_order_flow_imbalance(
    trades: Sequence[Dict[str, Any]],
    aggressive_threshold: float = 0.6,
) -> OrderFlowImbalance:
    """Compute order flow imbalance from recent trades.

    :param trades: List of trade dicts with ``"type"`` (buy/sell),
        ``"price"``, ``"amount"`` fields.
    :param aggressive_threshold: Threshold above which one side is
        flagged as aggressive.
    :returns: :class:`OrderFlowImbalance`.
    """
    buy_vol = 0.0
    sell_vol = 0.0
    count = 0

    for t in trades:
        if not isinstance(t, dict):
            continue
        trade_type = str(t.get("type", "")).lower()
        price = _safe_float(t.get("price", 0))
        raw_amount = t.get("amount")
        if raw_amount is None:
            raw_amount = t.get("vol", 0)
        amount = _safe_float(raw_amount)
        notional = price * amount if price > 0 else amount
        if trade_type in ("buy", "bid"):
            buy_vol += notional
        elif trade_type in ("sell", "ask"):
            sell_vol += notional
        count += 1

    total = buy_vol + sell_vol
    imbalance = (buy_vol - sell_vol) / total if total > 0 else 0.0
    net_flow = buy_vol - sell_vol

    buy_ratio = buy_vol / total if total > 0 else 0.5
    if buy_ratio >= aggressive_threshold:
        aggressive = "buy"
    elif (1 - buy_ratio) >= aggressive_threshold:
        aggressive = "sell"
    else:
        aggressive = None

    return OrderFlowImbalance(
        imbalance=imbalance,
        buy_volume=buy_vol,
        sell_volume=sell_vol,
        net_flow=net_flow,
        trade_count=count,
        aggressive_side=aggressive,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Buy vs Sell Pressure Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PressureAnalysis:
    """Combined buy/sell pressure from both orderbook and trade flow.

    ``pressure`` ranges from −1 (max sell pressure) to +1 (max buy pressure).
    Individual components are provided for transparency.
    """

    pressure: float
    book_imbalance: float   # from orderbook
    flow_imbalance: float   # from trade flow
    signal: str             # "buy_pressure", "sell_pressure", "neutral"


def analyze_pressure(
    depth: Dict[str, object],
    trades: Sequence[Dict[str, Any]],
    book_weight: float = 0.5,
    flow_weight: float = 0.5,
    threshold: float = 0.2,
    top_n: int = 20,
) -> PressureAnalysis:
    """Combine orderbook imbalance and trade flow into a pressure score.

    :param depth: Depth dict.
    :param trades: Recent trades.
    :param book_weight: Weight for the orderbook imbalance (0–1).
    :param flow_weight: Weight for the trade flow imbalance (0–1).
    :param threshold: Absolute pressure value above which a signal fires.
    :param top_n: Orderbook depth to use.
    :returns: :class:`PressureAnalysis`.
    """
    book = detect_imbalance(depth, top_n=top_n)
    flow = compute_order_flow_imbalance(trades)

    pressure = book.imbalance * book_weight + flow.imbalance * flow_weight

    if pressure >= threshold:
        signal = "buy_pressure"
    elif pressure <= -threshold:
        signal = "sell_pressure"
    else:
        signal = "neutral"

    return PressureAnalysis(
        pressure=pressure,
        book_imbalance=book.imbalance,
        flow_imbalance=flow.imbalance,
        signal=signal,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Spoofing Detection (enhanced)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SpoofingWall:
    """A single suspicious (potentially spoofed) wall."""

    side: str
    price: float
    volume: float
    distance_pct: float
    volume_ratio: float


@dataclass
class EnhancedSpoofingResult:
    """Enhanced spoofing detection with individual walls enumerated.

    Unlike the basic :class:`~bot.analysis.SpoofingResult`, this returns
    *all* suspicious walls and provides a ``risk_score`` (0–1).
    """

    detected: bool
    walls: List[SpoofingWall]
    risk_score: float


def detect_spoofing_enhanced(
    depth: Dict[str, object],
    volume_multiplier: float = 5.0,
    min_distance_pct: float = 0.03,
    top_n: int = 30,
) -> EnhancedSpoofingResult:
    """Enhanced spoofing detection that enumerates all suspicious walls.

    :param depth: Depth dict.
    :param volume_multiplier: Level-vol / avg-vol threshold.
    :param min_distance_pct: Minimum distance from top-of-book.
    :param top_n: Levels per side.
    :returns: :class:`EnhancedSpoofingResult`.
    """
    walls: List[SpoofingWall] = []

    for side_key, side_label in (("buy", "bid"), ("sell", "ask")):
        levels = _parse_levels(depth.get(side_key) or [], top_n)
        if len(levels) < 3:
            continue
        volumes = [v for _, v in levels]
        avg_vol = mean(volumes)
        if avg_vol <= 0:
            continue
        top_price = levels[0][0]
        if top_price <= 0:
            continue
        for price, vol in levels[1:]:
            ratio = vol / avg_vol
            distance = abs(price - top_price) / top_price
            if ratio >= volume_multiplier and distance >= min_distance_pct:
                walls.append(SpoofingWall(
                    side=side_label,
                    price=price,
                    volume=vol,
                    distance_pct=distance,
                    volume_ratio=ratio,
                ))

    risk_score = min(1.0, len(walls) * 0.25)
    return EnhancedSpoofingResult(
        detected=len(walls) > 0,
        walls=walls,
        risk_score=risk_score,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Slippage Prediction
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SlippagePrediction:
    """Predicted slippage for a given order size.

    ``estimated_slippage_pct`` is the expected deviation from mid-price.
    ``avg_fill_price`` is the volume-weighted average execution price.
    ``levels_consumed`` is how many orderbook levels would be filled.
    """

    side: str                   # "buy" or "sell"
    order_size: float           # requested notional or volume
    avg_fill_price: float
    estimated_slippage_pct: float
    levels_consumed: int
    fully_filled: bool


def predict_slippage(
    depth: Dict[str, object],
    order_size: float,
    side: str = "buy",
    top_n: int = 50,
) -> SlippagePrediction:
    """Predict slippage for a market order of given size.

    Walks the orderbook to simulate filling *order_size* volume, computing
    the volume-weighted average execution price.

    :param depth: Depth dict.
    :param order_size: Volume of coin to buy or sell.
    :param side: ``"buy"`` (walks ask side) or ``"sell"`` (walks bid side).
    :param top_n: Max levels to walk.
    :returns: :class:`SlippagePrediction`.
    """
    side_key = "sell" if side == "buy" else "buy"
    levels = _parse_levels(depth.get(side_key) or [], top_n)

    other_key = "buy" if side == "buy" else "sell"
    other_levels = _parse_levels(depth.get(other_key) or [], 1)

    best_bid = 0.0
    best_ask = 0.0
    if side == "buy":
        best_ask = levels[0][0] if levels else 0.0
        best_bid = other_levels[0][0] if other_levels else 0.0
    else:
        best_bid = levels[0][0] if levels else 0.0
        best_ask = other_levels[0][0] if other_levels else 0.0

    mid = (best_bid + best_ask) / 2 if best_bid and best_ask else (
        levels[0][0] if levels else 0.0
    )

    remaining = order_size
    cost = 0.0
    consumed = 0

    for price, vol in levels:
        if remaining <= 0:
            break
        fill = min(remaining, vol)
        cost += fill * price
        remaining -= fill
        consumed += 1

    filled_vol = order_size - remaining
    avg_price = cost / filled_vol if filled_vol > 0 else mid
    slippage = abs(avg_price - mid) / mid if mid > 0 else 0.0

    return SlippagePrediction(
        side=side,
        order_size=order_size,
        avg_fill_price=avg_price,
        estimated_slippage_pct=slippage,
        levels_consumed=consumed,
        fully_filled=remaining <= 0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Orderbook Heatmap Analysis
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeatmapBin:
    """A single price bin in the orderbook heatmap."""

    price_low: float
    price_high: float
    bid_volume: float
    ask_volume: float
    total_volume: float
    intensity: float  # 0–1 normalised


@dataclass
class HeatmapAnalysis:
    """Orderbook heatmap — volume distribution across price bins.

    ``bins`` are equal-width price slices.
    ``concentration_price`` is the bin centre with the highest combined volume.
    ``concentration_side`` is ``"bid"`` or ``"ask"`` (which side dominates
    at the concentration point).
    """

    bins: List[HeatmapBin]
    concentration_price: float
    concentration_side: Optional[str]
    num_bins: int


def analyze_heatmap(
    depth: Dict[str, object],
    num_bins: int = 20,
    top_n: int = 50,
) -> HeatmapAnalysis:
    """Build an orderbook heatmap showing volume concentration across price.

    :param depth: Depth dict.
    :param num_bins: Number of equal-width price bins.
    :param top_n: Max levels per side.
    :returns: :class:`HeatmapAnalysis`.
    """
    bids = _parse_levels(depth.get("buy") or [], top_n)
    asks = _parse_levels(depth.get("sell") or [], top_n)

    all_prices = [p for p, _ in bids] + [p for p, _ in asks]
    if not all_prices:
        return HeatmapAnalysis(
            bins=[], concentration_price=0.0,
            concentration_side=None, num_bins=0,
        )

    lo = min(all_prices)
    hi = max(all_prices)
    if hi == lo:
        total_vol = sum(v for _, v in bids) + sum(v for _, v in asks)
        return HeatmapAnalysis(
            bins=[HeatmapBin(lo, hi, sum(v for _, v in bids),
                             sum(v for _, v in asks), total_vol, 1.0)],
            concentration_price=lo,
            concentration_side="bid" if sum(v for _, v in bids) >= sum(v for _, v in asks) else "ask",
            num_bins=1,
        )

    width = (hi - lo) / num_bins
    bins: List[HeatmapBin] = []
    for i in range(num_bins):
        bl = lo + i * width
        bh = bl + width
        bins.append(HeatmapBin(bl, bh, 0.0, 0.0, 0.0, 0.0))

    def _place(price: float, volume: float, side: str) -> None:
        idx = min(int((price - lo) / width), num_bins - 1)
        b = bins[idx]
        if side == "bid":
            bins[idx] = HeatmapBin(
                b.price_low, b.price_high,
                b.bid_volume + volume, b.ask_volume,
                b.total_volume + volume, b.intensity,
            )
        else:
            bins[idx] = HeatmapBin(
                b.price_low, b.price_high,
                b.bid_volume, b.ask_volume + volume,
                b.total_volume + volume, b.intensity,
            )

    for p, v in bids:
        _place(p, v, "bid")
    for p, v in asks:
        _place(p, v, "ask")

    max_vol = max((b.total_volume for b in bins), default=1.0) or 1.0
    normalised: List[HeatmapBin] = []
    for b in bins:
        normalised.append(HeatmapBin(
            b.price_low, b.price_high,
            b.bid_volume, b.ask_volume,
            b.total_volume,
            b.total_volume / max_vol,
        ))

    hottest = max(normalised, key=lambda b: b.total_volume)
    conc_price = (hottest.price_low + hottest.price_high) / 2
    conc_side = "bid" if hottest.bid_volume >= hottest.ask_volume else "ask"

    return HeatmapAnalysis(
        bins=normalised,
        concentration_price=conc_price,
        concentration_side=conc_side,
        num_bins=num_bins,
    )
