"""Smart order execution module.

Provides 15 execution algorithm categories for the Indodax trading bot:

 1. Smart order routing
 2. Direct market access (DMA)
 3. Low-latency order execution
 4. Limit order execution
 5. Market order execution
 6. TWAP execution algorithm
 7. VWAP execution algorithm
 8. Iceberg order execution
 9. Adaptive order execution
10. Slippage protection
11. Partial fill handling
12. Order retry mechanism
13. Execution quality monitoring
14. Order batching
15. Latency optimization

Each algorithm is implemented as a pure function operating on standard
market data (candles, orderbook levels, order state) and returns typed
dataclasses.  All implementations use only the Python standard library.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .analysis import Candle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _weighted_avg(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights:
        return 0.0
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


# ═══════════════════════════════════════════════════════════════════════════
#  1. Smart Order Routing
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SmartRoute:
    """Result of smart order routing analysis.

    ``recommended_venue`` is the best execution venue.
    ``route_scores`` maps venue → quality score.
    ``estimated_cost`` is the expected execution cost (pct).
    """

    recommended_venue: str
    route_scores: Dict[str, float]
    estimated_cost: float
    estimated_fill_time: float
    reason: str


def smart_order_route(
    order_size: float,
    venues: Sequence[Dict[str, Any]],
    urgency: float = 0.5,
) -> SmartRoute:
    """Select the optimal execution venue for an order.

    Each venue dict should have ``"name"`` (str), ``"liquidity"`` (float),
    ``"fee"`` (float, pct), ``"latency_ms"`` (float), and
    ``"spread"`` (float, pct).

    :param order_size: Size of order in base currency.
    :param venues: Available execution venues with metrics.
    :param urgency: 0..1 where 1 means fastest execution preferred.
    :returns: :class:`SmartRoute`.
    """
    if not venues:
        return SmartRoute(
            recommended_venue="default",
            route_scores={},
            estimated_cost=0.0,
            estimated_fill_time=0.0,
            reason="no venues available",
        )

    scores: Dict[str, float] = {}
    for venue in venues:
        name = str(venue.get("name", "unknown"))
        liquidity = _safe_float(venue.get("liquidity", 0))
        fee = _safe_float(venue.get("fee", 0))
        latency = _safe_float(venue.get("latency_ms", 100))
        spread = _safe_float(venue.get("spread", 0))

        # Liquidity score: can we fill without moving the market?
        liq_score = min(1.0, liquidity / (order_size + 1e-9)) * 40

        # Cost score: lower fees and spread = better
        cost_score = max(0, 30 - (fee + spread) * 1000)

        # Speed score: lower latency = better, weighted by urgency
        speed_score = max(0, 30 - latency / 100) * urgency

        scores[name] = round(liq_score + cost_score + speed_score, 4)

    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_venue = next((v for v in venues if v.get("name") == best), venues[0])
    est_cost = _safe_float(best_venue.get("fee", 0)) + _safe_float(best_venue.get("spread", 0))
    est_time = _safe_float(best_venue.get("latency_ms", 100)) / 1000

    return SmartRoute(
        recommended_venue=best,
        route_scores=scores,
        estimated_cost=round(est_cost, 6),
        estimated_fill_time=round(est_time, 4),
        reason=f"route: {best} score={scores[best]:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Direct Market Access (DMA)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DMAOrder:
    """Direct market access order specification.

    ``order_type`` is ``"limit"`` or ``"market"``.
    ``pre_trade_checks`` lists validation results.
    """

    order_type: str
    price: float
    quantity: float
    side: str
    pre_trade_checks: List[Dict[str, Any]]
    passed_checks: bool
    estimated_fill_pct: float
    reason: str


def create_dma_order(
    side: str,
    price: float,
    quantity: float,
    orderbook_bids: Sequence[Tuple[float, float]],
    orderbook_asks: Sequence[Tuple[float, float]],
    max_order_pct: float = 0.1,
) -> DMAOrder:
    """Create a direct market access order with pre-trade validation.

    :param side: ``"buy"`` or ``"sell"``.
    :param price: Intended execution price.
    :param quantity: Order quantity.
    :param orderbook_bids: Current bids as (price, qty) tuples.
    :param orderbook_asks: Current asks as (price, qty) tuples.
    :param max_order_pct: Max order size as fraction of book liquidity.
    :returns: :class:`DMAOrder`.
    """
    checks: List[Dict[str, Any]] = []

    # Check 1: Price validity
    price_valid = price > 0
    checks.append({"check": "price_valid", "passed": price_valid, "value": price})

    # Check 2: Quantity validity
    qty_valid = quantity > 0
    checks.append({"check": "quantity_valid", "passed": qty_valid, "value": quantity})

    # Check 3: Market depth check
    if side == "buy":
        available = sum(q for _, q in orderbook_asks) if orderbook_asks else 0.0
    else:
        available = sum(q for _, q in orderbook_bids) if orderbook_bids else 0.0

    depth_ok = quantity <= available * max_order_pct if available > 0 else False
    checks.append({"check": "depth_sufficient", "passed": depth_ok, "available": round(available, 8)})

    # Check 4: Price reasonableness
    if side == "buy" and orderbook_asks:
        best_ask = orderbook_asks[0][0]
        price_reasonable = price <= best_ask * 1.05
    elif side == "sell" and orderbook_bids:
        best_bid = orderbook_bids[0][0]
        price_reasonable = price >= best_bid * 0.95
    else:
        price_reasonable = True
    checks.append({"check": "price_reasonable", "passed": price_reasonable})

    passed = all(c["passed"] for c in checks)
    est_fill = 0.95 if passed else 0.5

    order_type = "limit"

    return DMAOrder(
        order_type=order_type,
        price=price,
        quantity=quantity,
        side=side,
        pre_trade_checks=checks,
        passed_checks=passed,
        estimated_fill_pct=est_fill,
        reason=f"dma: {side} {quantity}@{price}, checks={'passed' if passed else 'failed'}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Low-Latency Order Execution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LowLatencyPlan:
    """Low-latency execution plan.

    ``optimizations`` lists applied latency reduction techniques.
    ``estimated_latency_ms`` is the expected round-trip time.
    """

    estimated_latency_ms: float
    optimizations: List[str]
    priority_level: str
    batch_mode: bool
    pre_computed_signature: bool
    reason: str


def plan_low_latency_execution(
    order_count: int = 1,
    current_latency_ms: float = 100.0,
    enable_batching: bool = True,
    enable_pre_compute: bool = True,
) -> LowLatencyPlan:
    """Plan a low-latency order execution strategy.

    Analyses current conditions and recommends optimizations to reduce
    execution latency.

    :param order_count: Number of orders to execute.
    :param current_latency_ms: Current measured round-trip latency.
    :param enable_batching: Allow order batching.
    :param enable_pre_compute: Allow pre-computing signatures.
    :returns: :class:`LowLatencyPlan`.
    """
    opts: List[str] = []
    latency = current_latency_ms

    # Connection pooling
    if current_latency_ms > 50:
        opts.append("connection_pooling")
        latency *= 0.8

    # Pre-compute signatures
    if enable_pre_compute:
        opts.append("pre_computed_signatures")
        latency *= 0.9

    # Batch mode
    batch = enable_batching and order_count > 1
    if batch:
        opts.append("order_batching")
        latency *= 0.7

    # Request pipelining
    if order_count > 3:
        opts.append("request_pipelining")
        latency *= 0.85

    # Payload minimization
    opts.append("payload_minimization")
    latency *= 0.95

    if latency < 20:
        priority = "ultra_low"
    elif latency < 50:
        priority = "low"
    elif latency < 100:
        priority = "normal"
    else:
        priority = "high"

    return LowLatencyPlan(
        estimated_latency_ms=round(latency, 2),
        optimizations=opts,
        priority_level=priority,
        batch_mode=batch,
        pre_computed_signature=enable_pre_compute,
        reason=f"latency: {latency:.1f}ms, {len(opts)} optimizations",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Limit Order Execution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LimitOrderPlan:
    """Limit order execution plan.

    ``price`` is the recommended limit price.
    ``placement_strategy`` describes where to place the order in the book.
    """

    price: float
    quantity: float
    side: str
    placement_strategy: str
    distance_from_mid: float
    expected_fill_time_s: float
    cancel_after_s: float
    reason: str


def plan_limit_order(
    side: str,
    quantity: float,
    candles: Sequence[Candle],
    best_bid: float,
    best_ask: float,
    aggressiveness: float = 0.5,
) -> LimitOrderPlan:
    """Plan a limit order with optimal price placement.

    :param side: ``"buy"`` or ``"sell"``.
    :param quantity: Order quantity.
    :param candles: Recent OHLCV candles for volatility estimation.
    :param best_bid: Current best bid price.
    :param best_ask: Current best ask price.
    :param aggressiveness: 0..1 where 1 means price at the top of book.
    :returns: :class:`LimitOrderPlan`.
    """
    mid = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else 0.0

    if mid == 0:
        return LimitOrderPlan(
            price=0.0, quantity=quantity, side=side,
            placement_strategy="none", distance_from_mid=0.0,
            expected_fill_time_s=0.0, cancel_after_s=300.0,
            reason="no valid mid price",
        )

    spread = (best_ask - best_bid) / mid if mid > 0 else 0.0

    # Estimate volatility
    if len(candles) >= 5:
        closes = [c.close for c in candles[-10:]]
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1] > 0]
        vol = pstdev(returns) if len(returns) > 1 else 0.01
    else:
        vol = 0.01

    # Offset from mid: more aggressive = closer to market
    offset = spread * (1 - aggressiveness) + vol * (1 - aggressiveness)

    if side == "buy":
        price = mid - mid * offset
        strategy = "bid_improvement" if aggressiveness > 0.7 else "passive_bid"
    else:
        price = mid + mid * offset
        strategy = "ask_improvement" if aggressiveness > 0.7 else "passive_ask"

    distance = abs(price - mid) / mid if mid > 0 else 0.0

    # Expected fill time: more aggressive = faster fill
    fill_time = 60 * (1 - aggressiveness) + 10
    cancel_after = fill_time * 5

    return LimitOrderPlan(
        price=round(price, 8),
        quantity=quantity,
        side=side,
        placement_strategy=strategy,
        distance_from_mid=round(distance, 6),
        expected_fill_time_s=round(fill_time, 2),
        cancel_after_s=round(cancel_after, 2),
        reason=f"limit: {side} {quantity}@{price:.2f}, strategy={strategy}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Market Order Execution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketOrderPlan:
    """Market order execution plan.

    ``expected_slippage`` is the estimated price impact.
    ``levels_consumed`` is how many orderbook levels will be taken.
    """

    side: str
    quantity: float
    expected_price: float
    expected_slippage: float
    levels_consumed: int
    total_cost: float
    should_split: bool
    reason: str


def plan_market_order(
    side: str,
    quantity: float,
    orderbook_bids: Sequence[Tuple[float, float]],
    orderbook_asks: Sequence[Tuple[float, float]],
    max_slippage_pct: float = 0.5,
) -> MarketOrderPlan:
    """Plan a market order and estimate execution costs.

    Walks the orderbook to calculate expected fill price and slippage.

    :param side: ``"buy"`` or ``"sell"``.
    :param quantity: Order quantity.
    :param orderbook_bids: Bids as (price, qty) tuples, best first.
    :param orderbook_asks: Asks as (price, qty) tuples, best first.
    :param max_slippage_pct: Max acceptable slippage percentage.
    :returns: :class:`MarketOrderPlan`.
    """
    levels = orderbook_asks if side == "buy" else orderbook_bids

    if not levels:
        return MarketOrderPlan(
            side=side, quantity=quantity, expected_price=0.0,
            expected_slippage=0.0, levels_consumed=0,
            total_cost=0.0, should_split=False,
            reason="no orderbook levels",
        )

    remaining = quantity
    total_value = 0.0
    levels_used = 0
    best_price = levels[0][0]

    for price, qty in levels:
        if remaining <= 0:
            break
        fill = min(remaining, qty)
        total_value += fill * price
        remaining -= fill
        levels_used += 1

    filled_qty = quantity - max(0, remaining)
    if filled_qty > 0:
        avg_price = total_value / filled_qty
        slippage = abs(avg_price - best_price) / best_price if best_price > 0 else 0.0
    else:
        avg_price = best_price
        slippage = 0.0

    should_split = slippage > max_slippage_pct / 100 or levels_used > 5

    return MarketOrderPlan(
        side=side,
        quantity=quantity,
        expected_price=round(avg_price, 8),
        expected_slippage=round(slippage * 100, 4),
        levels_consumed=levels_used,
        total_cost=round(total_value, 8),
        should_split=should_split,
        reason=f"market: {side} {quantity}, avg={avg_price:.4f}, slip={slippage:.4%}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6. TWAP Execution Algorithm
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TWAPPlan:
    """Time-weighted average price execution plan.

    ``slices`` contains the individual order slices with target times.
    ``total_duration_s`` is the full execution window.
    """

    total_quantity: float
    num_slices: int
    slice_quantity: float
    interval_seconds: float
    total_duration_s: float
    slices: List[Dict[str, Any]]
    randomize_size: bool
    reason: str


def plan_twap_execution(
    quantity: float,
    duration_minutes: float = 60.0,
    num_slices: int = 10,
    randomize: bool = True,
    variance_pct: float = 0.2,
) -> TWAPPlan:
    """Generate a TWAP execution schedule.

    Divides a large order into equal time-weighted slices to minimize
    market impact.

    :param quantity: Total quantity to execute.
    :param duration_minutes: Total execution window in minutes.
    :param num_slices: Number of child orders.
    :param randomize: Add random variation to slice sizes.
    :param variance_pct: Max variation from uniform slice (0..1).
    :returns: :class:`TWAPPlan`.
    """
    if quantity <= 0 or num_slices <= 0:
        return TWAPPlan(
            total_quantity=quantity, num_slices=0, slice_quantity=0.0,
            interval_seconds=0.0, total_duration_s=0.0,
            slices=[], randomize_size=randomize,
            reason="invalid parameters",
        )

    duration_s = duration_minutes * 60
    interval = duration_s / num_slices
    base_qty = quantity / num_slices

    slices: List[Dict[str, Any]] = []
    remaining = quantity
    for i in range(num_slices):
        if randomize and i < num_slices - 1:
            # Deterministic "random" variation using hash
            h = hashlib.md5(f"twap:{i}".encode()).hexdigest()
            var = (int(h[:8], 16) / 0xFFFFFFFF - 0.5) * 2 * variance_pct
            slice_qty = base_qty * (1 + var)
            slice_qty = max(base_qty * 0.5, min(base_qty * 1.5, slice_qty))
        else:
            slice_qty = remaining if i == num_slices - 1 else base_qty

        slice_qty = min(slice_qty, remaining)
        slices.append({
            "slice_index": i,
            "target_time_s": round(i * interval, 2),
            "quantity": round(slice_qty, 8),
            "status": "pending",
        })
        remaining -= slice_qty

    # Adjust last slice for any remainder
    if remaining > 0 and slices:
        slices[-1]["quantity"] = round(slices[-1]["quantity"] + remaining, 8)

    return TWAPPlan(
        total_quantity=quantity,
        num_slices=num_slices,
        slice_quantity=round(base_qty, 8),
        interval_seconds=round(interval, 2),
        total_duration_s=round(duration_s, 2),
        slices=slices,
        randomize_size=randomize,
        reason=f"twap: {num_slices} slices over {duration_minutes:.0f}min",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7. VWAP Execution Algorithm
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class VWAPPlan:
    """Volume-weighted average price execution plan.

    ``slices`` are sized proportionally to historical volume profile.
    ``target_vwap`` is the benchmark VWAP to beat.
    """

    total_quantity: float
    target_vwap: float
    num_slices: int
    slices: List[Dict[str, Any]]
    volume_profile: List[float]
    reason: str


def plan_vwap_execution(
    quantity: float,
    candles: Sequence[Candle],
    num_slices: int = 10,
) -> VWAPPlan:
    """Generate a VWAP execution schedule based on historical volume.

    Sizes order slices proportionally to the volume observed in each
    time bucket.

    :param quantity: Total quantity to execute.
    :param candles: Historical candles for volume profile.
    :param num_slices: Number of child orders.
    :returns: :class:`VWAPPlan`.
    """
    if quantity <= 0 or num_slices <= 0 or len(candles) < 2:
        return VWAPPlan(
            total_quantity=quantity, target_vwap=0.0,
            num_slices=0, slices=[], volume_profile=[],
            reason="insufficient data",
        )

    # Calculate VWAP from candles
    total_vol_price = sum(c.close * c.volume for c in candles)
    total_vol = sum(c.volume for c in candles)
    target_vwap = total_vol_price / total_vol if total_vol > 0 else 0.0

    # Build volume profile: divide candles into num_slices buckets
    bucket_size = max(1, len(candles) // num_slices)
    profile: List[float] = []
    for i in range(num_slices):
        start = i * bucket_size
        end = min(start + bucket_size, len(candles))
        bucket_vol = sum(c.volume for c in candles[start:end])
        profile.append(bucket_vol)

    total_profile = sum(profile) or 1.0
    norm_profile = [v / total_profile for v in profile]

    slices: List[Dict[str, Any]] = []
    remaining = quantity
    for i, weight in enumerate(norm_profile):
        slice_qty = quantity * weight
        slice_qty = min(slice_qty, remaining)
        slices.append({
            "slice_index": i,
            "quantity": round(slice_qty, 8),
            "volume_weight": round(weight, 6),
            "status": "pending",
        })
        remaining -= slice_qty

    if remaining > 0 and slices:
        slices[-1]["quantity"] = round(slices[-1]["quantity"] + remaining, 8)

    return VWAPPlan(
        total_quantity=quantity,
        target_vwap=round(target_vwap, 8),
        num_slices=num_slices,
        slices=slices,
        volume_profile=[round(v, 6) for v in norm_profile],
        reason=f"vwap: target={target_vwap:.2f}, {num_slices} slices",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. Iceberg Order Execution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class IcebergPlan:
    """Iceberg order execution plan.

    ``visible_quantity`` is the displayed portion.
    ``hidden_quantity`` is the undisclosed remainder.
    ``child_orders`` lists the scheduled visible tranches.
    """

    total_quantity: float
    visible_quantity: float
    hidden_quantity: float
    num_tranches: int
    child_orders: List[Dict[str, Any]]
    show_ratio: float
    reason: str


def plan_iceberg_order(
    quantity: float,
    show_ratio: float = 0.1,
    min_tranche: float = 0.0,
    price: float = 0.0,
) -> IcebergPlan:
    """Plan an iceberg order that hides the true order size.

    :param quantity: Total order quantity.
    :param show_ratio: Fraction of order visible at any time (0..1).
    :param min_tranche: Minimum tranche size.
    :param price: Target execution price.
    :returns: :class:`IcebergPlan`.
    """
    if quantity <= 0 or show_ratio <= 0:
        return IcebergPlan(
            total_quantity=quantity, visible_quantity=0.0,
            hidden_quantity=quantity, num_tranches=0,
            child_orders=[], show_ratio=show_ratio,
            reason="invalid parameters",
        )

    show_ratio = _clamp(show_ratio, 0.01, 1.0)
    visible = quantity * show_ratio
    if min_tranche > 0:
        visible = max(visible, min_tranche)
    visible = min(visible, quantity)

    hidden = quantity - visible
    num_tranches = max(1, math.ceil(quantity / visible))

    children: List[Dict[str, Any]] = []
    remaining = quantity
    for i in range(num_tranches):
        tranche_qty = min(visible, remaining)
        children.append({
            "tranche_index": i,
            "quantity": round(tranche_qty, 8),
            "price": price,
            "status": "pending",
        })
        remaining -= tranche_qty

    return IcebergPlan(
        total_quantity=quantity,
        visible_quantity=round(visible, 8),
        hidden_quantity=round(hidden, 8),
        num_tranches=num_tranches,
        child_orders=children,
        show_ratio=show_ratio,
        reason=f"iceberg: show={show_ratio:.0%}, {num_tranches} tranches",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Adaptive Order Execution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AdaptiveExecution:
    """Adaptive execution plan that adjusts to market conditions.

    ``strategy`` is the chosen execution approach.
    ``aggression_level`` is 0..1 indicating how aggressively to execute.
    ``adjustments`` lists condition-based parameter changes.
    """

    strategy: str
    aggression_level: float
    price_limit: float
    time_limit_s: float
    adjustments: List[Dict[str, Any]]
    reason: str


def plan_adaptive_execution(
    side: str,
    quantity: float,
    candles: Sequence[Candle],
    best_bid: float,
    best_ask: float,
    fill_urgency: float = 0.5,
) -> AdaptiveExecution:
    """Plan an adaptive execution that adjusts to current market conditions.

    :param side: ``"buy"`` or ``"sell"``.
    :param quantity: Order quantity.
    :param candles: Recent candles for condition assessment.
    :param best_bid: Current best bid.
    :param best_ask: Current best ask.
    :param fill_urgency: 0..1 indicating how urgently to fill.
    :returns: :class:`AdaptiveExecution`.
    """
    mid = (best_bid + best_ask) / 2 if (best_bid > 0 and best_ask > 0) else 0.0
    spread_pct = (best_ask - best_bid) / mid * 100 if mid > 0 else 0.0

    adjustments: List[Dict[str, Any]] = []

    # Assess volatility
    if len(candles) >= 5:
        closes = [c.close for c in candles[-10:]]
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
                   for i in range(1, len(closes)) if closes[i - 1] > 0]
        vol = pstdev(returns) if len(returns) > 1 else 0.01
    else:
        vol = 0.01

    # High volatility → more aggressive
    if vol > 0.02:
        adjustments.append({"condition": "high_volatility", "action": "increase_aggression", "vol": round(vol, 6)})
        aggression = min(1.0, fill_urgency + 0.2)
    else:
        aggression = fill_urgency

    # Wide spread → use limit orders
    if spread_pct > 0.5:
        adjustments.append({"condition": "wide_spread", "action": "use_limit", "spread_pct": round(spread_pct, 4)})
        strategy = "limit_with_chase"
    elif fill_urgency > 0.8:
        strategy = "aggressive_market"
    elif fill_urgency < 0.3:
        strategy = "passive_limit"
    else:
        strategy = "balanced"

    # Price limit
    buffer = vol * 2 + spread_pct / 100
    if side == "buy":
        price_limit = mid * (1 + buffer) if mid > 0 else 0.0
    else:
        price_limit = mid * (1 - buffer) if mid > 0 else 0.0

    time_limit = 300 * (1 - fill_urgency) + 30

    return AdaptiveExecution(
        strategy=strategy,
        aggression_level=round(aggression, 4),
        price_limit=round(price_limit, 8),
        time_limit_s=round(time_limit, 2),
        adjustments=adjustments,
        reason=f"adaptive: {strategy}, aggr={aggression:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Slippage Protection
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SlippageProtection:
    """Slippage analysis and protection recommendations.

    ``estimated_slippage_pct`` is the expected price impact.
    ``max_acceptable_pct`` is the configured tolerance.
    ``protections`` lists active protection mechanisms.
    """

    estimated_slippage_pct: float
    max_acceptable_pct: float
    is_safe: bool
    protections: List[str]
    recommended_order_type: str
    price_guard: float
    reason: str


def analyze_slippage(
    side: str,
    quantity: float,
    orderbook_bids: Sequence[Tuple[float, float]],
    orderbook_asks: Sequence[Tuple[float, float]],
    max_slippage_pct: float = 0.5,
) -> SlippageProtection:
    """Analyse expected slippage and recommend protections.

    :param side: ``"buy"`` or ``"sell"``.
    :param quantity: Order quantity.
    :param orderbook_bids: Bids as (price, qty) tuples.
    :param orderbook_asks: Asks as (price, qty) tuples.
    :param max_slippage_pct: Maximum acceptable slippage percentage.
    :returns: :class:`SlippageProtection`.
    """
    levels = orderbook_asks if side == "buy" else orderbook_bids

    if not levels:
        return SlippageProtection(
            estimated_slippage_pct=0.0, max_acceptable_pct=max_slippage_pct,
            is_safe=False, protections=[], recommended_order_type="limit",
            price_guard=0.0, reason="no orderbook data",
        )

    best_price = levels[0][0]
    remaining = quantity
    total_value = 0.0

    for price, qty in levels:
        if remaining <= 0:
            break
        fill = min(remaining, qty)
        total_value += fill * price
        remaining -= fill

    filled = quantity - max(0, remaining)
    avg_price = total_value / filled if filled > 0 else best_price
    slippage_pct = abs(avg_price - best_price) / best_price * 100 if best_price > 0 else 0.0

    protections: List[str] = []
    if slippage_pct > max_slippage_pct * 0.5:
        protections.append("price_limit_guard")
    if slippage_pct > max_slippage_pct * 0.3:
        protections.append("order_splitting")
    if remaining > 0:
        protections.append("partial_fill_handling")

    is_safe = slippage_pct <= max_slippage_pct

    if slippage_pct > max_slippage_pct:
        rec_type = "limit"
    elif slippage_pct > max_slippage_pct * 0.5:
        rec_type = "limit_ioc"
    else:
        rec_type = "market"

    price_guard = best_price * (1 + max_slippage_pct / 100) if side == "buy" else best_price * (1 - max_slippage_pct / 100)

    return SlippageProtection(
        estimated_slippage_pct=round(slippage_pct, 4),
        max_acceptable_pct=max_slippage_pct,
        is_safe=is_safe,
        protections=protections,
        recommended_order_type=rec_type,
        price_guard=round(price_guard, 8),
        reason=f"slippage: {slippage_pct:.4f}% (max {max_slippage_pct}%)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Partial Fill Handling
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PartialFillPlan:
    """Strategy for handling partial fills.

    ``action`` is ``"wait"``, ``"cancel_replace"``, ``"market_remaining"``,
    or ``"accept"``.
    ``filled_pct`` is the percentage already filled.
    """

    action: str
    filled_quantity: float
    remaining_quantity: float
    filled_pct: float
    new_price: Optional[float]
    timeout_s: float
    reason: str


def handle_partial_fill(
    original_quantity: float,
    filled_quantity: float,
    original_price: float,
    current_price: float,
    elapsed_seconds: float,
    max_wait_seconds: float = 300.0,
    min_fill_pct: float = 0.9,
) -> PartialFillPlan:
    """Decide how to handle a partially filled order.

    :param original_quantity: Original order quantity.
    :param filled_quantity: Quantity already filled.
    :param original_price: Original limit price.
    :param current_price: Current market price.
    :param elapsed_seconds: Time since order was placed.
    :param max_wait_seconds: Maximum time to wait for full fill.
    :param min_fill_pct: Minimum fill percentage to accept.
    :returns: :class:`PartialFillPlan`.
    """
    remaining = original_quantity - filled_quantity
    fill_pct = filled_quantity / original_quantity if original_quantity > 0 else 0.0

    if fill_pct >= min_fill_pct:
        return PartialFillPlan(
            action="accept",
            filled_quantity=filled_quantity,
            remaining_quantity=remaining,
            filled_pct=round(fill_pct, 4),
            new_price=None,
            timeout_s=0.0,
            reason=f"fill: {fill_pct:.0%} >= {min_fill_pct:.0%}, accepting",
        )

    price_drift = abs(current_price - original_price) / original_price if original_price > 0 else 0.0
    time_pct = elapsed_seconds / max_wait_seconds if max_wait_seconds > 0 else 1.0

    if time_pct >= 1.0:
        if fill_pct > 0.5:
            action = "accept"
            new_price = None
        else:
            action = "market_remaining"
            new_price = current_price
    elif price_drift > 0.01:
        action = "cancel_replace"
        new_price = current_price
    elif time_pct < 0.5:
        action = "wait"
        new_price = None
    else:
        action = "cancel_replace"
        new_price = current_price

    timeout = max(0, max_wait_seconds - elapsed_seconds)

    return PartialFillPlan(
        action=action,
        filled_quantity=filled_quantity,
        remaining_quantity=remaining,
        filled_pct=round(fill_pct, 4),
        new_price=round(new_price, 8) if new_price is not None else None,
        timeout_s=round(timeout, 2),
        reason=f"partial: {fill_pct:.0%} filled, action={action}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Order Retry Mechanism
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RetryPlan:
    """Order retry strategy.

    ``should_retry`` indicates whether to attempt the order again.
    ``delay_seconds`` is the recommended wait before retrying.
    ``adjustments`` lists parameter changes for the retry.
    """

    should_retry: bool
    attempt_number: int
    max_attempts: int
    delay_seconds: float
    adjustments: Dict[str, Any]
    backoff_factor: float
    reason: str


def plan_order_retry(
    error_type: str,
    attempt_number: int = 1,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
    backoff_factor: float = 2.0,
    original_price: float = 0.0,
    current_price: float = 0.0,
) -> RetryPlan:
    """Determine retry strategy for a failed order.

    :param error_type: Type of failure (e.g. "timeout", "rejected",
        "insufficient_funds", "rate_limit").
    :param attempt_number: Current attempt number (1-based).
    :param max_attempts: Maximum retry attempts allowed.
    :param base_delay_s: Base delay between retries.
    :param backoff_factor: Multiplier for exponential backoff.
    :param original_price: The original order price.
    :param current_price: Current market price.
    :returns: :class:`RetryPlan`.
    """
    # Non-retryable errors
    non_retryable = {"insufficient_funds", "invalid_pair", "account_locked"}
    if error_type in non_retryable:
        return RetryPlan(
            should_retry=False, attempt_number=attempt_number,
            max_attempts=max_attempts, delay_seconds=0.0,
            adjustments={}, backoff_factor=backoff_factor,
            reason=f"retry: {error_type} is non-retryable",
        )

    if attempt_number >= max_attempts:
        return RetryPlan(
            should_retry=False, attempt_number=attempt_number,
            max_attempts=max_attempts, delay_seconds=0.0,
            adjustments={}, backoff_factor=backoff_factor,
            reason=f"retry: max attempts ({max_attempts}) reached",
        )

    delay = base_delay_s * (backoff_factor ** (attempt_number - 1))
    adjustments: Dict[str, Any] = {}

    if error_type == "rate_limit":
        delay = max(delay, 5.0)
        adjustments["add_jitter"] = True

    if error_type == "timeout":
        adjustments["increase_timeout"] = True
        delay = max(delay, 2.0)

    if error_type == "rejected" and original_price > 0 and current_price > 0:
        adjustments["update_price"] = round(current_price, 8)

    if error_type == "partial_fill":
        adjustments["reduce_quantity"] = True

    return RetryPlan(
        should_retry=True,
        attempt_number=attempt_number,
        max_attempts=max_attempts,
        delay_seconds=round(delay, 2),
        adjustments=adjustments,
        backoff_factor=backoff_factor,
        reason=f"retry: attempt {attempt_number}/{max_attempts}, delay={delay:.1f}s",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Execution Quality Monitoring
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionQuality:
    """Execution quality metrics.

    ``implementation_shortfall`` measures cost vs. decision price.
    ``slippage_bps`` is the slippage in basis points.
    ``fill_rate`` is the fraction of intended quantity filled.
    """

    implementation_shortfall: float
    slippage_bps: float
    fill_rate: float
    execution_speed_s: float
    market_impact_pct: float
    timing_cost_pct: float
    overall_score: float
    reason: str


def monitor_execution_quality(
    intended_price: float,
    executed_price: float,
    intended_quantity: float,
    filled_quantity: float,
    execution_time_s: float,
    pre_trade_mid: float = 0.0,
    post_trade_mid: float = 0.0,
) -> ExecutionQuality:
    """Evaluate the quality of an order execution.

    :param intended_price: Price at decision time.
    :param executed_price: Average execution price.
    :param intended_quantity: Intended order quantity.
    :param filled_quantity: Actually filled quantity.
    :param execution_time_s: Time to fill in seconds.
    :param pre_trade_mid: Mid price before execution.
    :param post_trade_mid: Mid price after execution.
    :returns: :class:`ExecutionQuality`.
    """
    if intended_price <= 0:
        return ExecutionQuality(
            implementation_shortfall=0.0, slippage_bps=0.0,
            fill_rate=0.0, execution_speed_s=execution_time_s,
            market_impact_pct=0.0, timing_cost_pct=0.0,
            overall_score=0.0, reason="invalid intended price",
        )

    # Implementation shortfall
    shortfall = (executed_price - intended_price) / intended_price
    shortfall_pct = shortfall * 100

    # Slippage in basis points
    slippage_bps = abs(shortfall) * 10000

    # Fill rate
    fill_rate = filled_quantity / intended_quantity if intended_quantity > 0 else 0.0

    # Market impact
    if pre_trade_mid > 0 and post_trade_mid > 0:
        impact = abs(post_trade_mid - pre_trade_mid) / pre_trade_mid * 100
    else:
        impact = abs(shortfall_pct)

    # Timing cost
    timing_cost = abs(shortfall_pct) * 0.5

    # Overall score (0-100, higher is better)
    score = 100.0
    score -= min(30, slippage_bps / 10)  # Penalize slippage
    score -= min(20, (1 - fill_rate) * 20)  # Penalize unfilled
    score -= min(20, execution_time_s / 60 * 10)  # Penalize slow execution
    score -= min(15, impact * 5)  # Penalize market impact
    score = max(0, score)

    return ExecutionQuality(
        implementation_shortfall=round(shortfall_pct, 4),
        slippage_bps=round(slippage_bps, 2),
        fill_rate=round(fill_rate, 4),
        execution_speed_s=round(execution_time_s, 2),
        market_impact_pct=round(impact, 4),
        timing_cost_pct=round(timing_cost, 4),
        overall_score=round(score, 2),
        reason=f"quality: score={score:.0f}, slip={slippage_bps:.1f}bps",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 14. Order Batching
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OrderBatch:
    """Batch of orders optimized for execution.

    ``batches`` groups orders that can be sent together.
    ``estimated_savings_pct`` is the cost reduction from batching.
    """

    total_orders: int
    num_batches: int
    batches: List[List[Dict[str, Any]]]
    estimated_savings_pct: float
    execution_order: List[int]
    reason: str


def batch_orders(
    orders: Sequence[Dict[str, Any]],
    max_batch_size: int = 5,
    batch_by: str = "side",
) -> OrderBatch:
    """Group orders into optimized batches for execution.

    :param orders: List of order dicts with ``"side"``, ``"pair"``,
        ``"quantity"``, ``"price"`` fields.
    :param max_batch_size: Maximum orders per batch.
    :param batch_by: Grouping key (``"side"``, ``"pair"``, or ``"none"``).
    :returns: :class:`OrderBatch`.
    """
    if not orders:
        return OrderBatch(
            total_orders=0, num_batches=0, batches=[],
            estimated_savings_pct=0.0, execution_order=[],
            reason="no orders to batch",
        )

    # Group orders
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for i, order in enumerate(orders):
        order_copy = dict(order)
        order_copy["_index"] = i
        if batch_by == "side":
            key = str(order.get("side", "unknown"))
        elif batch_by == "pair":
            key = str(order.get("pair", "unknown"))
        else:
            key = "all"
        groups.setdefault(key, []).append(order_copy)

    # Split into max_batch_size chunks
    batches: List[List[Dict[str, Any]]] = []
    exec_order: List[int] = []
    for group in groups.values():
        for i in range(0, len(group), max_batch_size):
            chunk = group[i:i + max_batch_size]
            batches.append(chunk)
            exec_order.extend(o["_index"] for o in chunk)

    # Estimated savings: batching reduces overhead per order
    savings = min(0.5, len(orders) * 0.02) if len(orders) > 1 else 0.0

    return OrderBatch(
        total_orders=len(orders),
        num_batches=len(batches),
        batches=batches,
        estimated_savings_pct=round(savings, 4),
        execution_order=exec_order,
        reason=f"batch: {len(orders)} orders → {len(batches)} batches",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 15. Latency Optimization
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LatencyOptimization:
    """Latency optimization analysis and recommendations.

    ``bottlenecks`` lists identified latency issues.
    ``recommendations`` provides actionable improvement suggestions.
    ``estimated_improvement_pct`` is the expected latency reduction.
    """

    current_latency_ms: float
    optimized_latency_ms: float
    estimated_improvement_pct: float
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    priority: str
    reason: str


def optimize_latency(
    latency_samples: Sequence[float],
    network_latency_ms: float = 50.0,
    processing_time_ms: float = 10.0,
    serialization_time_ms: float = 5.0,
) -> LatencyOptimization:
    """Analyse execution latency and recommend optimizations.

    :param latency_samples: Recent latency measurements in milliseconds.
    :param network_latency_ms: Average network round-trip time.
    :param processing_time_ms: Server-side processing time.
    :param serialization_time_ms: Request/response serialization time.
    :returns: :class:`LatencyOptimization`.
    """
    if not latency_samples:
        return LatencyOptimization(
            current_latency_ms=0.0, optimized_latency_ms=0.0,
            estimated_improvement_pct=0.0, bottlenecks=[],
            recommendations=[], priority="low",
            reason="no latency data",
        )

    avg_latency = mean(latency_samples)
    p95 = sorted(latency_samples)[int(len(latency_samples) * 0.95)] if len(latency_samples) >= 20 else max(latency_samples)

    bottlenecks: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    savings = 0.0

    # Network bottleneck
    if network_latency_ms > 100:
        bottlenecks.append({"type": "network", "latency_ms": network_latency_ms, "severity": "high"})
        recommendations.append("use_connection_pooling")
        savings += network_latency_ms * 0.3

    if network_latency_ms > 50:
        recommendations.append("enable_keep_alive")
        savings += network_latency_ms * 0.1

    # Processing bottleneck
    if processing_time_ms > 20:
        bottlenecks.append({"type": "processing", "latency_ms": processing_time_ms, "severity": "medium"})
        recommendations.append("optimize_serialization")
        savings += processing_time_ms * 0.4

    # Serialization bottleneck
    if serialization_time_ms > 10:
        bottlenecks.append({"type": "serialization", "latency_ms": serialization_time_ms, "severity": "low"})
        recommendations.append("minimize_payload")
        savings += serialization_time_ms * 0.3

    # Jitter detection
    if len(latency_samples) > 5:
        jitter = pstdev(latency_samples)
        if jitter > avg_latency * 0.3:
            bottlenecks.append({"type": "jitter", "std_ms": round(jitter, 2), "severity": "medium"})
            recommendations.append("implement_retry_with_timeout")
            savings += jitter * 0.5

    # Pre-computation
    recommendations.append("pre_compute_signatures")
    savings += 5.0

    optimized = max(10, avg_latency - savings)
    improvement = (avg_latency - optimized) / avg_latency * 100 if avg_latency > 0 else 0.0

    if improvement > 30:
        priority = "high"
    elif improvement > 15:
        priority = "medium"
    else:
        priority = "low"

    return LatencyOptimization(
        current_latency_ms=round(avg_latency, 2),
        optimized_latency_ms=round(optimized, 2),
        estimated_improvement_pct=round(improvement, 2),
        bottlenecks=bottlenecks,
        recommendations=recommendations,
        priority=priority,
        reason=f"latency: {avg_latency:.0f}ms → {optimized:.0f}ms ({improvement:.0f}% improvement)",
    )
