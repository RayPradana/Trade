"""Market scanning module.

Provides 10 scanning categories for the Indodax trading bot:
 1. Multi-market scanning
 2. Liquidity filtering
 3. Volume filtering
 4. Volatility filtering
 5. Spread filtering
 6. Momentum scanning
 7. Breakout scanning
 8. Arbitrage opportunity scanning
 9. Trend scanning
10. Custom signal scanning

Each algorithm is implemented as a pure function operating on standard
market data (prices, depths, volumes) and returns typed dataclasses.
All implementations use only the Python standard library.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .analysis import Candle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float:
    """Convert *value* to float, returning ``0.0`` on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.debug("Failed to parse float from value=%s", value)
        return 0.0


def _returns(prices: Sequence[float]) -> List[float]:
    """Compute percentage returns from a price series."""
    if len(prices) < 2:
        return []
    result: List[float] = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        if prev == 0:
            continue
        result.append((prices[i] - prev) / prev)
    return result


def _simple_ma(prices: Sequence[float], period: int) -> float:
    """Return the simple moving average over the last *period* values."""
    if not prices or period <= 0:
        return 0.0
    window = prices[-period:]
    return mean(window) if window else 0.0


# ---------------------------------------------------------------------------
# 1. Multi-market scanning
# ---------------------------------------------------------------------------


@dataclass
class MultiMarketScanResult:
    market: str
    score: float
    signal: str
    metrics: Dict[str, float] = field(default_factory=dict)


def scan_multiple_markets(
    market_data: Dict[str, List[float]],
    min_score: float = 0.5,
) -> List[MultiMarketScanResult]:
    """Scan multiple markets and return scored opportunities."""
    results: List[MultiMarketScanResult] = []
    for market, prices in market_data.items():
        if len(prices) < 2:
            continue

        first = prices[0]
        last = prices[-1]
        trend = (last - first) / first if first != 0 else 0.0

        rets = _returns(prices)
        vol = pstdev(rets) if len(rets) >= 2 else 0.0

        last_5 = rets[-5:] if len(rets) >= 5 else rets
        momentum = mean(last_5) if last_5 else 0.0

        score = trend + momentum - vol

        if score > 0.5:
            signal = "buy"
        elif score < -0.5:
            signal = "sell"
        else:
            signal = "neutral"

        if abs(score) < min_score:
            continue

        results.append(
            MultiMarketScanResult(
                market=market,
                score=score,
                signal=signal,
                metrics={
                    "trend": trend,
                    "volatility": vol,
                    "momentum": momentum,
                },
            )
        )
    return results


# ---------------------------------------------------------------------------
# 2. Liquidity filtering
# ---------------------------------------------------------------------------


@dataclass
class LiquidityFilterResult:
    market: str
    bid_volume: float
    ask_volume: float
    total_liquidity: float
    spread_pct: float
    passed: bool


def filter_by_liquidity(
    markets: Dict[str, Dict],
    min_liquidity: float = 1000.0,
    max_spread_pct: float = 2.0,
) -> List[LiquidityFilterResult]:
    """Filter markets by depth-based liquidity and spread."""
    results: List[LiquidityFilterResult] = []
    for market, depth in markets.items():
        buys = depth.get("buy") or []
        sells = depth.get("sell") or []

        bid_volume = 0.0
        for level in buys:
            try:
                bid_volume += _safe_float(level[1])
            except (IndexError, TypeError):
                continue

        ask_volume = 0.0
        for level in sells:
            try:
                ask_volume += _safe_float(level[1])
            except (IndexError, TypeError):
                continue

        total_liquidity = bid_volume + ask_volume

        best_bid = _safe_float(buys[0][0]) if buys else 0.0
        best_ask = _safe_float(sells[0][0]) if sells else 0.0

        spread_pct = (
            (best_ask - best_bid) / best_bid * 100.0
            if best_bid > 0
            else 0.0
        )

        passed = total_liquidity >= min_liquidity and spread_pct <= max_spread_pct

        results.append(
            LiquidityFilterResult(
                market=market,
                bid_volume=bid_volume,
                ask_volume=ask_volume,
                total_liquidity=total_liquidity,
                spread_pct=spread_pct,
                passed=passed,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 3. Volume filtering
# ---------------------------------------------------------------------------


@dataclass
class VolumeFilterResult:
    market: str
    avg_volume: float
    current_volume: float
    volume_ratio: float
    passed: bool


def filter_by_volume(
    market_volumes: Dict[str, List[float]],
    min_avg_volume: float = 100.0,
    min_volume_ratio: float = 0.5,
) -> List[VolumeFilterResult]:
    """Filter markets by average volume and current-to-average ratio."""
    results: List[VolumeFilterResult] = []
    for market, volumes in market_volumes.items():
        if not volumes:
            continue

        avg_vol = mean(volumes)
        current_vol = volumes[-1]
        ratio = current_vol / avg_vol if avg_vol > 0 else 0.0

        passed = avg_vol >= min_avg_volume and ratio >= min_volume_ratio

        results.append(
            VolumeFilterResult(
                market=market,
                avg_volume=avg_vol,
                current_volume=current_vol,
                volume_ratio=ratio,
                passed=passed,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 4. Volatility filtering
# ---------------------------------------------------------------------------


@dataclass
class VolatilityFilterResult:
    market: str
    volatility: float
    annualized_vol: float
    passed: bool


def filter_by_volatility(
    market_prices: Dict[str, List[float]],
    min_vol: float = 0.001,
    max_vol: float = 0.1,
) -> List[VolatilityFilterResult]:
    """Filter markets whose volatility falls within a target range."""
    results: List[VolatilityFilterResult] = []
    for market, prices in market_prices.items():
        rets = _returns(prices)
        if len(rets) < 2:
            continue

        vol = pstdev(rets)
        annualized = vol * math.sqrt(365)

        passed = min_vol <= vol <= max_vol

        results.append(
            VolatilityFilterResult(
                market=market,
                volatility=vol,
                annualized_vol=annualized,
                passed=passed,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 5. Spread filtering
# ---------------------------------------------------------------------------


@dataclass
class SpreadFilterResult:
    market: str
    bid_price: float
    ask_price: float
    spread: float
    spread_pct: float
    passed: bool


def filter_by_spread(
    market_depths: Dict[str, Dict],
    max_spread_pct: float = 1.0,
) -> List[SpreadFilterResult]:
    """Filter markets by bid-ask spread percentage."""
    results: List[SpreadFilterResult] = []
    for market, depth in market_depths.items():
        buys = depth.get("buy") or []
        sells = depth.get("sell") or []

        bid_price = _safe_float(buys[0][0]) if buys else 0.0
        ask_price = _safe_float(sells[0][0]) if sells else 0.0

        spread = ask_price - bid_price
        spread_pct = spread / bid_price * 100.0 if bid_price > 0 else 0.0

        passed = spread_pct <= max_spread_pct

        results.append(
            SpreadFilterResult(
                market=market,
                bid_price=bid_price,
                ask_price=ask_price,
                spread=spread,
                spread_pct=spread_pct,
                passed=passed,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 6. Momentum scanning
# ---------------------------------------------------------------------------


@dataclass
class MomentumScanResult:
    market: str
    momentum: float
    roc: float
    signal: str
    strength: float


def scan_momentum(
    market_prices: Dict[str, List[float]],
    lookback: int = 10,
) -> List[MomentumScanResult]:
    """Scan markets for rate-of-change momentum signals."""
    results: List[MomentumScanResult] = []
    for market, prices in market_prices.items():
        if len(prices) < lookback + 1:
            continue

        current = prices[-1]
        old = prices[-lookback - 1]
        momentum = current - old
        roc = momentum / old if old != 0 else 0.0
        strength = abs(roc)

        if roc > 0.01:
            signal = "bullish"
        elif roc < -0.01:
            signal = "bearish"
        else:
            signal = "neutral"

        results.append(
            MomentumScanResult(
                market=market,
                momentum=momentum,
                roc=roc,
                signal=signal,
                strength=strength,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 7. Breakout scanning
# ---------------------------------------------------------------------------


@dataclass
class BreakoutScanResult:
    market: str
    breakout_type: str
    current_price: float
    level: float
    strength: float


def scan_breakouts(
    market_prices: Dict[str, List[float]],
    lookback: int = 20,
) -> List[BreakoutScanResult]:
    """Scan markets for price breakouts above resistance or below support."""
    results: List[BreakoutScanResult] = []
    for market, prices in market_prices.items():
        if len(prices) < lookback + 1:
            continue

        current = prices[-1]
        window = prices[-(lookback + 1) : -1]
        high = max(window)
        low = min(window)

        if current > high:
            breakout_type = "resistance_break"
            level = high
        elif current < low:
            breakout_type = "support_break"
            level = low
        else:
            breakout_type = "none"
            level = high  # reference for strength

        strength = abs(current - level) / level if level > 0 else 0.0

        results.append(
            BreakoutScanResult(
                market=market,
                breakout_type=breakout_type,
                current_price=current,
                level=level,
                strength=strength,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 8. Arbitrage opportunity scanning
# ---------------------------------------------------------------------------


@dataclass
class ArbitrageScanResult:
    market_a: str
    market_b: str
    price_a: float
    price_b: float
    spread_pct: float
    potential_profit: float
    actionable: bool


def scan_arbitrage(
    market_prices: Dict[str, float],
    fee_pct: float = 0.3,
) -> List[ArbitrageScanResult]:
    """Detect cross-market arbitrage opportunities."""
    results: List[ArbitrageScanResult] = []
    markets = list(market_prices.items())
    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            name_a, price_a = markets[i]
            name_b, price_b = markets[j]

            min_price = min(price_a, price_b)
            if min_price <= 0:
                continue

            spread_pct = abs(price_a - price_b) / min_price * 100.0
            potential_profit = spread_pct - 2 * fee_pct
            actionable = potential_profit > 0

            results.append(
                ArbitrageScanResult(
                    market_a=name_a,
                    market_b=name_b,
                    price_a=price_a,
                    price_b=price_b,
                    spread_pct=spread_pct,
                    potential_profit=potential_profit,
                    actionable=actionable,
                )
            )
    return results


# ---------------------------------------------------------------------------
# 9. Trend scanning
# ---------------------------------------------------------------------------


@dataclass
class TrendScanResult:
    market: str
    trend: str
    fast_ma: float
    slow_ma: float
    strength: float
    direction_score: float


def scan_trends(
    market_prices: Dict[str, List[float]],
    fast_period: int = 5,
    slow_period: int = 20,
) -> List[TrendScanResult]:
    """Detect market trends via fast/slow moving-average crossover."""
    results: List[TrendScanResult] = []
    for market, prices in market_prices.items():
        if len(prices) < slow_period:
            continue

        fast = _simple_ma(prices, fast_period)
        slow = _simple_ma(prices, slow_period)

        strength = abs(fast - slow) / slow if slow > 0 else 0.0
        direction_score = (fast - slow) / slow if slow > 0 else 0.0

        if slow > 0 and strength < 0.001:
            trend = "sideways"
        elif fast > slow:
            trend = "uptrend"
        else:
            trend = "downtrend"

        results.append(
            TrendScanResult(
                market=market,
                trend=trend,
                fast_ma=fast,
                slow_ma=slow,
                strength=strength,
                direction_score=direction_score,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 10. Custom signal scanning
# ---------------------------------------------------------------------------


@dataclass
class CustomSignalResult:
    market: str
    signal_name: str
    value: float
    triggered: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


def scan_custom_signals(
    market_data: Dict[str, List[float]],
    signals: List[Dict[str, Any]],
) -> List[CustomSignalResult]:
    """Apply user-defined signal functions to each market.

    Each entry in *signals* must contain:
    - ``"name"``  – human-readable signal name (str)
    - ``"condition"`` – ``Callable[[List[float]], Tuple[bool, float]]``
      that receives a price list and returns *(triggered, value)*.
    """
    results: List[CustomSignalResult] = []
    for market, prices in market_data.items():
        if not prices:
            continue
        for sig in signals:
            name = sig.get("name", "unnamed")
            condition: Optional[Callable[[List[float]], Tuple[bool, float]]] = sig.get(
                "condition"
            )
            if condition is None:
                continue
            try:
                triggered, value = condition(prices)
            except Exception:  # noqa: BLE001
                logger.debug(
                    "Custom signal '%s' raised for market=%s", name, market
                )
                continue

            results.append(
                CustomSignalResult(
                    market=market,
                    signal_name=name,
                    value=value,
                    triggered=triggered,
                )
            )
    return results
