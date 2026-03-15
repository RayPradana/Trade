"""Risk management & portfolio protection module.

Provides 17 risk-management categories for the Indodax trading bot:

 1. Position sizing algorithms
 2. Portfolio risk allocation
 3. Max position limit
 4. Daily loss limit
 5. Maximum drawdown protection
 6. Stop-loss (fixed)
 7. Stop-loss (trailing)
 8. Take-profit
 9. Dynamic risk adjustment
10. Volatility-based position sizing
11. Risk parity portfolio
12. Exposure limit per asset
13. Exposure limit per sector
14. Correlation risk monitoring
15. Capital allocation rules
16. Circuit breaker system
17. Strategy shutdown on anomaly

Each algorithm is implemented as a pure function operating on standard
market data (prices, balances, portfolio state) and returns typed
dataclasses.  All implementations use only the Python standard library.
"""

from __future__ import annotations

import logging
import math
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


def _returns(prices: Sequence[float]) -> List[float]:
    """Simple percentage returns from a price series."""
    if len(prices) < 2:
        return []
    return [
        (prices[i] - prices[i - 1]) / prices[i - 1]
        for i in range(1, len(prices))
        if prices[i - 1] != 0
    ]


def _volatility(prices: Sequence[float]) -> float:
    """Standard deviation of returns."""
    rets = _returns(prices)
    if len(rets) < 2:
        return 0.0
    return pstdev(rets)


def _max_drawdown(prices: Sequence[float]) -> float:
    """Maximum drawdown as a positive fraction (0..1)."""
    if len(prices) < 2:
        return 0.0
    peak = prices[0]
    dd = 0.0
    for p in prices:
        if p > peak:
            peak = p
        if peak > 0:
            dd = max(dd, (peak - p) / peak)
    return dd


# ═══════════════════════════════════════════════════════════════════════════
#  1. Position Sizing Algorithms
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PositionSize:
    """Result of position-sizing calculation.

    ``quantity`` is the recommended position size in base currency.
    ``method`` describes which algorithm was used.
    ``risk_amount`` is the capital at risk for this position.
    """

    quantity: float
    method: str
    risk_amount: float
    risk_pct: float
    account_balance: float
    reason: str


def calculate_position_size(
    account_balance: float,
    risk_per_trade_pct: float = 1.0,
    entry_price: float = 0.0,
    stop_loss_price: float = 0.0,
    method: str = "fixed_pct",
    volatility: float = 0.0,
) -> PositionSize:
    """Calculate the optimal position size for a trade.

    Supports ``"fixed_pct"`` (fixed percentage of capital),
    ``"kelly"`` (Kelly criterion approximation), and
    ``"volatility"`` (inverse-volatility sizing).

    :param account_balance: Total account equity.
    :param risk_per_trade_pct: Percentage of equity to risk (0-100).
    :param entry_price: Planned entry price.
    :param stop_loss_price: Stop-loss price.
    :param method: Sizing algorithm name.
    :param volatility: Annualised volatility (used by ``"volatility"`` method).
    :returns: :class:`PositionSize`.
    """
    if account_balance <= 0 or entry_price <= 0:
        return PositionSize(
            quantity=0.0, method=method, risk_amount=0.0,
            risk_pct=0.0, account_balance=account_balance,
            reason="invalid balance or entry price",
        )

    risk_frac = _clamp(risk_per_trade_pct, 0, 100) / 100
    risk_amount = account_balance * risk_frac

    if method == "kelly":
        # Simplified Kelly: f* = edge / odds.  We approximate with vol.
        edge = max(0.01, 0.5 - volatility)
        odds = max(0.01, volatility) if volatility > 0 else 1.0
        kelly_frac = _clamp(edge / odds, 0.0, 0.25)
        risk_amount = account_balance * kelly_frac
        qty = risk_amount / entry_price
    elif method == "volatility":
        if volatility > 0:
            target_risk = risk_amount
            qty = target_risk / (entry_price * volatility)
        else:
            qty = risk_amount / entry_price
    else:  # fixed_pct
        if stop_loss_price > 0 and stop_loss_price != entry_price:
            risk_per_unit = abs(entry_price - stop_loss_price)
            qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
        else:
            qty = risk_amount / entry_price

    qty = max(0.0, qty)
    actual_risk_pct = (risk_amount / account_balance * 100) if account_balance > 0 else 0.0

    return PositionSize(
        quantity=round(qty, 8),
        method=method,
        risk_amount=round(risk_amount, 2),
        risk_pct=round(actual_risk_pct, 4),
        account_balance=account_balance,
        reason=f"position_size: {method}, qty={qty:.8f}, risk={risk_amount:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Portfolio Risk Allocation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PortfolioAllocation:
    """Portfolio-level risk allocation result.

    ``allocations`` maps asset → weight (0..1).
    ``total_risk_pct`` is the aggregate portfolio risk.
    """

    allocations: Dict[str, float]
    total_risk_pct: float
    max_allocation: float
    diversification_score: float
    reason: str


def allocate_portfolio_risk(
    assets: Sequence[str],
    returns_map: Dict[str, Sequence[float]],
    max_single_asset_pct: float = 25.0,
    total_risk_budget_pct: float = 100.0,
) -> PortfolioAllocation:
    """Allocate risk budget across a portfolio of assets.

    Uses inverse-volatility weighting so lower-vol assets receive
    a larger allocation.

    :param assets: Asset identifiers.
    :param returns_map: Historical returns per asset.
    :param max_single_asset_pct: Cap per asset (0-100).
    :param total_risk_budget_pct: Total risk budget (0-100).
    :returns: :class:`PortfolioAllocation`.
    """
    if not assets:
        return PortfolioAllocation(
            allocations={}, total_risk_pct=0.0,
            max_allocation=0.0, diversification_score=0.0,
            reason="no assets provided",
        )

    inv_vols: Dict[str, float] = {}
    for asset in assets:
        rets = returns_map.get(asset, [])
        vol = pstdev(rets) if len(rets) >= 2 else 0.01
        inv_vols[asset] = 1.0 / max(vol, 1e-9)

    total_inv = sum(inv_vols.values()) or 1.0
    cap = max_single_asset_pct / 100

    raw: Dict[str, float] = {}
    for asset in assets:
        w = inv_vols[asset] / total_inv
        raw[asset] = min(w, cap)

    # Re-normalise after capping
    total_raw = sum(raw.values()) or 1.0
    allocations = {a: round(w / total_raw, 6) for a, w in raw.items()}

    max_alloc = max(allocations.values()) if allocations else 0.0
    # HHI-based diversification (0 = concentrated, 1 = diversified)
    hhi = sum(w ** 2 for w in allocations.values())
    div_score = 1.0 - hhi if len(allocations) > 1 else 0.0

    total_risk = total_risk_budget_pct * max_alloc

    return PortfolioAllocation(
        allocations=allocations,
        total_risk_pct=round(total_risk, 4),
        max_allocation=round(max_alloc, 6),
        diversification_score=round(div_score, 4),
        reason=f"portfolio: {len(assets)} assets, div={div_score:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Max Position Limit
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PositionLimitCheck:
    """Result of max-position-limit check.

    ``allowed`` is True when the new position respects the limit.
    ``current_exposure`` is the sum of existing positions.
    """

    allowed: bool
    current_exposure: float
    max_exposure: float
    remaining_capacity: float
    utilization_pct: float
    reason: str


def check_position_limit(
    current_positions: Dict[str, float],
    new_position_value: float,
    max_total_exposure: float,
    max_per_asset: float = 0.0,
    asset: str = "",
) -> PositionLimitCheck:
    """Check whether a new position respects exposure limits.

    :param current_positions: Map of asset → current position value.
    :param new_position_value: Value of the proposed new position.
    :param max_total_exposure: Maximum allowed aggregate exposure.
    :param max_per_asset: Optional per-asset limit.
    :param asset: Asset for per-asset check.
    :returns: :class:`PositionLimitCheck`.
    """
    current = sum(abs(v) for v in current_positions.values())
    proposed = current + abs(new_position_value)
    remaining = max(0, max_total_exposure - current)
    utilization = (current / max_total_exposure * 100) if max_total_exposure > 0 else 0.0

    allowed = proposed <= max_total_exposure

    # Per-asset check
    if allowed and max_per_asset > 0 and asset:
        asset_current = abs(current_positions.get(asset, 0.0))
        if asset_current + abs(new_position_value) > max_per_asset:
            allowed = False

    return PositionLimitCheck(
        allowed=allowed,
        current_exposure=round(current, 2),
        max_exposure=max_total_exposure,
        remaining_capacity=round(remaining, 2),
        utilization_pct=round(utilization, 4),
        reason=f"limit: {'allowed' if allowed else 'blocked'}, "
               f"util={utilization:.1f}%",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Daily Loss Limit
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DailyLossCheck:
    """Daily loss-limit evaluation.

    ``can_trade`` is False when the daily loss limit is breached.
    ``remaining_budget`` is how much more can be lost today.
    """

    can_trade: bool
    daily_pnl: float
    daily_loss_limit: float
    remaining_budget: float
    utilization_pct: float
    reason: str


def check_daily_loss_limit(
    daily_pnl: float,
    daily_loss_limit: float,
    pending_risk: float = 0.0,
) -> DailyLossCheck:
    """Check whether the daily loss limit has been reached.

    :param daily_pnl: Realised + unrealised P&L for the day.
    :param daily_loss_limit: Maximum acceptable daily loss (positive number).
    :param pending_risk: Additional risk from open orders.
    :returns: :class:`DailyLossCheck`.
    """
    limit = abs(daily_loss_limit)
    loss = -daily_pnl if daily_pnl < 0 else 0.0
    total_risk = loss + abs(pending_risk)
    remaining = max(0, limit - total_risk)
    # When limit == 0 the feature is disabled → always allow trading.
    can_trade = limit <= 0 or total_risk < limit
    utilization = (total_risk / limit * 100) if limit > 0 else 0.0

    return DailyLossCheck(
        can_trade=can_trade,
        daily_pnl=round(daily_pnl, 2),
        daily_loss_limit=round(limit, 2),
        remaining_budget=round(remaining, 2),
        utilization_pct=round(utilization, 4),
        reason=f"daily_loss: {'ok' if can_trade else 'limit_hit'}, "
               f"pnl={daily_pnl:.2f}, limit={limit:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Maximum Drawdown Protection
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DrawdownCheck:
    """Drawdown protection result.

    ``should_reduce`` is True when drawdown exceeds the warning threshold.
    ``should_stop`` is True when drawdown exceeds the hard stop threshold.
    """

    current_drawdown_pct: float
    max_drawdown_pct: float
    peak_equity: float
    current_equity: float
    should_reduce: bool
    should_stop: bool
    reason: str


def check_max_drawdown(
    equity_history: Sequence[float],
    max_drawdown_pct: float = 20.0,
    warning_pct: float = 15.0,
) -> DrawdownCheck:
    """Evaluate whether the portfolio drawdown is within limits.

    :param equity_history: Historical equity values (chronological).
    :param max_drawdown_pct: Hard-stop drawdown percentage.
    :param warning_pct: Reduced-risk threshold percentage.
    :returns: :class:`DrawdownCheck`.
    """
    if not equity_history:
        return DrawdownCheck(
            current_drawdown_pct=0.0, max_drawdown_pct=max_drawdown_pct,
            peak_equity=0.0, current_equity=0.0,
            should_reduce=False, should_stop=False,
            reason="no equity data",
        )

    peak = max(equity_history)
    current = equity_history[-1]
    dd = ((peak - current) / peak * 100) if peak > 0 else 0.0

    return DrawdownCheck(
        current_drawdown_pct=round(dd, 4),
        max_drawdown_pct=max_drawdown_pct,
        peak_equity=round(peak, 2),
        current_equity=round(current, 2),
        should_reduce=dd >= warning_pct,
        should_stop=dd >= max_drawdown_pct,
        reason=f"drawdown: {dd:.2f}% (warn={warning_pct}%, stop={max_drawdown_pct}%)",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6. Stop-Loss (Fixed)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class FixedStopLoss:
    """Fixed stop-loss result.

    ``triggered`` is True when the current price has breached the stop.
    ``stop_price`` is the fixed threshold.
    """

    triggered: bool
    stop_price: float
    current_price: float
    distance_pct: float
    loss_amount: float
    reason: str


def check_fixed_stop_loss(
    entry_price: float,
    current_price: float,
    stop_loss_pct: float = 5.0,
    position_size: float = 1.0,
    side: str = "long",
) -> FixedStopLoss:
    """Evaluate a fixed percentage stop-loss.

    :param entry_price: Position entry price.
    :param current_price: Current market price.
    :param stop_loss_pct: Stop-loss distance as percentage.
    :param position_size: Size of position in base currency.
    :param side: ``"long"`` or ``"short"``.
    :returns: :class:`FixedStopLoss`.
    """
    if entry_price <= 0:
        return FixedStopLoss(
            triggered=False, stop_price=0.0, current_price=current_price,
            distance_pct=0.0, loss_amount=0.0, reason="invalid entry price",
        )

    frac = stop_loss_pct / 100
    if side == "short":
        stop_price = entry_price * (1 + frac)
        triggered = current_price >= stop_price
        distance = (stop_price - current_price) / entry_price * 100
    else:
        stop_price = entry_price * (1 - frac)
        triggered = current_price <= stop_price
        distance = (current_price - stop_price) / entry_price * 100

    loss = abs(entry_price - current_price) * position_size if triggered else 0.0

    return FixedStopLoss(
        triggered=triggered,
        stop_price=round(stop_price, 8),
        current_price=current_price,
        distance_pct=round(distance, 4),
        loss_amount=round(loss, 2),
        reason=f"fixed_sl: {'TRIGGERED' if triggered else 'ok'}, "
               f"stop={stop_price:.2f}, dist={distance:.2f}%",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7. Stop-Loss (Trailing)
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TrailingStopLoss:
    """Trailing stop-loss result.

    ``stop_price`` is dynamically adjusted to trail the peak.
    ``triggered`` is True when the stop has been breached.
    """

    triggered: bool
    stop_price: float
    highest_price: float
    current_price: float
    distance_pct: float
    locked_profit_pct: float
    reason: str


def check_trailing_stop_loss(
    entry_price: float,
    current_price: float,
    highest_price: float,
    trail_pct: float = 3.0,
    side: str = "long",
) -> TrailingStopLoss:
    """Evaluate a trailing stop-loss that follows the price peak.

    :param entry_price: Position entry price.
    :param current_price: Current market price.
    :param highest_price: Highest (or lowest for shorts) observed price.
    :param trail_pct: Trailing distance as percentage.
    :param side: ``"long"`` or ``"short"``.
    :returns: :class:`TrailingStopLoss`.
    """
    if highest_price <= 0 or entry_price <= 0:
        return TrailingStopLoss(
            triggered=False, stop_price=0.0, highest_price=highest_price,
            current_price=current_price, distance_pct=0.0,
            locked_profit_pct=0.0, reason="invalid prices",
        )

    frac = trail_pct / 100
    if side == "short":
        lowest = highest_price  # for shorts highest_price = lowest seen
        stop_price = lowest * (1 + frac)
        triggered = current_price >= stop_price
        distance = (stop_price - current_price) / lowest * 100
        locked = (entry_price - lowest) / entry_price * 100 if entry_price > 0 else 0.0
    else:
        stop_price = highest_price * (1 - frac)
        triggered = current_price <= stop_price
        distance = (current_price - stop_price) / highest_price * 100
        locked = (highest_price - entry_price) / entry_price * 100 - trail_pct

    return TrailingStopLoss(
        triggered=triggered,
        stop_price=round(stop_price, 8),
        highest_price=round(highest_price, 8),
        current_price=current_price,
        distance_pct=round(distance, 4),
        locked_profit_pct=round(max(0, locked), 4),
        reason=f"trail_sl: {'TRIGGERED' if triggered else 'ok'}, "
               f"stop={stop_price:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. Take-Profit
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TakeProfit:
    """Take-profit evaluation.

    ``triggered`` is True when the profit target has been reached.
    ``profit_amount`` is the unrealised gain if triggered.
    """

    triggered: bool
    target_price: float
    current_price: float
    profit_pct: float
    profit_amount: float
    reason: str


def check_take_profit(
    entry_price: float,
    current_price: float,
    take_profit_pct: float = 5.0,
    position_size: float = 1.0,
    side: str = "long",
) -> TakeProfit:
    """Evaluate whether the take-profit target has been reached.

    :param entry_price: Position entry price.
    :param current_price: Current market price.
    :param take_profit_pct: Profit target percentage.
    :param position_size: Size of position.
    :param side: ``"long"`` or ``"short"``.
    :returns: :class:`TakeProfit`.
    """
    if entry_price <= 0:
        return TakeProfit(
            triggered=False, target_price=0.0, current_price=current_price,
            profit_pct=0.0, profit_amount=0.0,
            reason="invalid entry price",
        )

    frac = take_profit_pct / 100
    if side == "short":
        target = entry_price * (1 - frac)
        triggered = current_price <= target
        profit_pct_actual = (entry_price - current_price) / entry_price * 100
    else:
        target = entry_price * (1 + frac)
        triggered = current_price >= target
        profit_pct_actual = (current_price - entry_price) / entry_price * 100

    profit_amount = abs(current_price - entry_price) * position_size if triggered else 0.0

    return TakeProfit(
        triggered=triggered,
        target_price=round(target, 8),
        current_price=current_price,
        profit_pct=round(profit_pct_actual, 4),
        profit_amount=round(profit_amount, 2),
        reason=f"tp: {'TRIGGERED' if triggered else 'ok'}, "
               f"target={target:.2f}, pnl={profit_pct_actual:.2f}%",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Dynamic Risk Adjustment
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DynamicRiskAdjustment:
    """Dynamic risk-level adjustment recommendation.

    ``adjusted_risk_pct`` is the new per-trade risk percentage.
    ``scale_factor`` is the multiplier applied to the base risk.
    """

    adjusted_risk_pct: float
    base_risk_pct: float
    scale_factor: float
    conditions: List[str]
    reason: str


def adjust_risk_dynamically(
    base_risk_pct: float,
    recent_pnl: Sequence[float],
    current_drawdown_pct: float = 0.0,
    volatility: float = 0.0,
    win_rate: float = 0.5,
) -> DynamicRiskAdjustment:
    """Dynamically adjust per-trade risk based on conditions.

    Reduces risk during drawdowns and high volatility; increases
    during winning streaks with low volatility.

    :param base_risk_pct: Normal risk percentage.
    :param recent_pnl: Recent trade P&L values.
    :param current_drawdown_pct: Current drawdown percentage.
    :param volatility: Current realised volatility.
    :param win_rate: Recent win rate (0..1).
    :returns: :class:`DynamicRiskAdjustment`.
    """
    factor = 1.0
    conditions: List[str] = []

    # Drawdown penalty
    if current_drawdown_pct > 15:
        factor *= 0.5
        conditions.append("severe_drawdown")
    elif current_drawdown_pct > 10:
        factor *= 0.7
        conditions.append("moderate_drawdown")
    elif current_drawdown_pct > 5:
        factor *= 0.85
        conditions.append("mild_drawdown")

    # Volatility adjustment
    if volatility > 0.05:
        factor *= 0.7
        conditions.append("high_volatility")
    elif volatility > 0.03:
        factor *= 0.85
        conditions.append("elevated_volatility")

    # Win-rate bonus
    if win_rate > 0.65 and current_drawdown_pct < 5:
        factor *= 1.15
        conditions.append("strong_win_rate")

    # Losing streak penalty
    if len(recent_pnl) >= 3 and all(p < 0 for p in recent_pnl[-3:]):
        factor *= 0.6
        conditions.append("losing_streak")

    factor = _clamp(factor, 0.2, 1.5)
    adjusted = base_risk_pct * factor

    return DynamicRiskAdjustment(
        adjusted_risk_pct=round(adjusted, 4),
        base_risk_pct=base_risk_pct,
        scale_factor=round(factor, 4),
        conditions=conditions,
        reason=f"dyn_risk: {base_risk_pct:.1f}% → {adjusted:.2f}% (x{factor:.2f})",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Volatility-Based Position Sizing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class VolatilityPositionSize:
    """Position size derived from current volatility.

    ``quantity`` is the recommended size.
    ``atr_value`` is the Average True Range used.
    """

    quantity: float
    atr_value: float
    risk_per_unit: float
    volatility: float
    account_balance: float
    reason: str


def size_by_volatility(
    account_balance: float,
    candles: Sequence[Candle],
    risk_pct: float = 1.0,
    atr_multiplier: float = 2.0,
) -> VolatilityPositionSize:
    """Size a position using ATR-based volatility.

    :param account_balance: Account equity.
    :param candles: Recent OHLCV candles.
    :param risk_pct: Percentage of equity to risk (0-100).
    :param atr_multiplier: Multiplier for ATR to set stop distance.
    :returns: :class:`VolatilityPositionSize`.
    """
    if account_balance <= 0 or len(candles) < 2:
        return VolatilityPositionSize(
            quantity=0.0, atr_value=0.0, risk_per_unit=0.0,
            volatility=0.0, account_balance=account_balance,
            reason="insufficient data",
        )

    # Calculate ATR
    trs: List[float] = []
    for i in range(1, len(candles)):
        c = candles[i]
        prev_close = candles[i - 1].close
        tr = max(c.high - c.low, abs(c.high - prev_close), abs(c.low - prev_close))
        trs.append(tr)
    atr = mean(trs) if trs else 0.0

    risk_amount = account_balance * (risk_pct / 100)
    risk_per_unit = atr * atr_multiplier
    qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0

    closes = [c.close for c in candles]
    vol = _volatility(closes)

    return VolatilityPositionSize(
        quantity=round(qty, 8),
        atr_value=round(atr, 8),
        risk_per_unit=round(risk_per_unit, 8),
        volatility=round(vol, 6),
        account_balance=account_balance,
        reason=f"vol_size: atr={atr:.4f}, qty={qty:.8f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Risk Parity Portfolio
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RiskParityResult:
    """Risk-parity allocation where each asset contributes equal risk.

    ``weights`` maps asset → weight (0..1).
    ``risk_contributions`` maps asset → risk contribution fraction.
    """

    weights: Dict[str, float]
    risk_contributions: Dict[str, float]
    portfolio_volatility: float
    is_balanced: bool
    reason: str


def calculate_risk_parity(
    assets: Sequence[str],
    volatilities: Dict[str, float],
    target_volatility: float = 0.0,
) -> RiskParityResult:
    """Calculate risk-parity weights so each asset contributes equal risk.

    :param assets: Asset identifiers.
    :param volatilities: Per-asset annualised volatility.
    :param target_volatility: Optional target portfolio vol.
    :returns: :class:`RiskParityResult`.
    """
    if not assets:
        return RiskParityResult(
            weights={}, risk_contributions={},
            portfolio_volatility=0.0, is_balanced=False,
            reason="no assets",
        )

    inv: Dict[str, float] = {}
    for a in assets:
        v = volatilities.get(a, 0.01)
        inv[a] = 1.0 / max(v, 1e-9)

    total_inv = sum(inv.values()) or 1.0
    weights = {a: round(inv[a] / total_inv, 6) for a in assets}

    # Risk contribution = weight * vol
    contribs_raw = {a: weights[a] * volatilities.get(a, 0.01) for a in assets}
    total_contrib = sum(contribs_raw.values()) or 1.0
    risk_contribs = {a: round(c / total_contrib, 6) for a, c in contribs_raw.items()}

    port_vol = sum(weights[a] * volatilities.get(a, 0.01) for a in assets)

    # Balanced if max risk contrib deviation < 10%
    target_equal = 1.0 / len(assets) if assets else 0
    max_dev = max(abs(rc - target_equal) for rc in risk_contribs.values()) if risk_contribs else 0
    is_balanced = max_dev < 0.10

    return RiskParityResult(
        weights=weights,
        risk_contributions=risk_contribs,
        portfolio_volatility=round(port_vol, 6),
        is_balanced=is_balanced,
        reason=f"risk_parity: {len(assets)} assets, balanced={is_balanced}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Exposure Limit Per Asset
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AssetExposureCheck:
    """Per-asset exposure limit evaluation.

    ``within_limit`` is True when the asset exposure is acceptable.
    """

    asset: str
    current_exposure: float
    max_exposure: float
    within_limit: bool
    utilization_pct: float
    reason: str


def check_asset_exposure(
    asset: str,
    current_value: float,
    portfolio_value: float,
    max_asset_pct: float = 20.0,
) -> AssetExposureCheck:
    """Check whether an asset's exposure is within limits.

    :param asset: Asset identifier.
    :param current_value: Current position value for this asset.
    :param portfolio_value: Total portfolio value.
    :param max_asset_pct: Maximum allowed percentage for a single asset.
    :returns: :class:`AssetExposureCheck`.
    """
    if portfolio_value <= 0:
        return AssetExposureCheck(
            asset=asset, current_exposure=0.0, max_exposure=0.0,
            within_limit=True, utilization_pct=0.0,
            reason="zero portfolio value",
        )

    exposure_pct = abs(current_value) / portfolio_value * 100
    max_abs = portfolio_value * max_asset_pct / 100
    within = exposure_pct <= max_asset_pct

    return AssetExposureCheck(
        asset=asset,
        current_exposure=round(abs(current_value), 2),
        max_exposure=round(max_abs, 2),
        within_limit=within,
        utilization_pct=round(exposure_pct, 4),
        reason=f"asset_exp: {asset} {exposure_pct:.1f}% "
               f"(max {max_asset_pct}%)",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Exposure Limit Per Sector
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SectorExposureCheck:
    """Sector-level exposure analysis.

    ``sector_exposures`` maps sector → exposure percentage.
    ``breached_sectors`` lists sectors exceeding the limit.
    """

    sector_exposures: Dict[str, float]
    breached_sectors: List[str]
    max_sector_pct: float
    all_within_limit: bool
    reason: str


def check_sector_exposure(
    positions: Dict[str, float],
    asset_sectors: Dict[str, str],
    portfolio_value: float,
    max_sector_pct: float = 30.0,
) -> SectorExposureCheck:
    """Check whether sector-level exposures are within limits.

    :param positions: Map of asset → position value.
    :param asset_sectors: Map of asset → sector name.
    :param portfolio_value: Total portfolio value.
    :param max_sector_pct: Maximum allowed sector exposure (0-100).
    :returns: :class:`SectorExposureCheck`.
    """
    if portfolio_value <= 0:
        return SectorExposureCheck(
            sector_exposures={}, breached_sectors=[],
            max_sector_pct=max_sector_pct, all_within_limit=True,
            reason="zero portfolio",
        )

    sector_values: Dict[str, float] = {}
    for asset, value in positions.items():
        sector = asset_sectors.get(asset, "unknown")
        sector_values[sector] = sector_values.get(sector, 0.0) + abs(value)

    sector_pcts = {
        s: round(v / portfolio_value * 100, 4)
        for s, v in sector_values.items()
    }

    breached = [s for s, pct in sector_pcts.items() if pct > max_sector_pct]

    return SectorExposureCheck(
        sector_exposures=sector_pcts,
        breached_sectors=breached,
        max_sector_pct=max_sector_pct,
        all_within_limit=len(breached) == 0,
        reason=f"sector: {len(breached)} breached of {len(sector_pcts)} sectors",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 14. Correlation Risk Monitoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CorrelationRisk:
    """Correlation risk assessment.

    ``high_corr_pairs`` lists pairs whose correlation exceeds the threshold.
    ``avg_correlation`` is the mean pairwise correlation.
    """

    high_corr_pairs: List[Tuple[str, str, float]]
    avg_correlation: float
    max_correlation: float
    portfolio_corr_risk: str
    reason: str


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Simple Pearson correlation."""
    n = min(len(x), len(y))
    if n < 3:
        return 0.0
    mx = mean(x[:n])
    my = mean(y[:n])
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    dx = math.sqrt(sum((x[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((y[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def monitor_correlation_risk(
    assets: Sequence[str],
    returns_map: Dict[str, Sequence[float]],
    threshold: float = 0.7,
) -> CorrelationRisk:
    """Identify highly-correlated asset pairs in the portfolio.

    :param assets: Asset identifiers.
    :param returns_map: Historical returns per asset.
    :param threshold: Correlation threshold for alert (0..1).
    :returns: :class:`CorrelationRisk`.
    """
    if len(assets) < 2:
        return CorrelationRisk(
            high_corr_pairs=[], avg_correlation=0.0,
            max_correlation=0.0, portfolio_corr_risk="low",
            reason="fewer than 2 assets",
        )

    pairs: List[Tuple[str, str, float]] = []
    all_corrs: List[float] = []

    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            ra = list(returns_map.get(a, []))
            rb = list(returns_map.get(b, []))
            c = _pearson(ra, rb)
            all_corrs.append(abs(c))
            if abs(c) >= threshold:
                pairs.append((a, b, round(c, 4)))

    avg_c = mean(all_corrs) if all_corrs else 0.0
    max_c = max(all_corrs) if all_corrs else 0.0

    if max_c > 0.9:
        risk_level = "critical"
    elif max_c > 0.7:
        risk_level = "high"
    elif max_c > 0.5:
        risk_level = "moderate"
    else:
        risk_level = "low"

    return CorrelationRisk(
        high_corr_pairs=pairs,
        avg_correlation=round(avg_c, 4),
        max_correlation=round(max_c, 4),
        portfolio_corr_risk=risk_level,
        reason=f"corr: {len(pairs)} high pairs, risk={risk_level}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 15. Capital Allocation Rules
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CapitalAllocation:
    """Capital allocation across trading strategies.

    ``strategy_allocations`` maps strategy → allocated capital.
    ``reserve_amount`` is capital held in reserve.
    """

    strategy_allocations: Dict[str, float]
    reserve_amount: float
    reserve_pct: float
    total_deployed: float
    total_capital: float
    reason: str


def allocate_capital(
    total_capital: float,
    strategies: Sequence[str],
    strategy_weights: Dict[str, float] | None = None,
    reserve_pct: float = 10.0,
    min_allocation: float = 0.0,
) -> CapitalAllocation:
    """Allocate capital across trading strategies.

    :param total_capital: Total available capital.
    :param strategies: Strategy identifiers.
    :param strategy_weights: Optional custom weights (values summed & normalised).
    :param reserve_pct: Percentage kept as reserve (0-100).
    :param min_allocation: Minimum capital per strategy.
    :returns: :class:`CapitalAllocation`.
    """
    if total_capital <= 0 or not strategies:
        return CapitalAllocation(
            strategy_allocations={}, reserve_amount=0.0,
            reserve_pct=reserve_pct, total_deployed=0.0,
            total_capital=total_capital,
            reason="no capital or strategies",
        )

    reserve = total_capital * _clamp(reserve_pct, 0, 100) / 100
    deployable = total_capital - reserve

    if strategy_weights:
        total_w = sum(strategy_weights.get(s, 1.0) for s in strategies)
        if total_w == 0:
            total_w = 1.0
        raw = {s: deployable * strategy_weights.get(s, 1.0) / total_w for s in strategies}
    else:
        per = deployable / len(strategies)
        raw = {s: per for s in strategies}

    # Enforce minimum
    allocs: Dict[str, float] = {}
    for s in strategies:
        allocs[s] = round(max(raw[s], min_allocation), 2)

    total_deployed = sum(allocs.values())

    return CapitalAllocation(
        strategy_allocations=allocs,
        reserve_amount=round(reserve, 2),
        reserve_pct=reserve_pct,
        total_deployed=round(total_deployed, 2),
        total_capital=total_capital,
        reason=f"capital: {len(strategies)} strategies, "
               f"reserve={reserve:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 16. Circuit Breaker System
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CircuitBreakerStatus:
    """Circuit-breaker system status.

    ``is_tripped`` is True when trading should halt.
    ``triggers`` lists the conditions that activated the breaker.
    """

    is_tripped: bool
    triggers: List[str]
    severity: str
    cooldown_seconds: float
    can_resume: bool
    reason: str


def check_circuit_breaker(
    daily_loss_pct: float = 0.0,
    drawdown_pct: float = 0.0,
    consecutive_losses: int = 0,
    volatility_spike: bool = False,
    api_errors: int = 0,
    max_daily_loss_pct: float = 5.0,
    max_drawdown_pct: float = 20.0,
    max_consecutive_losses: int = 5,
    max_api_errors: int = 10,
) -> CircuitBreakerStatus:
    """Evaluate multiple circuit-breaker conditions.

    :param daily_loss_pct: Current daily loss as percentage.
    :param drawdown_pct: Current drawdown percentage.
    :param consecutive_losses: Number of consecutive losing trades.
    :param volatility_spike: Whether a vol-spike has been detected.
    :param api_errors: Number of recent API errors.
    :param max_daily_loss_pct: Threshold for daily-loss breaker.
    :param max_drawdown_pct: Threshold for drawdown breaker.
    :param max_consecutive_losses: Threshold for loss-streak breaker.
    :param max_api_errors: Threshold for API-error breaker.
    :returns: :class:`CircuitBreakerStatus`.
    """
    triggers: List[str] = []

    if daily_loss_pct >= max_daily_loss_pct:
        triggers.append("daily_loss_limit")
    if drawdown_pct >= max_drawdown_pct:
        triggers.append("max_drawdown")
    if consecutive_losses >= max_consecutive_losses:
        triggers.append("consecutive_losses")
    if volatility_spike:
        triggers.append("volatility_spike")
    if api_errors >= max_api_errors:
        triggers.append("api_errors")

    is_tripped = len(triggers) > 0

    if len(triggers) >= 3:
        severity = "critical"
        cooldown = 3600.0
    elif len(triggers) >= 2:
        severity = "high"
        cooldown = 1800.0
    elif len(triggers) == 1:
        severity = "medium"
        cooldown = 300.0
    else:
        severity = "low"
        cooldown = 0.0

    can_resume = not is_tripped or severity in ("low", "medium")

    return CircuitBreakerStatus(
        is_tripped=is_tripped,
        triggers=triggers,
        severity=severity,
        cooldown_seconds=cooldown,
        can_resume=can_resume,
        reason=f"circuit_breaker: {'TRIPPED' if is_tripped else 'ok'}, "
               f"triggers={triggers}",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 17. Strategy Shutdown on Anomaly
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AnomalyShutdown:
    """Anomaly-based strategy shutdown decision.

    ``should_shutdown`` is True when anomalies warrant halting.
    ``anomalies`` lists detected anomaly descriptions.
    """

    should_shutdown: bool
    anomalies: List[Dict[str, Any]]
    anomaly_score: float
    threshold: float
    recommended_action: str
    reason: str


def check_anomaly_shutdown(
    recent_returns: Sequence[float],
    expected_return: float = 0.0,
    volume_ratio: float = 1.0,
    spread_ratio: float = 1.0,
    anomaly_threshold: float = 3.0,
) -> AnomalyShutdown:
    """Detect anomalies that should trigger a strategy shutdown.

    Uses z-score analysis on returns and volume/spread deviations
    to identify abnormal market conditions.

    :param recent_returns: Recent period returns.
    :param expected_return: Expected average return.
    :param volume_ratio: Current volume / average volume.
    :param spread_ratio: Current spread / average spread.
    :param anomaly_threshold: Z-score threshold for anomaly.
    :returns: :class:`AnomalyShutdown`.
    """
    anomalies: List[Dict[str, Any]] = []
    scores: List[float] = []

    # Return anomaly (z-score)
    if len(recent_returns) >= 5:
        mu = mean(recent_returns)
        sigma = pstdev(recent_returns) if len(recent_returns) >= 2 else 0.01
        if sigma > 0:
            latest = recent_returns[-1]
            z = abs(latest - mu) / sigma
            scores.append(z)
            if z >= anomaly_threshold:
                anomalies.append({
                    "type": "return_anomaly",
                    "z_score": round(z, 4),
                    "value": round(latest, 6),
                })

    # Volume anomaly
    if volume_ratio > 3.0:
        v_score = volume_ratio / 3.0 * anomaly_threshold
        scores.append(v_score)
        anomalies.append({
            "type": "volume_spike",
            "ratio": round(volume_ratio, 2),
        })

    # Spread anomaly
    if spread_ratio > 3.0:
        s_score = spread_ratio / 3.0 * anomaly_threshold
        scores.append(s_score)
        anomalies.append({
            "type": "spread_blowout",
            "ratio": round(spread_ratio, 2),
        })

    overall_score = max(scores) if scores else 0.0
    should_shutdown = overall_score >= anomaly_threshold

    if should_shutdown:
        action = "shutdown"
    elif overall_score >= anomaly_threshold * 0.7:
        action = "reduce_exposure"
    else:
        action = "continue"

    return AnomalyShutdown(
        should_shutdown=should_shutdown,
        anomalies=anomalies,
        anomaly_score=round(overall_score, 4),
        threshold=anomaly_threshold,
        recommended_action=action,
        reason=f"anomaly: score={overall_score:.2f} "
               f"(threshold={anomaly_threshold}), action={action}",
    )
