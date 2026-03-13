"""Portfolio management & multi-asset optimisation module.

Provides 11 portfolio-management categories for the Indodax trading bot:

 1. Multi-asset portfolio management
 2. Portfolio rebalancing
 3. Portfolio diversification
 4. Asset allocation engine
 5. Correlation matrix monitoring
 6. Portfolio optimization
 7. Risk-adjusted return optimization
 8. Capital distribution across strategies
 9. Sector exposure monitoring
10. Cross-market hedging
11. Long / short portfolio balancing

Each algorithm is implemented as a pure function operating on standard
market data (prices, weights, returns) and returns typed dataclasses.
All implementations use only the Python standard library.
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


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
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


def _sharpe(returns: Sequence[float], risk_free: float = 0.0) -> float:
    """Annualised Sharpe ratio approximation."""
    if len(returns) < 2:
        return 0.0
    avg = mean(returns) - risk_free
    std = pstdev(returns)
    if std == 0:
        return 0.0
    return (avg / std) * math.sqrt(252)


def _sortino(returns: Sequence[float], risk_free: float = 0.0) -> float:
    """Annualised Sortino ratio approximation."""
    if len(returns) < 2:
        return 0.0
    avg = mean(returns) - risk_free
    down = [r for r in returns if r < 0]
    if len(down) < 2:
        return 0.0
    dd = pstdev(down)
    if dd == 0:
        return 0.0
    return (avg / dd) * math.sqrt(252)


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
#  1. Multi-Asset Portfolio Management
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MultiAssetPortfolio:
    """Snapshot of a multi-asset portfolio.

    ``holdings`` maps asset → current value.
    ``weights`` maps asset → portfolio weight (0..1).
    ``total_value`` is the aggregate portfolio value.
    """

    holdings: Dict[str, float]
    weights: Dict[str, float]
    total_value: float
    num_assets: int
    concentration_score: float
    reason: str


def evaluate_multi_asset_portfolio(
    holdings: Dict[str, float],
    prices: Dict[str, float] | None = None,
) -> MultiAssetPortfolio:
    """Evaluate a multi-asset portfolio's current state.

    :param holdings: Map of asset → quantity held.
    :param prices: Optional map of asset → current price.  When provided
        the position values are ``qty * price``; otherwise the values
        in *holdings* are treated as notional values directly.
    :returns: :class:`MultiAssetPortfolio`.
    """
    values: Dict[str, float] = {}
    for asset, qty in holdings.items():
        p = prices.get(asset, 1.0) if prices else 1.0
        values[asset] = abs(qty * p)

    total = sum(values.values()) or 1.0
    weights = {a: round(v / total, 6) for a, v in values.items()}

    # HHI concentration (0 = diversified, 1 = single-asset)
    hhi = sum(w ** 2 for w in weights.values())

    return MultiAssetPortfolio(
        holdings=values,
        weights=weights,
        total_value=round(total, 2),
        num_assets=len(values),
        concentration_score=round(hhi, 6),
        reason=f"portfolio: {len(values)} assets, "
               f"total={total:.2f}, hhi={hhi:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Portfolio Rebalancing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RebalanceAction:
    """Single rebalancing trade."""

    asset: str
    direction: str  # "buy" or "sell"
    amount: float
    current_weight: float
    target_weight: float


@dataclass
class RebalancePlan:
    """Complete rebalancing plan for the portfolio.

    ``actions`` is a list of :class:`RebalanceAction` trades needed.
    ``total_turnover`` is the aggregate absolute trade value.
    """

    actions: List[RebalanceAction]
    total_turnover: float
    max_drift: float
    needs_rebalance: bool
    reason: str


def plan_rebalance(
    current_weights: Dict[str, float],
    target_weights: Dict[str, float],
    portfolio_value: float,
    threshold_pct: float = 5.0,
) -> RebalancePlan:
    """Generate a rebalancing plan when weights drift beyond a threshold.

    :param current_weights: Map of asset → current weight (0..1).
    :param target_weights: Map of asset → target weight (0..1).
    :param portfolio_value: Total portfolio value.
    :param threshold_pct: Minimum drift (%) to trigger a rebalance.
    :returns: :class:`RebalancePlan`.
    """
    actions: List[RebalanceAction] = []
    max_drift = 0.0

    all_assets = set(current_weights) | set(target_weights)
    for asset in sorted(all_assets):
        cw = current_weights.get(asset, 0.0)
        tw = target_weights.get(asset, 0.0)
        drift = tw - cw
        max_drift = max(max_drift, abs(drift) * 100)
        if abs(drift) * 100 >= threshold_pct:
            trade_val = abs(drift) * portfolio_value
            actions.append(RebalanceAction(
                asset=asset,
                direction="buy" if drift > 0 else "sell",
                amount=round(trade_val, 2),
                current_weight=round(cw, 6),
                target_weight=round(tw, 6),
            ))

    turnover = sum(a.amount for a in actions)
    needs = len(actions) > 0

    return RebalancePlan(
        actions=actions,
        total_turnover=round(turnover, 2),
        max_drift=round(max_drift, 4),
        needs_rebalance=needs,
        reason=f"rebalance: {len(actions)} trades, "
               f"turnover={turnover:.2f}, max_drift={max_drift:.2f}%",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Portfolio Diversification
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class DiversificationScore:
    """Portfolio diversification assessment.

    ``score`` ranges from 0 (no diversification) to 1 (fully diversified).
    ``effective_assets`` is the inverse-HHI measure of how many
    "effective" independent positions the portfolio holds.
    """

    score: float
    effective_assets: float
    hhi: float
    num_assets: int
    recommendations: List[str]
    reason: str


def assess_diversification(
    weights: Dict[str, float],
    returns_map: Dict[str, Sequence[float]] | None = None,
) -> DiversificationScore:
    """Assess how well-diversified the portfolio is.

    :param weights: Map of asset → weight (0..1).
    :param returns_map: Optional historical returns for correlation check.
    :returns: :class:`DiversificationScore`.
    """
    if not weights:
        return DiversificationScore(
            score=0.0, effective_assets=0.0, hhi=1.0,
            num_assets=0, recommendations=["add assets"],
            reason="empty portfolio",
        )

    # Normalise
    total_w = sum(abs(w) for w in weights.values()) or 1.0
    normed = {a: abs(w) / total_w for a, w in weights.items()}

    hhi = sum(w ** 2 for w in normed.values())
    effective = 1.0 / hhi if hhi > 0 else 0.0
    n = len(normed)

    # Score: 1 - HHI  (0 = concentrated, approaches 1 with many equal wts)
    score = 1.0 - hhi

    recs: List[str] = []
    if n < 3:
        recs.append("increase_asset_count")
    if hhi > 0.5:
        recs.append("reduce_concentration")

    # Correlation-based penalty
    if returns_map and n >= 2:
        assets = list(normed.keys())
        corrs: List[float] = []
        for i in range(len(assets)):
            for j in range(i + 1, len(assets)):
                ra = list(returns_map.get(assets[i], []))
                rb = list(returns_map.get(assets[j], []))
                if ra and rb:
                    corrs.append(abs(_pearson(ra, rb)))
        if corrs:
            avg_corr = mean(corrs)
            if avg_corr > 0.7:
                score *= 0.7
                recs.append("high_avg_correlation")

    return DiversificationScore(
        score=round(_clamp(score, 0, 1), 4),
        effective_assets=round(effective, 2),
        hhi=round(hhi, 6),
        num_assets=n,
        recommendations=recs,
        reason=f"diversification: score={score:.2f}, "
               f"eff_assets={effective:.1f}, hhi={hhi:.4f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Asset Allocation Engine
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AssetAllocationPlan:
    """Allocation plan produced by the asset allocation engine.

    ``target_weights`` maps asset → recommended weight (0..1).
    ``method`` describes the allocation algorithm used.
    """

    target_weights: Dict[str, float]
    method: str
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    reason: str


def compute_asset_allocation(
    assets: Sequence[str],
    returns_map: Dict[str, Sequence[float]],
    method: str = "equal_weight",
    risk_free_rate: float = 0.0,
) -> AssetAllocationPlan:
    """Compute target asset allocation weights.

    Supported methods:
    - ``"equal_weight"``: 1/N allocation.
    - ``"inverse_vol"``: Inversely proportional to volatility.
    - ``"momentum"``: Weight by recent mean return.

    :param assets: Asset identifiers.
    :param returns_map: Historical returns per asset.
    :param method: Allocation algorithm.
    :param risk_free_rate: Daily risk-free rate.
    :returns: :class:`AssetAllocationPlan`.
    """
    if not assets:
        return AssetAllocationPlan(
            target_weights={}, method=method,
            expected_return=0.0, expected_volatility=0.0,
            sharpe_ratio=0.0, reason="no assets",
        )

    if method == "inverse_vol":
        inv_vols: Dict[str, float] = {}
        for a in assets:
            rets = list(returns_map.get(a, []))
            vol = pstdev(rets) if len(rets) >= 2 else 0.01
            inv_vols[a] = 1.0 / max(vol, 1e-9)
        total = sum(inv_vols.values()) or 1.0
        weights = {a: round(inv_vols[a] / total, 6) for a in assets}

    elif method == "momentum":
        scores: Dict[str, float] = {}
        for a in assets:
            rets = list(returns_map.get(a, []))
            scores[a] = max(0.0, mean(rets)) if rets else 0.0
        total = sum(scores.values()) or 1.0
        if total > 0:
            weights = {a: round(scores[a] / total, 6) for a in assets}
        else:
            per = 1.0 / len(assets)
            weights = {a: round(per, 6) for a in assets}

    else:  # equal_weight
        per = 1.0 / len(assets)
        weights = {a: round(per, 6) for a in assets}

    # Portfolio-level expected return & vol (simplified)
    port_ret = 0.0
    port_var = 0.0
    for a in assets:
        rets = list(returns_map.get(a, []))
        w = weights.get(a, 0.0)
        if rets:
            port_ret += w * mean(rets)
            port_var += (w * (pstdev(rets) if len(rets) >= 2 else 0.0)) ** 2
    port_vol = math.sqrt(port_var) if port_var > 0 else 0.0
    sr = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return AssetAllocationPlan(
        target_weights=weights,
        method=method,
        expected_return=round(port_ret * 252, 6),
        expected_volatility=round(port_vol * math.sqrt(252), 6),
        sharpe_ratio=round(sr * math.sqrt(252), 4),
        reason=f"allocation: {method}, {len(assets)} assets",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Correlation Matrix Monitoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class CorrelationMatrix:
    """Correlation matrix analysis result.

    ``matrix`` is a nested dict: ``matrix[A][B]`` = correlation.
    ``high_pairs`` lists pairs with correlation above the alert threshold.
    """

    matrix: Dict[str, Dict[str, float]]
    high_pairs: List[Tuple[str, str, float]]
    avg_correlation: float
    max_correlation: float
    risk_level: str
    reason: str


def compute_correlation_matrix(
    assets: Sequence[str],
    returns_map: Dict[str, Sequence[float]],
    alert_threshold: float = 0.7,
) -> CorrelationMatrix:
    """Build and analyse the pairwise correlation matrix.

    :param assets: Asset identifiers.
    :param returns_map: Historical returns per asset.
    :param alert_threshold: Absolute correlation threshold for alerts.
    :returns: :class:`CorrelationMatrix`.
    """
    matrix: Dict[str, Dict[str, float]] = {}
    high_pairs: List[Tuple[str, str, float]] = []
    all_corrs: List[float] = []

    for a in assets:
        matrix[a] = {}
        for b in assets:
            if a == b:
                matrix[a][b] = 1.0
            else:
                ra = list(returns_map.get(a, []))
                rb = list(returns_map.get(b, []))
                c = _pearson(ra, rb)
                matrix[a][b] = round(c, 4)

    # Collect unique pairs
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            a, b = assets[i], assets[j]
            c = matrix.get(a, {}).get(b, 0.0)
            all_corrs.append(abs(c))
            if abs(c) >= alert_threshold:
                high_pairs.append((a, b, round(c, 4)))

    avg_c = mean(all_corrs) if all_corrs else 0.0
    max_c = max(all_corrs) if all_corrs else 0.0

    if max_c > 0.9:
        risk = "critical"
    elif max_c > 0.7:
        risk = "high"
    elif max_c > 0.5:
        risk = "moderate"
    else:
        risk = "low"

    return CorrelationMatrix(
        matrix=matrix,
        high_pairs=high_pairs,
        avg_correlation=round(avg_c, 4),
        max_correlation=round(max_c, 4),
        risk_level=risk,
        reason=f"corr_matrix: {len(assets)} assets, "
               f"{len(high_pairs)} high pairs, risk={risk}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6. Portfolio Optimization
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OptimizedPortfolio:
    """Result of portfolio optimisation.

    ``weights`` maps asset → optimal weight.
    ``method`` describes the optimisation approach.
    """

    weights: Dict[str, float]
    method: str
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_est: float
    reason: str


def optimize_portfolio(
    assets: Sequence[str],
    returns_map: Dict[str, Sequence[float]],
    method: str = "min_variance",
    risk_free_rate: float = 0.0,
    max_weight: float = 0.40,
) -> OptimizedPortfolio:
    """Optimise portfolio weights.

    Supported methods:

    - ``"min_variance"``: Minimise estimated portfolio variance (inverse-vol
      proxy with weight caps).
    - ``"max_sharpe"``: Maximise Sharpe ratio via grid search over
      inverse-vol and momentum blends.
    - ``"equal_risk"``: Equal risk contribution (inverse-vol normalised).

    :param assets: Asset identifiers.
    :param returns_map: Historical returns per asset.
    :param method: Optimisation approach.
    :param risk_free_rate: Daily risk-free rate.
    :param max_weight: Maximum weight per asset (0..1).
    :returns: :class:`OptimizedPortfolio`.
    """
    if not assets:
        return OptimizedPortfolio(
            weights={}, method=method, expected_return=0.0,
            expected_volatility=0.0, sharpe_ratio=0.0,
            max_drawdown_est=0.0, reason="no assets",
        )

    vols: Dict[str, float] = {}
    means: Dict[str, float] = {}
    for a in assets:
        rets = list(returns_map.get(a, []))
        vols[a] = pstdev(rets) if len(rets) >= 2 else 0.01
        means[a] = mean(rets) if rets else 0.0

    def _inv_vol_weights() -> Dict[str, float]:
        inv = {a: 1.0 / max(vols[a], 1e-9) for a in assets}
        t = sum(inv.values()) or 1.0
        w = {a: inv[a] / t for a in assets}
        # Iteratively cap & redistribute to respect max_weight
        for _ in range(10):
            over = {a: v for a, v in w.items() if v > max_weight}
            if not over:
                break
            free = {a: v for a, v in w.items() if v <= max_weight}
            excess = sum(v - max_weight for v in over.values())
            for a in over:
                w[a] = max_weight
            if free:
                ft = sum(free.values()) or 1.0
                for a in free:
                    w[a] += excess * (free[a] / ft)
        # Final normalise to ensure sum == 1
        wt = sum(w.values()) or 1.0
        return {a: round(w[a] / wt, 6) for a in assets}

    def _momentum_weights() -> Dict[str, float]:
        scores = {a: max(0.0, means[a]) for a in assets}
        t = sum(scores.values())
        if t == 0:
            per = 1.0 / len(assets)
            return {a: round(per, 6) for a in assets}
        w = {a: scores[a] / t for a in assets}
        for _ in range(10):
            over = {a: v for a, v in w.items() if v > max_weight}
            if not over:
                break
            free = {a: v for a, v in w.items() if v <= max_weight}
            excess = sum(v - max_weight for v in over.values())
            for a in over:
                w[a] = max_weight
            if free:
                ft = sum(free.values()) or 1.0
                for a in free:
                    w[a] += excess * (free[a] / ft)
        wt = sum(w.values()) or 1.0
        return {a: round(w[a] / wt, 6) for a in assets}

    def _port_stats(w: Dict[str, float]) -> Tuple[float, float, float]:
        pr = sum(w.get(a, 0) * means[a] for a in assets)
        pv = math.sqrt(sum((w.get(a, 0) * vols[a]) ** 2 for a in assets))
        sr = (pr - risk_free_rate) / pv if pv > 0 else 0.0
        return pr, pv, sr

    if method == "max_sharpe":
        # Simple blend search: combine inv_vol and momentum
        best_sr = -999.0
        best_w: Dict[str, float] = {}
        iv = _inv_vol_weights()
        mom = _momentum_weights()
        for alpha_int in range(0, 11):
            alpha = alpha_int / 10.0
            blend = {}
            for a in assets:
                blend[a] = alpha * iv.get(a, 0) + (1 - alpha) * mom.get(a, 0)
            bt = sum(blend.values()) or 1.0
            blend = {a: blend[a] / bt for a in assets}
            _, _, sr = _port_stats(blend)
            if sr > best_sr:
                best_sr = sr
                best_w = {a: round(v, 6) for a, v in blend.items()}
        weights = best_w

    elif method == "equal_risk":
        weights = _inv_vol_weights()

    else:  # min_variance
        weights = _inv_vol_weights()

    pr, pv, sr = _port_stats(weights)
    # Rough drawdown estimate: vol * 2
    mdd = pv * 2 * math.sqrt(252)

    return OptimizedPortfolio(
        weights=weights,
        method=method,
        expected_return=round(pr * 252, 6),
        expected_volatility=round(pv * math.sqrt(252), 6),
        sharpe_ratio=round(sr * math.sqrt(252), 4),
        max_drawdown_est=round(mdd, 4),
        reason=f"optimized: {method}, {len(assets)} assets, "
               f"sharpe={sr * math.sqrt(252):.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7. Risk-Adjusted Return Optimization
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics for the portfolio.

    Contains Sharpe, Sortino, Calmar ratios plus per-asset breakdown.
    """

    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    per_asset: Dict[str, Dict[str, float]]
    best_risk_adjusted: str
    worst_risk_adjusted: str
    reason: str


def evaluate_risk_adjusted_returns(
    assets: Sequence[str],
    returns_map: Dict[str, Sequence[float]],
    prices_map: Dict[str, Sequence[float]] | None = None,
    risk_free_rate: float = 0.0,
) -> RiskAdjustedMetrics:
    """Evaluate risk-adjusted return metrics across assets.

    :param assets: Asset identifiers.
    :param returns_map: Historical returns per asset.
    :param prices_map: Optional price series for drawdown calc.
    :param risk_free_rate: Daily risk-free rate.
    :returns: :class:`RiskAdjustedMetrics`.
    """
    per_asset: Dict[str, Dict[str, float]] = {}
    best_a, best_sr = "", -999.0
    worst_a, worst_sr = "", 999.0

    for a in assets:
        rets = list(returns_map.get(a, []))
        sr = _sharpe(rets, risk_free_rate)
        so = _sortino(rets, risk_free_rate)
        # Calmar = annualised return / max drawdown
        ann_ret = mean(rets) * 252 if rets else 0.0
        prices = list((prices_map or {}).get(a, []))
        mdd = _max_drawdown(prices) if len(prices) >= 2 else 0.0
        calmar = ann_ret / mdd if mdd > 0 else 0.0

        per_asset[a] = {
            "sharpe": round(sr, 4),
            "sortino": round(so, 4),
            "calmar": round(calmar, 4),
            "ann_return": round(ann_ret, 6),
            "max_drawdown": round(mdd, 4),
        }
        if sr > best_sr:
            best_sr, best_a = sr, a
        if sr < worst_sr:
            worst_sr, worst_a = sr, a

    # Portfolio-level (equal-weight)
    all_rets: List[float] = []
    if assets:
        min_len = min(len(returns_map.get(a, [])) for a in assets)
        for i in range(min_len):
            r = mean(returns_map[a][i] for a in assets if len(returns_map.get(a, [])) > i)
            all_rets.append(r)

    port_sharpe = _sharpe(all_rets, risk_free_rate)
    port_sortino = _sortino(all_rets, risk_free_rate)
    port_ann = mean(all_rets) * 252 if all_rets else 0.0
    port_calmar = port_ann / 0.1 if port_ann > 0 else 0.0  # rough estimate

    return RiskAdjustedMetrics(
        sharpe_ratio=round(port_sharpe, 4),
        sortino_ratio=round(port_sortino, 4),
        calmar_ratio=round(port_calmar, 4),
        per_asset=per_asset,
        best_risk_adjusted=best_a,
        worst_risk_adjusted=worst_a,
        reason=f"risk_adj: sharpe={port_sharpe:.2f}, "
               f"best={best_a}, worst={worst_a}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. Capital Distribution Across Strategies
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyCapitalPlan:
    """Capital distribution plan across trading strategies.

    ``allocations`` maps strategy → allocated capital.
    ``performance_scores`` maps strategy → score used for ranking.
    """

    allocations: Dict[str, float]
    performance_scores: Dict[str, float]
    reserve_amount: float
    total_deployed: float
    total_capital: float
    reason: str


def distribute_capital(
    total_capital: float,
    strategies: Sequence[str],
    strategy_returns: Dict[str, Sequence[float]] | None = None,
    method: str = "equal",
    reserve_pct: float = 10.0,
    min_allocation: float = 0.0,
    max_allocation_pct: float = 50.0,
) -> StrategyCapitalPlan:
    """Distribute capital across trading strategies.

    Supported methods:
    - ``"equal"``: Equal capital per strategy.
    - ``"performance"``: Weight by recent Sharpe ratio.
    - ``"inverse_vol"``: Weight by inverse return-volatility.

    :param total_capital: Total available capital.
    :param strategies: Strategy identifiers.
    :param strategy_returns: Optional returns per strategy.
    :param method: Distribution method.
    :param reserve_pct: Cash reserve percentage (0-100).
    :param min_allocation: Minimum capital per strategy.
    :param max_allocation_pct: Maximum capital per strategy as pct.
    :returns: :class:`StrategyCapitalPlan`.
    """
    if total_capital <= 0 or not strategies:
        return StrategyCapitalPlan(
            allocations={}, performance_scores={},
            reserve_amount=0.0, total_deployed=0.0,
            total_capital=total_capital,
            reason="no capital or strategies",
        )

    reserve = total_capital * _clamp(reserve_pct, 0, 100) / 100
    deployable = total_capital - reserve
    max_per = deployable * _clamp(max_allocation_pct, 0, 100) / 100

    scores: Dict[str, float] = {}

    if method == "performance" and strategy_returns:
        for s in strategies:
            rets = list(strategy_returns.get(s, []))
            scores[s] = max(0.0, _sharpe(rets))
    elif method == "inverse_vol" and strategy_returns:
        for s in strategies:
            rets = list(strategy_returns.get(s, []))
            vol = pstdev(rets) if len(rets) >= 2 else 0.01
            scores[s] = 1.0 / max(vol, 1e-9)
    else:  # equal
        for s in strategies:
            scores[s] = 1.0

    total_score = sum(scores.values()) or 1.0
    raw: Dict[str, float] = {}
    for s in strategies:
        raw[s] = deployable * scores[s] / total_score

    # Apply min/max and re-normalise
    allocs: Dict[str, float] = {}
    for s in strategies:
        allocs[s] = round(_clamp(raw[s], min_allocation, max_per), 2)

    total_deployed = sum(allocs.values())

    return StrategyCapitalPlan(
        allocations=allocs,
        performance_scores={s: round(scores[s], 4) for s in strategies},
        reserve_amount=round(reserve, 2),
        total_deployed=round(total_deployed, 2),
        total_capital=total_capital,
        reason=f"capital_dist: {method}, {len(strategies)} strategies, "
               f"deployed={total_deployed:.2f}",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Sector Exposure Monitoring
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SectorExposure:
    """Sector-level exposure analysis.

    ``sector_weights`` maps sector → portfolio weight.
    ``overexposed`` lists sectors exceeding the limit.
    """

    sector_weights: Dict[str, float]
    sector_values: Dict[str, float]
    overexposed: List[str]
    underexposed: List[str]
    max_sector_weight: float
    portfolio_value: float
    reason: str


def monitor_sector_exposure(
    positions: Dict[str, float],
    asset_sectors: Dict[str, str],
    portfolio_value: float,
    max_sector_pct: float = 30.0,
    min_sector_pct: float = 5.0,
) -> SectorExposure:
    """Monitor portfolio exposure by sector.

    :param positions: Map of asset → position value.
    :param asset_sectors: Map of asset → sector name.
    :param portfolio_value: Total portfolio value.
    :param max_sector_pct: Maximum allowed sector weight (0-100).
    :param min_sector_pct: Minimum desired sector weight (0-100).
    :returns: :class:`SectorExposure`.
    """
    if portfolio_value <= 0:
        return SectorExposure(
            sector_weights={}, sector_values={},
            overexposed=[], underexposed=[],
            max_sector_weight=0.0,
            portfolio_value=0.0, reason="zero portfolio",
        )

    sector_vals: Dict[str, float] = {}
    for asset, val in positions.items():
        sector = asset_sectors.get(asset, "unknown")
        sector_vals[sector] = sector_vals.get(sector, 0.0) + abs(val)

    sector_wts = {
        s: round(v / portfolio_value * 100, 4)
        for s, v in sector_vals.items()
    }

    over = [s for s, w in sector_wts.items() if w > max_sector_pct]
    under = [s for s, w in sector_wts.items() if w < min_sector_pct]
    max_w = max(sector_wts.values()) if sector_wts else 0.0

    return SectorExposure(
        sector_weights=sector_wts,
        sector_values={s: round(v, 2) for s, v in sector_vals.items()},
        overexposed=over,
        underexposed=under,
        max_sector_weight=round(max_w, 4),
        portfolio_value=round(portfolio_value, 2),
        reason=f"sector: {len(sector_wts)} sectors, "
               f"{len(over)} over, {len(under)} under",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Cross-Market Hedging
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class HedgeRecommendation:
    """Cross-market hedging recommendation.

    ``hedge_pairs`` maps base asset → suggested hedge asset.
    ``hedge_ratios`` maps base asset → optimal hedge ratio.
    """

    hedge_pairs: Dict[str, str]
    hedge_ratios: Dict[str, float]
    portfolio_beta: float
    unhedged_risk: float
    hedged_risk: float
    risk_reduction_pct: float
    reason: str


def recommend_hedges(
    portfolio_assets: Sequence[str],
    hedge_candidates: Sequence[str],
    returns_map: Dict[str, Sequence[float]],
    portfolio_weights: Dict[str, float] | None = None,
) -> HedgeRecommendation:
    """Recommend cross-market hedging positions.

    For each portfolio asset, find the hedge candidate with the most
    negative correlation and compute the optimal hedge ratio.

    :param portfolio_assets: Assets currently held.
    :param hedge_candidates: Possible hedging instruments.
    :param returns_map: Historical returns per asset.
    :param portfolio_weights: Optional weights for portfolio assets.
    :returns: :class:`HedgeRecommendation`.
    """
    if not portfolio_assets or not hedge_candidates:
        return HedgeRecommendation(
            hedge_pairs={}, hedge_ratios={},
            portfolio_beta=0.0, unhedged_risk=0.0,
            hedged_risk=0.0, risk_reduction_pct=0.0,
            reason="no assets or hedge candidates",
        )

    hedge_pairs: Dict[str, str] = {}
    hedge_ratios: Dict[str, float] = {}

    for pa in portfolio_assets:
        best_h, best_corr = "", 1.0
        ra = list(returns_map.get(pa, []))
        if not ra:
            continue
        for hc in hedge_candidates:
            if hc == pa:
                continue
            rb = list(returns_map.get(hc, []))
            if not rb:
                continue
            c = _pearson(ra, rb)
            if c < best_corr:
                best_corr = c
                best_h = hc
        if best_h and best_corr < 0:
            hedge_pairs[pa] = best_h
            # Hedge ratio = -corr * vol_a / vol_h
            vol_a = pstdev(ra) if len(ra) >= 2 else 0.01
            rh = list(returns_map.get(best_h, []))
            vol_h = pstdev(rh) if len(rh) >= 2 else 0.01
            ratio = abs(best_corr) * vol_a / max(vol_h, 1e-9)
            hedge_ratios[pa] = round(_clamp(ratio, 0, 2), 4)

    # Simplified risk estimate
    wts = portfolio_weights or {a: 1.0 / len(portfolio_assets) for a in portfolio_assets}
    unhedged = 0.0
    for a in portfolio_assets:
        ra = list(returns_map.get(a, []))
        vol = pstdev(ra) if len(ra) >= 2 else 0.0
        unhedged += (wts.get(a, 0) * vol) ** 2
    unhedged_vol = math.sqrt(unhedged) if unhedged > 0 else 0.0

    # Estimate hedged risk (reduce by avg hedge ratio)
    avg_ratio = mean(hedge_ratios.values()) if hedge_ratios else 0.0
    hedged_vol = unhedged_vol * max(0, 1 - avg_ratio * 0.5)
    reduction = ((unhedged_vol - hedged_vol) / unhedged_vol * 100
                 if unhedged_vol > 0 else 0.0)

    return HedgeRecommendation(
        hedge_pairs=hedge_pairs,
        hedge_ratios=hedge_ratios,
        portfolio_beta=round(avg_ratio, 4),
        unhedged_risk=round(unhedged_vol, 6),
        hedged_risk=round(hedged_vol, 6),
        risk_reduction_pct=round(reduction, 2),
        reason=f"hedge: {len(hedge_pairs)} pairs, "
               f"risk_reduction={reduction:.1f}%",
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Long / Short Portfolio Balancing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LongShortBalance:
    """Long/short portfolio balance analysis.

    ``long_exposure`` and ``short_exposure`` are absolute values.
    ``net_exposure`` is long - short.  ``gross_exposure`` is long + short.
    """

    long_exposure: float
    short_exposure: float
    net_exposure: float
    gross_exposure: float
    long_pct: float
    short_pct: float
    net_pct: float
    is_balanced: bool
    long_assets: List[str]
    short_assets: List[str]
    recommendation: str
    reason: str


def analyse_long_short_balance(
    positions: Dict[str, float],
    portfolio_value: float,
    target_net_pct: float = 0.0,
    tolerance_pct: float = 10.0,
) -> LongShortBalance:
    """Analyse and assess the long/short balance of the portfolio.

    :param positions: Map of asset → signed position value (positive = long,
        negative = short).
    :param portfolio_value: Total portfolio value (for pct calculations).
    :param target_net_pct: Desired net exposure percentage.
    :param tolerance_pct: Allowed deviation from target.
    :returns: :class:`LongShortBalance`.
    """
    long_exp = sum(v for v in positions.values() if v > 0)
    short_exp = abs(sum(v for v in positions.values() if v < 0))
    net = long_exp - short_exp
    gross = long_exp + short_exp

    pv = portfolio_value if portfolio_value > 0 else 1.0
    long_pct = long_exp / pv * 100
    short_pct = short_exp / pv * 100
    net_pct = net / pv * 100

    deviation = abs(net_pct - target_net_pct)
    is_balanced = deviation <= tolerance_pct

    long_assets = sorted(a for a, v in positions.items() if v > 0)
    short_assets = sorted(a for a, v in positions.items() if v < 0)

    if is_balanced:
        rec = "maintain"
    elif net_pct > target_net_pct + tolerance_pct:
        rec = "increase_short_or_reduce_long"
    else:
        rec = "increase_long_or_reduce_short"

    return LongShortBalance(
        long_exposure=round(long_exp, 2),
        short_exposure=round(short_exp, 2),
        net_exposure=round(net, 2),
        gross_exposure=round(gross, 2),
        long_pct=round(long_pct, 4),
        short_pct=round(short_pct, 4),
        net_pct=round(net_pct, 4),
        is_balanced=is_balanced,
        long_assets=long_assets,
        short_assets=short_assets,
        recommendation=rec,
        reason=f"ls_balance: net={net_pct:.1f}%, "
               f"target={target_net_pct}% ±{tolerance_pct}%",
    )
