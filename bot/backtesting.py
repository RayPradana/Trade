"""Backtesting & simulation module.

Provides 13 backtesting categories for the Indodax trading bot:

 1. Historical backtesting
 2. Tick-level backtesting
 3. Walk-forward testing
 4. Monte Carlo simulation
 5. Parameter optimization
 6. Strategy robustness testing
 7. Stress testing
 8. Slippage simulation
 9. Transaction cost modeling
10. Liquidity simulation
11. Out-of-sample testing
12. Forward testing
13. Paper trading mode

Each algorithm is implemented as a pure function operating on standard
market data (prices, trades, candles) and returns typed dataclasses.
All implementations use only the Python standard library.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

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


def _max_drawdown(equity: Sequence[float]) -> float:
    """Maximum drawdown as a positive fraction."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    mdd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def _sharpe(returns: Sequence[float], risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio (daily returns assumed)."""
    if len(returns) < 2:
        return 0.0
    avg = mean(returns) - risk_free
    std = pstdev(returns)
    if std == 0:
        return 0.0
    return (avg / std) * math.sqrt(252)


def _sortino(returns: Sequence[float], risk_free: float = 0.0) -> float:
    """Annualized Sortino ratio."""
    if len(returns) < 2:
        return 0.0
    avg = mean(returns) - risk_free
    neg = [r for r in returns if r < 0]
    if len(neg) < 2:
        return 0.0 if avg <= 0 else float("inf")
    dsd = pstdev(neg)
    if dsd == 0:
        return 0.0
    return (avg / dsd) * math.sqrt(252)


def _equity_from_returns(
    returns: Sequence[float], initial: float = 10000.0
) -> List[float]:
    """Build an equity curve from a return series."""
    equity = [initial]
    for r in returns:
        equity.append(equity[-1] * (1.0 + r))
    return equity


# ═══════════════════════════════════════════════════════════════════════════
#  1. Historical Backtesting
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class BacktestTrade:
    """Single trade executed during a backtest."""

    entry_index: int
    exit_index: int
    side: str  # "long" | "short"
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    return_pct: float


@dataclass
class HistoricalBacktestResult:
    """Result of a historical bar-by-bar backtest."""

    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    total_trades: int
    win_rate: float
    profit_factor: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)


def run_historical_backtest(
    prices: Sequence[float],
    signals: Sequence[int],
    *,
    initial_capital: float = 10000.0,
    position_size: float = 1.0,
    trading_days_per_year: int = 252,
) -> HistoricalBacktestResult:
    """Run a bar-by-bar historical backtest.

    *signals* is a sequence aligned to *prices* where 1 = long entry,
    -1 = short entry, 0 = flat / exit.
    """
    if len(prices) < 2 or len(signals) < 2:
        return HistoricalBacktestResult(
            total_return=0.0,
            annualized_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            equity_curve=[initial_capital],
            trades=[],
        )

    n = min(len(prices), len(signals))
    equity = [initial_capital]
    trades: List[BacktestTrade] = []
    position: Optional[dict] = None  # {"side", "entry_idx", "entry_price", "qty"}
    daily_returns: List[float] = []

    for i in range(1, n):
        sig = signals[i]
        price = _safe_float(prices[i])
        prev_price = _safe_float(prices[i - 1])
        if prev_price == 0 or price == 0:
            equity.append(equity[-1])
            daily_returns.append(0.0)
            continue

        # Close existing position if signal flips
        if position is not None and (
            (position["side"] == "long" and sig != 1)
            or (position["side"] == "short" and sig != -1)
        ):
            ep = position["entry_price"]
            qty = position["qty"]
            if position["side"] == "long":
                pnl = (price - ep) * qty
            else:
                pnl = (ep - price) * qty
            ret_pct = pnl / (ep * qty) if (ep * qty) != 0 else 0.0
            trades.append(
                BacktestTrade(
                    entry_index=position["entry_idx"],
                    exit_index=i,
                    side=position["side"],
                    entry_price=ep,
                    exit_price=price,
                    quantity=qty,
                    pnl=pnl,
                    return_pct=ret_pct,
                )
            )
            position = None

        # Open new position
        if position is None and sig in (1, -1):
            qty = (equity[-1] * position_size) / price if price > 0 else 0.0
            position = {
                "side": "long" if sig == 1 else "short",
                "entry_idx": i,
                "entry_price": price,
                "qty": qty,
            }

        # Mark-to-market
        if position is not None:
            qty = position["qty"]
            if position["side"] == "long":
                daily_pnl = (price - prev_price) * qty
            else:
                daily_pnl = (prev_price - price) * qty
            equity.append(equity[-1] + daily_pnl)
        else:
            equity.append(equity[-1])

        prev_eq = equity[-2] if equity[-2] != 0 else 1.0
        daily_returns.append((equity[-1] - equity[-2]) / prev_eq)

    total_return = (equity[-1] - initial_capital) / initial_capital
    n_years = max(n / trading_days_per_year, 1e-9)
    ann_ret = (1.0 + total_return) ** (1.0 / n_years) - 1.0 if total_return > -1 else -1.0
    mdd = _max_drawdown(equity)
    sr = _sharpe(daily_returns)
    so = _sortino(daily_returns)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    gross_profit = sum(t.pnl for t in wins) if wins else 0.0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    return HistoricalBacktestResult(
        total_return=total_return,
        annualized_return=ann_ret,
        max_drawdown=mdd,
        sharpe_ratio=sr,
        sortino_ratio=so,
        total_trades=len(trades),
        win_rate=win_rate,
        profit_factor=profit_factor,
        equity_curve=equity,
        trades=trades,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  2. Tick-Level Backtesting
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class Tick:
    """Single market tick."""

    timestamp: float
    price: float
    volume: float
    side: str = "buy"  # "buy" | "sell"


@dataclass
class TickBacktestResult:
    """Result of a tick-level backtest."""

    total_return: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    avg_trade_duration: float
    equity_curve: List[float] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)


def run_tick_backtest(
    ticks: Sequence[Tick],
    signal_func: Callable[[Tick, List[Tick]], int],
    *,
    initial_capital: float = 10000.0,
    position_size: float = 1.0,
    lookback: int = 20,
) -> TickBacktestResult:
    """Run a tick-level backtest using *signal_func* to generate signals.

    *signal_func(current_tick, recent_ticks) -> int* where 1 = buy,
    -1 = sell, 0 = hold.
    """
    if not ticks:
        return TickBacktestResult(
            total_return=0.0,
            max_drawdown=0.0,
            total_trades=0,
            win_rate=0.0,
            avg_trade_duration=0.0,
            equity_curve=[initial_capital],
            trades=[],
        )

    equity = [initial_capital]
    trades: List[BacktestTrade] = []
    position: Optional[dict] = None
    recent: List[Tick] = []

    for idx, tick in enumerate(ticks):
        recent.append(tick)
        if len(recent) > lookback:
            recent = recent[-lookback:]

        sig = signal_func(tick, recent)

        # Close on opposite signal
        if position is not None:
            close_position = False
            if position["side"] == "long" and sig == -1:
                close_position = True
            elif position["side"] == "short" and sig == 1:
                close_position = True

            if close_position:
                ep = position["entry_price"]
                qty = position["qty"]
                if position["side"] == "long":
                    pnl = (tick.price - ep) * qty
                else:
                    pnl = (ep - tick.price) * qty
                ret_pct = pnl / (ep * qty) if (ep * qty) != 0 else 0.0
                trades.append(
                    BacktestTrade(
                        entry_index=position["entry_idx"],
                        exit_index=idx,
                        side=position["side"],
                        entry_price=ep,
                        exit_price=tick.price,
                        quantity=qty,
                        pnl=pnl,
                        return_pct=ret_pct,
                    )
                )
                equity.append(equity[-1] + pnl)
                position = None
                continue

        # Open new position
        if position is None and sig in (1, -1):
            qty = (equity[-1] * position_size) / tick.price if tick.price > 0 else 0.0
            position = {
                "side": "long" if sig == 1 else "short",
                "entry_idx": idx,
                "entry_price": tick.price,
                "qty": qty,
                "entry_ts": tick.timestamp,
            }

        equity.append(equity[-1])

    # Close remaining position at last tick
    if position is not None and ticks:
        last = ticks[-1]
        ep = position["entry_price"]
        qty = position["qty"]
        if position["side"] == "long":
            pnl = (last.price - ep) * qty
        else:
            pnl = (ep - last.price) * qty
        ret_pct = pnl / (ep * qty) if (ep * qty) != 0 else 0.0
        trades.append(
            BacktestTrade(
                entry_index=position["entry_idx"],
                exit_index=len(ticks) - 1,
                side=position["side"],
                entry_price=ep,
                exit_price=last.price,
                quantity=qty,
                pnl=pnl,
                return_pct=ret_pct,
            )
        )
        equity.append(equity[-1] + pnl)

    total_ret = (equity[-1] - initial_capital) / initial_capital
    mdd = _max_drawdown(equity)
    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    durations = [t.exit_index - t.entry_index for t in trades]
    avg_dur = mean(durations) if durations else 0.0

    return TickBacktestResult(
        total_return=total_ret,
        max_drawdown=mdd,
        total_trades=len(trades),
        win_rate=win_rate,
        avg_trade_duration=avg_dur,
        equity_curve=equity,
        trades=trades,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3. Walk-Forward Testing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class WalkForwardFold:
    """Result of a single walk-forward fold."""

    fold_index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    in_sample_return: float
    out_of_sample_return: float
    best_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregate walk-forward testing result."""

    folds: List[WalkForwardFold] = field(default_factory=list)
    combined_oos_return: float = 0.0
    avg_is_return: float = 0.0
    avg_oos_return: float = 0.0
    efficiency_ratio: float = 0.0
    is_robust: bool = False


def run_walk_forward_test(
    prices: Sequence[float],
    strategy_func: Callable[[Sequence[float], Dict[str, Any]], float],
    param_grid: Sequence[Dict[str, Any]],
    *,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    min_efficiency: float = 0.5,
) -> WalkForwardResult:
    """Walk-forward optimization.

    *strategy_func(price_window, params) -> return_pct* evaluates a strategy
    with given params on a price window.  *param_grid* is the list of
    parameter dictionaries to search over.
    """
    n = len(prices)
    if n < 10 or not param_grid:
        return WalkForwardResult()

    fold_size = n // n_folds
    if fold_size < 4:
        return WalkForwardResult()

    folds: List[WalkForwardFold] = []
    for fi in range(n_folds):
        start = fi * fold_size
        end = min(start + fold_size, n)
        split = start + int((end - start) * train_ratio)
        if split <= start or split >= end:
            continue

        train_prices = list(prices[start:split])
        test_prices = list(prices[split:end])

        # Optimize on training set
        best_ret = -float("inf")
        best_params: Dict[str, Any] = param_grid[0] if param_grid else {}
        for params in param_grid:
            ret = strategy_func(train_prices, params)
            if ret > best_ret:
                best_ret = ret
                best_params = params

        # Evaluate on test set
        oos_ret = strategy_func(test_prices, best_params)

        folds.append(
            WalkForwardFold(
                fold_index=fi,
                train_start=start,
                train_end=split,
                test_start=split,
                test_end=end,
                in_sample_return=best_ret,
                out_of_sample_return=oos_ret,
                best_params=dict(best_params),
            )
        )

    if not folds:
        return WalkForwardResult()

    avg_is = mean([f.in_sample_return for f in folds])
    avg_oos = mean([f.out_of_sample_return for f in folds])
    combined_oos = sum(f.out_of_sample_return for f in folds)
    eff = avg_oos / avg_is if avg_is != 0 else 0.0

    return WalkForwardResult(
        folds=folds,
        combined_oos_return=combined_oos,
        avg_is_return=avg_is,
        avg_oos_return=avg_oos,
        efficiency_ratio=eff,
        is_robust=eff >= min_efficiency,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  4. Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation of equity paths."""

    num_simulations: int
    median_return: float
    mean_return: float
    p5_return: float
    p95_return: float
    prob_profit: float
    worst_drawdown: float
    avg_drawdown: float
    paths: List[List[float]] = field(default_factory=list)


def run_monte_carlo(
    returns: Sequence[float],
    *,
    num_simulations: int = 1000,
    path_length: int = 252,
    initial_capital: float = 10000.0,
    seed: Optional[int] = None,
    store_paths: bool = False,
) -> MonteCarloResult:
    """Run Monte Carlo simulation by resampling historical returns."""
    if len(returns) < 2:
        return MonteCarloResult(
            num_simulations=0,
            median_return=0.0,
            mean_return=0.0,
            p5_return=0.0,
            p95_return=0.0,
            prob_profit=0.0,
            worst_drawdown=0.0,
            avg_drawdown=0.0,
        )

    rng = random.Random(seed)
    ret_list = list(returns)

    final_returns: List[float] = []
    drawdowns: List[float] = []
    paths: List[List[float]] = []

    for _ in range(num_simulations):
        equity = [initial_capital]
        for _ in range(path_length):
            r = rng.choice(ret_list)
            equity.append(equity[-1] * (1.0 + r))
        total_ret = (equity[-1] - initial_capital) / initial_capital
        final_returns.append(total_ret)
        drawdowns.append(_max_drawdown(equity))
        if store_paths:
            paths.append(equity)

    final_returns.sort()
    n = len(final_returns)
    p5_idx = max(0, int(n * 0.05))
    p95_idx = min(n - 1, int(n * 0.95))
    median_idx = n // 2

    return MonteCarloResult(
        num_simulations=num_simulations,
        median_return=final_returns[median_idx],
        mean_return=mean(final_returns),
        p5_return=final_returns[p5_idx],
        p95_return=final_returns[p95_idx],
        prob_profit=sum(1 for r in final_returns if r > 0) / n,
        worst_drawdown=max(drawdowns),
        avg_drawdown=mean(drawdowns),
        paths=paths,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  5. Parameter Optimization
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ParamOptResult:
    """Result from parameter optimization."""

    best_params: Dict[str, Any] = field(default_factory=dict)
    best_metric: float = 0.0
    all_results: List[Tuple[Dict[str, Any], float]] = field(default_factory=list)
    total_combinations: int = 0
    optimization_metric: str = "sharpe"


def optimize_parameters(
    prices: Sequence[float],
    strategy_func: Callable[[Sequence[float], Dict[str, Any]], float],
    param_grid: Sequence[Dict[str, Any]],
    *,
    metric: str = "return",
    top_n: int = 5,
) -> ParamOptResult:
    """Exhaustive grid-search parameter optimization.

    *strategy_func(prices, params) -> metric_value* evaluates the strategy
    for a given parameter set.
    """
    if not prices or not param_grid:
        return ParamOptResult(optimization_metric=metric)

    results: List[Tuple[Dict[str, Any], float]] = []
    for params in param_grid:
        val = strategy_func(prices, params)
        results.append((dict(params), val))

    results.sort(key=lambda x: x[1], reverse=True)
    best = results[0] if results else ({}, 0.0)

    return ParamOptResult(
        best_params=best[0],
        best_metric=best[1],
        all_results=results[:top_n],
        total_combinations=len(param_grid),
        optimization_metric=metric,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  6. Strategy Robustness Testing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class RobustnessResult:
    """Result of strategy robustness testing."""

    base_return: float
    avg_perturbed_return: float
    return_std: float
    degradation_pct: float
    is_robust: bool
    param_sensitivity: Dict[str, float] = field(default_factory=dict)
    perturbed_results: List[float] = field(default_factory=list)


def evaluate_strategy_robustness(
    prices: Sequence[float],
    strategy_func: Callable[[Sequence[float], Dict[str, Any]], float],
    base_params: Dict[str, Any],
    *,
    perturbation_pct: float = 0.1,
    num_perturbations: int = 50,
    max_degradation: float = 0.3,
    seed: Optional[int] = None,
) -> RobustnessResult:
    """Test strategy robustness via parameter perturbation."""
    if not prices:
        return RobustnessResult(
            base_return=0.0,
            avg_perturbed_return=0.0,
            return_std=0.0,
            degradation_pct=0.0,
            is_robust=False,
        )

    rng = random.Random(seed)
    base_ret = strategy_func(prices, base_params)

    perturbed: List[float] = []
    sensitivities: Dict[str, List[float]] = {k: [] for k in base_params}

    for _ in range(num_perturbations):
        p = {}
        for k, v in base_params.items():
            try:
                fv = float(v)
                delta = fv * perturbation_pct * (2 * rng.random() - 1)
                p[k] = fv + delta
            except (TypeError, ValueError):
                p[k] = v
        ret = strategy_func(prices, p)
        perturbed.append(ret)

        # Per-param sensitivity
        for k, v in base_params.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            sp = dict(base_params)
            delta = fv * perturbation_pct
            sp[k] = fv + delta
            sr = strategy_func(prices, sp)
            sensitivities[k].append(abs(sr - base_ret))

    avg_perturbed = mean(perturbed) if perturbed else 0.0
    ret_std = pstdev(perturbed) if len(perturbed) >= 2 else 0.0
    degradation = (
        (base_ret - avg_perturbed) / abs(base_ret)
        if base_ret != 0
        else 0.0
    )
    param_sens = {
        k: mean(v) if v else 0.0 for k, v in sensitivities.items()
    }

    return RobustnessResult(
        base_return=base_ret,
        avg_perturbed_return=avg_perturbed,
        return_std=ret_std,
        degradation_pct=degradation,
        is_robust=abs(degradation) <= max_degradation,
        param_sensitivity=param_sens,
        perturbed_results=perturbed,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  7. Stress Testing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class StressScenario:
    """A single stress scenario and its result."""

    name: str
    description: str
    return_pct: float
    max_drawdown: float
    survived: bool


@dataclass
class StressTestResult:
    """Aggregate stress test results."""

    scenarios: List[StressScenario] = field(default_factory=list)
    scenarios_passed: int = 0
    scenarios_total: int = 0
    worst_return: float = 0.0
    worst_drawdown: float = 0.0
    is_resilient: bool = False


def run_stress_test(
    prices: Sequence[float],
    strategy_func: Callable[[Sequence[float]], float],
    *,
    crash_pcts: Sequence[float] = (-0.10, -0.20, -0.30, -0.50),
    spike_pcts: Sequence[float] = (0.10, 0.20, 0.50),
    max_acceptable_loss: float = -0.40,
) -> StressTestResult:
    """Run stress tests with crash and spike scenarios."""
    if len(prices) < 5:
        return StressTestResult()

    scenarios: List[StressScenario] = []
    base_prices = list(prices)
    mid = len(base_prices) // 2

    # Crash scenarios
    for pct in crash_pcts:
        stressed = list(base_prices)
        for i in range(mid, len(stressed)):
            stressed[i] = stressed[i] * (1.0 + pct)
        ret = strategy_func(stressed)
        eq = _equity_from_returns(_returns(stressed))
        mdd = _max_drawdown(eq)
        survived = ret > max_acceptable_loss
        scenarios.append(
            StressScenario(
                name=f"crash_{int(abs(pct)*100)}pct",
                description=f"Market crash of {pct*100:.0f}%",
                return_pct=ret,
                max_drawdown=mdd,
                survived=survived,
            )
        )

    # Spike scenarios
    for pct in spike_pcts:
        stressed = list(base_prices)
        for i in range(mid, len(stressed)):
            stressed[i] = stressed[i] * (1.0 + pct)
        ret = strategy_func(stressed)
        eq = _equity_from_returns(_returns(stressed))
        mdd = _max_drawdown(eq)
        scenarios.append(
            StressScenario(
                name=f"spike_{int(pct*100)}pct",
                description=f"Market spike of {pct*100:.0f}%",
                return_pct=ret,
                max_drawdown=mdd,
                survived=True,
            )
        )

    # High-vol scenario
    stressed = list(base_prices)
    for i in range(1, len(stressed)):
        noise = ((i % 3) - 1) * 0.05 * stressed[i]
        stressed[i] = max(stressed[i] + noise, 0.01)
    ret = strategy_func(stressed)
    eq = _equity_from_returns(_returns(stressed))
    mdd = _max_drawdown(eq)
    scenarios.append(
        StressScenario(
            name="high_volatility",
            description="Artificially elevated volatility",
            return_pct=ret,
            max_drawdown=mdd,
            survived=ret > max_acceptable_loss,
        )
    )

    passed = sum(1 for s in scenarios if s.survived)
    worst_ret = min(s.return_pct for s in scenarios) if scenarios else 0.0
    worst_dd = max(s.max_drawdown for s in scenarios) if scenarios else 0.0

    return StressTestResult(
        scenarios=scenarios,
        scenarios_passed=passed,
        scenarios_total=len(scenarios),
        worst_return=worst_ret,
        worst_drawdown=worst_dd,
        is_resilient=passed == len(scenarios),
    )


# ═══════════════════════════════════════════════════════════════════════════
#  8. Slippage Simulation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class SlippageModel:
    """Configuration for a slippage model."""

    fixed_bps: float = 5.0  # basis points
    volume_impact_factor: float = 0.1
    spread_bps: float = 10.0


@dataclass
class SlippageResult:
    """Slippage simulation result."""

    total_slippage: float
    avg_slippage_per_trade: float
    slippage_as_pct_of_pnl: float
    net_return_before_slippage: float
    net_return_after_slippage: float
    trade_slippages: List[float] = field(default_factory=list)


def simulate_slippage(
    trades: Sequence[BacktestTrade],
    volumes: Optional[Sequence[float]] = None,
    *,
    model: Optional[SlippageModel] = None,
    initial_capital: float = 10000.0,
) -> SlippageResult:
    """Apply slippage to a sequence of backtest trades."""
    if model is None:
        model = SlippageModel()

    if not trades:
        return SlippageResult(
            total_slippage=0.0,
            avg_slippage_per_trade=0.0,
            slippage_as_pct_of_pnl=0.0,
            net_return_before_slippage=0.0,
            net_return_after_slippage=0.0,
        )

    trade_slips: List[float] = []
    total_slip = 0.0
    gross_pnl = 0.0

    for idx, trade in enumerate(trades):
        price = trade.entry_price
        qty = trade.quantity

        # Fixed component
        fixed_slip = price * (model.fixed_bps / 10000.0) * qty

        # Spread component
        spread_slip = price * (model.spread_bps / 10000.0) * qty * 0.5

        # Volume impact
        vol_slip = 0.0
        if volumes and idx < len(volumes) and volumes[idx] > 0:
            trade_val = price * qty
            participation = trade_val / volumes[idx]
            vol_slip = price * model.volume_impact_factor * participation * qty

        slip = fixed_slip + spread_slip + vol_slip
        trade_slips.append(slip)
        total_slip += slip
        gross_pnl += trade.pnl

    avg_slip = total_slip / len(trades)
    net_before = gross_pnl / initial_capital
    net_after = (gross_pnl - total_slip) / initial_capital
    slip_pct = total_slip / abs(gross_pnl) if gross_pnl != 0 else 0.0

    return SlippageResult(
        total_slippage=total_slip,
        avg_slippage_per_trade=avg_slip,
        slippage_as_pct_of_pnl=slip_pct,
        net_return_before_slippage=net_before,
        net_return_after_slippage=net_after,
        trade_slippages=trade_slips,
    )


# ═══════════════════════════════════════════════════════════════════════════
#  9. Transaction Cost Modeling
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class TransactionCostModel:
    """Fee model configuration."""

    maker_fee_pct: float = 0.1
    taker_fee_pct: float = 0.15
    min_fee: float = 0.0
    tax_pct: float = 0.0


@dataclass
class TransactionCostResult:
    """Transaction cost analysis result."""

    total_fees: float
    total_tax: float
    total_cost: float
    avg_cost_per_trade: float
    cost_as_pct_of_pnl: float
    net_return_before_costs: float
    net_return_after_costs: float
    fee_breakdown: Dict[str, float] = field(default_factory=dict)


def model_transaction_costs(
    trades: Sequence[BacktestTrade],
    *,
    cost_model: Optional[TransactionCostModel] = None,
    initial_capital: float = 10000.0,
    maker_ratio: float = 0.5,
) -> TransactionCostResult:
    """Compute transaction costs for a set of trades."""
    if cost_model is None:
        cost_model = TransactionCostModel()

    if not trades:
        return TransactionCostResult(
            total_fees=0.0,
            total_tax=0.0,
            total_cost=0.0,
            avg_cost_per_trade=0.0,
            cost_as_pct_of_pnl=0.0,
            net_return_before_costs=0.0,
            net_return_after_costs=0.0,
        )

    total_maker = 0.0
    total_taker = 0.0
    total_tax = 0.0
    gross_pnl = 0.0

    for trade in trades:
        notional = trade.entry_price * trade.quantity
        exit_notional = trade.exit_price * trade.quantity
        avg_notional = (notional + exit_notional) / 2.0

        maker_fee = avg_notional * (cost_model.maker_fee_pct / 100.0) * maker_ratio
        taker_fee = avg_notional * (cost_model.taker_fee_pct / 100.0) * (1 - maker_ratio)
        fee = max(maker_fee + taker_fee, cost_model.min_fee)
        # Double for entry + exit
        fee *= 2.0

        tax = 0.0
        if trade.pnl > 0:
            tax = trade.pnl * (cost_model.tax_pct / 100.0)

        total_maker += maker_fee * 2.0
        total_taker += taker_fee * 2.0
        total_tax += tax
        gross_pnl += trade.pnl

    total_fees = total_maker + total_taker
    total_cost = total_fees + total_tax
    avg_cost = total_cost / len(trades)
    cost_pct = total_cost / abs(gross_pnl) if gross_pnl != 0 else 0.0
    net_before = gross_pnl / initial_capital
    net_after = (gross_pnl - total_cost) / initial_capital

    return TransactionCostResult(
        total_fees=total_fees,
        total_tax=total_tax,
        total_cost=total_cost,
        avg_cost_per_trade=avg_cost,
        cost_as_pct_of_pnl=cost_pct,
        net_return_before_costs=net_before,
        net_return_after_costs=net_after,
        fee_breakdown={
            "maker_fees": total_maker,
            "taker_fees": total_taker,
            "tax": total_tax,
        },
    )


# ═══════════════════════════════════════════════════════════════════════════
# 10. Liquidity Simulation
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class LiquidityProfile:
    """Market liquidity profile."""

    avg_volume: float
    min_volume: float
    max_volume: float
    volume_std: float
    illiquid_periods: int


@dataclass
class LiquiditySimResult:
    """Result of liquidity simulation."""

    fill_rate: float
    avg_fill_time: float
    partial_fills: int
    rejected_orders: int
    total_orders: int
    liquidity_profile: Optional[LiquidityProfile] = None
    adjusted_returns: float = 0.0


def simulate_liquidity(
    trades: Sequence[BacktestTrade],
    volumes: Sequence[float],
    *,
    max_participation: float = 0.10,
    min_volume_threshold: float = 100.0,
    initial_capital: float = 10000.0,
) -> LiquiditySimResult:
    """Simulate liquidity constraints on trade execution."""
    if not trades or not volumes:
        return LiquiditySimResult(
            fill_rate=0.0,
            avg_fill_time=0.0,
            partial_fills=0,
            rejected_orders=0,
            total_orders=0,
        )

    vol_list = list(volumes)
    filled = 0
    partial = 0
    rejected = 0
    fill_times: List[float] = []
    adj_pnl = 0.0

    for idx, trade in enumerate(trades):
        vol_idx = min(trade.entry_index, len(vol_list) - 1)
        available = vol_list[vol_idx] if vol_idx >= 0 else 0.0
        trade_volume = trade.entry_price * trade.quantity

        if available < min_volume_threshold:
            rejected += 1
            continue

        max_fill = available * max_participation
        if trade_volume <= max_fill:
            filled += 1
            fill_times.append(1.0)
            adj_pnl += trade.pnl
        else:
            partial += 1
            fill_ratio = max_fill / trade_volume
            fill_times.append(1.0 / fill_ratio if fill_ratio > 0 else 10.0)
            adj_pnl += trade.pnl * fill_ratio

    total = len(trades)
    fill_rate = (filled + partial) / total if total > 0 else 0.0

    # Liquidity profile
    avg_vol = mean(vol_list) if vol_list else 0.0
    min_vol = min(vol_list) if vol_list else 0.0
    max_vol = max(vol_list) if vol_list else 0.0
    vol_std = pstdev(vol_list) if len(vol_list) >= 2 else 0.0
    illiquid = sum(1 for v in vol_list if v < min_volume_threshold)

    profile = LiquidityProfile(
        avg_volume=avg_vol,
        min_volume=min_vol,
        max_volume=max_vol,
        volume_std=vol_std,
        illiquid_periods=illiquid,
    )

    return LiquiditySimResult(
        fill_rate=fill_rate,
        avg_fill_time=mean(fill_times) if fill_times else 0.0,
        partial_fills=partial,
        rejected_orders=rejected,
        total_orders=total,
        liquidity_profile=profile,
        adjusted_returns=adj_pnl / initial_capital,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 11. Out-of-Sample Testing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class OutOfSampleResult:
    """Result from out-of-sample testing."""

    in_sample_return: float
    out_of_sample_return: float
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    degradation_pct: float
    is_overfit: bool
    split_index: int


def run_out_of_sample_test(
    prices: Sequence[float],
    strategy_func: Callable[[Sequence[float]], float],
    *,
    split_ratio: float = 0.7,
    max_degradation: float = 0.5,
    sharpe_func: Optional[Callable[[Sequence[float]], float]] = None,
) -> OutOfSampleResult:
    """Split data into in-sample and out-of-sample and compare."""
    n = len(prices)
    if n < 10:
        return OutOfSampleResult(
            in_sample_return=0.0,
            out_of_sample_return=0.0,
            in_sample_sharpe=0.0,
            out_of_sample_sharpe=0.0,
            degradation_pct=0.0,
            is_overfit=False,
            split_index=0,
        )

    split = int(n * split_ratio)
    is_prices = list(prices[:split])
    oos_prices = list(prices[split:])

    is_ret = strategy_func(is_prices)
    oos_ret = strategy_func(oos_prices)

    is_rets = _returns(is_prices)
    oos_rets = _returns(oos_prices)
    is_sharpe = _sharpe(is_rets) if len(is_rets) >= 2 else 0.0
    oos_sharpe = _sharpe(oos_rets) if len(oos_rets) >= 2 else 0.0

    degradation = (
        (is_ret - oos_ret) / abs(is_ret) if is_ret != 0 else 0.0
    )

    return OutOfSampleResult(
        in_sample_return=is_ret,
        out_of_sample_return=oos_ret,
        in_sample_sharpe=is_sharpe,
        out_of_sample_sharpe=oos_sharpe,
        degradation_pct=degradation,
        is_overfit=degradation > max_degradation,
        split_index=split,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 12. Forward Testing
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class ForwardTestResult:
    """Result from forward / live-paper testing."""

    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    total_trades: int
    win_rate: float
    avg_trade_pnl: float
    equity_curve: List[float] = field(default_factory=list)
    signals_generated: int = 0
    signals_executed: int = 0
    execution_rate: float = 0.0


def run_forward_test(
    prices: Sequence[float],
    signal_func: Callable[[Sequence[float]], int],
    *,
    initial_capital: float = 10000.0,
    lookback: int = 20,
    execution_delay: int = 1,
) -> ForwardTestResult:
    """Simulate forward testing with execution delay.

    *signal_func(recent_prices) -> int* where 1 = buy, -1 = sell, 0 = hold.
    Execution is delayed by *execution_delay* bars.
    """
    n = len(prices)
    if n < lookback + execution_delay + 2:
        return ForwardTestResult(
            total_return=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            total_trades=0,
            win_rate=0.0,
            avg_trade_pnl=0.0,
            equity_curve=[initial_capital],
        )

    equity = [initial_capital]
    trades: List[BacktestTrade] = []
    position: Optional[dict] = None
    pending_signal: Optional[Tuple[int, int]] = None  # (bar, signal)
    signals_gen = 0
    signals_exec = 0

    for i in range(lookback, n):
        window = list(prices[max(0, i - lookback) : i + 1])
        sig = signal_func(window)
        if sig != 0:
            signals_gen += 1
            pending_signal = (i + execution_delay, sig)

        price = _safe_float(prices[i])
        prev_price = _safe_float(prices[i - 1]) if i > 0 else price

        # Execute pending signal
        exec_sig = 0
        if pending_signal and i >= pending_signal[0]:
            exec_sig = pending_signal[1]
            pending_signal = None
            signals_exec += 1

        # Close existing position on opposite signal
        if position is not None and exec_sig != 0 and (
            (position["side"] == "long" and exec_sig == -1)
            or (position["side"] == "short" and exec_sig == 1)
        ):
            ep = position["entry_price"]
            qty = position["qty"]
            if position["side"] == "long":
                pnl = (price - ep) * qty
            else:
                pnl = (ep - price) * qty
            ret_pct = pnl / (ep * qty) if (ep * qty) != 0 else 0.0
            trades.append(
                BacktestTrade(
                    entry_index=position["entry_idx"],
                    exit_index=i,
                    side=position["side"],
                    entry_price=ep,
                    exit_price=price,
                    quantity=qty,
                    pnl=pnl,
                    return_pct=ret_pct,
                )
            )
            position = None

        # Open new position
        if position is None and exec_sig in (1, -1) and price > 0:
            qty = equity[-1] / price
            position = {
                "side": "long" if exec_sig == 1 else "short",
                "entry_idx": i,
                "entry_price": price,
                "qty": qty,
            }

        # Mark-to-market
        if position is not None and prev_price > 0:
            qty = position["qty"]
            if position["side"] == "long":
                daily_pnl = (price - prev_price) * qty
            else:
                daily_pnl = (prev_price - price) * qty
            equity.append(equity[-1] + daily_pnl)
        else:
            equity.append(equity[-1])

    total_ret = (equity[-1] - initial_capital) / initial_capital
    mdd = _max_drawdown(equity)
    daily_rets = _returns(equity)
    sr = _sharpe(daily_rets)
    wins = [t for t in trades if t.pnl > 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_pnl = mean([t.pnl for t in trades]) if trades else 0.0
    exec_rate = signals_exec / signals_gen if signals_gen > 0 else 0.0

    return ForwardTestResult(
        total_return=total_ret,
        max_drawdown=mdd,
        sharpe_ratio=sr,
        total_trades=len(trades),
        win_rate=win_rate,
        avg_trade_pnl=avg_pnl,
        equity_curve=equity,
        signals_generated=signals_gen,
        signals_executed=signals_exec,
        execution_rate=exec_rate,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 13. Paper Trading Mode
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class PaperOrder:
    """A simulated paper-trading order."""

    order_id: int
    timestamp: float
    side: str  # "buy" | "sell"
    price: float
    quantity: float
    filled: bool = False
    fill_price: float = 0.0
    slippage: float = 0.0


@dataclass
class PaperTradingState:
    """State of the paper trading engine."""

    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    orders: List[PaperOrder] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    total_fees: float = 0.0
    equity_curve: List[float] = field(default_factory=list)


class PaperTradingEngine:
    """Simulated paper trading engine.

    Provides a realistic paper-trading environment with order management,
    position tracking, slippage and fee simulation.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        fee_pct: float = 0.1,
        slippage_bps: float = 5.0,
    ) -> None:
        self.initial_capital = initial_capital
        self.fee_pct = fee_pct
        self.slippage_bps = slippage_bps
        self._next_oid = 1

        self.state = PaperTradingState(
            cash=initial_capital,
            equity_curve=[initial_capital],
        )

    def submit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        timestamp: Optional[float] = None,
    ) -> PaperOrder:
        """Submit a paper order."""
        ts = timestamp if timestamp is not None else time.time()
        slip_pct = self.slippage_bps / 10000.0
        if side == "buy":
            fill_price = price * (1.0 + slip_pct)
        else:
            fill_price = price * (1.0 - slip_pct)

        cost = fill_price * quantity
        fee = cost * (self.fee_pct / 100.0)

        order = PaperOrder(
            order_id=self._next_oid,
            timestamp=ts,
            side=side,
            price=price,
            quantity=quantity,
            filled=False,
            fill_price=fill_price,
            slippage=abs(fill_price - price) * quantity,
        )
        self._next_oid += 1

        # Check if fillable
        if side == "buy":
            if self.state.cash >= cost + fee:
                self.state.cash -= cost + fee
                self.state.positions[symbol] = (
                    self.state.positions.get(symbol, 0.0) + quantity
                )
                order.filled = True
                self.state.total_fees += fee
            else:
                order.filled = False
        elif side == "sell":
            current_pos = self.state.positions.get(symbol, 0.0)
            if current_pos >= quantity:
                self.state.cash += cost - fee
                self.state.positions[symbol] = current_pos - quantity
                order.filled = True
                self.state.total_fees += fee
            else:
                order.filled = False

        self.state.orders.append(order)
        return order

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """Compute current portfolio value."""
        pos_value = sum(
            qty * prices.get(sym, 0.0)
            for sym, qty in self.state.positions.items()
        )
        return self.state.cash + pos_value

    def update_equity(self, prices: Dict[str, float]) -> float:
        """Update the equity curve."""
        value = self.get_portfolio_value(prices)
        self.state.equity_curve.append(value)
        return value

    def get_state(self) -> PaperTradingState:
        """Return current paper trading state."""
        self.state.total_pnl = (
            self.state.equity_curve[-1] - self.initial_capital
            if self.state.equity_curve
            else 0.0
        )
        return self.state

    def reset(self) -> None:
        """Reset the paper trading engine."""
        self._next_oid = 1
        self.state = PaperTradingState(
            cash=self.initial_capital,
            equity_curve=[self.initial_capital],
        )
