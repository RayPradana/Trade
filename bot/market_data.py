"""Market data infrastructure for real-time and historical analysis.

Provides:
- Historical market data storage (HistoricalDataStore)
- Tick-level data processing (TickProcessor)
- Multi-exchange data aggregation (MultiExchangeAggregator)
- Orderbook depth analysis (DepthAnalyzer)
- Trade flow analysis (TradeFlowAnalyzer)
- Market microstructure analysis (MicrostructureAnalyzer)
- Liquidity monitoring (LiquidityMonitor)
- Volatility analysis (VolatilityAnalyzer)
- Spread monitoring (SpreadMonitor)
- Market regime detection (RegimeDetector)
- Cross-exchange price comparison (CrossExchangeComparator)
- Latency monitoring (LatencyMonitor)
- Data integrity validation (DataIntegrityValidator)
- Market anomaly detection (AnomalyDetector)
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Tick:
    """A single market tick (trade execution)."""
    timestamp: float
    price: float
    amount: float
    side: str  # "buy" or "sell"
    pair: str = ""


@dataclass
class DepthSnapshot:
    """Point-in-time orderbook snapshot."""
    timestamp: float
    bids: List[Tuple[float, float]]  # (price, amount)
    asks: List[Tuple[float, float]]
    pair: str = ""


@dataclass
class DepthMetrics:
    """Computed orderbook depth metrics."""
    bid_depth_idr: float
    ask_depth_idr: float
    imbalance: float  # (bid - ask) / (bid + ask), range -1..+1
    bid_levels: int
    ask_levels: int
    weighted_mid_price: float
    bid_wall_price: Optional[float] = None
    ask_wall_price: Optional[float] = None


@dataclass
class TradeFlowMetrics:
    """Metrics from trade flow analysis."""
    buy_volume: float
    sell_volume: float
    buy_count: int
    sell_count: int
    buy_ratio: float  # buy_volume / total_volume
    net_flow: float  # buy_volume - sell_volume
    vwap: float  # volume-weighted average price
    large_trade_count: int  # trades above threshold


@dataclass
class MicrostructureMetrics:
    """Market microstructure analysis results."""
    effective_spread: float
    realized_spread: float
    price_impact: float
    kyle_lambda: float  # price impact per unit volume
    order_flow_toxicity: float  # 0..1
    tick_direction_ratio: float  # fraction of upticks


@dataclass
class LiquidityMetrics:
    """Real-time liquidity monitoring results."""
    bid_liquidity_idr: float
    ask_liquidity_idr: float
    total_liquidity_idr: float
    liquidity_score: float  # 0..1 (1 = deep, 0 = thin)
    resilience: float  # how fast the book recovers
    concentration: float  # how spread out the liquidity is


@dataclass
class VolatilityMetrics:
    """Enhanced volatility analysis results."""
    realized_vol: float
    parkinson_vol: float  # high-low based estimator
    garman_klass_vol: float
    yang_zhang_vol: float  # drift-independent estimator
    vol_of_vol: float  # volatility clustering
    vol_regime: str  # "low", "normal", "high", "extreme"


@dataclass
class SpreadMetrics:
    """Spread monitoring results."""
    current_spread: float
    current_spread_pct: float
    avg_spread: float
    avg_spread_pct: float
    spread_volatility: float
    spread_z_score: float  # std deviations from mean
    is_wide: bool


@dataclass
class RegimeState:
    """Market regime detection result."""
    regime: str  # "trending_up", "trending_down", "ranging", "volatile", "quiet"
    confidence: float
    duration_seconds: float
    avg_volume: float
    volatility_level: str  # "low", "normal", "high"


@dataclass
class CrossExchangePrice:
    """Price from a single exchange."""
    exchange: str
    price: float
    volume_24h: float
    timestamp: float


@dataclass
class CrossExchangeComparison:
    """Cross-exchange price comparison result."""
    prices: List[CrossExchangePrice]
    max_spread_pct: float  # (max_price - min_price) / min_price
    arbitrage_opportunity: bool
    reference_price: float  # volume-weighted average


@dataclass
class LatencyStats:
    """API latency monitoring results."""
    last_latency_ms: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    total_requests: int
    error_count: int
    timeout_count: int


@dataclass
class DataIntegrityReport:
    """Data integrity validation result."""
    is_valid: bool
    issues: List[str]
    stale_pairs: List[str]
    gap_count: int
    duplicate_count: int
    out_of_order_count: int


@dataclass
class AnomalyAlert:
    """Market anomaly detection alert."""
    detected: bool
    anomaly_type: str  # "price_spike", "volume_spike", "spread_blowout", "flash_crash", "stale_data"
    severity: str  # "low", "medium", "high", "critical"
    pair: str
    description: str
    value: float
    threshold: float
    timestamp: float


# ---------------------------------------------------------------------------
# Historical Data Store
# ---------------------------------------------------------------------------

class HistoricalDataStore:
    """Persistent storage for tick-level and candle data.

    Stores data as JSON-lines files on disk, organised by pair and date.
    """

    def __init__(
        self,
        data_dir: str = "market_data",
        max_ticks_memory: int = 100_000,
        max_candles_memory: int = 10_000,
        flush_interval: int = 300,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_ticks_memory = max_ticks_memory
        self.max_candles_memory = max_candles_memory
        self.flush_interval = flush_interval
        self._ticks: Dict[str, Deque[Tick]] = {}
        self._candles: Dict[str, Deque[dict]] = {}
        self._last_flush: float = time.time()

    def add_tick(self, tick: Tick) -> None:
        """Store a tick in memory and flush to disk periodically."""
        pair = tick.pair
        if pair not in self._ticks:
            self._ticks[pair] = deque(maxlen=self.max_ticks_memory)
        self._ticks[pair].append(tick)
        if time.time() - self._last_flush >= self.flush_interval:
            self.flush()

    def add_candle(self, pair: str, candle: dict) -> None:
        """Store a candle dict in memory."""
        if pair not in self._candles:
            self._candles[pair] = deque(maxlen=self.max_candles_memory)
        self._candles[pair].append(candle)

    def get_ticks(
        self, pair: str, since: float = 0.0, limit: int = 0,
    ) -> List[Tick]:
        """Return ticks for *pair* since *since* timestamp."""
        buf = self._ticks.get(pair, deque())
        result = [t for t in buf if t.timestamp >= since]
        if limit > 0:
            result = result[-limit:]
        return result

    def get_candles(
        self, pair: str, since: float = 0.0, limit: int = 0,
    ) -> List[dict]:
        """Return candle dicts for *pair*."""
        buf = self._candles.get(pair, deque())
        result = [c for c in buf if c.get("timestamp", 0) >= since]
        if limit > 0:
            result = result[-limit:]
        return result

    def flush(self) -> None:
        """Flush in-memory ticks to disk as JSON-lines."""
        for pair, ticks in self._ticks.items():
            if not ticks:
                continue
            safe_pair = pair.replace("/", "_").replace("\\", "_")
            fpath = self.data_dir / f"{safe_pair}_ticks.jsonl"
            try:
                with open(fpath, "a") as f:
                    for t in ticks:
                        json.dump(
                            {
                                "ts": t.timestamp,
                                "p": t.price,
                                "a": t.amount,
                                "s": t.side,
                            },
                            f,
                        )
                        f.write("\n")
            except OSError as exc:
                logger.warning("Failed to flush ticks for %s: %s", pair, exc)
        self._last_flush = time.time()

    def pair_count(self) -> int:
        """Return the number of pairs with stored ticks."""
        return len(self._ticks)

    def tick_count(self, pair: str) -> int:
        """Return the number of ticks stored for *pair*."""
        return len(self._ticks.get(pair, deque()))


# ---------------------------------------------------------------------------
# Tick Processor
# ---------------------------------------------------------------------------

class TickProcessor:
    """Process tick-level data into actionable metrics.

    Maintains a rolling window per pair and computes statistics
    such as VWAP, tick-direction ratios, and trade imbalance.
    """

    def __init__(
        self,
        window_seconds: float = 60.0,
        max_ticks: int = 10_000,
        large_trade_threshold_idr: float = 10_000_000.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.max_ticks = max_ticks
        self.large_trade_threshold_idr = large_trade_threshold_idr
        self._ticks: Dict[str, Deque[Tick]] = {}

    def add_tick(self, tick: Tick) -> None:
        """Ingest a new tick and prune old ones."""
        pair = tick.pair
        if pair not in self._ticks:
            self._ticks[pair] = deque(maxlen=self.max_ticks)
        self._ticks[pair].append(tick)
        self._prune(pair)

    def _prune(self, pair: str) -> None:
        buf = self._ticks.get(pair, deque())
        if not buf:
            return
        cutoff = time.time() - self.window_seconds
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()

    def compute_vwap(self, pair: str) -> float:
        """Volume-weighted average price over the rolling window."""
        buf = self._ticks.get(pair, deque())
        if not buf:
            return 0.0
        total_pv = sum(t.price * t.amount for t in buf)
        total_v = sum(t.amount for t in buf)
        return total_pv / total_v if total_v > 0 else 0.0

    def compute_trade_flow(self, pair: str) -> TradeFlowMetrics:
        """Compute trade flow metrics from the rolling tick window."""
        buf = self._ticks.get(pair, deque())
        buy_vol = 0.0
        sell_vol = 0.0
        buy_count = 0
        sell_count = 0
        large_count = 0
        total_pv = 0.0
        total_v = 0.0

        for t in buf:
            trade_idr = t.price * t.amount
            total_pv += trade_idr
            total_v += t.amount
            if t.side == "buy":
                buy_vol += t.amount
                buy_count += 1
            else:
                sell_vol += t.amount
                sell_count += 1
            if trade_idr >= self.large_trade_threshold_idr:
                large_count += 1

        total_vol = buy_vol + sell_vol
        buy_ratio = buy_vol / total_vol if total_vol > 0 else 0.5
        vwap = total_pv / total_v if total_v > 0 else 0.0

        return TradeFlowMetrics(
            buy_volume=buy_vol,
            sell_volume=sell_vol,
            buy_count=buy_count,
            sell_count=sell_count,
            buy_ratio=round(buy_ratio, 4),
            net_flow=round(buy_vol - sell_vol, 8),
            vwap=round(vwap, 2),
            large_trade_count=large_count,
        )

    def tick_count(self, pair: str) -> int:
        """Return the number of ticks in the window for *pair*."""
        return len(self._ticks.get(pair, deque()))

    def compute_tick_direction_ratio(self, pair: str) -> float:
        """Fraction of ticks that are upticks (price higher than previous)."""
        buf = list(self._ticks.get(pair, deque()))
        if len(buf) < 2:
            return 0.5
        upticks = sum(
            1 for i in range(1, len(buf)) if buf[i].price > buf[i - 1].price
        )
        return round(upticks / (len(buf) - 1), 4)


# ---------------------------------------------------------------------------
# Multi-Exchange Data Aggregator
# ---------------------------------------------------------------------------

class MultiExchangeAggregator:
    """Aggregate price data from multiple exchanges.

    Each exchange feed pushes price updates; the aggregator computes
    a reference (volume-weighted) price and detects arbitrage spreads.
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 60.0,
        arbitrage_min_spread_pct: float = 0.005,
    ) -> None:
        self.stale_threshold_seconds = stale_threshold_seconds
        self.arbitrage_min_spread_pct = arbitrage_min_spread_pct
        # pair → exchange → CrossExchangePrice
        self._prices: Dict[str, Dict[str, CrossExchangePrice]] = {}

    def update_price(
        self,
        pair: str,
        exchange: str,
        price: float,
        volume_24h: float = 0.0,
    ) -> None:
        """Update the latest price for *pair* on *exchange*."""
        if pair not in self._prices:
            self._prices[pair] = {}
        self._prices[pair][exchange] = CrossExchangePrice(
            exchange=exchange,
            price=price,
            volume_24h=volume_24h,
            timestamp=time.time(),
        )

    def compare(self, pair: str) -> CrossExchangeComparison:
        """Compare prices across exchanges for *pair*."""
        now = time.time()
        entries = self._prices.get(pair, {})
        fresh = [
            ep
            for ep in entries.values()
            if (now - ep.timestamp) < self.stale_threshold_seconds
        ]
        if len(fresh) < 2:
            ref_price = fresh[0].price if fresh else 0.0
            return CrossExchangeComparison(
                prices=fresh,
                max_spread_pct=0.0,
                arbitrage_opportunity=False,
                reference_price=ref_price,
            )

        prices_sorted = sorted(fresh, key=lambda x: x.price)
        min_p = prices_sorted[0].price
        max_p = prices_sorted[-1].price
        spread_pct = (max_p - min_p) / min_p if min_p > 0 else 0.0

        # Volume-weighted reference
        total_vp = sum(ep.price * ep.volume_24h for ep in fresh)
        total_v = sum(ep.volume_24h for ep in fresh)
        ref_price = total_vp / total_v if total_v > 0 else statistics.mean(
            ep.price for ep in fresh
        )

        return CrossExchangeComparison(
            prices=fresh,
            max_spread_pct=round(spread_pct, 6),
            arbitrage_opportunity=spread_pct >= self.arbitrage_min_spread_pct,
            reference_price=round(ref_price, 2),
        )

    def exchange_count(self, pair: str) -> int:
        """Number of exchanges with data for *pair*."""
        return len(self._prices.get(pair, {}))


# ---------------------------------------------------------------------------
# Depth Analyzer
# ---------------------------------------------------------------------------

class DepthAnalyzer:
    """Analyse orderbook depth snapshots.

    Computes metrics like depth imbalance, weighted mid-price, wall
    detection, and depth at various price levels.
    """

    def __init__(
        self,
        wall_threshold_multiplier: float = 5.0,
        depth_levels: int = 20,
    ) -> None:
        self.wall_threshold_multiplier = wall_threshold_multiplier
        self.depth_levels = depth_levels

    def analyze(self, snapshot: DepthSnapshot) -> DepthMetrics:
        """Compute depth metrics from a snapshot."""
        bids = snapshot.bids[: self.depth_levels]
        asks = snapshot.asks[: self.depth_levels]

        bid_depth = sum(p * a for p, a in bids)
        ask_depth = sum(p * a for p, a in asks)
        total = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total if total > 0 else 0.0

        # Weighted mid-price
        best_bid = bids[0][0] if bids else 0.0
        best_ask = asks[0][0] if asks else 0.0
        if best_bid > 0 and best_ask > 0:
            bid_size = bids[0][1] if bids else 0.0
            ask_size = asks[0][1] if asks else 0.0
            total_size = bid_size + ask_size
            if total_size > 0:
                wmid = (best_bid * ask_size + best_ask * bid_size) / total_size
            else:
                wmid = (best_bid + best_ask) / 2
        else:
            wmid = best_bid or best_ask

        # Wall detection
        bid_wall = self._detect_wall(bids)
        ask_wall = self._detect_wall(asks)

        return DepthMetrics(
            bid_depth_idr=round(bid_depth, 2),
            ask_depth_idr=round(ask_depth, 2),
            imbalance=round(imbalance, 4),
            bid_levels=len(bids),
            ask_levels=len(asks),
            weighted_mid_price=round(wmid, 2),
            bid_wall_price=bid_wall,
            ask_wall_price=ask_wall,
        )

    def _detect_wall(
        self, levels: List[Tuple[float, float]],
    ) -> Optional[float]:
        """Return the price of a wall (outsized level) if found."""
        if len(levels) < 3:
            return None
        amounts = [a for _, a in levels]
        avg = statistics.mean(amounts)
        if avg <= 0:
            return None
        for price, amount in levels:
            if amount >= avg * self.wall_threshold_multiplier:
                return price
        return None


# ---------------------------------------------------------------------------
# Trade Flow Analyzer
# ---------------------------------------------------------------------------

class TradeFlowAnalyzer:
    """Analyse trade flow patterns for buy/sell pressure and large-order activity."""

    def __init__(
        self,
        window_seconds: float = 300.0,
        large_trade_multiplier: float = 5.0,
    ) -> None:
        self.window_seconds = window_seconds
        self.large_trade_multiplier = large_trade_multiplier
        self._trades: Dict[str, Deque[Tick]] = {}

    def add_trade(self, tick: Tick) -> None:
        """Ingest a trade."""
        pair = tick.pair
        if pair not in self._trades:
            self._trades[pair] = deque(maxlen=50_000)
        self._trades[pair].append(tick)

    def _window(self, pair: str) -> List[Tick]:
        buf = self._trades.get(pair, deque())
        cutoff = time.time() - self.window_seconds
        return [t for t in buf if t.timestamp >= cutoff]

    def analyze(self, pair: str) -> TradeFlowMetrics:
        """Compute trade flow metrics for *pair* within the rolling window."""
        window = self._window(pair)
        if not window:
            return TradeFlowMetrics(
                buy_volume=0, sell_volume=0, buy_count=0, sell_count=0,
                buy_ratio=0.5, net_flow=0, vwap=0, large_trade_count=0,
            )

        buy_vol = sell_vol = 0.0
        buy_count = sell_count = large_count = 0
        total_pv = total_v = 0.0

        avg_size = statistics.mean(t.amount for t in window) if window else 0.0
        large_thresh = avg_size * self.large_trade_multiplier

        for t in window:
            total_pv += t.price * t.amount
            total_v += t.amount
            if t.side == "buy":
                buy_vol += t.amount
                buy_count += 1
            else:
                sell_vol += t.amount
                sell_count += 1
            if t.amount >= large_thresh:
                large_count += 1

        total = buy_vol + sell_vol
        return TradeFlowMetrics(
            buy_volume=round(buy_vol, 8),
            sell_volume=round(sell_vol, 8),
            buy_count=buy_count,
            sell_count=sell_count,
            buy_ratio=round(buy_vol / total if total > 0 else 0.5, 4),
            net_flow=round(buy_vol - sell_vol, 8),
            vwap=round(total_pv / total_v if total_v > 0 else 0, 2),
            large_trade_count=large_count,
        )


# ---------------------------------------------------------------------------
# Market Microstructure Analyzer
# ---------------------------------------------------------------------------

class MicrostructureAnalyzer:
    """Analyze market microstructure from tick and orderbook data.

    Computes effective spread, realized spread, price impact,
    Kyle's lambda (price impact per unit volume), and order-flow toxicity.
    """

    def __init__(self, lookback_ticks: int = 100) -> None:
        self.lookback_ticks = lookback_ticks

    def analyze(
        self,
        ticks: Sequence[Tick],
        mid_prices: Sequence[float],
    ) -> MicrostructureMetrics:
        """Compute microstructure metrics.

        :param ticks: Recent ticks (most recent last).
        :param mid_prices: Mid-prices corresponding to each tick's time.
        """
        n = min(len(ticks), len(mid_prices), self.lookback_ticks)
        if n < 2:
            return MicrostructureMetrics(
                effective_spread=0.0, realized_spread=0.0,
                price_impact=0.0, kyle_lambda=0.0,
                order_flow_toxicity=0.0, tick_direction_ratio=0.5,
            )
        recent_ticks = list(ticks[-n:])
        recent_mids = list(mid_prices[-n:])

        # Effective spread: 2 * |trade_price - mid_price|
        eff_spreads = [
            2 * abs(t.price - m) for t, m in zip(recent_ticks, recent_mids)
        ]
        avg_eff_spread = statistics.mean(eff_spreads)

        # Tick direction ratio
        upticks = sum(
            1 for i in range(1, len(recent_ticks))
            if recent_ticks[i].price > recent_ticks[i - 1].price
        )
        tick_dir_ratio = upticks / (len(recent_ticks) - 1)

        # Price impact: change in mid-price after a trade
        impacts = [
            abs(recent_mids[i] - recent_mids[i - 1])
            for i in range(1, len(recent_mids))
        ]
        avg_impact = statistics.mean(impacts) if impacts else 0.0

        # Kyle's lambda: regression of price change on signed volume
        signed_volumes = []
        price_changes = []
        for i in range(1, len(recent_ticks)):
            sign = 1.0 if recent_ticks[i].side == "buy" else -1.0
            signed_volumes.append(sign * recent_ticks[i].amount)
            price_changes.append(recent_mids[i] - recent_mids[i - 1])

        kyle_lambda = 0.0
        if signed_volumes and any(sv != 0 for sv in signed_volumes):
            sv_mean = statistics.mean(signed_volumes)
            pc_mean = statistics.mean(price_changes)
            numerator = sum(
                (sv - sv_mean) * (pc - pc_mean)
                for sv, pc in zip(signed_volumes, price_changes)
            )
            denominator = sum((sv - sv_mean) ** 2 for sv in signed_volumes)
            kyle_lambda = numerator / denominator if denominator != 0 else 0.0

        # Order flow toxicity: fraction of trades on the "wrong" side of mid
        toxic_count = sum(
            1
            for t, m in zip(recent_ticks, recent_mids)
            if (t.side == "buy" and t.price > m)
            or (t.side == "sell" and t.price < m)
        )
        toxicity = toxic_count / len(recent_ticks) if recent_ticks else 0.0

        # Realized spread (simplified): effective spread minus price impact
        realized = max(0.0, avg_eff_spread - avg_impact)

        return MicrostructureMetrics(
            effective_spread=round(avg_eff_spread, 6),
            realized_spread=round(realized, 6),
            price_impact=round(avg_impact, 6),
            kyle_lambda=round(kyle_lambda, 8),
            order_flow_toxicity=round(toxicity, 4),
            tick_direction_ratio=round(tick_dir_ratio, 4),
        )


# ---------------------------------------------------------------------------
# Liquidity Monitor
# ---------------------------------------------------------------------------

class LiquidityMonitor:
    """Monitor real-time liquidity conditions across pairs.

    Tracks depth, spread, and trade frequency to produce a
    composite liquidity score.
    """

    def __init__(
        self,
        min_depth_idr: float = 50_000_000.0,
        min_trade_frequency: float = 1.0,
        history_size: int = 100,
    ) -> None:
        self.min_depth_idr = min_depth_idr
        self.min_trade_frequency = min_trade_frequency
        self.history_size = history_size
        self._depth_history: Dict[str, Deque[float]] = {}
        self._last_trade_ts: Dict[str, float] = {}
        self._trade_counts: Dict[str, Deque[float]] = {}

    def update_depth(self, pair: str, total_depth_idr: float) -> None:
        """Record a depth snapshot for *pair*."""
        if pair not in self._depth_history:
            self._depth_history[pair] = deque(maxlen=self.history_size)
        self._depth_history[pair].append(total_depth_idr)

    def record_trade(self, pair: str) -> None:
        """Record that a trade occurred for *pair*."""
        now = time.time()
        if pair not in self._trade_counts:
            self._trade_counts[pair] = deque(maxlen=self.history_size)
        self._trade_counts[pair].append(now)
        self._last_trade_ts[pair] = now

    def assess(self, pair: str) -> LiquidityMetrics:
        """Compute a liquidity assessment for *pair*."""
        depths = self._depth_history.get(pair, deque())
        current_depth = depths[-1] if depths else 0.0
        avg_depth = statistics.mean(depths) if depths else 0.0

        # Score from 0-1 based on depth relative to minimum
        depth_score = min(1.0, current_depth / self.min_depth_idr) if self.min_depth_idr > 0 else 1.0

        # Resilience: how stable the depth is
        if len(depths) >= 2:
            vol_depth = statistics.stdev(depths) / statistics.mean(depths) if statistics.mean(depths) > 0 else 1.0
            resilience = max(0.0, 1.0 - vol_depth)
        else:
            resilience = 0.5

        # Concentration: how evenly distributed (placeholder using depth history)
        concentration = 0.5  # default
        if depths:
            max_d = max(depths)
            concentration = 1.0 - (max_d / sum(depths)) if sum(depths) > 0 else 0.0

        # Trade frequency
        trade_ts = self._trade_counts.get(pair, deque())
        now = time.time()
        recent_trades = [ts for ts in trade_ts if now - ts < 60]
        freq = len(recent_trades) / 60.0 if recent_trades else 0.0
        freq_score = min(1.0, freq / self.min_trade_frequency) if self.min_trade_frequency > 0 else 1.0

        liquidity_score = round(0.6 * depth_score + 0.2 * resilience + 0.2 * freq_score, 4)

        return LiquidityMetrics(
            bid_liquidity_idr=round(current_depth * 0.5, 2),
            ask_liquidity_idr=round(current_depth * 0.5, 2),
            total_liquidity_idr=round(current_depth, 2),
            liquidity_score=liquidity_score,
            resilience=round(resilience, 4),
            concentration=round(concentration, 4),
        )


# ---------------------------------------------------------------------------
# Volatility Analyzer
# ---------------------------------------------------------------------------

class VolatilityAnalyzer:
    """Enhanced volatility analysis using multiple estimators.

    Supports realized, Parkinson, Garman-Klass, and Yang-Zhang estimators,
    plus volatility-of-volatility and regime classification.
    """

    def __init__(
        self,
        window: int = 20,
        vol_low_threshold: float = 0.005,
        vol_high_threshold: float = 0.03,
        vol_extreme_threshold: float = 0.06,
    ) -> None:
        self.window = window
        self.vol_low = vol_low_threshold
        self.vol_high = vol_high_threshold
        self.vol_extreme = vol_extreme_threshold

    def analyze(
        self,
        closes: Sequence[float],
        highs: Optional[Sequence[float]] = None,
        lows: Optional[Sequence[float]] = None,
        opens: Optional[Sequence[float]] = None,
    ) -> VolatilityMetrics:
        """Compute volatility metrics from price series."""
        n = len(closes)
        if n < 2:
            return VolatilityMetrics(
                realized_vol=0.0, parkinson_vol=0.0, garman_klass_vol=0.0,
                yang_zhang_vol=0.0, vol_of_vol=0.0, vol_regime="low",
            )

        window_closes = list(closes[-self.window:]) if n > self.window else list(closes)

        # Realized volatility (close-to-close)
        returns = [
            math.log(window_closes[i] / window_closes[i - 1])
            for i in range(1, len(window_closes))
            if window_closes[i - 1] > 0 and window_closes[i] > 0
        ]
        realized = statistics.stdev(returns) if len(returns) >= 2 else 0.0

        # Parkinson (high-low)
        parkinson = 0.0
        if highs and lows and len(highs) >= 2 and len(lows) >= 2:
            w_highs = list(highs[-self.window:])
            w_lows = list(lows[-self.window:])
            hl_logs = [
                math.log(h / l) ** 2
                for h, l in zip(w_highs, w_lows)
                if h > 0 and l > 0 and h >= l
            ]
            if hl_logs:
                parkinson = math.sqrt(sum(hl_logs) / (4 * len(hl_logs) * math.log(2)))

        # Garman-Klass
        gk_vol = 0.0
        if highs and lows and opens and len(opens) >= 2:
            w_opens = list(opens[-self.window:])
            w_highs = list(highs[-self.window:])
            w_lows = list(lows[-self.window:])
            w_closes = window_closes
            min_len = min(len(w_opens), len(w_highs), len(w_lows), len(w_closes))
            gk_sum = 0.0
            gk_count = 0
            for i in range(min_len):
                o, h, l, c = w_opens[i], w_highs[i], w_lows[i], w_closes[i]
                if o > 0 and h > 0 and l > 0 and c > 0 and h >= l:
                    hl = math.log(h / l) ** 2
                    co = math.log(c / o) ** 2
                    gk_sum += 0.5 * hl - (2 * math.log(2) - 1) * co
                    gk_count += 1
            if gk_count > 0:
                gk_vol = math.sqrt(gk_sum / gk_count)

        # Yang-Zhang (simplified: uses overnight + Rogers-Satchell + open-close)
        yz_vol = 0.0
        if opens and len(opens) >= 3 and len(window_closes) >= 3:
            w_opens = list(opens[-self.window:])
            oc_returns = [
                math.log(w_opens[i] / window_closes[i - 1])
                for i in range(1, min(len(w_opens), len(window_closes)))
                if w_opens[i] > 0 and window_closes[i - 1] > 0
            ]
            if len(oc_returns) >= 2:
                overnight_var = statistics.variance(oc_returns)
                yz_vol = math.sqrt(overnight_var + realized ** 2)

        # Volatility of volatility
        vol_of_vol = 0.0
        if len(returns) >= 6:
            half = len(returns) // 2
            sub_vols = []
            for start in range(0, len(returns) - half + 1, max(1, half // 2)):
                chunk = returns[start : start + half]
                if len(chunk) >= 2:
                    sub_vols.append(statistics.stdev(chunk))
            if len(sub_vols) >= 2:
                vol_of_vol = statistics.stdev(sub_vols)

        # Regime classification
        if realized >= self.vol_extreme:
            regime = "extreme"
        elif realized >= self.vol_high:
            regime = "high"
        elif realized >= self.vol_low:
            regime = "normal"
        else:
            regime = "low"

        return VolatilityMetrics(
            realized_vol=round(realized, 6),
            parkinson_vol=round(parkinson, 6),
            garman_klass_vol=round(gk_vol, 6),
            yang_zhang_vol=round(yz_vol, 6),
            vol_of_vol=round(vol_of_vol, 6),
            vol_regime=regime,
        )


# ---------------------------------------------------------------------------
# Spread Monitor
# ---------------------------------------------------------------------------

class SpreadMonitor:
    """Monitor bid-ask spread conditions for anomaly detection.

    Tracks spread history per pair and flags widening or abnormal spreads.
    """

    def __init__(
        self,
        history_size: int = 200,
        z_score_threshold: float = 2.5,
        wide_spread_pct: float = 0.005,
    ) -> None:
        self.history_size = history_size
        self.z_score_threshold = z_score_threshold
        self.wide_spread_pct = wide_spread_pct
        self._spreads: Dict[str, Deque[float]] = {}

    def record(self, pair: str, spread_pct: float) -> None:
        """Record a spread observation."""
        if pair not in self._spreads:
            self._spreads[pair] = deque(maxlen=self.history_size)
        self._spreads[pair].append(spread_pct)

    def assess(self, pair: str, current_spread_pct: float) -> SpreadMetrics:
        """Assess the current spread for *pair*."""
        history = list(self._spreads.get(pair, deque()))
        if not history:
            return SpreadMetrics(
                current_spread=current_spread_pct,
                current_spread_pct=current_spread_pct,
                avg_spread=current_spread_pct,
                avg_spread_pct=current_spread_pct,
                spread_volatility=0.0,
                spread_z_score=0.0,
                is_wide=current_spread_pct >= self.wide_spread_pct,
            )
        avg = statistics.mean(history)
        std = statistics.stdev(history) if len(history) >= 2 else 0.0
        z_score = (current_spread_pct - avg) / std if std > 0 else 0.0

        return SpreadMetrics(
            current_spread=current_spread_pct,
            current_spread_pct=current_spread_pct,
            avg_spread=round(avg, 6),
            avg_spread_pct=round(avg, 6),
            spread_volatility=round(std, 6),
            spread_z_score=round(z_score, 2),
            is_wide=current_spread_pct >= self.wide_spread_pct or z_score >= self.z_score_threshold,
        )


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """Detect the current market regime from price and volume data.

    Classifies the market into: trending_up, trending_down, ranging,
    volatile, or quiet based on recent price action and volume.
    """

    def __init__(
        self,
        lookback: int = 50,
        trend_threshold: float = 0.02,
        vol_threshold: float = 0.03,
    ) -> None:
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        self.vol_threshold = vol_threshold
        self._regime_start: Dict[str, float] = {}
        self._last_regime: Dict[str, str] = {}

    def detect(
        self,
        pair: str,
        closes: Sequence[float],
        volumes: Sequence[float],
    ) -> RegimeState:
        """Detect the current regime for *pair*."""
        if len(closes) < 3:
            return RegimeState(
                regime="quiet", confidence=0.0, duration_seconds=0.0,
                avg_volume=0.0, volatility_level="low",
            )

        window = list(closes[-self.lookback:])
        vol_window = list(volumes[-self.lookback:]) if volumes else []

        # Trend: simple slope over the window
        if window[0] > 0:
            total_return = (window[-1] - window[0]) / window[0]
        else:
            total_return = 0.0

        # Volatility
        returns = [
            (window[i] - window[i - 1]) / window[i - 1]
            for i in range(1, len(window))
            if window[i - 1] > 0
        ]
        vol = statistics.stdev(returns) if len(returns) >= 2 else 0.0
        avg_vol = statistics.mean(vol_window) if vol_window else 0.0

        # Regime classification
        if vol >= self.vol_threshold:
            regime = "volatile"
            confidence = min(1.0, vol / self.vol_threshold)
        elif total_return > self.trend_threshold:
            regime = "trending_up"
            confidence = min(1.0, total_return / self.trend_threshold)
        elif total_return < -self.trend_threshold:
            regime = "trending_down"
            confidence = min(1.0, abs(total_return) / self.trend_threshold)
        elif vol < self.vol_threshold * 0.3:
            regime = "quiet"
            confidence = 0.8
        else:
            regime = "ranging"
            confidence = 0.6

        # Track duration
        now = time.time()
        prev = self._last_regime.get(pair)
        if prev != regime:
            self._regime_start[pair] = now
        self._last_regime[pair] = regime
        duration = now - self._regime_start.get(pair, now)

        # Volatility level
        if vol >= self.vol_threshold:
            vol_level = "high"
        elif vol >= self.vol_threshold * 0.5:
            vol_level = "normal"
        else:
            vol_level = "low"

        return RegimeState(
            regime=regime,
            confidence=round(confidence, 4),
            duration_seconds=round(duration, 2),
            avg_volume=round(avg_vol, 2),
            volatility_level=vol_level,
        )


# ---------------------------------------------------------------------------
# Cross-Exchange Comparator
# ---------------------------------------------------------------------------

class CrossExchangeComparator:
    """Compare prices across exchanges for arbitrage and reference pricing.

    Wraps MultiExchangeAggregator with additional analysis capabilities.
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 60.0,
        significant_spread_pct: float = 0.005,
    ) -> None:
        self._aggregator = MultiExchangeAggregator(
            stale_threshold_seconds=stale_threshold_seconds,
            arbitrage_min_spread_pct=significant_spread_pct,
        )
        self._spread_history: Dict[str, Deque[float]] = {}

    def update(
        self,
        pair: str,
        exchange: str,
        price: float,
        volume_24h: float = 0.0,
    ) -> None:
        """Push a price update from *exchange*."""
        self._aggregator.update_price(pair, exchange, price, volume_24h)

    def compare(self, pair: str) -> CrossExchangeComparison:
        """Get cross-exchange comparison."""
        result = self._aggregator.compare(pair)
        if pair not in self._spread_history:
            self._spread_history[pair] = deque(maxlen=200)
        self._spread_history[pair].append(result.max_spread_pct)
        return result

    def avg_spread(self, pair: str) -> float:
        """Average cross-exchange spread."""
        hist = self._spread_history.get(pair, deque())
        return round(statistics.mean(hist), 6) if hist else 0.0


# ---------------------------------------------------------------------------
# Latency Monitor
# ---------------------------------------------------------------------------

class LatencyMonitor:
    """Monitor API request latency for health and performance tracking."""

    def __init__(self, max_history: int = 1000) -> None:
        self.max_history = max_history
        self._latencies: Deque[float] = deque(maxlen=max_history)
        self._error_count: int = 0
        self._timeout_count: int = 0
        self._total_requests: int = 0

    def record(self, latency_ms: float) -> None:
        """Record a successful request latency."""
        self._latencies.append(latency_ms)
        self._total_requests += 1

    def record_error(self) -> None:
        """Record a request error."""
        self._error_count += 1
        self._total_requests += 1

    def record_timeout(self) -> None:
        """Record a request timeout."""
        self._timeout_count += 1
        self._total_requests += 1

    def stats(self) -> LatencyStats:
        """Compute latency statistics."""
        if not self._latencies:
            return LatencyStats(
                last_latency_ms=0, avg_latency_ms=0, p95_latency_ms=0,
                p99_latency_ms=0, max_latency_ms=0, total_requests=self._total_requests,
                error_count=self._error_count, timeout_count=self._timeout_count,
            )
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        p95_idx = min(int(n * 0.95), n - 1)
        p99_idx = min(int(n * 0.99), n - 1)
        return LatencyStats(
            last_latency_ms=round(sorted_lat[-1], 2),
            avg_latency_ms=round(statistics.mean(sorted_lat), 2),
            p95_latency_ms=round(sorted_lat[p95_idx], 2),
            p99_latency_ms=round(sorted_lat[p99_idx], 2),
            max_latency_ms=round(max(sorted_lat), 2),
            total_requests=self._total_requests,
            error_count=self._error_count,
            timeout_count=self._timeout_count,
        )


# ---------------------------------------------------------------------------
# Data Integrity Validator
# ---------------------------------------------------------------------------

class DataIntegrityValidator:
    """Validate market data for staleness, gaps, duplicates, and ordering."""

    def __init__(
        self,
        stale_threshold_seconds: float = 120.0,
        max_gap_seconds: float = 300.0,
    ) -> None:
        self.stale_threshold_seconds = stale_threshold_seconds
        self.max_gap_seconds = max_gap_seconds
        self._last_update: Dict[str, float] = {}

    def record_update(self, pair: str) -> None:
        """Record that fresh data was received for *pair*."""
        self._last_update[pair] = time.time()

    def validate_ticks(
        self,
        pair: str,
        ticks: Sequence[Tick],
    ) -> DataIntegrityReport:
        """Validate a sequence of ticks for integrity issues."""
        issues: List[str] = []
        gaps = 0
        duplicates = 0
        out_of_order = 0

        for i in range(1, len(ticks)):
            # Ordering
            if ticks[i].timestamp < ticks[i - 1].timestamp:
                out_of_order += 1
            # Gaps
            gap = ticks[i].timestamp - ticks[i - 1].timestamp
            if gap > self.max_gap_seconds:
                gaps += 1
            # Duplicates
            if (
                ticks[i].timestamp == ticks[i - 1].timestamp
                and ticks[i].price == ticks[i - 1].price
                and ticks[i].amount == ticks[i - 1].amount
            ):
                duplicates += 1

        # Staleness check
        stale_pairs: List[str] = []
        now = time.time()
        last_update = self._last_update.get(pair, 0.0)
        if last_update > 0 and (now - last_update) > self.stale_threshold_seconds:
            stale_pairs.append(pair)
            issues.append(f"stale_data: {pair} not updated for {now - last_update:.0f}s")

        if gaps > 0:
            issues.append(f"gaps: {gaps} gaps > {self.max_gap_seconds}s")
        if duplicates > 0:
            issues.append(f"duplicates: {duplicates}")
        if out_of_order > 0:
            issues.append(f"out_of_order: {out_of_order}")

        # Price sanity
        if ticks:
            prices = [t.price for t in ticks if t.price > 0]
            if prices:
                median_p = statistics.median(prices)
                for t in ticks:
                    if t.price > 0 and abs(t.price - median_p) / median_p > 0.5:
                        issues.append(f"price_outlier: {t.price} vs median {median_p:.2f}")
                        break

        return DataIntegrityReport(
            is_valid=len(issues) == 0,
            issues=issues,
            stale_pairs=stale_pairs,
            gap_count=gaps,
            duplicate_count=duplicates,
            out_of_order_count=out_of_order,
        )

    def check_staleness(self, pairs: Sequence[str]) -> List[str]:
        """Return list of pairs whose data is stale."""
        now = time.time()
        stale = []
        for pair in pairs:
            ts = self._last_update.get(pair, 0.0)
            if ts > 0 and (now - ts) > self.stale_threshold_seconds:
                stale.append(pair)
        return stale


# ---------------------------------------------------------------------------
# Market Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """Detect market anomalies such as price spikes, volume surges,
    spread blowouts, and flash crashes.
    """

    def __init__(
        self,
        price_spike_threshold: float = 0.05,
        volume_spike_threshold: float = 5.0,
        spread_blowout_threshold: float = 3.0,
        flash_crash_threshold: float = 0.10,
        window_size: int = 100,
    ) -> None:
        self.price_spike_threshold = price_spike_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.spread_blowout_threshold = spread_blowout_threshold
        self.flash_crash_threshold = flash_crash_threshold
        self.window_size = window_size
        self._price_history: Dict[str, Deque[float]] = {}
        self._volume_history: Dict[str, Deque[float]] = {}
        self._spread_history: Dict[str, Deque[float]] = {}

    def record(
        self,
        pair: str,
        price: float,
        volume: float = 0.0,
        spread_pct: float = 0.0,
    ) -> None:
        """Record market data for anomaly detection."""
        for store, val in [
            (self._price_history, price),
            (self._volume_history, volume),
            (self._spread_history, spread_pct),
        ]:
            if pair not in store:
                store[pair] = deque(maxlen=self.window_size)
            store[pair].append(val)

    def detect(self, pair: str) -> List[AnomalyAlert]:
        """Detect anomalies for *pair* from recorded data."""
        alerts: List[AnomalyAlert] = []
        now = time.time()

        # Price spike / flash crash
        prices = list(self._price_history.get(pair, deque()))
        if len(prices) >= 3:
            recent = prices[-1]
            prev = prices[-2]
            if prev > 0:
                change = (recent - prev) / prev
                if change >= self.price_spike_threshold:
                    alerts.append(AnomalyAlert(
                        detected=True, anomaly_type="price_spike",
                        severity="high" if change >= self.price_spike_threshold * 2 else "medium",
                        pair=pair,
                        description=f"Price spiked {change:.2%}",
                        value=change, threshold=self.price_spike_threshold,
                        timestamp=now,
                    ))
                if change <= -self.flash_crash_threshold:
                    alerts.append(AnomalyAlert(
                        detected=True, anomaly_type="flash_crash",
                        severity="critical",
                        pair=pair,
                        description=f"Flash crash {change:.2%}",
                        value=abs(change), threshold=self.flash_crash_threshold,
                        timestamp=now,
                    ))

        # Volume spike
        volumes = list(self._volume_history.get(pair, deque()))
        if len(volumes) >= 5:
            avg_vol = statistics.mean(volumes[:-1])
            if avg_vol > 0:
                ratio = volumes[-1] / avg_vol
                if ratio >= self.volume_spike_threshold:
                    alerts.append(AnomalyAlert(
                        detected=True, anomaly_type="volume_spike",
                        severity="medium",
                        pair=pair,
                        description=f"Volume {ratio:.1f}× average",
                        value=ratio, threshold=self.volume_spike_threshold,
                        timestamp=now,
                    ))

        # Spread blowout
        spreads = list(self._spread_history.get(pair, deque()))
        if len(spreads) >= 5:
            avg_spread = statistics.mean(spreads[:-1])
            if avg_spread > 0:
                ratio = spreads[-1] / avg_spread
                if ratio >= self.spread_blowout_threshold:
                    alerts.append(AnomalyAlert(
                        detected=True, anomaly_type="spread_blowout",
                        severity="high",
                        pair=pair,
                        description=f"Spread {ratio:.1f}× average",
                        value=ratio, threshold=self.spread_blowout_threshold,
                        timestamp=now,
                    ))

        return alerts


# ---------------------------------------------------------------------------
# Composite MarketDataFeed
# ---------------------------------------------------------------------------

class MarketDataFeed:
    """Composite manager that wires together all market-data components.

    Typical usage::

        feed = MarketDataFeed.from_config(config)
        feed.on_tick(tick)
        feed.on_depth(depth_snapshot)
        alerts = feed.check_anomalies("btc_idr")
    """

    def __init__(
        self,
        *,
        historical_store: Optional[HistoricalDataStore] = None,
        tick_processor: Optional[TickProcessor] = None,
        multi_exchange: Optional[MultiExchangeAggregator] = None,
        depth_analyzer: Optional[DepthAnalyzer] = None,
        trade_flow_analyzer: Optional[TradeFlowAnalyzer] = None,
        microstructure: Optional[MicrostructureAnalyzer] = None,
        liquidity_monitor: Optional[LiquidityMonitor] = None,
        volatility_analyzer: Optional[VolatilityAnalyzer] = None,
        spread_monitor: Optional[SpreadMonitor] = None,
        regime_detector: Optional[RegimeDetector] = None,
        cross_exchange: Optional[CrossExchangeComparator] = None,
        latency_monitor: Optional[LatencyMonitor] = None,
        integrity_validator: Optional[DataIntegrityValidator] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
    ) -> None:
        self.historical_store = historical_store or HistoricalDataStore()
        self.tick_processor = tick_processor or TickProcessor()
        self.multi_exchange = multi_exchange or MultiExchangeAggregator()
        self.depth_analyzer = depth_analyzer or DepthAnalyzer()
        self.trade_flow_analyzer = trade_flow_analyzer or TradeFlowAnalyzer()
        self.microstructure = microstructure or MicrostructureAnalyzer()
        self.liquidity_monitor = liquidity_monitor or LiquidityMonitor()
        self.volatility_analyzer = volatility_analyzer or VolatilityAnalyzer()
        self.spread_monitor = spread_monitor or SpreadMonitor()
        self.regime_detector = regime_detector or RegimeDetector()
        self.cross_exchange = cross_exchange or CrossExchangeComparator()
        self.latency_monitor = latency_monitor or LatencyMonitor()
        self.integrity_validator = integrity_validator or DataIntegrityValidator()
        self.anomaly_detector = anomaly_detector or AnomalyDetector()

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Any) -> "MarketDataFeed":
        """Create a *MarketDataFeed* from a :class:`BotConfig`."""
        data_dir = getattr(config, "market_data_dir", "market_data")
        max_ticks = getattr(config, "market_data_max_ticks", 100_000)
        tick_window = getattr(config, "market_data_tick_window", 60.0)
        large_trade_idr = getattr(config, "market_data_large_trade_idr", 10_000_000.0)
        stale_sec = getattr(config, "market_data_stale_seconds", 120.0)
        anomaly_price = getattr(config, "market_data_anomaly_price_pct", 0.05)
        anomaly_volume = getattr(config, "market_data_anomaly_volume_mult", 5.0)
        anomaly_spread = getattr(config, "market_data_anomaly_spread_mult", 3.0)
        anomaly_crash = getattr(config, "market_data_anomaly_crash_pct", 0.10)

        return cls(
            historical_store=HistoricalDataStore(
                data_dir=data_dir,
                max_ticks_memory=max_ticks,
            ),
            tick_processor=TickProcessor(
                window_seconds=tick_window,
                max_ticks=max_ticks,
                large_trade_threshold_idr=large_trade_idr,
            ),
            depth_analyzer=DepthAnalyzer(),
            trade_flow_analyzer=TradeFlowAnalyzer(),
            microstructure=MicrostructureAnalyzer(),
            liquidity_monitor=LiquidityMonitor(),
            volatility_analyzer=VolatilityAnalyzer(),
            spread_monitor=SpreadMonitor(),
            regime_detector=RegimeDetector(),
            cross_exchange=CrossExchangeComparator(),
            latency_monitor=LatencyMonitor(),
            integrity_validator=DataIntegrityValidator(
                stale_threshold_seconds=stale_sec,
            ),
            anomaly_detector=AnomalyDetector(
                price_spike_threshold=anomaly_price,
                volume_spike_threshold=anomaly_volume,
                spread_blowout_threshold=anomaly_spread,
                flash_crash_threshold=anomaly_crash,
            ),
        )

    # ── Event handlers ────────────────────────────────────────────────────

    def on_tick(self, tick: Tick) -> None:
        """Process an incoming tick through all components."""
        self.historical_store.add_tick(tick)
        self.tick_processor.add_tick(tick)
        self.trade_flow_analyzer.add_trade(tick)
        self.liquidity_monitor.record_trade(tick.pair)
        self.integrity_validator.record_update(tick.pair)
        self.anomaly_detector.record(
            tick.pair, tick.price, tick.amount,
        )

    def on_depth(self, snapshot: DepthSnapshot) -> DepthMetrics:
        """Process an orderbook depth snapshot."""
        metrics = self.depth_analyzer.analyze(snapshot)
        self.liquidity_monitor.update_depth(
            snapshot.pair, metrics.bid_depth_idr + metrics.ask_depth_idr,
        )
        if metrics.bid_depth_idr + metrics.ask_depth_idr > 0:
            best_bid = snapshot.bids[0][0] if snapshot.bids else 0
            best_ask = snapshot.asks[0][0] if snapshot.asks else 0
            if best_bid > 0 and best_ask > 0:
                spread_pct = (best_ask - best_bid) / best_bid
                self.spread_monitor.record(snapshot.pair, spread_pct)
                self.anomaly_detector.record(
                    snapshot.pair, (best_bid + best_ask) / 2,
                    spread_pct=spread_pct,
                )
        return metrics

    def on_exchange_price(
        self, pair: str, exchange: str, price: float, volume_24h: float = 0,
    ) -> None:
        """Record a price update from an external exchange."""
        self.cross_exchange.update(pair, exchange, price, volume_24h)

    def record_latency(self, latency_ms: float) -> None:
        """Record an API request latency."""
        self.latency_monitor.record(latency_ms)

    def check_anomalies(self, pair: str) -> List[AnomalyAlert]:
        """Check for market anomalies on *pair*."""
        return self.anomaly_detector.detect(pair)

    def check_integrity(self, pair: str) -> DataIntegrityReport:
        """Validate data integrity for *pair*."""
        ticks = self.historical_store.get_ticks(pair)
        return self.integrity_validator.validate_ticks(pair, ticks)

    def get_latency_stats(self) -> LatencyStats:
        """Get current latency statistics."""
        return self.latency_monitor.stats()

    def get_liquidity(self, pair: str) -> LiquidityMetrics:
        """Get liquidity assessment for *pair*."""
        return self.liquidity_monitor.assess(pair)

    def get_spread_metrics(self, pair: str, current_spread_pct: float) -> SpreadMetrics:
        """Get spread assessment for *pair*."""
        return self.spread_monitor.assess(pair, current_spread_pct)

    def get_regime(
        self, pair: str, closes: Sequence[float], volumes: Sequence[float],
    ) -> RegimeState:
        """Detect market regime for *pair*."""
        return self.regime_detector.detect(pair, closes, volumes)

    def get_cross_exchange(self, pair: str) -> CrossExchangeComparison:
        """Get cross-exchange comparison for *pair*."""
        return self.cross_exchange.compare(pair)

    def get_trade_flow(self, pair: str) -> TradeFlowMetrics:
        """Get trade flow metrics for *pair*."""
        return self.trade_flow_analyzer.analyze(pair)
