"""Tests for bot/execution.py – Smart Order Execution module.

Covers all 15 execution algorithm categories with focused tests for
each function and dataclass.
"""

from __future__ import annotations

import unittest
from typing import List, Sequence, Tuple

from bot.analysis import Candle
from bot.execution import (
    AdaptiveExecution,
    DMAOrder,
    ExecutionQuality,
    IcebergPlan,
    LatencyOptimization,
    LimitOrderPlan,
    LowLatencyPlan,
    MarketOrderPlan,
    OrderBatch,
    PartialFillPlan,
    RetryPlan,
    SlippageProtection,
    SmartRoute,
    TWAPPlan,
    VWAPPlan,
    analyze_slippage,
    batch_orders,
    create_dma_order,
    handle_partial_fill,
    monitor_execution_quality,
    optimize_latency,
    plan_adaptive_execution,
    plan_iceberg_order,
    plan_limit_order,
    plan_low_latency_execution,
    plan_market_order,
    plan_order_retry,
    plan_twap_execution,
    plan_vwap_execution,
    smart_order_route,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_candles(
    prices: Sequence[float] | None = None,
    volume: float = 100.0,
    spread: float = 10.0,
) -> List[Candle]:
    if prices is None:
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110]
    candles = []
    for i, p in enumerate(prices):
        candles.append(Candle(
            timestamp=1000 + i * 60,
            open=p - spread / 4,
            high=p + spread / 2,
            low=p - spread / 2,
            close=p,
            volume=volume,
        ))
    return candles


def _make_orderbook(
    mid: float = 100.0,
    spread: float = 0.2,
    levels: int = 5,
    qty_per_level: float = 10.0,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    bids = [(round(mid - spread / 2 - i * 0.1, 2), qty_per_level) for i in range(levels)]
    asks = [(round(mid + spread / 2 + i * 0.1, 2), qty_per_level) for i in range(levels)]
    return bids, asks


# ═══════════════════════════════════════════════════════════════════════════
#  1. Smart Order Routing
# ═══════════════════════════════════════════════════════════════════════════

class TestSmartOrderRouting(unittest.TestCase):
    def test_no_venues(self):
        result = smart_order_route(100, [])
        self.assertEqual(result.recommended_venue, "default")
        self.assertEqual(result.route_scores, {})

    def test_single_venue(self):
        venues = [{"name": "exchange_a", "liquidity": 1000, "fee": 0.001, "latency_ms": 50, "spread": 0.001}]
        result = smart_order_route(10, venues)
        self.assertEqual(result.recommended_venue, "exchange_a")
        self.assertIn("exchange_a", result.route_scores)

    def test_best_venue_selected(self):
        venues = [
            {"name": "fast", "liquidity": 500, "fee": 0.003, "latency_ms": 10, "spread": 0.002},
            {"name": "cheap", "liquidity": 2000, "fee": 0.0005, "latency_ms": 100, "spread": 0.0005},
        ]
        result = smart_order_route(50, venues, urgency=0.2)
        self.assertIsInstance(result, SmartRoute)
        self.assertIn(result.recommended_venue, {"fast", "cheap"})
        self.assertTrue(result.estimated_cost >= 0)

    def test_estimated_fill_time(self):
        venues = [{"name": "v", "liquidity": 100, "fee": 0.001, "latency_ms": 200, "spread": 0.001}]
        result = smart_order_route(10, venues)
        self.assertGreater(result.estimated_fill_time, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  2. Direct Market Access (DMA)
# ═══════════════════════════════════════════════════════════════════════════

class TestDMAOrder(unittest.TestCase):
    def test_buy_order_passes_checks(self):
        bids, asks = _make_orderbook(100, 0.2, 5, 100)
        result = create_dma_order("buy", 100.1, 5, bids, asks)
        self.assertTrue(result.passed_checks)
        self.assertEqual(result.side, "buy")

    def test_sell_order(self):
        bids, asks = _make_orderbook(100, 0.2, 5, 100)
        result = create_dma_order("sell", 99.9, 5, bids, asks)
        self.assertIsInstance(result, DMAOrder)
        self.assertEqual(result.side, "sell")

    def test_invalid_price_fails(self):
        bids, asks = _make_orderbook()
        result = create_dma_order("buy", -1, 10, bids, asks)
        self.assertFalse(result.passed_checks)

    def test_empty_orderbook(self):
        result = create_dma_order("buy", 100, 10, [], [])
        self.assertFalse(result.passed_checks)

    def test_excessive_size_fails(self):
        bids, asks = _make_orderbook(100, 0.2, 2, 5)
        result = create_dma_order("buy", 100.1, 100, bids, asks, max_order_pct=0.1)
        self.assertFalse(result.passed_checks)


# ═══════════════════════════════════════════════════════════════════════════
#  3. Low-Latency Order Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestLowLatencyExecution(unittest.TestCase):
    def test_basic_plan(self):
        result = plan_low_latency_execution()
        self.assertIsInstance(result, LowLatencyPlan)
        self.assertGreater(len(result.optimizations), 0)

    def test_batching_with_multiple_orders(self):
        result = plan_low_latency_execution(order_count=5, enable_batching=True)
        self.assertTrue(result.batch_mode)
        self.assertIn("order_batching", result.optimizations)

    def test_no_batching_single_order(self):
        result = plan_low_latency_execution(order_count=1, enable_batching=True)
        self.assertFalse(result.batch_mode)

    def test_low_latency_priority(self):
        result = plan_low_latency_execution(current_latency_ms=20.0)
        self.assertIn(result.priority_level, {"ultra_low", "low", "normal", "high"})


# ═══════════════════════════════════════════════════════════════════════════
#  4. Limit Order Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestLimitOrderExecution(unittest.TestCase):
    def test_buy_limit(self):
        candles = _make_candles()
        result = plan_limit_order("buy", 10, candles, 99.9, 100.1)
        self.assertIsInstance(result, LimitOrderPlan)
        self.assertEqual(result.side, "buy")
        self.assertGreater(result.price, 0)

    def test_sell_limit(self):
        candles = _make_candles()
        result = plan_limit_order("sell", 10, candles, 99.9, 100.1)
        self.assertEqual(result.side, "sell")
        self.assertGreater(result.price, 0)

    def test_aggressive_placement(self):
        candles = _make_candles()
        passive = plan_limit_order("buy", 10, candles, 99.9, 100.1, aggressiveness=0.1)
        aggressive = plan_limit_order("buy", 10, candles, 99.9, 100.1, aggressiveness=0.9)
        self.assertGreater(aggressive.price, passive.price)

    def test_no_mid_price(self):
        result = plan_limit_order("buy", 10, [], 0.0, 0.0)
        self.assertEqual(result.price, 0.0)


# ═══════════════════════════════════════════════════════════════════════════
#  5. Market Order Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestMarketOrderExecution(unittest.TestCase):
    def test_buy_market_order(self):
        bids, asks = _make_orderbook(100, 0.2, 5, 10)
        result = plan_market_order("buy", 5, bids, asks)
        self.assertIsInstance(result, MarketOrderPlan)
        self.assertEqual(result.side, "buy")
        self.assertGreater(result.expected_price, 0)

    def test_sell_market_order(self):
        bids, asks = _make_orderbook(100, 0.2, 5, 10)
        result = plan_market_order("sell", 5, bids, asks)
        self.assertEqual(result.side, "sell")

    def test_empty_orderbook(self):
        result = plan_market_order("buy", 10, [], [])
        self.assertEqual(result.expected_price, 0.0)

    def test_large_order_slippage(self):
        bids, asks = _make_orderbook(100, 0.2, 5, 2)
        result = plan_market_order("buy", 50, bids, asks)
        self.assertGreater(result.levels_consumed, 0)

    def test_should_split_large_order(self):
        bids, asks = _make_orderbook(100, 0.5, 10, 1)
        result = plan_market_order("buy", 100, bids, asks, max_slippage_pct=0.01)
        # With 10 levels of 1 qty each, consuming all should trigger split
        self.assertTrue(result.should_split or result.levels_consumed > 5)


# ═══════════════════════════════════════════════════════════════════════════
#  6. TWAP Execution Algorithm
# ═══════════════════════════════════════════════════════════════════════════

class TestTWAPExecution(unittest.TestCase):
    def test_basic_twap(self):
        result = plan_twap_execution(100, duration_minutes=60, num_slices=10)
        self.assertIsInstance(result, TWAPPlan)
        self.assertEqual(result.num_slices, 10)
        self.assertEqual(len(result.slices), 10)

    def test_slice_quantities_sum(self):
        result = plan_twap_execution(100, num_slices=5, randomize=False)
        total = sum(s["quantity"] for s in result.slices)
        self.assertAlmostEqual(total, 100, places=4)

    def test_randomized_slices_differ(self):
        result = plan_twap_execution(100, num_slices=10, randomize=True)
        quantities = [s["quantity"] for s in result.slices]
        # At least some slices should differ
        self.assertTrue(len(set(round(q, 4) for q in quantities)) > 1)

    def test_invalid_params(self):
        result = plan_twap_execution(0, num_slices=10)
        self.assertEqual(result.num_slices, 0)
        self.assertEqual(len(result.slices), 0)

    def test_interval_seconds(self):
        result = plan_twap_execution(100, duration_minutes=60, num_slices=6)
        self.assertAlmostEqual(result.interval_seconds, 600.0, places=1)


# ═══════════════════════════════════════════════════════════════════════════
#  7. VWAP Execution Algorithm
# ═══════════════════════════════════════════════════════════════════════════

class TestVWAPExecution(unittest.TestCase):
    def test_basic_vwap(self):
        candles = _make_candles()
        result = plan_vwap_execution(100, candles, num_slices=5)
        self.assertIsInstance(result, VWAPPlan)
        self.assertGreater(result.target_vwap, 0)
        self.assertEqual(len(result.slices), 5)

    def test_volume_weighted_slicing(self):
        # Create candles with varying volume
        prices = [100] * 10
        candles = []
        for i, p in enumerate(prices):
            vol = 200 if i < 5 else 50
            candles.append(Candle(timestamp=1000 + i * 60, open=p, high=p + 1, low=p - 1, close=p, volume=vol))
        result = plan_vwap_execution(100, candles, num_slices=2)
        # First slice should have more quantity due to higher volume
        if len(result.slices) == 2:
            self.assertGreater(result.slices[0]["quantity"], result.slices[1]["quantity"])

    def test_insufficient_data(self):
        result = plan_vwap_execution(100, [])
        self.assertEqual(result.num_slices, 0)

    def test_quantities_sum(self):
        candles = _make_candles()
        result = plan_vwap_execution(50, candles, num_slices=5)
        total = sum(s["quantity"] for s in result.slices)
        self.assertAlmostEqual(total, 50, places=4)


# ═══════════════════════════════════════════════════════════════════════════
#  8. Iceberg Order Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestIcebergOrder(unittest.TestCase):
    def test_basic_iceberg(self):
        result = plan_iceberg_order(1000, show_ratio=0.1, price=100.0)
        self.assertIsInstance(result, IcebergPlan)
        self.assertEqual(result.total_quantity, 1000)
        self.assertAlmostEqual(result.visible_quantity, 100, places=4)

    def test_hidden_quantity(self):
        result = plan_iceberg_order(1000, show_ratio=0.2)
        self.assertAlmostEqual(result.hidden_quantity, 800, places=4)

    def test_tranches_cover_full_order(self):
        result = plan_iceberg_order(100, show_ratio=0.25)
        total = sum(t["quantity"] for t in result.child_orders)
        self.assertAlmostEqual(total, 100, places=4)

    def test_invalid_params(self):
        result = plan_iceberg_order(0)
        self.assertEqual(result.num_tranches, 0)

    def test_full_visibility(self):
        result = plan_iceberg_order(100, show_ratio=1.0)
        self.assertEqual(result.num_tranches, 1)
        self.assertAlmostEqual(result.visible_quantity, 100, places=4)


# ═══════════════════════════════════════════════════════════════════════════
#  9. Adaptive Order Execution
# ═══════════════════════════════════════════════════════════════════════════

class TestAdaptiveExecution(unittest.TestCase):
    def test_basic_adaptive(self):
        candles = _make_candles()
        result = plan_adaptive_execution("buy", 10, candles, 99.9, 100.1)
        self.assertIsInstance(result, AdaptiveExecution)
        self.assertIn(result.strategy, {"balanced", "limit_with_chase", "aggressive_market", "passive_limit"})

    def test_high_urgency(self):
        candles = _make_candles()
        result = plan_adaptive_execution("buy", 10, candles, 99.9, 100.1, fill_urgency=0.9)
        self.assertGreaterEqual(result.aggression_level, 0.5)

    def test_wide_spread(self):
        candles = _make_candles()
        result = plan_adaptive_execution("buy", 10, candles, 95.0, 105.0, fill_urgency=0.5)
        self.assertIsNotNone(result.strategy)

    def test_no_candle_data(self):
        result = plan_adaptive_execution("sell", 10, [], 99.9, 100.1)
        self.assertIsInstance(result, AdaptiveExecution)


# ═══════════════════════════════════════════════════════════════════════════
# 10. Slippage Protection
# ═══════════════════════════════════════════════════════════════════════════

class TestSlippageProtection(unittest.TestCase):
    def test_safe_order(self):
        bids, asks = _make_orderbook(100, 0.2, 10, 100)
        result = analyze_slippage("buy", 5, bids, asks, max_slippage_pct=1.0)
        self.assertIsInstance(result, SlippageProtection)
        self.assertTrue(result.is_safe)

    def test_dangerous_order(self):
        bids, asks = _make_orderbook(100, 0.5, 3, 1)
        result = analyze_slippage("buy", 50, bids, asks, max_slippage_pct=0.01)
        # With only 3 qty available, large order = high slippage
        self.assertFalse(result.is_safe)

    def test_empty_orderbook(self):
        result = analyze_slippage("buy", 10, [], [])
        self.assertFalse(result.is_safe)

    def test_protections_listed(self):
        bids, asks = _make_orderbook(100, 0.5, 3, 2)
        result = analyze_slippage("buy", 20, bids, asks, max_slippage_pct=0.1)
        self.assertIsInstance(result.protections, list)

    def test_price_guard(self):
        bids, asks = _make_orderbook(100, 0.2, 5, 50)
        result = analyze_slippage("buy", 5, bids, asks, max_slippage_pct=0.5)
        self.assertGreater(result.price_guard, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 11. Partial Fill Handling
# ═══════════════════════════════════════════════════════════════════════════

class TestPartialFillHandling(unittest.TestCase):
    def test_accept_near_full_fill(self):
        result = handle_partial_fill(100, 95, 100.0, 100.5, 60)
        self.assertEqual(result.action, "accept")
        self.assertGreaterEqual(result.filled_pct, 0.9)

    def test_wait_early_fill(self):
        result = handle_partial_fill(100, 30, 100.0, 100.0, 30, max_wait_seconds=300)
        self.assertEqual(result.action, "wait")

    def test_cancel_replace_on_drift(self):
        result = handle_partial_fill(100, 30, 100.0, 105.0, 100, max_wait_seconds=300)
        self.assertEqual(result.action, "cancel_replace")
        self.assertIsNotNone(result.new_price)

    def test_market_remaining_on_timeout(self):
        result = handle_partial_fill(100, 20, 100.0, 101.0, 300, max_wait_seconds=300)
        self.assertEqual(result.action, "market_remaining")

    def test_remaining_quantity(self):
        result = handle_partial_fill(100, 60, 100.0, 100.0, 100)
        self.assertAlmostEqual(result.remaining_quantity, 40, places=4)


# ═══════════════════════════════════════════════════════════════════════════
# 12. Order Retry Mechanism
# ═══════════════════════════════════════════════════════════════════════════

class TestOrderRetry(unittest.TestCase):
    def test_retry_timeout(self):
        result = plan_order_retry("timeout", attempt_number=1, max_attempts=3)
        self.assertTrue(result.should_retry)
        self.assertGreater(result.delay_seconds, 0)

    def test_no_retry_insufficient_funds(self):
        result = plan_order_retry("insufficient_funds")
        self.assertFalse(result.should_retry)

    def test_no_retry_max_attempts(self):
        result = plan_order_retry("timeout", attempt_number=3, max_attempts=3)
        self.assertFalse(result.should_retry)

    def test_rate_limit_longer_delay(self):
        result = plan_order_retry("rate_limit", attempt_number=1)
        self.assertGreaterEqual(result.delay_seconds, 5.0)

    def test_exponential_backoff(self):
        r1 = plan_order_retry("rejected", attempt_number=1, base_delay_s=1.0, backoff_factor=2.0)
        r2 = plan_order_retry("rejected", attempt_number=2, base_delay_s=1.0, backoff_factor=2.0)
        self.assertGreater(r2.delay_seconds, r1.delay_seconds)

    def test_rejected_price_update(self):
        result = plan_order_retry("rejected", original_price=100.0, current_price=101.0)
        self.assertIn("update_price", result.adjustments)


# ═══════════════════════════════════════════════════════════════════════════
# 13. Execution Quality Monitoring
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutionQuality(unittest.TestCase):
    def test_perfect_execution(self):
        result = monitor_execution_quality(100.0, 100.0, 10, 10, 1.0)
        self.assertIsInstance(result, ExecutionQuality)
        self.assertAlmostEqual(result.slippage_bps, 0, places=1)
        self.assertAlmostEqual(result.fill_rate, 1.0, places=4)

    def test_high_slippage(self):
        result = monitor_execution_quality(100.0, 102.0, 10, 10, 5.0)
        self.assertGreater(result.slippage_bps, 0)
        self.assertLess(result.overall_score, 100)

    def test_partial_fill(self):
        result = monitor_execution_quality(100.0, 100.0, 10, 5, 1.0)
        self.assertAlmostEqual(result.fill_rate, 0.5, places=4)

    def test_invalid_price(self):
        result = monitor_execution_quality(0.0, 100.0, 10, 10, 1.0)
        self.assertEqual(result.overall_score, 0.0)

    def test_market_impact(self):
        result = monitor_execution_quality(100.0, 100.5, 10, 10, 2.0,
                                            pre_trade_mid=100.0, post_trade_mid=100.8)
        self.assertGreater(result.market_impact_pct, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 14. Order Batching
# ═══════════════════════════════════════════════════════════════════════════

class TestOrderBatching(unittest.TestCase):
    def test_empty_orders(self):
        result = batch_orders([])
        self.assertEqual(result.total_orders, 0)
        self.assertEqual(result.num_batches, 0)

    def test_group_by_side(self):
        orders = [
            {"side": "buy", "pair": "btc_idr", "quantity": 1, "price": 100},
            {"side": "sell", "pair": "btc_idr", "quantity": 2, "price": 101},
            {"side": "buy", "pair": "eth_idr", "quantity": 3, "price": 50},
        ]
        result = batch_orders(orders, batch_by="side")
        self.assertEqual(result.total_orders, 3)
        self.assertGreater(result.num_batches, 0)

    def test_max_batch_size(self):
        orders = [{"side": "buy", "pair": "btc_idr", "quantity": i, "price": 100} for i in range(10)]
        result = batch_orders(orders, max_batch_size=3, batch_by="none")
        for batch in result.batches:
            self.assertLessEqual(len(batch), 3)

    def test_savings_multiple_orders(self):
        orders = [{"side": "buy", "pair": "btc_idr", "quantity": 1, "price": 100} for _ in range(5)]
        result = batch_orders(orders)
        self.assertGreater(result.estimated_savings_pct, 0)


# ═══════════════════════════════════════════════════════════════════════════
# 15. Latency Optimization
# ═══════════════════════════════════════════════════════════════════════════

class TestLatencyOptimization(unittest.TestCase):
    def test_no_samples(self):
        result = optimize_latency([])
        self.assertEqual(result.current_latency_ms, 0.0)
        self.assertEqual(result.priority, "low")

    def test_basic_optimization(self):
        samples = [80, 85, 90, 75, 100, 95, 88]
        result = optimize_latency(samples)
        self.assertIsInstance(result, LatencyOptimization)
        self.assertGreater(result.current_latency_ms, 0)
        self.assertGreater(len(result.recommendations), 0)

    def test_high_latency_bottlenecks(self):
        samples = [200, 250, 180, 220, 190]
        result = optimize_latency(samples, network_latency_ms=150)
        self.assertTrue(any(b["type"] == "network" for b in result.bottlenecks))

    def test_improvement_percentage(self):
        samples = [100, 110, 90, 105, 95]
        result = optimize_latency(samples, network_latency_ms=100, processing_time_ms=30)
        self.assertGreaterEqual(result.estimated_improvement_pct, 0)
        self.assertLessEqual(result.optimized_latency_ms, result.current_latency_ms)

    def test_jitter_detection(self):
        samples = [50, 200, 30, 180, 40, 190, 35, 175, 45, 195]
        result = optimize_latency(samples)
        self.assertTrue(any(b["type"] == "jitter" for b in result.bottlenecks))


if __name__ == "__main__":
    unittest.main()
