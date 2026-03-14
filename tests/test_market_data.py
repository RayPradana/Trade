"""Tests for bot.market_data module."""
from __future__ import annotations

import time

import pytest

from bot.market_data import (
    AnomalyDetector,
    CrossExchangeComparator,
    DataIntegrityValidator,
    DepthAnalyzer,
    DepthSnapshot,
    HistoricalDataStore,
    LatencyMonitor,
    LiquidityMonitor,
    MarketDataFeed,
    MicrostructureAnalyzer,
    MultiExchangeAggregator,
    RegimeDetector,
    SpreadMonitor,
    Tick,
    TickProcessor,
    TradeFlowAnalyzer,
    VolatilityAnalyzer,
)


# ---------------------------------------------------------------------------
# HistoricalDataStore
# ---------------------------------------------------------------------------

class TestHistoricalDataStore:
    def test_add_and_get_ticks(self, tmp_path):
        store = HistoricalDataStore(data_dir=str(tmp_path / "data"))
        tick = Tick(timestamp=1000, price=100, amount=1.0, side="buy", pair="btc_idr")
        store.add_tick(tick)
        ticks = store.get_ticks("btc_idr")
        assert len(ticks) == 1
        assert ticks[0].price == 100

    def test_get_ticks_since(self, tmp_path):
        store = HistoricalDataStore(data_dir=str(tmp_path / "data"))
        store.add_tick(Tick(timestamp=100, price=50, amount=1, side="buy", pair="x"))
        store.add_tick(Tick(timestamp=200, price=60, amount=1, side="sell", pair="x"))
        store.add_tick(Tick(timestamp=300, price=70, amount=1, side="buy", pair="x"))
        assert len(store.get_ticks("x", since=200)) == 2
        assert len(store.get_ticks("x", since=250)) == 1

    def test_get_ticks_limit(self, tmp_path):
        store = HistoricalDataStore(data_dir=str(tmp_path / "data"))
        for i in range(10):
            store.add_tick(Tick(timestamp=i, price=i, amount=1, side="buy", pair="p"))
        assert len(store.get_ticks("p", limit=3)) == 3

    def test_add_candle(self, tmp_path):
        store = HistoricalDataStore(data_dir=str(tmp_path / "data"))
        store.add_candle("btc_idr", {"timestamp": 1000, "close": 100})
        candles = store.get_candles("btc_idr")
        assert len(candles) == 1
        assert candles[0]["close"] == 100

    def test_flush_to_disk(self, tmp_path):
        store = HistoricalDataStore(data_dir=str(tmp_path / "data"))
        store.add_tick(Tick(timestamp=1, price=10, amount=1, side="buy", pair="btc_idr"))
        store.flush()
        files = list((tmp_path / "data").glob("*.jsonl"))
        assert len(files) == 1

    def test_pair_count_and_tick_count(self, tmp_path):
        store = HistoricalDataStore(data_dir=str(tmp_path / "data"))
        store.add_tick(Tick(timestamp=1, price=10, amount=1, side="buy", pair="a"))
        store.add_tick(Tick(timestamp=2, price=20, amount=1, side="sell", pair="b"))
        assert store.pair_count() == 2
        assert store.tick_count("a") == 1
        assert store.tick_count("b") == 1


# ---------------------------------------------------------------------------
# TickProcessor
# ---------------------------------------------------------------------------

class TestTickProcessor:
    def test_compute_vwap(self):
        tp = TickProcessor(window_seconds=9999)
        now = time.time()
        tp.add_tick(Tick(timestamp=now, price=100, amount=2, side="buy", pair="p"))
        tp.add_tick(Tick(timestamp=now, price=200, amount=1, side="sell", pair="p"))
        # VWAP = (100*2 + 200*1) / (2+1) = 400/3 ≈ 133.33
        vwap = tp.compute_vwap("p")
        assert abs(vwap - 133.33) < 0.1

    def test_compute_trade_flow(self):
        tp = TickProcessor(window_seconds=9999)
        now = time.time()
        tp.add_tick(Tick(timestamp=now, price=100, amount=3, side="buy", pair="p"))
        tp.add_tick(Tick(timestamp=now, price=100, amount=1, side="sell", pair="p"))
        flow = tp.compute_trade_flow("p")
        assert flow.buy_volume == 3
        assert flow.sell_volume == 1
        assert flow.buy_ratio == 0.75
        assert flow.buy_count == 1
        assert flow.sell_count == 1

    def test_empty_pair(self):
        tp = TickProcessor()
        assert tp.compute_vwap("x") == 0.0
        flow = tp.compute_trade_flow("x")
        assert flow.buy_ratio == 0.5

    def test_tick_direction_ratio(self):
        tp = TickProcessor(window_seconds=9999)
        now = time.time()
        tp.add_tick(Tick(timestamp=now, price=100, amount=1, side="buy", pair="p"))
        tp.add_tick(Tick(timestamp=now, price=110, amount=1, side="buy", pair="p"))
        tp.add_tick(Tick(timestamp=now, price=120, amount=1, side="buy", pair="p"))
        # 2 upticks out of 2 transitions = 1.0
        assert tp.compute_tick_direction_ratio("p") == 1.0


# ---------------------------------------------------------------------------
# MultiExchangeAggregator
# ---------------------------------------------------------------------------

class TestMultiExchangeAggregator:
    def test_single_exchange(self):
        agg = MultiExchangeAggregator()
        agg.update_price("btc_idr", "indodax", 500_000_000, 100)
        result = agg.compare("btc_idr")
        assert result.reference_price == 500_000_000
        assert not result.arbitrage_opportunity

    def test_arbitrage_detection(self):
        agg = MultiExchangeAggregator(arbitrage_min_spread_pct=0.005)
        agg.update_price("btc_idr", "exA", 1000, 100)
        agg.update_price("btc_idr", "exB", 1100, 100)
        result = agg.compare("btc_idr")
        assert result.max_spread_pct > 0.005
        assert result.arbitrage_opportunity

    def test_no_arbitrage(self):
        agg = MultiExchangeAggregator(arbitrage_min_spread_pct=0.01)
        agg.update_price("btc_idr", "exA", 1000, 100)
        agg.update_price("btc_idr", "exB", 1005, 100)
        result = agg.compare("btc_idr")
        assert not result.arbitrage_opportunity


# ---------------------------------------------------------------------------
# DepthAnalyzer
# ---------------------------------------------------------------------------

class TestDepthAnalyzer:
    def test_basic_analysis(self):
        da = DepthAnalyzer()
        snap = DepthSnapshot(
            timestamp=time.time(),
            bids=[(100, 10), (99, 5), (98, 3)],
            asks=[(101, 8), (102, 4), (103, 2)],
            pair="btc_idr",
        )
        metrics = da.analyze(snap)
        assert metrics.bid_levels == 3
        assert metrics.ask_levels == 3
        assert metrics.bid_depth_idr > 0
        assert metrics.ask_depth_idr > 0
        assert -1 <= metrics.imbalance <= 1

    def test_weighted_mid_price(self):
        da = DepthAnalyzer()
        snap = DepthSnapshot(
            timestamp=time.time(),
            bids=[(100, 10)],
            asks=[(102, 10)],
            pair="p",
        )
        metrics = da.analyze(snap)
        assert metrics.weighted_mid_price == 101  # symmetric

    def test_wall_detection(self):
        da = DepthAnalyzer(wall_threshold_multiplier=3.0)
        snap = DepthSnapshot(
            timestamp=time.time(),
            bids=[(100, 1), (99, 1), (98, 1), (97, 50)],  # wall at 97
            asks=[(101, 1), (102, 1), (103, 1)],
            pair="p",
        )
        metrics = da.analyze(snap)
        assert metrics.bid_wall_price == 97


# ---------------------------------------------------------------------------
# TradeFlowAnalyzer
# ---------------------------------------------------------------------------

class TestTradeFlowAnalyzer:
    def test_basic_flow(self):
        tfa = TradeFlowAnalyzer(window_seconds=9999)
        now = time.time()
        tfa.add_trade(Tick(timestamp=now, price=100, amount=5, side="buy", pair="p"))
        tfa.add_trade(Tick(timestamp=now, price=100, amount=3, side="sell", pair="p"))
        metrics = tfa.analyze("p")
        assert metrics.buy_volume == 5
        assert metrics.sell_volume == 3
        assert metrics.net_flow > 0

    def test_empty(self):
        tfa = TradeFlowAnalyzer()
        metrics = tfa.analyze("x")
        assert metrics.buy_ratio == 0.5


# ---------------------------------------------------------------------------
# MicrostructureAnalyzer
# ---------------------------------------------------------------------------

class TestMicrostructureAnalyzer:
    def test_basic_metrics(self):
        ma = MicrostructureAnalyzer()
        now = time.time()
        ticks = [
            Tick(timestamp=now + i, price=100 + i * 0.1, amount=1, side="buy", pair="p")
            for i in range(20)
        ]
        mids = [100 + i * 0.1 for i in range(20)]
        metrics = ma.analyze(ticks, mids)
        assert metrics.tick_direction_ratio > 0
        assert metrics.order_flow_toxicity >= 0

    def test_insufficient_data(self):
        ma = MicrostructureAnalyzer()
        metrics = ma.analyze([], [])
        assert metrics.effective_spread == 0.0


# ---------------------------------------------------------------------------
# LiquidityMonitor
# ---------------------------------------------------------------------------

class TestLiquidityMonitor:
    def test_basic_assessment(self):
        lm = LiquidityMonitor(min_depth_idr=1_000_000)
        lm.update_depth("p", 500_000)
        metrics = lm.assess("p")
        assert metrics.liquidity_score < 1.0
        assert metrics.total_liquidity_idr == 500_000

    def test_full_liquidity(self):
        lm = LiquidityMonitor(min_depth_idr=100)
        lm.update_depth("p", 200)
        metrics = lm.assess("p")
        assert metrics.liquidity_score > 0


# ---------------------------------------------------------------------------
# VolatilityAnalyzer
# ---------------------------------------------------------------------------

class TestVolatilityAnalyzer:
    def test_low_vol(self):
        va = VolatilityAnalyzer()
        # Flat prices → low vol
        closes = [100] * 25
        metrics = va.analyze(closes)
        assert metrics.realized_vol == 0.0
        assert metrics.vol_regime == "low"

    def test_high_vol(self):
        va = VolatilityAnalyzer(vol_high_threshold=0.01)
        closes = [100, 110, 90, 115, 85, 120, 80, 125, 75, 130, 70]
        metrics = va.analyze(closes)
        assert metrics.realized_vol > 0
        assert metrics.vol_regime in ("high", "extreme")

    def test_with_ohlc(self):
        va = VolatilityAnalyzer()
        opens = [100, 102, 98, 105, 95]
        highs = [105, 108, 103, 110, 100]
        lows = [95, 97, 93, 100, 90]
        closes = [102, 98, 105, 95, 97]
        metrics = va.analyze(closes, highs=highs, lows=lows, opens=opens)
        assert metrics.parkinson_vol > 0
        assert metrics.garman_klass_vol > 0


# ---------------------------------------------------------------------------
# SpreadMonitor
# ---------------------------------------------------------------------------

class TestSpreadMonitor:
    def test_basic_assessment(self):
        sm = SpreadMonitor(wide_spread_pct=0.01)
        sm.record("p", 0.005)
        sm.record("p", 0.006)
        sm.record("p", 0.004)
        metrics = sm.assess("p", 0.005)
        assert metrics.avg_spread > 0
        assert not metrics.is_wide

    def test_wide_spread(self):
        sm = SpreadMonitor(wide_spread_pct=0.005)
        sm.record("p", 0.002)
        metrics = sm.assess("p", 0.01)
        assert metrics.is_wide

    def test_z_score(self):
        sm = SpreadMonitor(z_score_threshold=2.0)
        # Record slightly varying spreads so stdev > 0
        for i in range(20):
            sm.record("p", 0.001 + i * 0.0001)
        metrics = sm.assess("p", 0.05)  # way above mean
        assert metrics.spread_z_score > 2.0


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    def test_trending_up(self):
        rd = RegimeDetector(trend_threshold=0.01, vol_threshold=0.05)
        closes = list(range(100, 120))  # steady rise
        volumes = [100] * 20
        state = rd.detect("p", closes, volumes)
        assert state.regime == "trending_up"

    def test_trending_down(self):
        rd = RegimeDetector(trend_threshold=0.01, vol_threshold=0.05)
        closes = list(range(120, 100, -1))
        volumes = [100] * 20
        state = rd.detect("p", closes, volumes)
        assert state.regime == "trending_down"

    def test_volatile(self):
        rd = RegimeDetector(vol_threshold=0.01)
        closes = [100, 120, 80, 130, 70, 140, 60, 150, 50, 160]
        volumes = [100] * 10
        state = rd.detect("p", closes, volumes)
        assert state.regime == "volatile"


# ---------------------------------------------------------------------------
# CrossExchangeComparator
# ---------------------------------------------------------------------------

class TestCrossExchangeComparator:
    def test_compare(self):
        cec = CrossExchangeComparator()
        cec.update("btc_idr", "exA", 1000, 100)
        cec.update("btc_idr", "exB", 1050, 100)
        result = cec.compare("btc_idr")
        assert result.max_spread_pct > 0
        assert len(result.prices) == 2

    def test_avg_spread(self):
        cec = CrossExchangeComparator()
        cec.update("p", "a", 100, 50)
        cec.update("p", "b", 110, 50)
        cec.compare("p")
        assert cec.avg_spread("p") > 0


# ---------------------------------------------------------------------------
# LatencyMonitor
# ---------------------------------------------------------------------------

class TestLatencyMonitor:
    def test_basic_stats(self):
        lm = LatencyMonitor()
        for ms in [10, 20, 30, 50, 100]:
            lm.record(ms)
        stats = lm.stats()
        assert stats.avg_latency_ms > 0
        assert stats.max_latency_ms == 100
        assert stats.total_requests == 5

    def test_error_tracking(self):
        lm = LatencyMonitor()
        lm.record(10)
        lm.record_error()
        lm.record_timeout()
        stats = lm.stats()
        assert stats.error_count == 1
        assert stats.timeout_count == 1
        assert stats.total_requests == 3

    def test_empty(self):
        lm = LatencyMonitor()
        stats = lm.stats()
        assert stats.avg_latency_ms == 0


# ---------------------------------------------------------------------------
# DataIntegrityValidator
# ---------------------------------------------------------------------------

class TestDataIntegrityValidator:
    def test_valid_data(self):
        div = DataIntegrityValidator()
        div.record_update("p")
        ticks = [
            Tick(timestamp=i * 10, price=100, amount=1, side="buy", pair="p")
            for i in range(10)
        ]
        report = div.validate_ticks("p", ticks)
        assert report.is_valid
        assert report.gap_count == 0
        assert report.duplicate_count == 0

    def test_gap_detection(self):
        div = DataIntegrityValidator(max_gap_seconds=100)
        ticks = [
            Tick(timestamp=0, price=100, amount=1, side="buy", pair="p"),
            Tick(timestamp=200, price=100, amount=1, side="buy", pair="p"),
        ]
        report = div.validate_ticks("p", ticks)
        assert report.gap_count == 1
        assert not report.is_valid

    def test_duplicate_detection(self):
        div = DataIntegrityValidator()
        ticks = [
            Tick(timestamp=100, price=50, amount=1, side="buy", pair="p"),
            Tick(timestamp=100, price=50, amount=1, side="buy", pair="p"),
        ]
        report = div.validate_ticks("p", ticks)
        assert report.duplicate_count == 1

    def test_out_of_order(self):
        div = DataIntegrityValidator()
        ticks = [
            Tick(timestamp=200, price=50, amount=1, side="buy", pair="p"),
            Tick(timestamp=100, price=50, amount=1, side="buy", pair="p"),
        ]
        report = div.validate_ticks("p", ticks)
        assert report.out_of_order_count == 1

    def test_staleness_check(self):
        div = DataIntegrityValidator(stale_threshold_seconds=1)
        div._last_update["stale_pair"] = time.time() - 100
        stale = div.check_staleness(["stale_pair", "fresh_pair"])
        assert "stale_pair" in stale
        assert "fresh_pair" not in stale


# ---------------------------------------------------------------------------
# AnomalyDetector
# ---------------------------------------------------------------------------

class TestAnomalyDetector:
    def test_no_anomaly(self):
        ad = AnomalyDetector()
        for i in range(10):
            ad.record("p", 100 + i * 0.01, volume=100)
        alerts = ad.detect("p")
        assert len(alerts) == 0

    def test_price_spike(self):
        ad = AnomalyDetector(price_spike_threshold=0.02)
        ad.record("p", 100)
        ad.record("p", 100)
        ad.record("p", 110)  # 10% spike
        alerts = ad.detect("p")
        types = [a.anomaly_type for a in alerts]
        assert "price_spike" in types

    def test_flash_crash(self):
        ad = AnomalyDetector(flash_crash_threshold=0.05)
        ad.record("p", 100)
        ad.record("p", 100)
        ad.record("p", 80)  # 20% crash
        alerts = ad.detect("p")
        types = [a.anomaly_type for a in alerts]
        assert "flash_crash" in types

    def test_volume_spike(self):
        ad = AnomalyDetector(volume_spike_threshold=3.0)
        for _ in range(10):
            ad.record("p", 100, volume=10)
        ad.record("p", 100, volume=100)  # 10x spike
        alerts = ad.detect("p")
        types = [a.anomaly_type for a in alerts]
        assert "volume_spike" in types

    def test_spread_blowout(self):
        ad = AnomalyDetector(spread_blowout_threshold=2.0)
        for _ in range(10):
            ad.record("p", 100, spread_pct=0.001)
        ad.record("p", 100, spread_pct=0.01)  # 10x blowout
        alerts = ad.detect("p")
        types = [a.anomaly_type for a in alerts]
        assert "spread_blowout" in types


# ---------------------------------------------------------------------------
# MarketDataFeed (composite)
# ---------------------------------------------------------------------------

class TestMarketDataFeed:
    def test_from_config_default(self):
        """MarketDataFeed.from_config creates all components."""
        from bot.config import BotConfig
        cfg = BotConfig(api_key=None, market_data_enabled=True)
        feed = MarketDataFeed.from_config(cfg)
        assert feed.historical_store is not None
        assert feed.tick_processor is not None
        assert feed.anomaly_detector is not None

    def test_on_tick(self):
        feed = MarketDataFeed()
        tick = Tick(timestamp=time.time(), price=100, amount=1, side="buy", pair="p")
        feed.on_tick(tick)
        assert feed.historical_store.tick_count("p") == 1

    def test_on_depth(self):
        feed = MarketDataFeed()
        snap = DepthSnapshot(
            timestamp=time.time(),
            bids=[(100, 10), (99, 5)],
            asks=[(101, 8), (102, 4)],
            pair="p",
        )
        metrics = feed.on_depth(snap)
        assert metrics.bid_depth_idr > 0

    def test_record_latency(self):
        feed = MarketDataFeed()
        feed.record_latency(50)
        feed.record_latency(100)
        stats = feed.get_latency_stats()
        assert stats.total_requests == 2
        assert stats.avg_latency_ms == 75.0

    def test_check_anomalies_empty(self):
        feed = MarketDataFeed()
        alerts = feed.check_anomalies("nonexistent")
        assert alerts == []

    def test_get_trade_flow_empty(self):
        feed = MarketDataFeed()
        flow = feed.get_trade_flow("p")
        assert flow.buy_ratio == 0.5


# ---------------------------------------------------------------------------
# Trader integration: stale order, absorption, sweep/trap guards
# ---------------------------------------------------------------------------

class TestTraderDetectorWiring:
    """Test that new analysis detectors are correctly wired into Trader."""

    def _make_trader(self, **overrides):
        from bot.config import BotConfig
        from bot.trader import Trader
        defaults = dict(
            api_key=None,
            dry_run=True,
            initial_capital=1_000_000,
            pair="btc_idr",
            real_time=False,
            min_coin_price_idr=0,
        )
        defaults.update(overrides)
        cfg = BotConfig(**defaults)
        return Trader(cfg)

    def test_analyze_includes_new_keys(self):
        """analyze_market returns new detector results when enabled."""
        from unittest.mock import patch
        trader = self._make_trader(
            liquidity_sweep_enabled=True,
            smart_money_enabled=True,
            volume_accel_enabled=True,
            micro_trend_enabled=True,
        )
        # Provide sufficient mock data
        mock_ticker = {"ticker": {"last": "1000", "buy": "999", "sell": "1001",
                                  "high": "1100", "low": "900", "vol_idr": "1000000"}}
        mock_depth = {"buy": [["999", "10"]], "sell": [["1001", "10"]]}
        mock_trades = [{"date": "1000", "price": "1000", "amount": "1", "type": "buy", "tid": "1"}] * 5

        with patch.object(trader.client, "get_ticker", return_value=mock_ticker), \
             patch.object(trader.client, "get_depth", return_value=mock_depth), \
             patch.object(trader.client, "get_trades", return_value=mock_trades), \
             patch.object(trader.client, "get_ohlc", return_value={"ohlc": [
                 [1000 + i * 60, str(100 + i), str(105 + i), str(95 + i), str(102 + i), str(1000)]
                 for i in range(50)
             ]}):
            snapshot = trader.analyze_market("btc_idr")
            # New keys should be present
            for key in ("liquidity_sweep", "smart_money", "volume_accel", "micro_trend"):
                assert key in snapshot

    def test_stale_order_cancellation_disabled(self):
        """When stale_order_seconds=0, no orders are cancelled."""
        trader = self._make_trader(stale_order_seconds=0)
        # Should be a no-op, return 0
        assert trader._cancel_stale_orders("btc_idr") == 0

    def test_stale_order_cancellation_dry_run(self):
        """Stale order cancel is skipped in dry_run mode."""
        trader = self._make_trader(stale_order_seconds=60, dry_run=True)
        assert trader._cancel_stale_orders("btc_idr") == 0

    def test_prev_depth_dict_initialized(self):
        """_prev_depth dict is available on Trader."""
        trader = self._make_trader()
        assert isinstance(trader._prev_depth, dict)
