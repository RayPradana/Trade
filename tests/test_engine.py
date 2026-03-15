"""Tests for the bot.engine low-latency architecture module."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from bot.engine import (
    DataCache,
    ExecutionEngine,
    MarketDataEngine,
    RiskEngine,
    SignalQueue,
    StrategyEngine,
    TradingOrchestrator,
    TradingSignal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_snap(pair: str = "btc_idr", price: float = 100_000.0, action: str = "buy") -> Dict[str, Any]:
    decision = MagicMock()
    decision.action = action
    return {
        "pair": pair,
        "price": price,
        "decision": decision,
    }


def _make_config(**kwargs) -> MagicMock:
    cfg = MagicMock()
    cfg.pair = "btc_idr"
    cfg.trailing_stop_pct = 0.03
    cfg.trailing_tp_pct = 0.02
    cfg.initial_capital = 1_000_000.0
    cfg.max_daily_loss_pct = 0.0
    cfg.circuit_breaker_max_errors = 0
    cfg.market_data_engine_interval = 0.05
    cfg.strategy_engine_interval = 0.05
    cfg.engine_mode = True
    for k, v in kwargs.items():
        setattr(cfg, k, v)
    return cfg


def _make_tracker(base_position: float = 0.0, has_pending_buy: bool = False) -> MagicMock:
    t = MagicMock()
    t.base_position = base_position
    t.has_pending_buy = has_pending_buy
    t.has_pending_sell = False
    t.stop_reason = MagicMock(return_value=None)
    t.update_trailing_stop = MagicMock()
    t.activate_trailing_tp = MagicMock()
    t.as_dict = MagicMock(return_value={"realized_pnl": 0.0, "equity": 1_000_000.0})
    return t


def _make_trader(**kwargs) -> MagicMock:
    trader = MagicMock()
    trader._all_pairs = ["btc_idr"]
    trader.active_positions = {}
    tracker = _make_tracker()
    trader._active_tracker = MagicMock(return_value=tracker)
    trader.at_max_positions = MagicMock(return_value=False)
    trader.analyze_market = MagicMock(return_value=_make_snap())
    trader.maybe_execute = MagicMock(return_value={"action": "hold"})
    trader.force_sell = MagicMock(return_value={"status": "simulated"})
    trader.resume_pending_buy = MagicMock(return_value={"status": "none"})
    trader.resume_pending_sell = MagicMock(return_value={"status": "none"})
    trader.check_momentum_exit = MagicMock(return_value=False)
    trader.check_post_entry_dump = MagicMock(return_value=False)
    trader.evaluate_dynamic_tp = MagicMock(return_value=None)
    for k, v in kwargs.items():
        setattr(trader, k, v)
    return trader


# ===========================================================================
# DataCache tests
# ===========================================================================

class TestDataCache:
    def test_put_and_get(self):
        cache = DataCache()
        snap = _make_snap()
        cache.put("btc_idr", snap)
        assert cache.get("btc_idr") is snap

    def test_get_missing_returns_none(self):
        cache = DataCache()
        assert cache.get("btc_idr") is None

    def test_get_all_returns_copy(self):
        cache = DataCache()
        snap = _make_snap()
        cache.put("btc_idr", snap)
        result = cache.get_all()
        assert "btc_idr" in result
        assert result["btc_idr"] is snap

    def test_put_replaces_existing(self):
        cache = DataCache()
        snap1 = _make_snap(price=100.0)
        snap2 = _make_snap(price=200.0)
        cache.put("btc_idr", snap1)
        cache.put("btc_idr", snap2)
        assert cache.get("btc_idr")["price"] == 200.0

    def test_age_seconds_fresh(self):
        cache = DataCache()
        cache.put("btc_idr", _make_snap())
        assert cache.age_seconds("btc_idr") < 1.0

    def test_age_seconds_missing_is_inf(self):
        cache = DataCache()
        assert cache.age_seconds("btc_idr") == float("inf")

    def test_pairs(self):
        cache = DataCache()
        cache.put("btc_idr", _make_snap())
        cache.put("eth_idr", _make_snap())
        assert set(cache.pairs()) == {"btc_idr", "eth_idr"}

    def test_thread_safety(self):
        """Concurrent reads and writes must not raise exceptions."""
        cache = DataCache()
        errors = []

        def writer():
            for i in range(100):
                try:
                    cache.put("btc_idr", _make_snap(price=float(i)))
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(100):
                try:
                    cache.get("btc_idr")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# SignalQueue tests
# ===========================================================================

class TestSignalQueue:
    def _make_signal(self, pair: str = "btc_idr", stype: str = "buy") -> TradingSignal:
        return TradingSignal(pair=pair, snapshot=_make_snap(pair=pair), signal_type=stype)

    def test_put_and_get(self):
        sq = SignalQueue()
        sig = self._make_signal()
        sq.put(sig)
        result = sq.get(timeout=0.1)
        assert result is sig

    def test_get_empty_returns_none(self):
        sq = SignalQueue()
        assert sq.get(timeout=0.05) is None

    def test_size(self):
        sq = SignalQueue()
        sq.put(self._make_signal())
        sq.put(self._make_signal(pair="eth_idr"))
        assert sq.size() == 2

    def test_full_queue_drops_oldest(self):
        sq = SignalQueue(maxsize=2)
        for _ in range(5):
            sq.put(self._make_signal())
        # Should not raise and queue should be at most maxsize
        assert sq.size() <= 2

    def test_thread_safety(self):
        sq = SignalQueue()
        errors = []
        results = []
        lock = threading.Lock()

        def producer():
            for i in range(50):
                try:
                    sq.put(self._make_signal(pair=f"pair_{i}"))
                except Exception as e:
                    errors.append(e)

        def consumer():
            for _ in range(50):
                try:
                    sig = sq.get(timeout=0.1)
                    if sig is not None:
                        with lock:
                            results.append(sig)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=producer) for _ in range(3)]
        threads += [threading.Thread(target=consumer) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# ===========================================================================
# RiskEngine tests
# ===========================================================================

class TestRiskEngine:
    def test_approve_valid_signal(self):
        config = _make_config()
        risk = RiskEngine(config)
        trader = _make_trader()
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, reason = risk.approve(sig, trader)
        assert ok is True
        assert reason == "approved"

    def test_reject_zero_price(self):
        config = _make_config()
        risk = RiskEngine(config)
        trader = _make_trader()
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=0.0), signal_type="buy")
        ok, reason = risk.approve(sig, trader)
        assert ok is False
        assert reason == "invalid_price"

    def test_reject_negative_price(self):
        config = _make_config()
        risk = RiskEngine(config)
        trader = _make_trader()
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=-1.0), signal_type="buy")
        ok, reason = risk.approve(sig, trader)
        assert ok is False

    def test_circuit_breaker_tripped(self):
        config = _make_config(circuit_breaker_max_errors=3)
        risk = RiskEngine(config)
        trader = _make_trader()
        trader._consecutive_errors = 5
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, reason = risk.approve(sig, trader)
        assert ok is False
        assert reason == "circuit_breaker_tripped"

    def test_circuit_breaker_not_tripped_below_threshold(self):
        config = _make_config(circuit_breaker_max_errors=3)
        risk = RiskEngine(config)
        trader = _make_trader()
        trader._consecutive_errors = 2
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, _ = risk.approve(sig, trader)
        assert ok is True

    def test_circuit_breaker_disabled(self):
        config = _make_config(circuit_breaker_max_errors=0)
        risk = RiskEngine(config)
        trader = _make_trader()
        trader._consecutive_errors = 999
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, _ = risk.approve(sig, trader)
        assert ok is True

    def test_daily_loss_cap_reached(self):
        config = _make_config(max_daily_loss_pct=0.05, initial_capital=1_000_000.0)
        risk = RiskEngine(config)
        trader = _make_trader()
        tracker = _make_tracker()
        tracker.daily_loss = MagicMock(return_value=60_000.0)  # exceeds 5% of 1M
        trader._active_tracker = MagicMock(return_value=tracker)
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, reason = risk.approve(sig, trader)
        assert ok is False
        assert reason == "daily_loss_cap_reached"

    def test_daily_loss_cap_not_reached(self):
        config = _make_config(max_daily_loss_pct=0.05, initial_capital=1_000_000.0)
        risk = RiskEngine(config)
        trader = _make_trader()
        tracker = _make_tracker()
        tracker.daily_loss = MagicMock(return_value=10_000.0)  # only 1% loss, under 5% cap
        trader._active_tracker = MagicMock(return_value=tracker)
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, _ = risk.approve(sig, trader)
        assert ok is True

    def test_daily_loss_cap_disabled(self):
        # max_daily_loss_pct=0 → daily loss check is skipped entirely
        config = _make_config(max_daily_loss_pct=0.0, initial_capital=1_000_000.0)
        risk = RiskEngine(config)
        trader = _make_trader()
        tracker = _make_tracker()
        tracker.daily_loss = MagicMock(return_value=999_999.0)
        trader._active_tracker = MagicMock(return_value=tracker)
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, reason = risk.approve(sig, trader)
        assert ok is True
        assert reason == "approved"


# ===========================================================================
# MarketDataEngine tests
# ===========================================================================

class TestMarketDataEngine:
    def test_start_stop(self):
        cache = DataCache()
        trader = _make_trader()
        config = _make_config()
        engine = MarketDataEngine(trader, config, cache, interval=0.05)
        engine.start()
        assert engine.is_alive()
        time.sleep(0.15)
        engine.stop()
        assert not engine.is_alive()

    def test_populates_cache(self):
        cache = DataCache()
        config = _make_config()
        trader = _make_trader()
        trader.analyze_market = MagicMock(return_value=_make_snap("btc_idr", 100.0))
        engine = MarketDataEngine(trader, config, cache, interval=0.05)
        engine.start()
        time.sleep(0.25)
        engine.stop()
        assert cache.get("btc_idr") is not None

    def test_active_pairs_fetched_first(self):
        """Active pairs should be first in the fetch order."""
        cache = DataCache()
        config = _make_config()
        trader = _make_trader()
        trader.active_positions = {"eth_idr": _make_tracker(base_position=1.0)}
        trader._all_pairs = ["btc_idr", "eth_idr"]
        engine = MarketDataEngine(trader, config, cache, interval=0.05)
        pairs = engine._get_pairs()
        assert pairs[0] == "eth_idr"

    def test_error_does_not_crash_engine(self):
        """An exception in analyze_market should be swallowed."""
        cache = DataCache()
        config = _make_config()
        trader = _make_trader()
        trader.analyze_market = MagicMock(side_effect=RuntimeError("API down"))
        engine = MarketDataEngine(trader, config, cache, interval=0.05)
        engine.start()
        time.sleep(0.2)
        engine.stop()
        assert engine.error_count > 0


# ===========================================================================
# StrategyEngine tests
# ===========================================================================

class TestStrategyEngine:
    def test_buy_signal_emitted(self):
        cache = DataCache()
        snap = _make_snap("btc_idr", 100.0, action="buy")
        cache.put("btc_idr", snap)
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.at_max_positions = MagicMock(return_value=False)
        engine = StrategyEngine(trader, config, cache, sq, interval=0.05)
        engine.start()
        time.sleep(0.25)
        engine.stop()
        sig = sq.get(timeout=0.1)
        assert sig is not None
        assert sig.signal_type == "buy"

    def test_no_signal_when_hold(self):
        cache = DataCache()
        snap = _make_snap("btc_idr", 100.0, action="hold")
        cache.put("btc_idr", snap)
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.at_max_positions = MagicMock(return_value=False)
        engine = StrategyEngine(trader, config, cache, sq, interval=0.05)
        engine.start()
        time.sleep(0.2)
        engine.stop()
        # "hold" should not emit any signal
        assert sq.get(timeout=0.05) is None

    def test_no_signal_when_at_max_positions(self):
        cache = DataCache()
        snap = _make_snap("btc_idr", 100.0, action="buy")
        cache.put("btc_idr", snap)
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.at_max_positions = MagicMock(return_value=True)
        engine = StrategyEngine(trader, config, cache, sq, interval=0.05)
        engine.start()
        time.sleep(0.2)
        engine.stop()
        assert sq.get(timeout=0.05) is None

    def test_exit_signal_for_held_position(self):
        cache = DataCache()
        snap = _make_snap("btc_idr", 100.0, action="hold")
        cache.put("btc_idr", snap)
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        tracker = _make_tracker(base_position=1.0)
        tracker.stop_reason = MagicMock(return_value="stop_loss_hit")
        trader._active_tracker = MagicMock(return_value=tracker)
        engine = StrategyEngine(trader, config, cache, sq, interval=0.05)
        engine.start()
        time.sleep(0.25)
        engine.stop()
        sig = sq.get(timeout=0.1)
        assert sig is not None
        assert sig.signal_type == "exit"

    def test_resume_buy_signal(self):
        cache = DataCache()
        snap = _make_snap("btc_idr", 100.0, action="hold")
        cache.put("btc_idr", snap)
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        tracker = _make_tracker(base_position=0.0, has_pending_buy=True)
        trader._active_tracker = MagicMock(return_value=tracker)
        engine = StrategyEngine(trader, config, cache, sq, interval=0.05)
        engine.start()
        time.sleep(0.25)
        engine.stop()
        sig = sq.get(timeout=0.1)
        assert sig is not None
        assert sig.signal_type == "resume_buy"

    def test_error_does_not_crash_engine(self):
        cache = DataCache()
        snap = _make_snap("btc_idr", 100.0)
        cache.put("btc_idr", snap)
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader._active_tracker = MagicMock(side_effect=RuntimeError("crash"))
        engine = StrategyEngine(trader, config, cache, sq, interval=0.05)
        engine.start()
        time.sleep(0.2)
        engine.stop()
        assert engine.error_count > 0


# ===========================================================================
# ExecutionEngine tests
# ===========================================================================

class TestExecutionEngine:
    def test_buy_signal_calls_maybe_execute(self):
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.maybe_execute = MagicMock(return_value={"action": "buy", "status": "simulated", "amount": 0.001})
        risk = RiskEngine(config)
        notifications: List[str] = []
        engine = ExecutionEngine(
            trader, config, sq, risk,
            notify_fn=lambda t: notifications.append(t),
        )
        engine.start()
        sq.put(TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy"))
        time.sleep(0.3)
        engine.stop()
        trader.maybe_execute.assert_called()
        assert any("BUY" in n for n in notifications)

    def test_exit_signal_calls_force_sell(self):
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.force_sell = MagicMock(return_value={"status": "simulated", "amount": 0.001})
        risk = RiskEngine(config)
        engine = ExecutionEngine(trader, config, sq, risk)
        engine.start()
        sq.put(TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="exit"))
        time.sleep(0.3)
        engine.stop()
        trader.force_sell.assert_called()

    def test_resume_buy_signal(self):
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.resume_pending_buy = MagicMock(return_value={"status": "resumed", "amount": 0.001})
        risk = RiskEngine(config)
        engine = ExecutionEngine(trader, config, sq, risk)
        engine.start()
        sq.put(TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="resume_buy"))
        time.sleep(0.3)
        engine.stop()
        trader.resume_pending_buy.assert_called()

    def test_resume_sell_signal(self):
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.resume_pending_sell = MagicMock(return_value={"status": "resumed", "amount": 0.001})
        risk = RiskEngine(config)
        engine = ExecutionEngine(trader, config, sq, risk)
        engine.start()
        sq.put(TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="resume_sell"))
        time.sleep(0.3)
        engine.stop()
        trader.resume_pending_sell.assert_called()

    def test_risk_rejection_skips_execution(self):
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        risk = RiskEngine(config)
        engine = ExecutionEngine(trader, config, sq, risk)
        engine.start()
        # Zero price → rejected by RiskEngine
        sq.put(TradingSignal(
            pair="btc_idr",
            snapshot=_make_snap(price=0.0),
            signal_type="buy",
        ))
        time.sleep(0.3)
        engine.stop()
        trader.maybe_execute.assert_not_called()

    def test_on_outcome_callback(self):
        sq = SignalQueue()
        config = _make_config()
        trader = _make_trader()
        trader.maybe_execute = MagicMock(return_value={"action": "buy", "status": "simulated", "amount": 0.001})
        risk = RiskEngine(config)
        outcomes: list = []
        engine = ExecutionEngine(
            trader, config, sq, risk,
            on_outcome=lambda sig, out: outcomes.append((sig, out)),
        )
        engine.start()
        sq.put(TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy"))
        time.sleep(0.3)
        engine.stop()
        assert len(outcomes) == 1
        assert outcomes[0][0].signal_type == "buy"


# ===========================================================================
# TradingOrchestrator tests
# ===========================================================================

class TestTradingOrchestrator:
    def test_start_stop_all_engines(self):
        config = _make_config()
        trader = _make_trader()
        orch = TradingOrchestrator(trader, config)
        orch.start()
        health = orch.health()
        assert health["market_data_engine"] == "running"
        assert health["strategy_engine"] == "running"
        assert health["execution_engine"] == "running"
        orch.stop()
        health = orch.health()
        assert health["market_data_engine"] == "stopped"
        assert health["strategy_engine"] == "stopped"
        assert health["execution_engine"] == "stopped"

    def test_health_structure(self):
        config = _make_config()
        trader = _make_trader()
        orch = TradingOrchestrator(trader, config)
        h = orch.health()
        assert "cache_pairs" in h
        assert "signal_queue_size" in h
        assert "market_data_engine" in h
        assert "strategy_engine" in h
        assert "execution_engine" in h
        assert "market_data_errors" in h
        assert "strategy_errors" in h
        assert "execution_errors" in h

    def test_run_returns_on_shutdown(self):
        config = _make_config()
        trader = _make_trader()
        orch = TradingOrchestrator(trader, config)
        orch.start()
        shutdown = threading.Event()
        t = threading.Thread(
            target=orch.run,
            args=(shutdown,),
            kwargs={"poll_interval": 0.05},
            daemon=True,
        )
        t.start()
        shutdown.set()
        t.join(timeout=2.0)
        assert not t.is_alive(), "run() did not return after shutdown event"
        orch.stop()

    def test_notify_fn_invoked(self):
        config = _make_config()
        trader = _make_trader()
        trader.maybe_execute = MagicMock(
            return_value={"action": "buy", "status": "simulated", "amount": 0.001}
        )
        notifications: List[str] = []
        orch = TradingOrchestrator(
            trader, config,
            notify_fn=lambda t: notifications.append(t),
        )
        # Manually push a buy signal to the signal queue
        orch.start()
        orch.signal_queue.put(
            TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        )
        time.sleep(0.5)
        orch.stop()
        assert any("BUY" in n for n in notifications)

    def test_full_pipeline_buy(self):
        """End-to-end: MarketDataEngine → DataCache → StrategyEngine → SignalQueue → ExecutionEngine."""
        config = _make_config()
        trader = _make_trader()
        # Simulate a buy decision
        snap = _make_snap("btc_idr", 100.0, action="buy")
        trader.analyze_market = MagicMock(return_value=snap)
        trader.at_max_positions = MagicMock(return_value=False)
        outcomes: List[Any] = []
        orch = TradingOrchestrator(
            trader, config,
            on_outcome=lambda sig, out: outcomes.append((sig, out)),
        )
        orch.start()
        time.sleep(0.6)
        orch.stop()
        # maybe_execute should have been called at least once by ExecutionEngine
        assert trader.maybe_execute.called or len(outcomes) > 0


# ===========================================================================
# Config field tests
# ===========================================================================

class TestEngineConfigFields:
    def test_default_engine_mode(self):
        from bot.config import BotConfig
        cfg = BotConfig(api_key=None)
        assert cfg.engine_mode is True

    def test_default_intervals(self):
        from bot.config import BotConfig
        cfg = BotConfig(api_key=None)
        assert cfg.market_data_engine_interval == 2.0
        assert cfg.strategy_engine_interval == 3.0

    def test_engine_mode_false(self):
        from bot.config import BotConfig
        cfg = BotConfig(api_key=None, engine_mode=False)
        assert cfg.engine_mode is False

    def test_from_env_engine_mode_env_var(self, monkeypatch):
        monkeypatch.setenv("ENGINE_MODE", "false")
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("INDODAX_KEY", "test")
        from bot.config import BotConfig
        cfg = BotConfig.from_env()
        assert cfg.engine_mode is False

    def test_from_env_engine_mode_true(self, monkeypatch):
        monkeypatch.setenv("ENGINE_MODE", "true")
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("INDODAX_KEY", "test")
        from bot.config import BotConfig
        cfg = BotConfig.from_env()
        assert cfg.engine_mode is True

    def test_from_env_market_data_engine_interval(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_ENGINE_INTERVAL", "5.0")
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("INDODAX_KEY", "test")
        from bot.config import BotConfig
        cfg = BotConfig.from_env()
        assert cfg.market_data_engine_interval == 5.0

    def test_from_env_strategy_engine_interval(self, monkeypatch):
        monkeypatch.setenv("STRATEGY_ENGINE_INTERVAL", "7.5")
        monkeypatch.setenv("DRY_RUN", "true")
        monkeypatch.setenv("INDODAX_KEY", "test")
        from bot.config import BotConfig
        cfg = BotConfig.from_env()
        assert cfg.strategy_engine_interval == 7.5


# ===========================================================================
# _run_engine_monitor cycle display tests
# ===========================================================================

class TestEngineMonitorCycleDisplay:
    """Engine monitor loop displays all cached pairs + portfolio when available,
    and shows a 'waiting' message when the cache is empty."""

    def _make_minimal_snap(self, pair: str = "btc_idr", price: float = 100_000.0, action: str = "hold"):
        decision = MagicMock()
        decision.action = action
        decision.confidence = 0.5
        decision.reason = "test"
        decision.mode = "day_trading"
        return {
            "pair": pair,
            "price": price,
            "decision": decision,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
        }

    def _make_orchestrator(self, snaps: dict | None = None):
        """Build a mock orchestrator whose cache contains ``snaps`` (pair→snap dict).

        If snaps is None the cache is empty.
        """
        snaps = snaps or {}
        orch = MagicMock()
        orch.cache = MagicMock()
        orch.cache.pairs = MagicMock(return_value=list(snaps.keys()))
        orch.cache.get = MagicMock(side_effect=lambda p: snaps.get(p))
        orch.health = MagicMock(return_value={
            "market_data_engine": "running",
            "strategy_engine": "running",
            "execution_engine": "running",
            "cache_pairs": list(snaps.keys()),
            "signal_queue_size": 0,
        })
        # Default: no recent execution outcomes buffered
        orch.pop_recent_outcomes = MagicMock(return_value=[])
        return orch

    def _run_once(self, orchestrator, pair="btc_idr", active_positions=None, trader=None):
        import main
        import threading
        config = MagicMock()
        config.pair = pair
        config.run_once = True
        config.strategy_engine_interval = 0.0
        config.cycle_summary_interval = 9999
        config.state_path = None
        config.state_backup_interval = 0
        config.trailing_stop_pct = 0.0
        config.trailing_tp_pct = 0.0
        config.initial_capital = 1_000_000.0
        if trader is None:
            trader = MagicMock()
            trader.active_positions = active_positions or {}
            trader.at_max_positions = MagicMock(return_value=False)
            trader.portfolio_snapshot = MagicMock(return_value={
                "equity": 1_000_000.0, "cash": 1_000_000.0,
                "base_position": 0.0, "realized_pnl": 0.0,
                "trade_count": 0, "win_rate": 0.0,
            })
        journal = MagicMock()
        journal.summary_str = MagicMock(return_value="0 sells | PnL=+0.00")
        shutdown = threading.Event()
        tasks = []
        autonomous_state = MagicMock()
        polling_config = MagicMock()
        main._run_engine_monitor(
            orchestrator=orchestrator,
            trader=trader,
            config=config,
            journal=journal,
            shutdown=shutdown,
            tasks=tasks,
            autonomous_state=autonomous_state,
            start_time=0.0,
            crash_history=[],
            error_log=[],
            components=[],
            polling_config=polling_config,
            now_ms=lambda: 0,
        )

    # ── basic display ─────────────────────────────────────────────────────

    def test_log_signal_called_when_cache_has_snapshot(self):
        """When cache has a snapshot, _log_signal should be called with it."""
        import main
        snap = self._make_minimal_snap()
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        with patch.object(main, "_log_signal") as mock_log_signal:
            self._run_once(orch)
        mock_log_signal.assert_called_once_with(snap)

    def test_waiting_message_logged_when_cache_empty(self):
        """When cache is empty, a 'Waiting for market data' message is logged."""
        import logging
        import main
        orch = self._make_orchestrator(snaps={})
        with patch.object(main, "_log_signal") as mock_log_signal:
            with self.assertLogs_compat(level="INFO") as cm:
                self._run_once(orch)
        mock_log_signal.assert_not_called()
        assert any("Waiting" in msg or "waiting" in msg for msg in cm.output)

    def assertLogs_compat(self, level="INFO"):
        """Helper to use assertLogs in pytest context."""
        import logging

        class _CM:
            def __enter__(self_):
                self_.handler = _LogCapture()
                logging.getLogger().addHandler(self_.handler)
                self_.handler.setLevel(level)
                self_.old_level = logging.getLogger().level
                logging.getLogger().setLevel(level)
                self_.output = self_.handler.records
                return self_

            def __exit__(self_, *args):
                logging.getLogger().removeHandler(self_.handler)
                logging.getLogger().setLevel(self_.old_level)

        class _LogCapture(logging.Handler):
            def __init__(self):
                super().__init__()
                self.records = []

            def emit(self, record):
                self.records.append(self.format(record))

        return _CM()

    def test_cache_get_called_for_cached_pair(self):
        """orchestrator.cache.get must be called for each pair returned by cache.pairs()."""
        import main
        snap = self._make_minimal_snap()
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        with patch.object(main, "_log_signal"):
            self._run_once(orch, pair="btc_idr")
        orch.cache.get.assert_any_call("btc_idr")

    def test_display_error_does_not_crash_monitor(self):
        """If _log_signal raises, the engine monitor must not crash."""
        import main
        snap = self._make_minimal_snap()
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        with patch.object(main, "_log_signal", side_effect=RuntimeError("display boom")):
            # Should not raise
            self._run_once(orch)

    # ── multi-pair display ────────────────────────────────────────────────

    def test_all_cached_pairs_are_displayed(self):
        """Every pair in the cache must have _log_signal called for it."""
        import main
        snap_btc = self._make_minimal_snap("btc_idr", 100_000.0)
        snap_eth = self._make_minimal_snap("eth_idr", 5_000.0)
        snap_sol = self._make_minimal_snap("sol_idr", 3_000.0)
        orch = self._make_orchestrator(snaps={
            "btc_idr": snap_btc,
            "eth_idr": snap_eth,
            "sol_idr": snap_sol,
        })
        calls = []
        with patch.object(main, "_log_signal", side_effect=lambda s: calls.append(s["pair"])):
            self._run_once(orch)
        assert set(calls) == {"btc_idr", "eth_idr", "sol_idr"}

    def test_only_pairs_in_cache_are_displayed(self):
        """config.pair should NOT be added artificially if it isn't in the cache."""
        import main
        snap_eth = self._make_minimal_snap("eth_idr", 5_000.0)
        # config.pair = "btc_idr" but cache only has "eth_idr"
        orch = self._make_orchestrator(snaps={"eth_idr": snap_eth})
        calls = []
        with patch.object(main, "_log_signal", side_effect=lambda s: calls.append(s["pair"])):
            self._run_once(orch, pair="btc_idr")
        assert calls == ["eth_idr"]

    # ── portfolio / holding display ───────────────────────────────────────

    def test_log_holding_called_for_active_position(self):
        """_log_holding must be called for a pair where the trader holds a position."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0)
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        # Build a tracker with an open position
        tracker = MagicMock()
        tracker.base_position = 0.001
        tracker.as_dict = MagicMock(return_value={
            "equity": 1_100_000.0, "cash": 1_000_000.0,
            "base_position": 0.001, "realized_pnl": 0.0,
            "avg_cost": 100_000.0, "trade_count": 1, "win_rate": 1.0,
        })
        active_positions = {"btc_idr": tracker}
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_holding") as mock_holding:
                with patch.object(main, "_log_portfolio"):
                    self._run_once(orch, active_positions=active_positions)
        mock_holding.assert_called_once()
        call_args = mock_holding.call_args
        assert call_args[0][0] == "btc_idr"  # pair argument

    def test_log_holding_not_called_when_no_position(self):
        """_log_holding must NOT be called when base_position == 0."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0)
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        tracker = MagicMock()
        tracker.base_position = 0.0
        active_positions = {"btc_idr": tracker}
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_holding") as mock_holding:
                self._run_once(orch, active_positions=active_positions)
        mock_holding.assert_not_called()

    def test_log_portfolio_called_when_positions_held(self):
        """_log_portfolio must be called once when trader has active positions."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0)
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        tracker = MagicMock()
        tracker.base_position = 0.001
        tracker.as_dict = MagicMock(return_value={
            "equity": 1_100_000.0, "cash": 1_000_000.0,
            "base_position": 0.001, "realized_pnl": 0.0,
            "avg_cost": 100_000.0, "trade_count": 1, "win_rate": 1.0,
        })
        active_positions = {"btc_idr": tracker}
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_holding"):
                with patch.object(main, "_log_portfolio") as mock_portfolio:
                    self._run_once(orch, active_positions=active_positions)
        mock_portfolio.assert_called_once()

    def test_log_portfolio_called_even_when_no_positions(self):
        """_log_portfolio must always be called when the cache is non-empty,
        even when there are no active positions (flat / startup state)."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0)
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_portfolio") as mock_portfolio:
                self._run_once(orch, active_positions={})
        mock_portfolio.assert_called_once()

    def test_log_portfolio_not_called_when_cache_empty(self):
        """_log_portfolio must NOT be called when the cache is empty."""
        import main
        orch = self._make_orchestrator(snaps={})
        with patch.object(main, "_log_portfolio") as mock_portfolio:
            self._run_once(orch, active_positions={})
        mock_portfolio.assert_not_called()

    # ── outcome display (skipped / placed) ────────────────────────────────

    def test_log_outcome_called_for_hold_decision(self):
        """_log_outcome must be called with status='skipped' for HOLD decisions."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0, action="hold")
        snap["decision"].reason = "rug_pull_risk:dead_coin_trades=0"
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        # No recent execution outcomes buffered
        orch.pop_recent_outcomes = MagicMock(return_value=[])
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_outcome") as mock_outcome:
                with patch.object(main, "_log_portfolio"):
                    self._run_once(orch)
        mock_outcome.assert_called_once()
        call_kwargs = mock_outcome.call_args[0][0]
        assert call_kwargs["status"] == "skipped"
        assert call_kwargs["action"] == "hold"
        assert "rug_pull_risk" in call_kwargs.get("reason", "")

    def test_log_outcome_called_for_each_hold_pair(self):
        """_log_outcome must be called once per HOLD pair."""
        import main
        snap_btc = self._make_minimal_snap("btc_idr", 100_000.0, action="hold")
        snap_eth = self._make_minimal_snap("eth_idr", 5_000.0, action="hold")
        orch = self._make_orchestrator(snaps={
            "btc_idr": snap_btc,
            "eth_idr": snap_eth,
        })
        orch.pop_recent_outcomes = MagicMock(return_value=[])
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_outcome") as mock_outcome:
                with patch.object(main, "_log_portfolio"):
                    self._run_once(orch)
        assert mock_outcome.call_count == 2

    def test_log_outcome_shows_exec_outcome_over_synthetic_skip(self):
        """When an execution outcome is buffered for a pair, that takes
        priority over the synthetic 'skipped' outcome for a HOLD decision."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0, action="buy")
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        placed_outcome = {
            "action": "buy",
            "status": "placed",
            "amount": 0.001,
            "price": 100_000.0,
        }
        orch.pop_recent_outcomes = MagicMock(
            return_value=[("btc_idr", placed_outcome)]
        )
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_outcome") as mock_outcome:
                with patch.object(main, "_log_portfolio"):
                    self._run_once(orch)
        mock_outcome.assert_called_once()
        call_kwargs = mock_outcome.call_args[0][0]
        assert call_kwargs["status"] == "placed"

    def test_log_outcome_skipped_max_positions_for_buy_decision(self):
        """When decision is 'buy' but trader is at max positions, _log_outcome
        must show status='skipped' with reason='max_positions'."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0, action="buy")
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        orch.pop_recent_outcomes = MagicMock(return_value=[])
        trader = MagicMock()
        trader.active_positions = {}
        trader.at_max_positions = MagicMock(return_value=True)
        trader.portfolio_snapshot = MagicMock(return_value={
            "equity": 1_000_000.0, "cash": 1_000_000.0,
            "base_position": 0.0, "realized_pnl": 0.0,
            "trade_count": 0, "win_rate": 0.0,
        })
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_outcome") as mock_outcome:
                with patch.object(main, "_log_portfolio"):
                    self._run_once(orch, trader=trader)
        mock_outcome.assert_called_once()
        call_kwargs = mock_outcome.call_args[0][0]
        assert call_kwargs["status"] == "skipped"
        assert call_kwargs["reason"] == "max_positions"

    def test_log_outcome_skipped_no_position_for_sell_decision(self):
        """When decision is 'sell' and the trader holds no position for that pair,
        _log_outcome must show status='skipped' with reason containing 'no_position'."""
        import main
        snap = self._make_minimal_snap("btc_idr", 100_000.0, action="sell")
        orch = self._make_orchestrator(snaps={"btc_idr": snap})
        orch.pop_recent_outcomes = MagicMock(return_value=[])
        trader = MagicMock()
        # No active positions
        trader.active_positions = {}
        trader.at_max_positions = MagicMock(return_value=False)
        trader.portfolio_snapshot = MagicMock(return_value={
            "equity": 1_000_000.0, "cash": 1_000_000.0,
            "base_position": 0.0, "realized_pnl": 0.0,
            "trade_count": 0, "win_rate": 0.0,
        })
        with patch.object(main, "_log_signal"):
            with patch.object(main, "_log_outcome") as mock_outcome:
                with patch.object(main, "_log_portfolio"):
                    self._run_once(orch, trader=trader)
        mock_outcome.assert_called_once()
        call_kwargs = mock_outcome.call_args[0][0]
        assert call_kwargs["action"] == "sell"
        assert call_kwargs["status"] == "skipped"
        assert "no_position" in call_kwargs.get("reason", "")

    def test_pop_recent_outcomes_integration(self):
        """TradingOrchestrator.pop_recent_outcomes must start empty and
        accumulate outcomes pushed by the internal callback."""
        config = _make_config()
        trader = _make_trader()
        orch = TradingOrchestrator(trader, config)
        # Should start empty
        assert orch.pop_recent_outcomes() == []
        # Simulate the internal chained callback by pushing directly
        with orch._outcome_lock:
            orch._recent_outcomes.append(("btc_idr", {"status": "placed"}))
        result = orch.pop_recent_outcomes()
        assert result == [("btc_idr", {"status": "placed"})]
        # After pop, buffer is cleared
        assert orch.pop_recent_outcomes() == []

    def test_on_outcome_callback_still_invoked_via_chained(self):
        """External on_outcome callback passed to TradingOrchestrator must
        still be invoked (chained through) even though outcomes are also
        buffered internally."""
        config = _make_config()
        trader = _make_trader()
        trader.maybe_execute = MagicMock(
            return_value={"action": "buy", "status": "simulated", "amount": 0.001}
        )
        ext_calls: list = []
        orch = TradingOrchestrator(
            trader, config,
            on_outcome=lambda sig, out: ext_calls.append((sig.pair, out)),
        )
        orch.start()
        orch.signal_queue.put(
            TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        )
        import time as _time
        _time.sleep(0.4)
        orch.stop()
        # External callback must have been invoked
        assert len(ext_calls) >= 1
        # Outcomes must also have been buffered (or already popped by nothing)
        # — just verify no exception occurred


class TestInitializePairs:
    """Trader.initialize_pairs() loads the pair list from the exchange exactly
    once and is idempotent on subsequent calls."""

    def _make_trader(self, pairs_response):
        """Build a minimal Trader mock backed by a fake client."""
        from unittest.mock import MagicMock, patch
        from bot.config import BotConfig
        config = BotConfig(api_key=None, pair="btc_idr")
        client = MagicMock()
        client.get_pairs = MagicMock(return_value=pairs_response)
        client.configure_caches = MagicMock()
        with patch("bot.trader.Trader.__init__", return_value=None):
            from bot.trader import Trader
            t = object.__new__(Trader)
            t.config = config
            t.client = client
            t._all_pairs = None
            t._pair_min_order_cache_cycles = 0
        return t, client

    def test_initialize_pairs_loads_all_pairs(self):
        """initialize_pairs populates _all_pairs from the exchange."""
        from bot.trader import Trader
        t, client = self._make_trader([
            {"name": "btc_idr"}, {"name": "eth_idr"}, {"name": "sol_idr"},
        ])
        t.initialize_pairs()
        assert t._all_pairs == ["btc_idr", "eth_idr", "sol_idr"]
        client.get_pairs.assert_called_once()

    def test_initialize_pairs_is_idempotent(self):
        """initialize_pairs must not call the API again when already populated."""
        from bot.trader import Trader
        t, client = self._make_trader([{"name": "btc_idr"}])
        t._all_pairs = ["btc_idr", "eth_idr"]  # already loaded
        t.initialize_pairs()
        client.get_pairs.assert_not_called()
        assert t._all_pairs == ["btc_idr", "eth_idr"]

    def test_initialize_pairs_fallback_on_error(self):
        """initialize_pairs falls back to config.pair when the API fails."""
        import requests
        from bot.trader import Trader
        t, client = self._make_trader([])
        client.get_pairs.side_effect = requests.RequestException("timeout")
        t.initialize_pairs()
        assert t._all_pairs == ["btc_idr"]

    def test_scan_and_choose_uses_initialize_pairs(self):
        """scan_and_choose must not call get_pairs a second time when
        initialize_pairs was already called before it."""
        from bot.trader import Trader
        t, client = self._make_trader([{"name": "btc_idr"}, {"name": "eth_idr"}])
        t.initialize_pairs()
        assert client.get_pairs.call_count == 1
        # Subsequent calls to initialize_pairs (as done inside scan_and_choose)
        # must not trigger another API call.
        t.initialize_pairs()
        assert client.get_pairs.call_count == 1


# ===========================================================================
# Dead-coin trade-count filter — missing ticker field must not block all pairs
# ===========================================================================

class TestRugPullTradeCountFilter:
    """detect_rug_pull_risk must only apply the trade-count filter when the
    ticker actually exposes a trade_count / count field.  Indodax tickers do
    not include this field, so the filter must be silently skipped to prevent
    every live pair from being flagged as a dead coin."""

    def test_missing_trade_count_field_does_not_flag_dead(self):
        """When the ticker has no trade_count / count field, min_trades_24h
        check must be skipped — even if set to a large value."""
        from bot.analysis import detect_rug_pull_risk
        ticker = {"ticker": {"high": "100", "last": "95", "vol_idr": "5000000"}}
        result = detect_rug_pull_risk(ticker, min_trades_24h=1000)
        assert not result.detected, (
            f"Expected no detection (field missing), got reason={result.reason!r}"
        )

    def test_explicit_zero_trade_count_flags_dead(self):
        """When the ticker explicitly provides trade_count=0, the filter
        must trigger for min_trades_24h > 0."""
        from bot.analysis import detect_rug_pull_risk
        ticker = {"ticker": {"high": "10", "last": "9.5", "vol_idr": "500000",
                              "trade_count": "0"}}
        result = detect_rug_pull_risk(ticker, min_trades_24h=10)
        assert result.detected
        assert "dead_coin_trades" in result.reason

    def test_sufficient_trade_count_does_not_flag(self):
        """When trade_count field is present and above threshold, no flag."""
        from bot.analysis import detect_rug_pull_risk
        ticker = {"ticker": {"high": "100", "last": "95", "vol_idr": "5000000",
                              "count": "500"}}
        result = detect_rug_pull_risk(ticker, min_trades_24h=100)
        assert not result.detected

    def test_btc_like_ticker_no_trade_count_passes(self):
        """Ticker without trade_count (like Indodax BTC) must pass all checks
        when volume and price-drop checks are also satisfied."""
        from bot.analysis import detect_rug_pull_risk
        # Typical Indodax btc_idr ticker shape — no trade_count field
        ticker = {"ticker": {"high": "1600000000", "last": "1550000000",
                              "vol_idr": "50000000000"}}
        result = detect_rug_pull_risk(
            ticker,
            max_drop_24h_pct=0.50,
            min_volume_24h_idr=1_000_000,
            min_trades_24h=100,  # non-zero but field missing → must be skipped
        )
        assert not result.detected, (
            f"btc_idr-like ticker was incorrectly blocked: {result.reason!r}"
        )


# ===========================================================================
# TradingSignal metadata field
# ===========================================================================

class TestTradingSignalMetadata:
    """TradingSignal must carry an optional metadata dict."""

    def test_default_metadata_is_empty(self):
        snap = _make_snap()
        sig = TradingSignal(pair="btc_idr", snapshot=snap, signal_type="buy")
        assert sig.metadata == {}

    def test_metadata_stores_partial_tp_info(self):
        snap = _make_snap()
        sig = TradingSignal(
            pair="btc_idr",
            snapshot=snap,
            signal_type="partial_tp",
            metadata={"fraction": 0.5, "level": 2},
        )
        assert sig.metadata["fraction"] == 0.5
        assert sig.metadata["level"] == 2

    def test_metadata_stores_cancel_reason(self):
        snap = _make_snap()
        sig = TradingSignal(
            pair="btc_idr",
            snapshot=snap,
            signal_type="cancel_buy",
            metadata={"reason": "sell signal (bearish candle)"},
        )
        assert "sell signal" in sig.metadata["reason"]


# ===========================================================================
# StrategyEngine — re-analysis gate for resume_buy
# ===========================================================================

class TestStrategyEngineResumeBuyCancelGate:
    """StrategyEngine must emit cancel_buy (not resume_buy) when market
    conditions turn bearish while a pending buy is open."""

    def _make_strategy_engine(self, snap, tracker) -> StrategyEngine:
        config = _make_config()
        config.trailing_stop_pct = 0.0
        config.trailing_tp_pct = 0.0
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        cache = DataCache()
        signals = SignalQueue()
        se = StrategyEngine(trader, config, cache, signals)
        return se

    def test_sell_decision_cancels_pending_buy(self):
        """Sell signal while buy is pending → cancel_buy signal emitted."""
        snap = _make_snap(action="sell")
        tracker = _make_tracker(base_position=0.0, has_pending_buy=True)
        tracker.base_position = 0.0
        se = self._make_strategy_engine(snap, tracker)
        signals = SignalQueue()
        se._signals = signals
        se._evaluate("btc_idr", snap)
        sig = signals.get(timeout=0.1)
        assert sig is not None
        assert sig.signal_type == "cancel_buy"
        assert "sell signal" in sig.metadata.get("reason", "")

    def test_trending_down_regime_cancels_pending_buy_when_no_position(self):
        """trending_down regime ≥ 0.5 while no position → cancel_buy signal."""
        snap = _make_snap(action="hold")
        regime = MagicMock()
        regime.regime = "trending_down"
        regime.strength = 0.7
        snap["regime"] = regime
        tracker = _make_tracker(base_position=0.0, has_pending_buy=True)
        se = self._make_strategy_engine(snap, tracker)
        signals = SignalQueue()
        se._signals = signals
        se._evaluate("btc_idr", snap)
        sig = signals.get(timeout=0.1)
        assert sig is not None
        assert sig.signal_type == "cancel_buy"
        assert "trending_down" in sig.metadata.get("reason", "")

    def test_trending_down_does_not_cancel_when_position_held(self):
        """When base_position > 0 (partial fill), bearish regime must NOT cancel
        the pending buy — we already hold coins and must let the buy complete."""
        snap = _make_snap(action="hold")
        regime = MagicMock()
        regime.regime = "trending_down"
        regime.strength = 0.8
        snap["regime"] = regime
        tracker = _make_tracker(base_position=0.001, has_pending_buy=True)
        se = self._make_strategy_engine(snap, tracker)
        signals = SignalQueue()
        se._signals = signals
        se._evaluate("btc_idr", snap)
        sig = signals.get(timeout=0.1)
        assert sig is not None
        # Must still resume, not cancel
        assert sig.signal_type == "resume_buy"

    def test_hold_decision_resumes_pending_buy(self):
        """Neutral market (hold decision) keeps the resume_buy signal path."""
        snap = _make_snap(action="hold")
        tracker = _make_tracker(base_position=0.0, has_pending_buy=True)
        se = self._make_strategy_engine(snap, tracker)
        signals = SignalQueue()
        se._signals = signals
        se._evaluate("btc_idr", snap)
        sig = signals.get(timeout=0.1)
        assert sig is not None
        assert sig.signal_type == "resume_buy"

    def test_trending_down_low_strength_resumes_pending_buy(self):
        """trending_down with strength < 0.5 must NOT trigger cancel."""
        snap = _make_snap(action="hold")
        regime = MagicMock()
        regime.regime = "trending_down"
        regime.strength = 0.4  # below threshold
        snap["regime"] = regime
        tracker = _make_tracker(base_position=0.0, has_pending_buy=True)
        se = self._make_strategy_engine(snap, tracker)
        signals = SignalQueue()
        se._signals = signals
        se._evaluate("btc_idr", snap)
        sig = signals.get(timeout=0.1)
        assert sig is not None
        assert sig.signal_type == "resume_buy"


# ===========================================================================
# StrategyEngine — partial take-profit signals
# ===========================================================================

class TestStrategyEnginePartialTP:
    """StrategyEngine must emit partial_tp signals when partial TP conditions
    are met for an open position."""

    def _evaluate(self, snap, tracker, **cfg_overrides):
        defaults = {
            "partial_tp_fraction": 0.0,
            "partial_tp2_fraction": 0.0,
            "partial_tp2_target_pct": 0.0,
            "partial_tp3_fraction": 0.0,
            "partial_tp3_target_pct": 0.0,
            "trailing_stop_pct": 0.0,
            "trailing_tp_pct": 0.0,
        }
        defaults.update(cfg_overrides)
        config = _make_config(**defaults)
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        cache = DataCache()
        signals = SignalQueue()
        se = StrategyEngine(trader, config, cache, signals)
        se._evaluate("btc_idr", snap)
        return signals.get(timeout=0.1)

    def test_partial_tp1_emitted_when_price_at_tp(self):
        """partial_tp level 1 must be emitted when price >= decision.take_profit."""
        snap = _make_snap(price=110_000.0, action="hold")
        snap["decision"].take_profit = 105_000.0
        tracker = _make_tracker(base_position=0.001)
        tracker.partial_tp_taken = False
        tracker.stop_reason = MagicMock(return_value=None)
        sig = self._evaluate(snap, tracker, partial_tp_fraction=0.5)
        assert sig is not None
        assert sig.signal_type == "partial_tp"
        assert sig.metadata["level"] == 1
        assert sig.metadata["fraction"] == 0.5

    def test_partial_tp1_not_emitted_when_price_below_tp(self):
        """partial_tp level 1 must NOT be emitted when price < decision.take_profit."""
        snap = _make_snap(price=100_000.0, action="hold")
        snap["decision"].take_profit = 110_000.0
        tracker = _make_tracker(base_position=0.001)
        tracker.partial_tp_taken = False
        tracker.stop_reason = MagicMock(return_value=None)
        sig = self._evaluate(snap, tracker, partial_tp_fraction=0.5)
        # Should be None (holding, no signal)
        assert sig is None

    def test_partial_tp1_not_emitted_when_already_taken(self):
        """partial_tp level 1 must NOT repeat after the flag is set."""
        snap = _make_snap(price=115_000.0, action="hold")
        snap["decision"].take_profit = 105_000.0
        tracker = _make_tracker(base_position=0.001)
        tracker.partial_tp_taken = True  # already taken
        tracker.stop_reason = MagicMock(return_value=None)
        sig = self._evaluate(snap, tracker, partial_tp_fraction=0.5)
        assert sig is None

    def test_partial_tp2_emitted_when_price_above_avg_cost_target(self):
        """partial_tp level 2 must be emitted when price >= avg_cost × (1 + target)."""
        snap = _make_snap(price=120_000.0, action="hold")
        snap["decision"].take_profit = None  # no level-1 TP
        tracker = _make_tracker(base_position=0.001)
        tracker.partial_tp_taken = True   # level 1 done
        tracker.partial_tp2_taken = False
        tracker.avg_cost = 100_000.0
        tracker.stop_reason = MagicMock(return_value=None)
        # 120_000 >= 100_000 * (1 + 0.15)
        sig = self._evaluate(
            snap, tracker,
            partial_tp2_fraction=0.3,
            partial_tp2_target_pct=0.15,
        )
        assert sig is not None
        assert sig.signal_type == "partial_tp"
        assert sig.metadata["level"] == 2
        assert sig.metadata["fraction"] == 0.3

    def test_partial_tp3_emitted_correctly(self):
        """partial_tp level 3 must be emitted when price >= avg_cost × (1 + target)."""
        snap = _make_snap(price=130_000.0, action="hold")
        snap["decision"].take_profit = None
        tracker = _make_tracker(base_position=0.001)
        tracker.partial_tp_taken = True
        tracker.partial_tp2_taken = True
        tracker.partial_tp3_taken = False
        tracker.avg_cost = 100_000.0
        tracker.stop_reason = MagicMock(return_value=None)
        # 130_000 >= 100_000 * (1 + 0.25)
        sig = self._evaluate(
            snap, tracker,
            partial_tp3_fraction=0.2,
            partial_tp3_target_pct=0.25,
        )
        assert sig is not None
        assert sig.signal_type == "partial_tp"
        assert sig.metadata["level"] == 3
        assert sig.metadata["fraction"] == 0.2


# ===========================================================================
# ExecutionEngine — cancel_buy and partial_tp signal handling
# ===========================================================================

class TestExecutionEngineCancelBuyAndPartialTP:
    """ExecutionEngine must correctly handle cancel_buy and partial_tp signals."""

    def _make_exec_engine(self, trader):
        config = _make_config()
        signals = SignalQueue()
        risk = RiskEngine(config)
        return ExecutionEngine(trader, config, signals, risk), signals

    def test_cancel_buy_calls_cancel_methods(self):
        """cancel_buy signal must cancel orders and clear the pending buy state."""
        tracker = _make_tracker(base_position=0.0, has_pending_buy=True)
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader._cancel_open_orders = MagicMock()
        trader._persist_after_trade = MagicMock()
        trader._resume_buy_retry_counts = {}
        trader.multi_manager = None

        exec_eng, signals = self._make_exec_engine(trader)
        snap = _make_snap(price=100_000.0, action="sell")
        signal = TradingSignal(
            pair="btc_idr",
            snapshot=snap,
            signal_type="cancel_buy",
            metadata={"reason": "sell signal (bearish)"},
        )
        exec_eng._process(signal)
        trader._cancel_open_orders.assert_called_once_with("btc_idr", ("buy",))
        tracker.cancel_pending_buy.assert_called_once()

    def test_cancel_buy_returns_cancelled_outcome(self):
        """cancel_buy must produce an outcome with action=cancel_buy + status=cancelled."""
        outcomes = []
        tracker = _make_tracker(base_position=0.0, has_pending_buy=True)
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader._cancel_open_orders = MagicMock()
        trader._persist_after_trade = MagicMock()
        trader._resume_buy_retry_counts = {}
        trader.multi_manager = None

        config = _make_config()
        signals = SignalQueue()
        risk = RiskEngine(config)
        exec_eng = ExecutionEngine(
            trader, config, signals, risk,
            on_outcome=lambda sig, out: outcomes.append(out),
        )
        snap = _make_snap(price=100_000.0, action="sell")
        signal = TradingSignal(
            pair="btc_idr",
            snapshot=snap,
            signal_type="cancel_buy",
            metadata={"reason": "sell signal"},
        )
        exec_eng._process(signal)
        assert len(outcomes) == 1
        assert outcomes[0]["action"] == "cancel_buy"
        assert outcomes[0]["status"] == "cancelled"

    def test_partial_tp_calls_partial_take_profit(self):
        """partial_tp signal must invoke trader.partial_take_profit with correct fraction."""
        tracker = _make_tracker(base_position=0.001)
        tracker.as_dict = MagicMock(return_value={"realized_pnl": 1000.0})
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader.partial_take_profit = MagicMock(
            return_value={"status": "partial_tp", "amount": 0.0005}
        )

        exec_eng, _ = self._make_exec_engine(trader)
        snap = _make_snap(price=115_000.0, action="hold")
        signal = TradingSignal(
            pair="btc_idr",
            snapshot=snap,
            signal_type="partial_tp",
            metadata={"fraction": 0.5, "level": 1},
        )
        exec_eng._process(signal)
        trader.partial_take_profit.assert_called_once_with(snap, 0.5)

    def test_partial_tp_level2_sets_partial_tp2_taken(self):
        """After level-2 partial TP, tracker.partial_tp2_taken must be set to True."""
        tracker = _make_tracker(base_position=0.001)
        tracker.as_dict = MagicMock(return_value={"realized_pnl": 500.0})
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader.partial_take_profit = MagicMock(
            return_value={"status": "partial_tp", "amount": 0.0003}
        )

        exec_eng, _ = self._make_exec_engine(trader)
        snap = _make_snap(price=120_000.0, action="hold")
        signal = TradingSignal(
            pair="btc_idr",
            snapshot=snap,
            signal_type="partial_tp",
            metadata={"fraction": 0.3, "level": 2},
        )
        exec_eng._process(signal)
        assert tracker.partial_tp2_taken is True

    def test_partial_tp_level3_sets_partial_tp3_taken(self):
        """After level-3 partial TP, tracker.partial_tp3_taken must be set to True."""
        tracker = _make_tracker(base_position=0.001)
        tracker.as_dict = MagicMock(return_value={"realized_pnl": 750.0})
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader.partial_take_profit = MagicMock(
            return_value={"status": "partial_tp", "amount": 0.0002}
        )

        exec_eng, _ = self._make_exec_engine(trader)
        snap = _make_snap(price=130_000.0, action="hold")
        signal = TradingSignal(
            pair="btc_idr",
            snapshot=snap,
            signal_type="partial_tp",
            metadata={"fraction": 0.2, "level": 3},
        )
        exec_eng._process(signal)
        assert tracker.partial_tp3_taken is True


class TestExecutionEngineReentryAnalysis:
    """ExecutionEngine must call analyze_reentry_opportunity after a successful exit."""

    def _make_exec_engine(self, trader):
        config = _make_config()
        signals = SignalQueue()
        risk = RiskEngine(config)
        return ExecutionEngine(trader, config, signals, risk), signals

    def test_reentry_analysis_called_after_exit(self):
        """After a successful 'exit' signal (status != pending_sell),
        ExecutionEngine must call trader.analyze_reentry_opportunity with the snapshot."""
        tracker = _make_tracker(base_position=0.001)
        tracker.as_dict = MagicMock(return_value={"realized_pnl": 5000.0})
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader.force_sell = MagicMock(return_value={"status": "simulated", "amount": 0.001})
        trader.analyze_reentry_opportunity = MagicMock(return_value={"reentry": False})

        exec_eng, _ = self._make_exec_engine(trader)
        snap = _make_snap(price=120_000.0, action="sell")
        signal = TradingSignal(pair="btc_idr", snapshot=snap, signal_type="exit")
        exec_eng._process(signal)

        trader.analyze_reentry_opportunity.assert_called_once_with(snap)

    def test_reentry_analysis_not_called_when_pending_sell(self):
        """When force_sell returns 'pending_sell', the exit was not completed —
        analyze_reentry_opportunity must NOT be called."""
        tracker = _make_tracker(base_position=0.001)
        tracker.as_dict = MagicMock(return_value={"realized_pnl": 0.0})
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader.force_sell = MagicMock(return_value={"status": "pending_sell"})
        trader.analyze_reentry_opportunity = MagicMock(return_value={"reentry": False})

        exec_eng, _ = self._make_exec_engine(trader)
        snap = _make_snap(price=120_000.0, action="sell")
        signal = TradingSignal(pair="btc_idr", snapshot=snap, signal_type="exit")
        exec_eng._process(signal)

        trader.analyze_reentry_opportunity.assert_not_called()

    def test_reentry_opportunity_notifies_when_detected(self):
        """When analyze_reentry_opportunity returns reentry=True,
        the notify function must be called with a RE-ENTRY message."""
        tracker = _make_tracker(base_position=0.001)
        tracker.as_dict = MagicMock(return_value={"realized_pnl": 2000.0})
        trader = _make_trader()
        trader._active_tracker = MagicMock(return_value=tracker)
        trader.force_sell = MagicMock(return_value={"status": "simulated", "amount": 0.001})
        trader.analyze_reentry_opportunity = MagicMock(return_value={
            "reentry": True,
            "target_price": 115_000.0,
            "regime": "trending_up",
            "imbalance": 0.65,
        })
        notifications: list = []

        config = _make_config()
        signals = SignalQueue()
        risk = RiskEngine(config)
        exec_eng = ExecutionEngine(
            trader, config, signals, risk,
            notify_fn=lambda text: notifications.append(text),
        )
        snap = _make_snap(price=120_000.0, action="sell")
        signal = TradingSignal(pair="btc_idr", snapshot=snap, signal_type="exit")
        exec_eng._process(signal)

        assert len(notifications) >= 2  # EXIT + RE-ENTRY
        re_entry_msg = next(
            (n for n in notifications if "RE-ENTRY" in n), None
        )
        assert re_entry_msg is not None
        assert "115,000" in re_entry_msg or "115000" in re_entry_msg.replace(",", "")
