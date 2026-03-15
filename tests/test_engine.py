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
        tracker._daily_pnl = -60_000.0  # exceeds 5% of 1M
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
        tracker._daily_pnl = -10_000.0  # only 1% loss, under 5% cap
        trader._active_tracker = MagicMock(return_value=tracker)
        sig = TradingSignal(pair="btc_idr", snapshot=_make_snap(price=100.0), signal_type="buy")
        ok, _ = risk.approve(sig, trader)
        assert ok is True


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
