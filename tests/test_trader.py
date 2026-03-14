import logging
import time
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any

import requests

from bot.analysis import WhaleActivity, Candle, OrderbookInsight, TrendResult
from bot.config import BotConfig
from bot.strategies import StrategyDecision
from bot.trader import Trader
from bot.tracking import PortfolioTracker
from bot.grid import GridPlan, GridOrder


class VolStub:
    def __init__(self, volatility: float) -> None:
        self.volatility = volatility


class StubTrader(Trader):
    """Trader that returns pre-built snapshots without making real API calls."""

    class _Client:
        def __init__(self, pairs: list[str]) -> None:
            self._pairs = pairs

        def get_pairs(self) -> list[dict]:
            return [{"name": p} for p in self._pairs]

        def get_summaries(self) -> dict:
            return {}

    def __init__(self, config: BotConfig, snapshots: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(config, client=self._Client(list(snapshots.keys())))
        self._snapshots = snapshots

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
        key = pair or self.config.pair
        return self._snapshots[key]


class GuardedTrader(Trader):
    """Trader with stubbed client to test balance guards."""

    class _Client:
        _pair_min_order: Dict[str, Any] = {}

        def get_depth(self, pair: str, count: int = 5) -> Dict[str, Any]:
            return {"buy": [["100", "1"]], "sell": [["100.05", "1"]]}

        def get_summaries(self) -> dict:
            return {}

        def get_pair_min_order(self, pair: str) -> Dict[str, float]:
            return {"min_coin": 0.0, "min_idr": 0.0}

        def load_pair_min_orders(self) -> None:
            pass

    def __init__(self, config: BotConfig) -> None:
        super().__init__(config, client=self._Client())


class AutoPairsTrader(Trader):
    """Trader that auto-loads pairs from stubbed client and analyzes provided snapshots."""

    class _Client:
        def __init__(self, pairs: list[str]) -> None:
            self._pairs = pairs

        def get_pairs(self) -> list[Dict[str, Any]]:
            return [{"name": p} for p in self._pairs]

        def get_summaries(self) -> dict:
            return {}

    def __init__(self, config: BotConfig, snapshots: Dict[str, Dict[str, Any]]) -> None:
        super().__init__(config, client=self._Client(list(snapshots.keys())))
        self._snapshots = snapshots

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
        key = pair or self.config.pair
        return self._snapshots[key]


class AllFailTrader(Trader):
    """Trader that always fails to analyze markets to simulate network/API outages."""

    class _Client:
        def get_pairs(self) -> list[dict]:
            return [{"name": "a_idr"}, {"name": "b_idr"}]

        def get_summaries(self) -> dict:
            return {}

    def __init__(self, config: BotConfig) -> None:
        super().__init__(config, client=self._Client())

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
        raise requests.RequestException("network unavailable")


class TraderSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_scan_and_choose_picks_best_confidence(self) -> None:
        config = BotConfig(api_key=None)
        snapshots = {
            "a_idr": {
                "pair": "a_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="buy",
                    confidence=0.4,
                    reason="low",
                    target_price=100,
                    amount=0.1,
                    stop_loss=99,
                    take_profit=101,
                ),
            },
            "b_idr": {
                "pair": "b_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="buy",
                    confidence=0.8,
                    reason="high",
                    target_price=100,
                    amount=0.1,
                    stop_loss=99,
                    take_profit=101,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "b_idr")
        self.assertEqual(snapshot["decision"].confidence, 0.8)

    def test_scan_and_choose_without_manual_input_uses_auto_pairs(self) -> None:
        config = BotConfig(api_key=None, pair="manual_idr")
        snapshots = {
            "auto_a": {
                "pair": "auto_a",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="swing_trading",
                    action="buy",
                    confidence=0.3,  # below min_confidence=0.52 → no early exit here
                    reason="weak",
                    target_price=100,
                    amount=0.1,
                    stop_loss=95,
                    take_profit=110,
                ),
            },
            "auto_b": {
                "pair": "auto_b",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="buy",
                    confidence=0.9,  # above threshold → serial early exit
                    reason="better",
                    target_price=100,
                    amount=0.1,
                    stop_loss=95,
                    take_profit=110,
                ),
            },
        }
        trader = AutoPairsTrader(config, snapshots)
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "auto_b")
        self.assertEqual(snapshot["decision"].confidence, 0.9)
        self.assertEqual(trader.config.pair, "manual_idr")  # config stays as fallback

    def test_scan_and_choose_falls_back_when_all_hold(self) -> None:
        config = BotConfig(api_key=None, pair="fallback_idr")
        snapshots = {
            "a_idr": {
                "pair": "a_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="hold",
                    confidence=0.4,
                    reason="hold",
                    target_price=100,
                    amount=0.0,
                    stop_loss=None,
                    take_profit=None,
                ),
            },
            "b_idr": {
                "pair": "b_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="hold",
                    confidence=0.3,
                    reason="hold",
                    target_price=100,
                    amount=0.0,
                    stop_loss=None,
                    take_profit=None,
                ),
            },
            "fallback_idr": {
                "pair": "fallback_idr",
                "price": 120.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="position_trading",
                    action="buy",
                    confidence=0.6,
                    reason="fallback",
                    target_price=120,
                    amount=0.2,
                    stop_loss=110,
                    take_profit=130,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "fallback_idr")
        self.assertEqual(snapshot["decision"].action, "buy")

    def test_scan_and_choose_ignores_sell_when_no_position(self) -> None:
        """A SELL signal must not be returned by scan_and_choose when the bot
        holds no open position.  Without this guard the scanner would lock onto
        the first overbought pair every cycle, skip it as "insufficient
        position" in maybe_execute, and never look for a BUY opportunity."""
        config = BotConfig(api_key=None)
        snapshots = {
            "sell_idr": {
                "pair": "sell_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="position_trading",
                    action="sell",
                    confidence=0.9,  # high confidence but no position
                    reason="overbought",
                    target_price=100,
                    amount=0.5,
                    stop_loss=None,
                    take_profit=None,
                ),
            },
            "buy_idr": {
                "pair": "buy_idr",
                "price": 100.0,
                "trend": None,
                "orderbook": None,
                "volatility": None,
                "levels": None,
                "decision": StrategyDecision(
                    mode="day_trading",
                    action="buy",
                    confidence=0.55,  # lower confidence but actionable
                    reason="good entry",
                    target_price=110,
                    amount=0.2,
                    stop_loss=95,
                    take_profit=115,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        # No position held — sell_idr's SELL should be skipped
        self.assertEqual(trader.tracker.base_position, 0.0)
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "buy_idr")
        self.assertEqual(snapshot["decision"].action, "buy")

    def test_scan_and_choose_raises_when_all_pairs_fail(self) -> None:
        config = BotConfig(api_key=None)
        trader = AllFailTrader(config)
        with self.assertRaises(RuntimeError) as ctx:
            trader.scan_and_choose()
        self.assertIn("a_idr", str(ctx.exception))

    def test_maybe_execute_limits_buy_amount_by_available_cash(self) -> None:
        config = BotConfig(
            api_key=None,
            dry_run=True,
            initial_capital=50.0,
            max_loss_pct=0.9,
            target_profit_pct=1.0,
            min_order_idr=1.0,  # disable minimum-order guard (not under test here)
        )
        trader = GuardedTrader(config)
        trader.tracker.cash = 50.0  # very small cash
        decision = StrategyDecision(
            mode="day_trading",
            action="buy",
            confidence=1.0,
            reason="test",
            target_price=100,
            amount=10.0,
            stop_loss=95.0,
            take_profit=105.0,
        )
        snapshot = {
            "pair": "btc_idr",
            "price": 100.0,
            "decision": decision,
        }
        outcome = trader.maybe_execute(snapshot)
        self.assertEqual(outcome["status"], "simulated")
        self.assertLessEqual(outcome["amount"], 0.5)  # 50 cash / 100 price

    def test_staged_entries_reduce_full_allocation_on_high_vol(self) -> None:
        config = BotConfig(
            api_key=None,
            dry_run=True,
            initial_capital=1000.0,
            max_loss_pct=0.9,
            target_profit_pct=1.0,
            staged_entry_steps=3,
            staged_entry_min_equity=0,  # disable small-equity single-step override
            min_order_idr=1.0,  # disable minimum-order guard (not under test here)
        )
        trader = GuardedTrader(config)
        trader.tracker.cash = 1000.0
        decision = StrategyDecision(
            mode="swing_trading",
            action="buy",
            confidence=0.7,
            reason="test-staged",
            target_price=100,
            amount=3.0,
            stop_loss=95.0,
            take_profit=110.0,
        )
        snapshot = {
            "pair": "btc_idr",
            "price": 100.0,
            "decision": decision,
            "volatility": VolStub(0.03),  # high vol triggers staging
        }
        outcome = trader.maybe_execute(snapshot)
        self.assertEqual(outcome["status"], "simulated")
        self.assertIn("executed_steps", outcome)
        self.assertGreater(len(outcome["executed_steps"]), 1)
        self.assertLessEqual(outcome["amount"], decision.amount)

    def test_staged_entry_collapsed_to_single_step_for_small_equity(self) -> None:
        """When cash < staged_entry_min_equity, staged entry is forced to 1 step."""
        config = BotConfig(
            api_key=None,
            dry_run=True,
            initial_capital=500_000.0,
            max_loss_pct=0.9,
            target_profit_pct=1.0,
            staged_entry_steps=3,
            staged_entry_min_equity=1_000_000.0,  # threshold above current cash
            min_order_idr=1.0,
        )
        trader = GuardedTrader(config)
        trader.tracker.cash = 500_000.0  # below threshold
        decision = StrategyDecision(
            mode="day_trading",
            action="buy",
            confidence=0.6,
            reason="test-small-equity",
            target_price=100,
            amount=3.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {
            "pair": "btc_idr",
            "price": 100.0,
            "decision": decision,
            "volatility": VolStub(0.03),  # high vol — would normally trigger 3 steps
        }
        outcome = trader.maybe_execute(snapshot)
        self.assertEqual(outcome["status"], "simulated")
        self.assertIn("executed_steps", outcome)
        self.assertEqual(len(outcome["executed_steps"]), 1, "small equity must use single-step entry")

    def test_staged_entry_uses_multiple_steps_above_equity_threshold(self) -> None:
        """When cash >= staged_entry_min_equity, multi-step staged entry is used."""
        config = BotConfig(
            api_key=None,
            dry_run=True,
            initial_capital=2_000_000.0,
            max_loss_pct=0.9,
            target_profit_pct=1.0,
            staged_entry_steps=3,
            staged_entry_min_equity=1_000_000.0,  # threshold below current cash
            min_order_idr=1.0,
        )
        trader = GuardedTrader(config)
        trader.tracker.cash = 2_000_000.0  # above threshold
        decision = StrategyDecision(
            mode="day_trading",
            action="buy",
            confidence=0.6,
            reason="test-large-equity",
            target_price=100,
            amount=3.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {
            "pair": "btc_idr",
            "price": 100.0,
            "decision": decision,
            "volatility": VolStub(0.03),  # high vol triggers staging
        }
        outcome = trader.maybe_execute(snapshot)
        self.assertEqual(outcome["status"], "simulated")
        self.assertIn("executed_steps", outcome)
        self.assertGreater(len(outcome["executed_steps"]), 1, "large equity must use multi-step entry")

    def test_force_sell_liquidates_entire_position(self) -> None:
        config = BotConfig(
            api_key=None,
            dry_run=True,
            initial_capital=1000.0,
            max_loss_pct=0.9,
            target_profit_pct=2.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        # Simulate an open position: bought 5 units at 90
        trader.tracker.record_trade("buy", 90.0, 5.0)
        self.assertAlmostEqual(trader.tracker.base_position, 5.0)
        decision = StrategyDecision(
            mode="day_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=100,
            amount=5.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "btc_idr", "price": 100.0, "decision": decision}
        outcome = trader.force_sell(snapshot)
        self.assertEqual(outcome["status"], "force_sold")
        self.assertEqual(outcome["action"], "sell")
        self.assertAlmostEqual(outcome["amount"], 5.0, places=5)
        self.assertEqual(trader.tracker.base_position, 0.0)

    def test_force_sell_returns_no_position_when_not_holding(self) -> None:
        config = BotConfig(api_key=None, dry_run=True)
        trader = GuardedTrader(config)
        decision = StrategyDecision(
            mode="day_trading",
            action="hold",
            confidence=0.5,
            reason="test",
            target_price=100,
            amount=0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "btc_idr", "price": 100.0, "decision": decision}
        outcome = trader.force_sell(snapshot)
        self.assertEqual(outcome["status"], "no_position")

    def test_force_sell_cancels_unfilled_buy_and_returns_no_position(self) -> None:
        """force_sell must gracefully handle unfilled buy limit orders.

        When the exchange balance is 0 (buy was placed but not filled), the bot
        should cancel pending buy orders and roll back the phantom position
        instead of raising a RuntimeError.
        """
        import unittest.mock as mock

        cancelled_orders: list = []

        class _LiveClient(GuardedTrader._Client):
            def get_account_info(self):
                # Exchange shows 0 WTEC — buy limit order not yet filled
                return {"return": {"balance": {"wtec": "0", "idr": "459125"}}}

            def open_orders(self, pair: str):
                return {
                    "return": {
                        "orders": [{"order_id": "1975685", "type": "buy", "price": "3.0"}]
                    }
                }

            def cancel_order(self, pair: str, order_id: str, order_type: str | None = None):
                cancelled_orders.append(order_id)
                return {"success": 1}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        # Simulate bot having recorded a buy of 13625 WTEC at Rp 3
        trader.tracker.record_trade("buy", 3.0, 13625.0)
        self.assertEqual(trader.tracker.base_position, 13625.0)
        post_buy_cash = 500_000.0 - 3.0 * 13625.0
        self.assertAlmostEqual(trader.tracker.cash, post_buy_cash, places=2)

        decision = StrategyDecision(
            mode="swing_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=3.0,
            amount=13625.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "wtec_idr", "price": 3.0, "decision": decision}
        outcome = trader.force_sell(snapshot)

        # Should return no_position (not raise, not place a sell order)
        self.assertEqual(outcome["status"], "no_position")
        self.assertAlmostEqual(outcome["amount"], 0.0)
        # The phantom position must be cleared
        self.assertEqual(trader.tracker.base_position, 0.0)
        # Cash should be restored to pre-buy level
        self.assertAlmostEqual(trader.tracker.cash, 500_000.0, places=2)
        # The pending buy order should have been cancelled
        self.assertIn("1975685", cancelled_orders)

    def test_force_sell_uses_actual_balance_when_partial_fill(self) -> None:
        """force_sell uses the real exchange balance when partially filled."""
        import unittest.mock as mock

        placed_sell_amount: list = []

        class _LiveClient(GuardedTrader._Client):
            def get_account_info(self):
                # Exchange shows only 5000 WTEC instead of tracked 13625
                return {"return": {"balance": {"wtec": "5000", "idr": "459125"}}}

            def open_orders(self, pair: str):
                return {"return": {"orders": [{"order_id": "9999", "type": "buy"}]}}

            def cancel_order(self, pair: str, order_id: str, order_type: str | None = None):
                return {"success": 1}

            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                placed_sell_amount.append(amount)
                return {"success": 1, "return": {}}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        trader.tracker.record_trade("buy", 3.0, 13625.0)

        decision = StrategyDecision(
            mode="swing_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=3.0,
            amount=13625.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "wtec_idr", "price": 3.0, "decision": decision}
        outcome = trader.force_sell(snapshot)

        self.assertEqual(outcome["status"], "force_sold")
        # Sell should use actual exchange balance, not tracked amount
        self.assertAlmostEqual(outcome["amount"], 5000.0, places=2)
        self.assertEqual(placed_sell_amount, [5000.0])

    def test_force_sell_cancels_existing_orders_before_selling(self) -> None:
        """force_sell must cancel pending buy/sell orders so exits don't loop."""
        cancelled: list = []
        created: list = []

        class _LiveClient(GuardedTrader._Client):
            def get_account_info(self):
                return {"return": {"balance": {"btc": "5.0", "idr": "0"}}}

            def open_orders(self, pair: str):
                return {
                    "return": {
                        "orders": [
                            {"order_id": "11", "type": "sell"},
                            {"order_id": "22", "type": "buy"},
                        ]
                    }
                }

            def cancel_order(self, pair: str, order_id: str, order_type: str | None = None):
                cancelled.append((order_id, order_type))
                return {"success": 1}

            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["105.0", "1"]]}

            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                created.append((pair, order_type, price, amount))
                return {"success": 1, "return": {}}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=1_000_000.0,
            min_order_idr=0.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        trader.tracker.record_trade("buy", 100.0, 5.0)

        decision = StrategyDecision(
            mode="day_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=105.0,
            amount=5.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "btc_idr", "price": 105.0, "decision": decision}

        outcome = trader.force_sell(snapshot)

        self.assertEqual(outcome["status"], "force_sold")
        self.assertEqual(trader.tracker.base_position, 0.0)
        self.assertCountEqual(cancelled, [("11", "sell"), ("22", "buy")])
        self.assertEqual(len(created), 1)
        _, order_type, _, amount = created[0]
        self.assertEqual(order_type, "sell")
        self.assertAlmostEqual(amount, 5.0)

    def test_force_sell_below_minimum_clears_dust(self) -> None:
        """When sell value is below min IDR, force_sell must clear the dust position."""

        class _LiveClient(GuardedTrader._Client):
            cancel_called = False
            order_called = False

            def get_depth(self, *a, **kw):
                return {"buy": [["1689", "1000"]], "sell": [["1690", "1000"]]}

            def get_account_info(self):
                # Exchange balance matches tracked amount
                return {"return": {"balance": {"doge": "2.27518843", "idr": "500000"}}}

            def get_pair_min_order(self, pair: str):
                return {"min_idr": 30_000.0}

            def cancel_order(self, *a, **kw):
                _LiveClient.cancel_called = True
                return {"success": 1}

            def create_order(self, *a, **kw):
                _LiveClient.order_called = True
                return {"success": 1}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            min_order_idr=30_000.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        # Position worth ~3.8K IDR (< 30K minimum)
        trader.tracker.record_trade("buy", 1689.0, 2.27518843)

        snapshot = {"pair": "doge_idr", "price": 1689.0, "decision": StrategyDecision(
            mode="scalping", action="sell", confidence=0.9, reason="exit",
            target_price=1689.0, amount=trader.tracker.base_position,
            stop_loss=None, take_profit=None,
        )}

        outcome = trader.force_sell(snapshot)

        self.assertEqual(outcome["status"], "dust_cleared")
        # Position must be cleared since it's unsellable dust
        self.assertAlmostEqual(trader.tracker.base_position, 0.0, places=8)
        self.assertFalse(_LiveClient.order_called)

    def test_analyze_with_retry_succeeds_after_429(self) -> None:
        """_analyze_with_retry must back off and succeed when the first call gets a 429."""
        config = BotConfig(api_key=None, scan_request_delay=0.0)
        trader = GuardedTrader(config)
        # Simulate a snapshot that would be returned on success
        success_snapshot: Dict[str, Any] = {
            "pair": "btc_idr", "price": 100.0, "decision": StrategyDecision(
                mode="scalping", action="buy", confidence=0.7, reason="ok",
                target_price=100, amount=0.01, stop_loss=99, take_profit=101,
            ),
        }
        calls: list[int] = []

        def fake_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            calls.append(1)
            if len(calls) == 1:
                # Build a minimal HTTPError with status 429
                resp = requests.Response()
                resp.status_code = 429
                raise requests.HTTPError(response=resp)
            return success_snapshot

        trader.analyze_market = fake_analyze  # type: ignore[method-assign]
        import unittest.mock as mock
        with mock.patch("bot.trader.time.sleep") as mock_sleep:
            result = trader._analyze_with_retry("btc_idr")
        self.assertEqual(result["pair"], "btc_idr")
        self.assertEqual(len(calls), 2)  # failed once, succeeded on retry
        # Verify exponential backoff: first retry = BACKOFF_BASE * 2^0 = 2.0s
        mock_sleep.assert_called_once_with(trader._SCAN_BACKOFF_BASE)

    def test_analyze_with_retry_raises_after_max_retries(self) -> None:
        """After MAX_SCAN_RETRIES 429s, _analyze_with_retry must re-raise the last error.

        Crucially, sleep() must NOT be called after the final attempt: the
        backoff delay only makes sense when a subsequent retry will follow it.
        Sleeping after the last failure would block the scan for no benefit.
        """
        config = BotConfig(api_key=None, scan_request_delay=0.0)
        trader = GuardedTrader(config)
        sleep_calls: list[float] = []

        def always_429(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            resp = requests.Response()
            resp.status_code = 429
            raise requests.HTTPError(response=resp)

        trader.analyze_market = always_429  # type: ignore[method-assign]
        import unittest.mock as mock
        with mock.patch("bot.trader.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with self.assertRaises(requests.HTTPError):
                trader._analyze_with_retry("btc_idr")
        # Sleep happens only between retries (before each retry), not after the
        # final failed attempt — so the number of sleeps is MAX_SCAN_RETRIES - 1.
        self.assertEqual(len(sleep_calls), trader._MAX_SCAN_RETRIES - 1)
        for i, delay in enumerate(sleep_calls):
            expected = min(trader._SCAN_BACKOFF_BASE * (2 ** i), trader._SCAN_BACKOFF_MAX)
            self.assertAlmostEqual(delay, expected)

    def test_scan_request_delay_is_honoured(self) -> None:
        """scan_and_choose must call time.sleep(scan_request_delay) before each pair it analyzes."""
        config = BotConfig(api_key=None, scan_request_delay=0.5)
        snapshots = {
            "a_idr": {
                "pair": "a_idr", "price": 1.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None,
                "decision": StrategyDecision(
                    # hold → no early exit; delay is still applied before analysing this pair
                    mode="scalping", action="hold", confidence=0.1, reason="quiet",
                    target_price=1, amount=0, stop_loss=None, take_profit=None,
                ),
            },
            "b_idr": {
                "pair": "b_idr", "price": 2.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.6, reason="ok",
                    target_price=2, amount=5, stop_loss=1.98, take_profit=2.02,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        import unittest.mock as mock
        with mock.patch("bot.trader.time.sleep") as mock_sleep:
            trader.scan_and_choose()
        # sleep must fire once per pair analyzed:
        # a_idr (hold, delay fires) → b_idr (buy ≥ min_confidence → early exit, delay fires)
        self.assertEqual(mock_sleep.call_count, 2)
        for call_args in mock_sleep.call_args_list:
            self.assertEqual(call_args, mock.call(0.5))

    def test_scan_and_choose_uses_summaries_to_prefetch_tickers(self) -> None:
        """scan_and_choose must pass the feed-cached ticker as prefetched_ticker to analyze_market.

        In serial mode the loop exits on the first pair whose signal meets the
        confidence threshold, so only one pair is analyzed before early-exit.
        The important invariant is that the returned pair received a non-None
        prefetched ticker sourced from the multi-pair feed.
        """
        import unittest.mock as mock

        config = BotConfig(api_key=None, scan_request_delay=0.0)
        received_prefetched: list[Any] = []

        class SummaryClient:
            def get_pairs(self) -> list[dict]:
                return [{"name": "btc_idr"}, {"name": "eth_idr"}]

            def get_summaries(self) -> dict:
                return {
                    "tickers": {
                        "btcidr": {"last": "1000000000", "high": "1100000000"},
                        "ethidr": {"last": "50000000", "high": "55000000"},
                    }
                }

        class SummaryTrader(Trader):
            def analyze_market(
                self,
                pair: str | None = None,
                prefetched_ticker: Dict[str, Any] | None = None,
                skip_depth: bool = False,
                skip_trades: bool = False,
            ) -> Dict[str, Any]:
                received_prefetched.append((pair, prefetched_ticker))
                return {
                    "pair": pair, "price": 100.0, "trend": None, "orderbook": None,
                    "volatility": None, "levels": None, "indicators": None,
                    "decision": StrategyDecision(
                        mode="scalping", action="buy", confidence=0.7, reason="ok",
                        target_price=100, amount=1, stop_loss=99, take_profit=101,
                    ),
                }

        trader = SummaryTrader(config, client=SummaryClient())
        with mock.patch("bot.trader.time.sleep"):
            returned_pair, _ = trader.scan_and_choose()

        # Serial early exit: at least 1 pair must have been analyzed
        self.assertGreaterEqual(len(received_prefetched), 1)
        # The returned pair must have received a non-None prefetched ticker from the feed
        returned_entry = next((r for r in received_prefetched if r[0] == returned_pair), None)
        self.assertIsNotNone(returned_entry, f"Returned pair {returned_pair} not in analyzed list")
        assert returned_entry is not None  # narrowing for type checker
        pair_name, ticker = returned_entry
        self.assertIsNotNone(ticker, f"Expected prefetched_ticker for {pair_name}")
        self.assertIn("last", ticker)

    def test_analyze_with_retry_handles_runtimeerror_429(self) -> None:
        """_analyze_with_retry must retry when the client raises RuntimeError wrapping a 429.

        In production, IndodaxClient._handle_response() wraps requests.HTTPError
        inside RuntimeError using ``raise RuntimeError(msg) from exc``.  The
        retry logic detects 429 by inspecting ``__cause__``.
        """
        import unittest.mock as mock

        config = BotConfig(api_key=None, scan_request_delay=0.0)
        trader = GuardedTrader(config)
        calls: list[int] = []
        success_snapshot: Dict[str, Any] = {
            "pair": "btc_idr", "price": 1.0, "decision": StrategyDecision(
                mode="scalping", action="buy", confidence=0.6, reason="ok",
                target_price=1, amount=1, stop_loss=0.99, take_profit=1.01,
            ),
        }

        def fake_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            calls.append(1)
            if len(calls) == 1:
                # This is what IndodaxClient actually raises (RuntimeError wrapping HTTPError)
                resp = requests.Response()
                resp.status_code = 429
                original = requests.HTTPError(response=resp)
                raise RuntimeError(f"HTTP error: 429 Client Error: Too Many Requests for url: https://indodax.com/api/ticker/btc_idr") from original
            return success_snapshot

        trader.analyze_market = fake_analyze  # type: ignore[method-assign]
        with mock.patch("bot.trader.time.sleep"):
            result = trader._analyze_with_retry("btc_idr")
        self.assertEqual(result["pair"], "btc_idr")
        self.assertEqual(len(calls), 2)

    def test_scan_and_choose_falls_back_when_summaries_fails(self) -> None:
        """When get_summaries() fails, scan_and_choose must still work using per-pair ticker calls."""
        config = BotConfig(api_key=None, scan_request_delay=0.0)
        snapshots = {
            "btc_idr": {
                "pair": "btc_idr", "price": 1.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.7, reason="ok",
                    target_price=1, amount=1, stop_loss=0.99, take_profit=1.01,
                ),
            },
        }

        class BrokenSummariesClient:
            def get_pairs(self) -> list[dict]:
                return [{"name": "btc_idr"}]

            def get_summaries(self) -> dict:
                raise RuntimeError("summaries unavailable")

        class BrokenSummariesTrader(StubTrader):
            pass

        trader = BrokenSummariesTrader(config, snapshots)
        trader.client = BrokenSummariesClient()  # type: ignore[assignment]
        pair, snapshot = trader.scan_and_choose()
        self.assertEqual(pair, "btc_idr")
        self.assertIsNotNone(snapshot)

    def test_pairs_per_cycle_scans_only_window(self) -> None:
        """When pairs_per_cycle > 0 only that many pairs must be analyzed per call."""
        import unittest.mock as mock

        all_pairs = ["a_idr", "b_idr", "c_idr", "d_idr", "e_idr"]
        snapshots = {
            p: {
                "pair": p, "price": 100.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.5, reason="ok",
                    target_price=100, amount=1, stop_loss=99, take_profit=101,
                ),
            }
            for p in all_pairs
        }
        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=2)
        trader = StubTrader(config, snapshots)

        analyzed_on_first: list[str] = []
        original_analyze = trader.analyze_market

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            if pair:
                analyzed_on_first.append(pair)
            return original_analyze(pair, prefetched_ticker)

        trader.analyze_market = recording_analyze  # type: ignore[method-assign]
        with mock.patch("bot.trader.time.sleep"):
            trader.scan_and_choose()

        # Must have analyzed exactly pairs_per_cycle=2 pairs (not all 5)
        self.assertEqual(len(analyzed_on_first), 2)

    def test_pairs_per_cycle_offset_advances_each_call(self) -> None:
        """_scan_offset must advance by pairs_per_cycle on each scan_and_choose() call."""
        all_pairs = ["a_idr", "b_idr", "c_idr", "d_idr"]
        snapshots = {
            p: {
                "pair": p, "price": 100.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="hold", confidence=0.2, reason="quiet",
                    target_price=100, amount=0, stop_loss=None, take_profit=None,
                ),
            }
            for p in all_pairs
        }
        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=2)
        trader = StubTrader(config, snapshots)

        import unittest.mock as mock
        with mock.patch("bot.trader.time.sleep"):
            trader.scan_and_choose()  # first call: analyzes [0,1], offset → 2
            self.assertEqual(trader._scan_offset, 2)
            trader.scan_and_choose()  # second call: analyzes [2,3], offset → 0 (wraps)
            self.assertEqual(trader._scan_offset, 0)

    def test_pairs_per_cycle_zero_scans_all_pairs(self) -> None:
        """When pairs_per_cycle=0 all pairs must be scanned each cycle."""
        import unittest.mock as mock

        all_pairs = ["a_idr", "b_idr", "c_idr"]
        snapshots = {
            p: {
                "pair": p, "price": 100.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.5, reason="ok",
                    target_price=100, amount=1, stop_loss=99, take_profit=101,
                ),
            }
            for p in all_pairs
        }
        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0)
        trader = StubTrader(config, snapshots)

        analyzed: list[str] = []
        original_analyze = trader.analyze_market

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            if pair:
                analyzed.append(pair)
            return original_analyze(pair, prefetched_ticker)

        trader.analyze_market = recording_analyze  # type: ignore[method-assign]
        with mock.patch("bot.trader.time.sleep"):
            trader.scan_and_choose()

        # All 3 pairs must have been analyzed
        self.assertEqual(len(analyzed), 3)
        self.assertEqual(set(analyzed), set(all_pairs))

    def test_multi_feed_started_on_first_scan(self) -> None:
        """_multi_feed must be initialized after the first scan_and_choose() call."""
        import unittest.mock as mock

        config = BotConfig(api_key=None, scan_request_delay=0.0)
        snapshots = {
            "btc_idr": {
                "pair": "btc_idr", "price": 1.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.7, reason="ok",
                    target_price=1, amount=1, stop_loss=0.99, take_profit=1.01,
                ),
            }
        }
        trader = StubTrader(config, snapshots)
        self.assertIsNone(trader._multi_feed)
        with mock.patch("bot.trader.time.sleep"):
            trader.scan_and_choose()
        self.assertIsNotNone(trader._multi_feed)

    def test_scan_skips_pairs_absent_from_seeded_feed(self) -> None:
        """When the feed is seeded and a pair is not in the cache, it must be
        skipped (no REST ticker call) rather than falling through to the REST API."""
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0)
        all_pairs = ["btc_idr", "dogs_idr"]

        # btc_idr is in the feed; dogs_idr is not (absent from summaries)
        ticker_cache = {"btc_idr": {"last": "1000000000"}}
        snapshots = {
            "btc_idr": {
                "pair": "btc_idr", "price": 1.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.7, reason="ok",
                    target_price=1, amount=1, stop_loss=0.99, take_profit=1.01,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        trader._all_pairs = all_pairs

        # Build a seeded feed that only knows about btc_idr
        feed = MultiPairFeed(all_pairs, mock.MagicMock(), websocket_enabled=False, summaries_interval=9999)
        with feed._lock:
            feed._cache.update(ticker_cache)  # seed directly – no REST call
        trader._multi_feed = feed

        analyzed: list[str] = []
        original_analyze = trader.analyze_market

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            if pair:
                analyzed.append(pair)
            return original_analyze(pair, prefetched_ticker)

        trader.analyze_market = recording_analyze  # type: ignore[method-assign]
        with mock.patch("bot.trader.time.sleep"):
            trader.scan_and_choose()

        # dogs_idr must NOT have been analyzed (no REST fallback)
        self.assertNotIn("dogs_idr", analyzed)
        # btc_idr must have been analyzed normally
        self.assertIn("btc_idr", analyzed)

    def test_scan_does_not_skip_when_feed_unseeded(self) -> None:
        """When the feed has no cached data (summaries failed), pairs must NOT be
        skipped — the REST fallback must still work."""
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0)
        all_pairs = ["btc_idr"]
        snapshots = {
            "btc_idr": {
                "pair": "btc_idr", "price": 1.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.7, reason="ok",
                    target_price=1, amount=1, stop_loss=0.99, take_profit=1.01,
                ),
            },
        }
        trader = StubTrader(config, snapshots)
        trader._all_pairs = all_pairs

        # Build an empty (un-seeded) feed
        feed = MultiPairFeed(all_pairs, mock.MagicMock(), websocket_enabled=False, summaries_interval=9999)
        self.assertFalse(feed.is_seeded)  # confirm unseeded
        trader._multi_feed = feed

        analyzed: list[str] = []
        original_analyze = trader.analyze_market

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            if pair:
                analyzed.append(pair)
            return original_analyze(pair, prefetched_ticker)

        trader.analyze_market = recording_analyze  # type: ignore[method-assign]
        with mock.patch("bot.trader.time.sleep"):
            trader.scan_and_choose()

        # btc_idr must still be analyzed even with no cache (REST fallback)
        self.assertIn("btc_idr", analyzed)

    def test_serial_scan_exits_early_on_first_valid_signal(self) -> None:
        """scan_and_choose must stop after the first pair that meets min_confidence.

        Three pairs: first two are below threshold (no exit), third meets it
        (exit).  The fourth pair must NEVER be analyzed.
        """
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        all_pairs = ["p1_idr", "p2_idr", "p3_idr", "p4_idr"]

        def make_snap(pair: str, action: str, conf: float) -> Dict[str, Any]:
            return {
                "pair": pair, "price": 100.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action=action, confidence=conf, reason="test",
                    target_price=100, amount=1, stop_loss=99, take_profit=101,
                ),
            }

        snapshots = {
            "p1_idr": make_snap("p1_idr", "buy", 0.3),   # below threshold → no exit
            "p2_idr": make_snap("p2_idr", "buy", 0.4),   # below threshold → no exit
            "p3_idr": make_snap("p3_idr", "buy", 0.8),   # above threshold → EXIT HERE
            "p4_idr": make_snap("p4_idr", "buy", 0.9),   # must not be reached
        }
        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0)
        trader = StubTrader(config, snapshots)
        trader._all_pairs = all_pairs

        # Seeded feed so all pairs have cached tickers (no skip)
        feed = MultiPairFeed(all_pairs, mock.MagicMock(), websocket_enabled=False, summaries_interval=9999)
        for p in all_pairs:
            feed._apply_ws_message_for_pair(p, {"last": "100"})
        self.assertTrue(feed.is_seeded)
        trader._multi_feed = feed

        analyzed: list[str] = []
        original_analyze = trader.analyze_market

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            if pair and pair != config.pair:
                analyzed.append(pair)
            return original_analyze(pair, prefetched_ticker)

        trader.analyze_market = recording_analyze  # type: ignore[method-assign]
        with mock.patch("bot.trader.time.sleep"):
            returned_pair, snapshot = trader.scan_and_choose()

        # p3_idr triggers the early exit
        self.assertEqual(returned_pair, "p3_idr")
        self.assertEqual(snapshot["decision"].confidence, 0.8)
        # p4_idr must not have been analyzed
        self.assertNotIn("p4_idr", analyzed)
        # p1, p2, p3 must have been analyzed in order
        self.assertIn("p1_idr", analyzed)
        self.assertIn("p2_idr", analyzed)
        self.assertIn("p3_idr", analyzed)

    def test_scan_sorts_liquid_pairs_first(self) -> None:
        """Pairs must be sorted by descending IDR volume before the scan loop."""
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        # eth_idr has 10× more IDR volume than btc_idr → must be scanned first
        all_pairs = ["btc_idr", "eth_idr", "xrp_idr"]
        snapshots = {
            p: {
                "pair": p, "price": 100.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None, "indicators": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.8, reason="ok",
                    target_price=100, amount=1, stop_loss=99, take_profit=101,
                ),
            }
            for p in all_pairs
        }
        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0)
        trader = StubTrader(config, snapshots)
        trader._all_pairs = all_pairs

        # Seed the feed with explicit volumes: eth highest, btc mid, xrp lowest
        feed = MultiPairFeed(all_pairs, mock.MagicMock(), websocket_enabled=False, summaries_interval=9999)
        feed._apply_ws_message_for_pair("btc_idr", {"last": "1000000", "vol_idr": "500000000"})
        feed._apply_ws_message_for_pair("eth_idr", {"last": "50000", "vol_idr": "5000000000"})
        feed._apply_ws_message_for_pair("xrp_idr", {"last": "1000", "vol_idr": "100000000"})
        trader._multi_feed = feed

        analysis_order: list[str] = []
        original_analyze = trader.analyze_market

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False, skip_trades: bool = False) -> Dict[str, Any]:
            if pair and pair in all_pairs:
                analysis_order.append(pair)
            return original_analyze(pair, prefetched_ticker)

        trader.analyze_market = recording_analyze  # type: ignore[method-assign]
        with mock.patch("bot.trader.time.sleep"):
            # Serial early exit fires on the first pair analyzed (eth_idr, highest vol)
            returned_pair, _ = trader.scan_and_choose()

        # eth_idr (highest volume) must be analyzed first and returned via early exit
        self.assertEqual(returned_pair, "eth_idr")
        self.assertEqual(analysis_order[0], "eth_idr")


class InsufficientDataTests(unittest.TestCase):
    """Tests for insufficient_data handling in analyze_market and scan_and_choose."""

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def _make_snapshot(self, pair: str, action: str = "hold", confidence: float = 0.3,
                       insufficient: bool = False) -> Dict[str, Any]:
        return {
            "pair": pair,
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": insufficient,
            "decision": StrategyDecision(
                mode="position_trading",
                action=action,
                confidence=confidence,
                reason="test",
                target_price=100,
                amount=0.1,
                stop_loss=None,
                take_profit=None,
            ),
        }

    def test_scan_skips_insufficient_data_pairs(self) -> None:
        """Pairs flagged insufficient_data must not influence best_pair selection."""
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0, min_candles=20)

        # a_idr has insufficient data; b_idr has valid "hold" data
        snapshots = {
            "a_idr": self._make_snapshot("a_idr", action="hold", confidence=0.9, insufficient=True),
            "b_idr": self._make_snapshot("b_idr", action="hold", confidence=0.4, insufficient=False),
        }

        class _Client:
            def get_pairs(self): return [{"name": p} for p in snapshots]
            def get_summaries(self): return {}

        class _StubTrader(AutoPairsTrader):
            pass

        trader = _StubTrader(config, snapshots)
        feed = MultiPairFeed(
            list(snapshots), mock.MagicMock(), websocket_enabled=False, summaries_interval=9999
        )
        feed._apply_ws_message_for_pair("a_idr", {"last": "100", "vol_idr": "100", "trade_count": 10})
        feed._apply_ws_message_for_pair("b_idr", {"last": "100", "vol_idr": "50", "trade_count": 5})
        trader._multi_feed = feed

        with mock.patch("bot.trader.time.sleep"):
            returned_pair, snapshot = trader.scan_and_choose()

        # a_idr must be skipped; b_idr (valid hold) must be the fallback result
        self.assertEqual(returned_pair, "b_idr")
        self.assertFalse(snapshot.get("insufficient_data"))

    def test_scan_uses_best_hold_without_extra_rest_call(self) -> None:
        """scan_and_choose returns the best hold snapshot without re-calling analyze_market."""
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0)

        snapshots = {
            "a_idr": self._make_snapshot("a_idr", action="hold", confidence=0.4),
            "b_idr": self._make_snapshot("b_idr", action="hold", confidence=0.6),
        }
        trader = AutoPairsTrader(config, snapshots)
        feed = MultiPairFeed(
            list(snapshots), mock.MagicMock(), websocket_enabled=False, summaries_interval=9999
        )
        feed._apply_ws_message_for_pair("a_idr", {"last": "100", "vol_idr": "100"})
        feed._apply_ws_message_for_pair("b_idr", {"last": "100", "vol_idr": "50"})
        trader._multi_feed = feed

        analyze_calls: list[str] = []
        orig = trader.analyze_market

        def tracking_analyze(pair=None, prefetched_ticker=None, skip_depth=False, skip_trades=False):
            analyze_calls.append(pair or "")
            return orig(pair, prefetched_ticker)

        trader.analyze_market = tracking_analyze  # type: ignore[method-assign]

        with mock.patch("bot.trader.time.sleep"):
            returned_pair, snapshot = trader.scan_and_choose()

        # Both pairs are analyzed once during the scan loop (expected).
        # The best hold (a_idr has higher IDR vol → scanned first → b_idr second)
        # should be returned WITHOUT an additional fallback REST call.
        # Total analyze_market calls must be exactly 2 (one per pair in scan loop).
        self.assertEqual(len(analyze_calls), 2, f"Expected 2 analyze calls, got {analyze_calls}")
        # Returned pair is the one with highest _score (b_idr has confidence=0.6 > 0.4)
        self.assertEqual(returned_pair, "b_idr")

    def test_scan_fallback_uses_retry_wrapper(self) -> None:
        """Fallback analysis must go through _analyze_with_retry (handles 429s)."""
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        config = BotConfig(api_key=None, scan_request_delay=0.0, pairs_per_cycle=0)

        class RecordingTrader(AutoPairsTrader):
            def __init__(self, cfg, snaps):
                super().__init__(cfg, snaps)
                self.calls: list[str] = []

            def _analyze_with_retry(self, pair, prefetched_ticker=None, skip_depth=False, skip_trades=False):
                self.calls.append(pair)
                return self._snapshots[pair]

        # All pairs flagged insufficient → fallback path triggers extra retry call
        snapshots = {
            "a_idr": self._make_snapshot("a_idr", insufficient=True),
            "b_idr": self._make_snapshot("b_idr", insufficient=True),
        }
        trader = RecordingTrader(config, snapshots)
        feed = MultiPairFeed(
            list(snapshots), mock.MagicMock(), websocket_enabled=False, summaries_interval=9999
        )
        feed._apply_ws_message_for_pair("a_idr", {"last": "100", "vol_idr": "100"})
        feed._apply_ws_message_for_pair("b_idr", {"last": "100", "vol_idr": "50"})
        trader._multi_feed = feed

        with mock.patch("bot.trader.time.sleep"):
            returned_pair, snapshot = trader.scan_and_choose()

        # _analyze_with_retry is called once per pair during scan + once for fallback
        self.assertEqual(trader.calls, ["a_idr", "b_idr", "a_idr"])
        self.assertEqual(returned_pair, "a_idr")
        self.assertTrue(snapshot.get("insufficient_data"))

    def test_get_ohlc_in_client(self) -> None:
        """IndodaxClient.get_ohlc builds URL params correctly."""
        import unittest.mock as mock
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient()
        mock_response = mock.MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {"Time": 1000, "Open": 100.0, "High": 110.0, "Low": 90.0, "Close": 105.0, "Volume": "50"}
        ]
        with mock.patch.object(client.session, "get", return_value=mock_response) as mock_get:
            result = client.get_ohlc("btc_idr", tf="15", limit=50)
        call_kwargs = mock_get.call_args
        url = call_kwargs[0][0]
        params = call_kwargs[1]["params"]
        self.assertIn("/tradingview/history_v2", url)
        self.assertEqual(params["tf"], "15")
        self.assertEqual(params["symbol"], "BTCIDR")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["Close"], 105.0)

    def test_analyze_market_sets_insufficient_data_flag(self) -> None:
        """analyze_market sets insufficient_data=True when candle count < min_candles."""
        import unittest.mock as mock
        from bot.indodax_client import IndodaxClient

        config = BotConfig(api_key=None, min_candles=20, dry_run=True)

        class _NoOhlcClient(IndodaxClient):
            def get_ticker(self, pair):
                return {"ticker": {"last": "50000"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                return []  # No trades → empty candles
            def get_ohlc(self, pair, tf="15", *, limit=200, to_ts=None):
                return []  # OHLC also fails

        trader = Trader(config, client=_NoOhlcClient())
        with mock.patch("bot.trader.time.sleep"):
            snapshot = trader.analyze_market("btc_idr")

        self.assertTrue(snapshot["insufficient_data"])

    def test_config_trade_count_and_min_candles(self) -> None:
        """BotConfig trade_count and min_candles fields are configurable."""
        config = BotConfig(api_key=None, trade_count=500, min_candles=30)
        self.assertEqual(config.trade_count, 500)
        self.assertEqual(config.min_candles, 30)

    def test_config_defaults(self) -> None:
        """Default values for trade_count and min_candles are sensible."""
        config = BotConfig(api_key=None)
        self.assertEqual(config.trade_count, 1000)
        self.assertEqual(config.min_candles, 20)


class MinVolumeFilterTests(unittest.TestCase):
    """Tests that MIN_VOLUME_IDR correctly filters low-volume pairs from the scan."""

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def _make_snapshot(self, pair: str, action: str = "buy", conf: float = 0.9) -> dict:
        return {
            "pair": pair,
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "decision": StrategyDecision(
                mode="day_trading",
                action=action,
                confidence=conf,
                reason="test",
                target_price=100,
                amount=0.1,
                stop_loss=90,
                take_profit=110,
            ),
        }

    def test_low_volume_pair_is_skipped(self) -> None:
        """A pair whose 24-h volume is below min_volume_idr must be skipped."""
        import unittest.mock as mock
        from bot.realtime import MultiPairFeed

        config = BotConfig(
            api_key=None,
            min_volume_idr=1_000_000.0,
            scan_request_delay=0.0,
            pairs_per_cycle=0,
        )
        snapshots = {
            "btc_idr": self._make_snapshot("btc_idr", "buy", 0.9),
            "low_idr": self._make_snapshot("low_idr", "buy", 0.95),
        }

        trader = AutoPairsTrader(config, snapshots)
        feed = MultiPairFeed(
            list(snapshots), mock.MagicMock(), websocket_enabled=False, summaries_interval=9999
        )
        # btc_idr has high volume; low_idr has none
        feed._apply_ws_message_for_pair("btc_idr", {"last": "500000000", "vol_idr": "5000000000"})
        feed._apply_ws_message_for_pair("low_idr", {"last": "100", "vol_idr": "0"})
        trader._multi_feed = feed

        with mock.patch("bot.trader.time.sleep"):
            chosen_pair, _ = trader.scan_and_choose()
        # low_idr volume is 0 < 1_000_000 → must be filtered out
        self.assertEqual(chosen_pair, "btc_idr")


class RiskExposureCapTests(unittest.TestCase):
    """Tests for MAX_EXPOSURE_PER_COIN_PCT and MAX_DAILY_LOSS_PCT caps."""

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def _make_snapshot(self, pair: str = "btc_idr", action: str = "buy", conf: float = 0.9) -> dict:
        return {
            "pair": pair,
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action=action,
                confidence=conf,
                reason="test",
                target_price=100,
                amount=0.1,
                stop_loss=90,
                take_profit=110,
            ),
        }

    def test_exposure_cap_skips_buy_when_over_limit(self):
        """maybe_execute should skip buy when per-coin exposure cap is reached."""
        config = BotConfig(api_key=None, max_exposure_per_coin_pct=0.05, dry_run=True, multi_position_enabled=False)
        trader = Trader(config)
        # Simulate: 1000 coins at price 100 = 100_000 exposure on initial capital 1_000_000 → 10%
        trader.tracker.base_position = 1000.0
        trader.tracker.avg_cost = 100.0
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["99", "1"]], "sell": [["101", "1"]]},
        })()

        snap = self._make_snapshot(action="buy")
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("exposure_cap", outcome["reason"])

    def test_exposure_cap_zero_means_no_cap(self):
        """max_exposure_per_coin_pct=0 should never block a buy."""
        config = BotConfig(api_key=None, max_exposure_per_coin_pct=0.0, dry_run=True)
        trader = Trader(config)
        trader.tracker.base_position = 1000.0
        trader.tracker.avg_cost = 100.0
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["99", "1"]], "sell": [["101", "1"]]},
        })()
        snap = self._make_snapshot(action="buy")
        # Should not return "skipped" due to exposure cap
        outcome = trader.maybe_execute(snap)
        self.assertNotEqual(outcome.get("reason", ""), "exposure_cap")


class AdaptiveIntervalTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_returns_config_interval_when_disabled(self):
        config = BotConfig(api_key=None, interval_seconds=300, adaptive_interval_enabled=False)
        trader = Trader(config)
        self.assertEqual(trader._effective_interval(), 300)

    def test_returns_config_interval_when_low_volatility(self):
        from bot.analysis import VolatilityStats
        config = BotConfig(api_key=None, interval_seconds=300, adaptive_interval_enabled=True, adaptive_interval_min_seconds=30)
        trader = Trader(config)
        snapshot = {"volatility": VolatilityStats(volatility=0.005, avg_volume=0.0)}
        self.assertEqual(trader._effective_interval(snapshot), 300)

    def test_returns_min_interval_when_high_volatility(self):
        from bot.analysis import VolatilityStats
        config = BotConfig(api_key=None, interval_seconds=300, adaptive_interval_enabled=True, adaptive_interval_min_seconds=30)
        trader = Trader(config)
        snapshot = {"volatility": VolatilityStats(volatility=0.05, avg_volume=0.0)}
        self.assertEqual(trader._effective_interval(snapshot), 30)


class PartialTakeProfitTests(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _make_snapshot(self, pair="btc_idr", price=110.0):
        return {
            "pair": pair,
            "price": price,
        }

    def test_partial_tp_sells_fraction_dry_run(self):
        config = BotConfig(api_key=None, partial_tp_fraction=0.5, dry_run=True, multi_position_enabled=False)
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 2.0)  # buy 2 coins
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["109", "1"]], "sell": []},
        })()
        outcome = trader.partial_take_profit(self._make_snapshot(), fraction=0.5)
        self.assertEqual(outcome["status"], "partial_tp")
        self.assertAlmostEqual(outcome["amount"], 1.0)  # 50% of 2
        self.assertTrue(trader.tracker.partial_tp_taken)
        # Position should now be 1.0 (half sold)
        self.assertAlmostEqual(trader.tracker.base_position, 1.0)

    def test_partial_tp_no_position_returns_no_position(self):
        config = BotConfig(api_key=None, dry_run=True)
        trader = Trader(config)
        outcome = trader.partial_take_profit(self._make_snapshot(), fraction=0.5)
        self.assertEqual(outcome["status"], "no_position")

    def test_partial_tp_invalid_fraction(self):
        config = BotConfig(api_key=None, dry_run=True, multi_position_enabled=False)
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        outcome = trader.partial_take_profit(self._make_snapshot(), fraction=0.0)
        self.assertEqual(outcome["status"], "invalid_fraction")


class ReEntryCooldownTraderTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _make_snapshot(self, pair="btc_idr", action="buy", conf=0.9):
        return {
            "pair": pair,
            "price": 90.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action=action,
                confidence=conf,
                reason="test",
                target_price=90,
                amount=0.1,
                stop_loss=80,
                take_profit=100,
            ),
        }

    def test_re_entry_blocked_within_cooldown(self):
        import time
        config = BotConfig(api_key=None, re_entry_cooldown_seconds=3600, dry_run=True, multi_position_enabled=False)
        trader = Trader(config)
        trader.tracker.last_sell_price = 100.0
        trader.tracker.last_sell_time = time.time()  # just sold
        outcome = trader.maybe_execute(self._make_snapshot(action="buy"))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("re_entry", outcome["reason"])

    def test_re_entry_allowed_after_cooldown(self):
        import time
        config = BotConfig(api_key=None, re_entry_cooldown_seconds=1, dry_run=True)
        trader = Trader(config)
        trader.tracker.last_sell_price = 100.0
        trader.tracker.last_sell_time = time.time() - 5  # 5s ago, cooldown = 1s
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["89", "1"]], "sell": [["91", "1"]]},
        })()
        outcome = trader.maybe_execute(self._make_snapshot(action="buy"))
        # Should not be blocked by re-entry cooldown (may be skipped for other reasons)
        self.assertNotIn("re_entry", outcome.get("reason", ""))


class LiquidityDepthFilterTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_thin_market_skipped(self):
        config = BotConfig(api_key=None, min_liquidity_depth_idr=1_000_000_000, dry_run=True)
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {
                "buy": [["100", "10"]], "sell": [["101", "10"]]
            },  # only 2010 IDR depth — way below 1B threshold
        })()
        snap = {
            "pair": "btc_idr",
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="buy",
                confidence=0.9,
                reason="test",
                target_price=100,
                amount=0.1,
                stop_loss=90,
                take_profit=110,
            ),
        }
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("thin_market", outcome["reason"])

    def test_missing_depth_keys_does_not_block_trade(self):
        """When depth API returns no buy/sell keys, trade must NOT be blocked."""
        config = BotConfig(api_key=None, min_liquidity_depth_idr=50_000_000, dry_run=True)
        trader = Trader(config)
        # Simulate an API error response: no buy/sell keys at all
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"error": "pair not found"},
        })()
        snap = {
            "pair": "ogn_idr",
            "price": 532.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="buy",
                confidence=0.9,
                reason="test",
                target_price=532.0,
                amount=10.0,
                stop_loss=480.0,
                take_profit=600.0,
            ),
        }
        outcome = trader.maybe_execute(snap)
        # Should NOT be blocked for thin_market when depth is unavailable
        self.assertNotIn("thin_market", outcome.get("reason", ""))

    def test_liquidity_depth_idr_returns_none_for_empty_dict(self):
        """_liquidity_depth_idr must return None when depth has no orderbook keys."""
        from bot.trader import Trader
        trader = Trader(BotConfig(api_key=None, dry_run=True))
        self.assertIsNone(trader._liquidity_depth_idr({}, 100.0))
        self.assertIsNone(trader._liquidity_depth_idr({"error": "not found"}, 100.0))
        self.assertIsNone(trader._liquidity_depth_idr({"ticker": {"last": "532"}}, 100.0))

    def test_liquidity_depth_idr_one_side_only_not_none(self):
        """When only buy OR sell key exists, return a calculated total (not None)."""
        from bot.trader import Trader
        trader = Trader(BotConfig(api_key=None, dry_run=True))
        # Only bids present — should compute buy-side depth, not return None
        result = trader._liquidity_depth_idr({"buy": [["532", "100"]]}, 532.0)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 532 * 100, places=0)
        # Only asks present — should compute ask-side depth, not return None
        result_ask = trader._liquidity_depth_idr({"sell": [["533", "50"]]}, 532.0)
        self.assertIsNotNone(result_ask)
        self.assertAlmostEqual(result_ask, 533 * 50, places=0)

    def test_liquidity_depth_idr_returns_zero_for_empty_lists(self):
        """_liquidity_depth_idr returns 0 when buy/sell keys exist but are empty."""
        from bot.trader import Trader
        trader = Trader(BotConfig(api_key=None, dry_run=True))
        self.assertEqual(trader._liquidity_depth_idr({"buy": [], "sell": []}, 100.0), 0.0)

    def test_liquidity_depth_idr_computes_correctly(self):
        """_liquidity_depth_idr sums price × volume for all levels."""
        from bot.trader import Trader
        trader = Trader(BotConfig(api_key=None, dry_run=True))
        depth = {
            "buy": [["532", "1000"], ["531", "500"]],
            "sell": [["533", "800"]],
        }
        expected = 532 * 1000 + 531 * 500 + 533 * 800
        self.assertAlmostEqual(
            trader._liquidity_depth_idr(depth, 532.0), expected, places=0
        )


def _make_buy_snap(
    price: float = 100.0,
    action: str = "buy",
    confidence: float = 0.9,
    trend=None,
) -> Dict[str, Any]:
    """Helper to create a minimal buy snapshot for maybe_execute tests."""
    return {
        "pair": "btc_idr",
        "price": price,
        "trend": trend,
        "orderbook": None,
        "volatility": None,
        "levels": None,
        "indicators": None,
        "volume_24h_idr": 10_000_000.0,
        "trades_24h": 1_000,
        "insufficient_data": False,
        "grid_plan": None,
        "decision": StrategyDecision(
            mode="scalping",
            action=action,
            confidence=confidence,
            reason="test",
            target_price=price,
            amount=0.1,
            stop_loss=price * 0.95,
            take_profit=price * 1.05,
        ),
    }


class SpreadFilterTest(unittest.TestCase):
    """Tests for MAX_SPREAD_PCT spread filter in maybe_execute."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader_with_depth(self, bid: float, ask: float, max_spread_pct: float) -> Trader:
        config = BotConfig(api_key=None, max_spread_pct=max_spread_pct, dry_run=True)
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {
                "buy": [[str(bid), "10"]], "sell": [[str(ask), "10"]],
            },
        })()
        return trader

    def test_wide_spread_skips_buy(self):
        """When spread > max_spread_pct the buy must be skipped."""
        # bid=100, ask=103 → spread=3% > limit of 0.2%
        trader = self._trader_with_depth(bid=100.0, ask=103.0, max_spread_pct=0.002)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("spread_too_wide", outcome["reason"])

    def test_wide_spread_skips_sell(self):
        """Spread filter also applies to sell actions."""
        trader = self._trader_with_depth(bid=100.0, ask=103.0, max_spread_pct=0.002)
        # pre-load a position so sell isn't blocked by insufficient balance
        trader.tracker.record_trade("buy", 90.0, 0.1)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0, action="sell"))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("spread_too_wide", outcome["reason"])

    def test_tight_spread_allows_trade(self):
        """When spread is within limit the trade must proceed past the spread check."""
        # bid=100, ask=100.1 → spread=0.1% < limit of 0.2%
        trader = self._trader_with_depth(bid=100.0, ask=100.1, max_spread_pct=0.002)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))
        # Status may be skipped for other reasons (balance) but NOT spread
        self.assertNotIn("spread_too_wide", outcome.get("reason", ""))

    def test_spread_filter_disabled(self):
        """When max_spread_pct=0 the spread filter is disabled."""
        trader = self._trader_with_depth(bid=100.0, ask=200.0, max_spread_pct=0.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))
        self.assertNotIn("spread_too_wide", outcome.get("reason", ""))


class AdaptiveConfidenceTests(unittest.TestCase):
    """Adaptive confidence threshold should react to trend strength."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_strong_trend_enforces_adaptive_floor(self):
        class _DepthStubClient:
            def get_depth(self, *a, **kw):
                return {"buy": [["100", "10"]], "sell": [["100.05", "10"]]}

        trader = Trader(BotConfig(api_key=None, dry_run=True, min_order_idr=0, pair_min_order_cache_enabled=False))
        trader.client = _DepthStubClient()
        strong_trend = TrendResult(direction="up", strength=0.02, fast_ma=101.0, slow_ma=100.0)
        snap = _make_buy_snap(price=100.0, confidence=0.28, trend=strong_trend)
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("confidence", outcome["reason"])

    def test_strong_trend_allows_confidence_that_would_otherwise_skip(self):
        class _DepthStubClient:
            def get_depth(self, *a, **kw):
                return {"buy": [["100", "10"]], "sell": [["100.05", "10"]]}

        trader = Trader(BotConfig(api_key=None, dry_run=True, min_order_idr=0, pair_min_order_cache_enabled=False))
        trader.client = _DepthStubClient()
        strong_trend = TrendResult(direction="up", strength=0.02, fast_ma=101.0, slow_ma=100.0)
        snap = _make_buy_snap(price=100.0, confidence=0.32, trend=strong_trend)
        outcome = trader.maybe_execute(snap)
        # May still skip for other reasons, but should not be blocked by confidence threshold
        self.assertNotIn("confidence", outcome.get("reason", ""))

    def test_flat_trend_uses_higher_floor_and_skips(self):
        trader = Trader(BotConfig(api_key=None, dry_run=True))
        flat_trend = TrendResult(direction="flat", strength=0.0, fast_ma=100.0, slow_ma=100.0)
        snap = _make_buy_snap(price=100.0, confidence=0.35, trend=flat_trend)
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("threshold 0.4", outcome["reason"])


class MinBuyPriceFilterTest(unittest.TestCase):
    """Tests for MIN_BUY_PRICE_IDR soft filter in maybe_execute.

    Coins below the threshold are checked for orderbook quality instead of
    being hard-skipped.  Stuck/illiquid coins fail; active cheap coins pass.
    """

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader(self, min_price: float, coin_price: float, bids=None, asks=None, min_coin_price: float = 0.0, **cfg) -> Trader:
        config = BotConfig(
            api_key=None,
            min_coin_price_idr=min_coin_price,
            min_buy_price_idr=min_price,
            dry_run=True,
            **cfg,
        )
        trader = Trader(config)
        _bids = bids if bids is not None else [[str(coin_price), "100"]]
        _asks = asks if asks is not None else [[str(coin_price * 1.001), "100"]]
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": _bids, "sell": _asks},
        })()
        return trader

    def test_cheap_coin_thin_book_skipped(self):
        """Buy must be skipped when the orderbook has fewer levels than small_coin_min_bid_levels."""
        # Only 1 bid level; default small_coin_min_bid_levels=3
        trader = self._trader(min_price=10.0, coin_price=4.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("small_coin_thin_book", outcome["reason"])

    def test_cheap_coin_good_ob_allowed(self):
        """Cheap coin with sufficient bid levels, depth and tight spread must NOT be blocked."""
        # 5 bid levels with realistic volumes → total depth ~175K IDR > 50K default
        # depth = 4.0*10000 + 3.9*12000 + 3.8*8000 + 3.7*9000 + 3.6*7000 ≈ 175K IDR
        bids = [["4.0", "10000"], ["3.9", "12000"], ["3.8", "8000"], ["3.7", "9000"], ["3.6", "7000"]]
        asks = [["4.01", "100"]]
        trader = self._trader(min_price=10.0, coin_price=4.0, bids=bids, asks=asks)
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0))
        self.assertNotIn("small_coin_thin_book", outcome.get("reason", ""))
        self.assertNotIn("small_coin_illiquid", outcome.get("reason", ""))

    def test_cheap_coin_low_volume_blocked(self):
        """Cheap coin with healthy OB but low 24-h volume must be blocked."""
        bids = [["4.0", "10000"], ["3.9", "12000"], ["3.8", "8000"], ["3.7", "9000"], ["3.6", "7000"]]
        asks = [["4.01", "100"]]
        trader = self._trader(min_price=10.0, coin_price=4.0, bids=bids, asks=asks)
        snap = _make_buy_snap(price=4.0)
        snap["volume_24h_idr"] = 100_000.0  # below 1_000_000 default
        snap["trades_24h"] = 200  # above trades threshold so only volume fails
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("small_coin_low_volume", outcome["reason"])

    def test_cheap_coin_low_trades_blocked(self):
        """Cheap coin with healthy OB but low trade count must be blocked."""
        bids = [["4.0", "10000"], ["3.9", "12000"], ["3.8", "8000"], ["3.7", "9000"], ["3.6", "7000"]]
        asks = [["4.01", "100"]]
        trader = self._trader(min_price=10.0, coin_price=4.0, bids=bids, asks=asks)
        snap = _make_buy_snap(price=4.0)
        snap["volume_24h_idr"] = 5_000_000.0  # above default
        snap["trades_24h"] = 10  # below 50 default
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("small_coin_low_trades", outcome["reason"])

    def test_cheap_coin_illiquid_depth_skipped(self):
        """Buy must be skipped when total IDR bid depth is below small_coin_min_depth_idr."""
        # 5 bid levels but tiny depth: total ~10 IDR
        bids = [["4", "1"], ["3", "1"], ["2", "1"], ["1", "1"], ["0.5", "1"]]
        asks = [["5", "1"]]
        trader = self._trader(
            min_price=100.0, coin_price=4.0, bids=bids, asks=asks,
            small_coin_min_bid_levels=0,  # disable level check
            small_coin_min_depth_idr=1_000_000.0,  # require 1M IDR
            small_coin_max_spread_pct=0,  # disable spread check so depth check runs
        )
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("small_coin_illiquid", outcome["reason"])

    def test_cheap_coin_wide_spread_skipped(self):
        """Buy must be skipped when bid-ask spread exceeds small_coin_max_spread_pct."""
        # bid=4, ask=6 → spread = 2/4 = 50% > 10% threshold
        bids = [["4", "100000"], ["3.5", "200000"], ["3", "300000"], ["2.5", "400000"], ["2", "500000"]]
        asks = [["6", "50"]]
        trader = self._trader(
            min_price=100.0, coin_price=4.0, bids=bids, asks=asks,
            small_coin_min_bid_levels=0,  # disable level check
            small_coin_min_depth_idr=0,   # disable depth check so spread check runs
            small_coin_max_spread_pct=0.10,  # max 10% spread
        )
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("small_coin_wide_spread", outcome["reason"])

    def test_price_below_hard_floor_blocked(self):
        """Coins below min_coin_price_idr are skipped outright (no quality checks)."""
        trader = self._trader(min_price=10.0, min_coin_price=50.0, coin_price=4.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("price_below_min_coin", outcome["reason"])

    def test_price_exactly_at_threshold_allowed(self):
        """Coin priced exactly at the threshold must NOT trigger the quality check."""
        trader = self._trader(min_price=10.0, coin_price=10.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=10.0))
        self.assertNotIn("small_coin_thin_book", outcome.get("reason", ""))

    def test_price_above_threshold_allowed(self):
        """Coin priced above the threshold must NOT trigger the quality check."""
        trader = self._trader(min_price=10.0, coin_price=50.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=50.0))
        self.assertNotIn("small_coin_thin_book", outcome.get("reason", ""))

    def test_filter_disabled_when_zero(self):
        """When min_buy_price_idr=0 the filter must be fully disabled."""
        trader = self._trader(min_price=0.0, coin_price=1.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=1.0))
        self.assertNotIn("small_coin", outcome.get("reason", ""))

    def test_sell_not_blocked_by_price_filter(self):
        """Quality check must only apply to buy signals, not sells."""
        trader = self._trader(min_price=10.0, coin_price=4.0)
        trader.tracker.record_trade("buy", 4.0, 1000.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0, action="sell"))
        self.assertNotIn("small_coin", outcome.get("reason", ""))

    def test_sell_not_blocked_by_hard_floor(self):
        """Hard floor must not block sells so existing positions can exit."""
        trader = self._trader(min_price=10.0, min_coin_price=50.0, coin_price=4.0)
        trader.tracker.record_trade("buy", 4.0, 1000.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0, action="sell"))
        # It may still skip for other reasons (e.g. no position), but must not be
        # blocked by the hard floor filter.
        self.assertNotIn("price_below_min_coin", outcome.get("reason", ""))


class PreScanCheapCoinFilterTest(unittest.TestCase):
    """Tests for the pre-scan cheap coin filter in scan_and_choose().

    Cheap coins (price < min_buy_price_idr) with thin/inactive orderbooks must
    be skipped during scanning so they never appear as 'best hold' candidates.
    """

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_sepi_cheap_coin_skipped_before_analysis(self):
        """A cheap coin with a thin WS orderbook must be skipped before full analysis.

        The pre-scan filter checks the real-time WS depth data.  When the
        orderbook fails quality checks (fewer than min bid levels), the pair is
        skipped without calling analyze_market.
        """
        from bot.realtime import MultiPairFeed

        analyzed_pairs: list[str] = []

        class ScanClient:
            def get_pairs(self) -> list[dict]:
                return [{"name": "meme_idr"}, {"name": "btc_idr"}]

            def get_summaries(self) -> dict:
                return {
                    "tickers": {
                        "memeidr": {"last": "10", "high": "11", "low": "10"},
                        "btcidr": {"last": "1000000000", "high": "1100000000", "low": "900000000"},
                    }
                }

        config = BotConfig(
            api_key=None,
            min_buy_price_idr=100.0,   # meme_idr (10 IDR) is below threshold
            small_coin_min_bid_levels=3,  # require at least 3 bid levels
            small_coin_min_depth_idr=50000.0,
            small_coin_max_spread_pct=0.05,
            dry_run=True,
        )

        class ScanTrader(Trader):
            def analyze_market(self, pair=None, prefetched_ticker=None, skip_depth=False, skip_trades=False):
                analyzed_pairs.append(pair)
                return {
                    "pair": pair, "price": 1000000000.0, "trend": None,
                    "orderbook": None, "volatility": None, "levels": None, "indicators": None,
                    "decision": StrategyDecision(
                        mode="scalping", action="hold", confidence=0.0, reason="wait",
                        target_price=1000000000, amount=0, stop_loss=0, take_profit=0,
                    ),
                }

        trader = ScanTrader(config, client=ScanClient())

        # Seed the multi-pair feed so it is "seeded" and provides prefetched tickers
        trader._multi_feed = MultiPairFeed(
            pairs=["meme_idr", "btc_idr"],
            client=ScanClient(),
            websocket_enabled=False,
        )
        # Inject ticker data directly into the feed cache
        trader._multi_feed._apply_ws_message_for_pair(
            "meme_idr",
            {"last": "10", "high": "11", "low": "10", "vol_idr": "200000", "trade_count": "10"},
        )
        trader._multi_feed._apply_ws_message_for_pair(
            "btc_idr",
            {"last": "1000000000", "high": "1100000000", "vol_idr": "5000000000", "trade_count": "5000"},
        )
        trader._all_pairs = ["meme_idr", "btc_idr"]

        # Inject thin orderbook for meme_idr (only 1 bid level → thin book)
        with trader._multi_feed._lock:
            trader._multi_feed._depth_cache["meme_idr"] = {
                "buy": [["10", "100"]],  # only 1 bid level < 3 required
                "sell": [["11", "50"]],
            }

        trader.scan_and_choose()

        # meme_idr must have been skipped by the pre-scan filter (not analyzed)
        self.assertNotIn("meme_idr", analyzed_pairs)
        # btc_idr (above threshold) must still be analyzed normally
        self.assertIn("btc_idr", analyzed_pairs)

    def test_hard_floor_skips_ultra_cheap_coin_pre_scan(self):
        """Coins below min_coin_price_idr are dropped before analysis."""
        from bot.realtime import MultiPairFeed

        analyzed_pairs: list[str] = []

        class ScanClient:
            def get_pairs(self) -> list[dict]:
                return [{"name": "shan_idr"}, {"name": "btc_idr"}]

            def get_summaries(self) -> dict:
                return {
                    "tickers": {
                        "shanidr": {"last": "2", "high": "3", "low": "2"},
                        "btcidr": {"last": "1000000000", "high": "1100000000", "low": "900000000"},
                    }
                }

        config = BotConfig(
            api_key=None,
            min_coin_price_idr=50.0,
            min_buy_price_idr=100.0,
            small_coin_min_bid_levels=3,
            small_coin_min_depth_idr=50000.0,
            small_coin_max_spread_pct=0.05,
            dry_run=True,
        )

        class ScanTrader(Trader):
            def analyze_market(self, pair=None, prefetched_ticker=None, skip_depth=False, skip_trades=False):
                analyzed_pairs.append(pair)
                return {
                    "pair": pair, "price": 1000000000.0, "trend": None,
                    "orderbook": None, "volatility": None, "levels": None, "indicators": None,
                    "decision": StrategyDecision(
                        mode="scalping", action="hold", confidence=0.0, reason="wait",
                        target_price=1000000000, amount=0, stop_loss=0, take_profit=0,
                    ),
                }

        trader = ScanTrader(config, client=ScanClient())

        trader._multi_feed = MultiPairFeed(
            pairs=["shan_idr", "btc_idr"],
            client=ScanClient(),
            websocket_enabled=False,
        )
        trader._multi_feed._apply_ws_message_for_pair(
            "shan_idr",
            {"last": "2", "high": "3", "low": "2", "vol_idr": "200000", "trade_count": "10"},
        )
        trader._multi_feed._apply_ws_message_for_pair(
            "btc_idr",
            {"last": "1000000000", "high": "1100000000", "low": "900000000", "vol_idr": "5000000000", "trade_count": "5000"},
        )
        trader._all_pairs = ["shan_idr", "btc_idr"]

        trader.scan_and_choose()

        self.assertNotIn("shan_idr", analyzed_pairs)
        self.assertIn("btc_idr", analyzed_pairs)

    def test_active_cheap_coin_not_skipped(self):
        """A cheap coin with a healthy WS orderbook must NOT be filtered out pre-scan."""
        from bot.realtime import MultiPairFeed

        analyzed_pairs: list[str] = []

        class ScanClient:
            def get_pairs(self) -> list[dict]:
                return [{"name": "shib_idr"}]

            def get_summaries(self) -> dict:
                return {"tickers": {"shibidr": {"last": "50", "high": "55", "low": "48"}}}

        config = BotConfig(
            api_key=None,
            min_buy_price_idr=100.0,
            small_coin_min_bid_levels=3,
            small_coin_min_depth_idr=50000.0,
            small_coin_max_spread_pct=0.05,
            dry_run=True,
        )

        class ScanTrader(Trader):
            def analyze_market(self, pair=None, prefetched_ticker=None, skip_depth=False, skip_trades=False):
                analyzed_pairs.append(pair)
                return {
                    "pair": pair, "price": 50.0, "trend": None,
                    "orderbook": None, "volatility": None, "levels": None, "indicators": None,
                    "decision": StrategyDecision(
                        mode="scalping", action="hold", confidence=0.0, reason="wait",
                        target_price=50, amount=0, stop_loss=0, take_profit=0,
                    ),
                }

        trader = ScanTrader(config, client=ScanClient())
        trader._multi_feed = MultiPairFeed(
            pairs=["shib_idr"],
            client=ScanClient(),
            websocket_enabled=False,
        )
        trader._multi_feed._apply_ws_message_for_pair(
            "shib_idr",
            {"last": "50", "high": "55", "vol_idr": "5000000", "trade_count": "200"},
        )
        trader._all_pairs = ["shib_idr"]

        # Inject a healthy orderbook with 5 levels and >50K IDR depth, tight spread
        with trader._multi_feed._lock:
            trader._multi_feed._depth_cache["shib_idr"] = {
                "buy": [
                    ["50", "10000"], ["49.5", "12000"], ["49", "8000"],
                    ["48.5", "9000"], ["48", "7000"],
                ],  # total depth ~2.3M IDR > 50K, 5 levels ≥ 3
                "sell": [["50.5", "5000"]],  # spread = 0.5/50 = 1% < 5%
            }

        trader.scan_and_choose()

        # shib_idr has good OB → must be analyzed (not pre-filtered)
        self.assertIn("shib_idr", analyzed_pairs)


class TickMoveFilterTest(unittest.TestCase):
    """Tests for MAX_TICK_MOVE_PCT filter in maybe_execute."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader(self, max_tick: float, bids: list, asks: list, price: float) -> Trader:
        config = BotConfig(api_key=None, max_tick_move_pct=max_tick, min_buy_price_idr=0, min_coin_price_idr=0, dry_run=True)
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": bids, "sell": asks},
        })()
        return trader

    def test_large_tick_skips_buy(self):
        """Buy must be skipped when tick (4→5 IDR = 25%) exceeds max_tick_move_pct."""
        # Two bid levels 5 and 4 → tick = 1/5 = 20% > 8% limit
        bids = [["5", "100"], ["4", "200"]]
        asks = [["6", "50"]]
        trader = self._trader(max_tick=0.08, bids=bids, asks=asks, price=5.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=5.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("tick_too_large", outcome["reason"])

    def test_small_tick_allows_buy(self):
        """Buy must proceed when tick is within the max_tick_move_pct limit."""
        # Two bid levels 100 and 99.5 → tick = 0.5/100 = 0.5% < 8% limit
        bids = [["100", "500"], ["99.5", "500"]]
        asks = [["100.1", "200"]]
        trader = self._trader(max_tick=0.08, bids=bids, asks=asks, price=100.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))
        self.assertNotIn("tick_too_large", outcome.get("reason", ""))

    def test_tick_fallback_to_spread(self):
        """When only one bid level exists, tick is estimated from bid-ask spread."""
        # Only one bid; ask=6, bid=4 → spread fallback = 2/4 = 50% > 8%
        bids = [["4", "100"]]
        asks = [["6", "50"]]
        trader = self._trader(max_tick=0.08, bids=bids, asks=asks, price=4.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=4.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("tick_too_large", outcome["reason"])


class Depth429FallbackTest(unittest.TestCase):
    """analyze_market must tolerate depth 429s by using WS/cache instead of failing."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _candles(self):
        return [
            Candle(timestamp=1, open=10, high=11, low=9, close=10.5, volume=1000),
            Candle(timestamp=2, open=10.5, high=11, low=10, close=10.8, volume=900),
        ]

    def test_depth_prefers_ws_cache_and_skips_rest(self):
        ws_depth = {"buy": [["10", "1"]], "sell": [["11", "1"]]}
        ticker = {"last": "10", "high": "11", "low": "9", "vol_idr": "1000000", "trade_count": "100"}

        class _Client:
            depth_calls = 0

            def get_ticker(self, pair):
                return ticker

            def get_depth(self, pair, count=200):
                _Client.depth_calls += 1
                raise RuntimeError("HTTP error: 429 Client Error: Too Many Requests for url: depth")

            def get_trades(self, pair, count=200):
                return []

            def get_ohlc(self, pair, tf=None, limit=None):
                return []

        class _MultiFeed:
            def get_depth(self, pair):
                return ws_depth

            def get_trades(self, pair):
                return []

            def get_ticker(self, pair):
                return ticker

        class _Trader(Trader):
            def _fetch_candles(self, pair, trades, use_cache=False):
                return Depth429FallbackTest._candles(self)

            def _get_reference_pair_trend(self, pair):
                return None

        config = BotConfig(api_key=None, dry_run=True, min_candles=1)
        trader = _Trader(config, client=_Client())
        trader._multi_feed = _MultiFeed()

        snap = trader.analyze_market("cst_idr")
        self.assertGreater(snap["orderbook"].bid_volume, 0.0, "should use WS depth after 429")
        self.assertEqual(_Client.depth_calls, 0, "should avoid REST depth when WS snapshot is available")

    def test_depth_429_with_position_snapshot_falls_back_to_empty(self):
        ticker = {"last": "10", "high": "11", "low": "9", "vol_idr": "1000000", "trade_count": "50"}

        class _Client:
            depth_calls = 0

            def get_ticker(self, pair):
                return ticker

            def get_depth(self, pair, count=200):
                _Client.depth_calls += 1
                raise RuntimeError("HTTP error: 429 Client Error: Too Many Requests for url: depth")

            def get_trades(self, pair, count=200):
                return []

            def get_ohlc(self, pair, tf=None, limit=None):
                return []

        class _PositionFeed:
            has_snapshot = True

            def snapshot(self):
                return {"ticker": ticker, "trades": []}

        class _Trader(Trader):
            def _fetch_candles(self, pair, trades, use_cache=False):
                return Depth429FallbackTest._candles(self)

            def _get_reference_pair_trend(self, pair):
                return None

        config = BotConfig(api_key=None, dry_run=True, min_candles=1)
        trader = _Trader(config, client=_Client())
        trader._position_feeds["cst_idr"] = _PositionFeed()

        snap = trader.analyze_market("cst_idr")
        self.assertIsInstance(snap["orderbook"], OrderbookInsight)
        self.assertEqual(_Client.depth_calls, 1)


class TopVolumeAutoSelectorTest(unittest.TestCase):
    """Tests for enhanced _refresh_dynamic_pairs top-volume pair selector."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _make_feed_with_tickers(self, ticker_map: dict):
        """Build a minimal mock MultiPairFeed with a populated _cache."""
        import threading
        feed = type("FakeFeed", (), {
            "_cache": ticker_map,
            "_lock": threading.Lock(),
            "get_ticker": lambda self, p: self._cache.get(p),
            "get_depth": lambda self, p: None,
            "get_trades": lambda self, p: None,
            "subscribe_depth_pairs": lambda self, pairs: None,
            "is_seeded": True,
        })()
        return feed

    def _trader_with_feed(self, feed, **cfg_kwargs) -> Trader:
        config = BotConfig(api_key=None, dry_run=True, **cfg_kwargs)
        trader = Trader(config)
        trader._multi_feed = feed
        return trader

    def test_low_volume_pairs_excluded_from_watchlist(self):
        """Pairs below top_volume_min_volume_idr must not appear in the watchlist."""
        tickers = {
            "btc_idr": {"vol_idr": "5000000000", "last": "1000000000", "high": "1100000000", "low": "900000000"},
            "skya_idr": {"vol_idr": "1000000", "last": "17", "high": "18", "low": "16"},  # tiny volume
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            top_volume_min_volume_idr=100_000_000,  # 100M IDR minimum
        )
        trader._refresh_dynamic_pairs()
        self.assertIn("btc_idr", trader._all_pairs)
        self.assertNotIn("skya_idr", trader._all_pairs)

    def test_stagnant_pairs_excluded_from_watchlist(self):
        """Pairs with < top_volume_min_price_change_24h_pct movement must be excluded."""
        tickers = {
            "btc_idr": {"vol_idr": "5000000000", "last": "1000000", "open": "950000",
                        "high": "1050000", "low": "940000"},
            "dent_idr": {"vol_idr": "200000000", "last": "4", "open": "4",  # 0% change
                         "high": "4", "low": "4"},
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            top_volume_min_price_change_24h_pct=0.005,  # require 0.5% movement
        )
        trader._refresh_dynamic_pairs()
        self.assertIn("btc_idr", trader._all_pairs)
        self.assertNotIn("dent_idr", trader._all_pairs)

    def test_active_pairs_pass_all_filters(self):
        """Pairs meeting both volume and price-change criteria must be in the watchlist."""
        tickers = {
            "eth_idr": {"vol_idr": "800000000", "last": "50000", "open": "47000",
                        "high": "52000", "low": "46000"},
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            top_volume_min_volume_idr=100_000_000,
            top_volume_min_price_change_24h_pct=0.005,
        )
        trader._refresh_dynamic_pairs()
        self.assertIn("eth_idr", trader._all_pairs)

    def test_top_n_limits_watchlist_size(self):
        """Watchlist must never exceed dynamic_pairs_top_n entries."""
        tickers = {
            f"coin{i}_idr": {
                "vol_idr": str(1_000_000_000 - i * 1_000_000),
                "last": "1000", "high": "1100", "low": "900",
            }
            for i in range(30)
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(feed, dynamic_pairs_top_n=10)
        trader._refresh_dynamic_pairs()
        self.assertLessEqual(len(trader._all_pairs), 10)

    def test_filters_disabled_when_zero(self):
        """When all filter thresholds are 0, all pairs must pass to the ranking step."""
        tickers = {
            "btc_idr": {"vol_idr": "5000000000", "last": "1000000", "high": "1100000", "low": "900000"},
            "skya_idr": {"vol_idr": "500", "last": "17", "open": "17", "high": "17", "low": "17"},
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            top_volume_min_volume_idr=0,
            top_volume_min_price_change_24h_pct=0,
            min_buy_price_idr=0,
        )
        trader._refresh_dynamic_pairs()
        self.assertIn("btc_idr", trader._all_pairs)
        self.assertIn("skya_idr", trader._all_pairs)

    def test_watchlist_empty_candidates_keeps_existing(self):
        """When all pairs are filtered out, the existing watchlist must be preserved."""
        tickers = {
            "dent_idr": {"vol_idr": "100", "last": "4", "open": "4", "high": "4", "low": "4"},
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            top_volume_min_volume_idr=1_000_000_000,  # impossibly high
        )
        trader._all_pairs = ["btc_idr"]  # pre-existing watchlist
        trader._refresh_dynamic_pairs()
        self.assertEqual(trader._all_pairs, ["btc_idr"])  # unchanged

    def test_selected_pairs_logged_in_order(self):
        """Top-N pairs must be ranked by composite score (higher vol×volatility first)."""
        tickers = {
            "lowvol_idr": {"vol_idr": "100000000", "last": "100", "high": "101", "low": "99"},
            "highvol_idr": {"vol_idr": "9000000000", "last": "100", "high": "120", "low": "80"},
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(feed, dynamic_pairs_top_n=20)
        trader._refresh_dynamic_pairs()
        pairs = trader._all_pairs
        self.assertIsNotNone(pairs)
        # highvol pair should rank above lowvol pair
        self.assertLess(pairs.index("highvol_idr"), pairs.index("lowvol_idr"))

    def test_low_price_pairs_excluded_from_watchlist(self):
        """Cheap coins that are stuck (high==low) must be dropped from the watchlist."""
        tickers = {
            "btc_idr": {
                "vol_idr": "5000000000", "last": "1000000000",
                "high": "1100000000", "low": "900000000",
            },
            # DENT-like coin: price = 4 IDR, high == low → completely stuck
            "dent_idr": {
                "vol_idr": "200000000", "last": "4",
                "high": "4", "low": "4",  # stuck: no movement
            },
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            min_buy_price_idr=100.0,
        )
        trader._refresh_dynamic_pairs()
        self.assertIn("btc_idr", trader._all_pairs)
        self.assertNotIn("dent_idr", trader._all_pairs)

    def test_cheap_active_coin_stays_in_watchlist(self):
        """Cheap coin with price movement (high != low) must stay on watchlist."""
        tickers = {
            "shib_idr": {
                "vol_idr": "300000000", "last": "50",
                "high": "55", "low": "48",  # active: high != low
            },
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            min_buy_price_idr=100.0,
        )
        trader._refresh_dynamic_pairs()
        self.assertIn("shib_idr", trader._all_pairs)

    def test_cheap_dead_coin_excluded_from_watchlist(self):
        """Cheap coin with zero volume (dead) must be dropped from watchlist."""
        tickers = {
            "dead_idr": {
                "vol_idr": "0", "last": "20",
                "high": "22", "low": "18",  # price moved but no volume
            },
            "btc_idr": {
                "vol_idr": "5000000000", "last": "500000000",
                "high": "550000000", "low": "480000000",
            },
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            min_buy_price_idr=100.0,
        )
        trader._refresh_dynamic_pairs()
        self.assertIsNotNone(trader._all_pairs)
        self.assertIn("btc_idr", trader._all_pairs)
        self.assertNotIn("dead_idr", trader._all_pairs)

    def test_low_price_filter_disabled_when_zero(self):
        """min_buy_price_idr=0 must not filter any pairs by price."""
        tickers = {
            "dent_idr": {
                "vol_idr": "200000000", "last": "4",
                "high": "5", "low": "3",
            },
        }
        feed = self._make_feed_with_tickers(tickers)
        trader = self._trader_with_feed(
            feed,
            dynamic_pairs_top_n=20,
            min_buy_price_idr=0,  # disabled
        )
        trader._refresh_dynamic_pairs()
        self.assertIn("dent_idr", trader._all_pairs)


class SellWallGuardTest(unittest.TestCase):
    """Tests for ORDERBOOK_WALL_THRESHOLD sell-wall guard in maybe_execute."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader_with_depth(self, bid_vol: float, ask_vol: float, threshold: float) -> Trader:
        config = BotConfig(api_key=None, orderbook_wall_threshold=threshold, dry_run=True)
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {
                "buy": [["100", str(bid_vol)]],
                "sell": [["100.1", str(ask_vol)]],
            },
        })()
        return trader

    def test_dominant_sell_wall_blocks_buy(self):
        """When ask/bid volume ratio ≥ threshold the buy must be blocked."""
        # ask=500 units, bid=100 units → ratio=5.0 ≥ threshold=5.0
        trader = self._trader_with_depth(bid_vol=100.0, ask_vol=500.0, threshold=5.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("sell_wall", outcome["reason"])

    def test_balanced_book_allows_buy(self):
        """When ask/bid ratio < threshold the buy must NOT be blocked by wall guard."""
        # ask=200, bid=100 → ratio=2.0 < threshold=5.0
        trader = self._trader_with_depth(bid_vol=100.0, ask_vol=200.0, threshold=5.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))
        self.assertNotIn("sell_wall", outcome.get("reason", ""))

    def test_wall_guard_does_not_block_sell(self):
        """The sell-wall guard only applies to buy orders."""
        trader = self._trader_with_depth(bid_vol=100.0, ask_vol=500.0, threshold=5.0)
        trader.tracker.record_trade("buy", 90.0, 0.1)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0, action="sell"))
        self.assertNotIn("sell_wall", outcome.get("reason", ""))

    def test_wall_guard_disabled(self):
        """When threshold=0 the sell-wall guard is disabled."""
        trader = self._trader_with_depth(bid_vol=1.0, ask_vol=1000.0, threshold=0.0)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))
        self.assertNotIn("sell_wall", outcome.get("reason", ""))


class MinOrderIdrTest(unittest.TestCase):
    """Tests for MIN_ORDER_IDR guard in maybe_execute."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _snap(self, price: float, action: str = "buy") -> Dict[str, Any]:
        return {
            "pair": "pixel_idr",
            "price": price,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="position_trading",
                action=action,
                confidence=0.9,
                reason="test",
                target_price=price,
                amount=1.0,  # very small: 1 coin × price IDR
                stop_loss=price * 0.95,
                take_profit=price * 1.05,
            ),
        }

    def test_buy_below_minimum_skipped(self):
        """An order whose total IDR value is below min_order_idr and cannot be
        bumped up (insufficient available cash) must be skipped."""
        config = BotConfig(api_key=None, min_order_idr=10_000, dry_run=True,
                           initial_capital=5_000)  # capital < min_order_idr → cannot bump up
        trader = Trader(config)
        trader.client = type("_C", (), {
            # bid=ask=253 so slippage guard passes
            "get_depth": lambda self, *a, **kw: {"buy": [["253", "100"]], "sell": [["253", "100"]]},
        })()
        # price=253, amount=1 → total=253 IDR < 10,000 IDR
        # capital=5000 → max affordable=5000/253≈19.7 coins → value≈5000 < 10,000
        snap = self._snap(price=253.0)
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("order_below_minimum", outcome["reason"])

    def test_buy_below_minimum_bumped_up(self):
        """When the order value is below min_order_idr but capital can cover
        the minimum, the order amount should be bumped up instead of skipped."""
        config = BotConfig(api_key=None, min_order_idr=10_000, dry_run=True,
                           initial_capital=100_000)
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["253", "1000"]], "sell": [["253", "1000"]]},
        })()
        # price=253, amount=1 → 253 IDR < 10,000, but capital can cover min
        snap = self._snap(price=253.0)
        outcome = trader.maybe_execute(snap)
        # Should proceed (bumped up to minimum) instead of being skipped
        self.assertNotEqual(outcome["status"], "skipped")

    def test_buy_above_minimum_proceeds(self):
        """An order whose total IDR value meets or exceeds min_order_idr must proceed."""
        config = BotConfig(api_key=None, min_order_idr=10_000, dry_run=True,
                           initial_capital=100_000)
        trader = Trader(config)
        trader.client = type("_C", (), {
            # bid=ask=253 so slippage guard passes
            "get_depth": lambda self, *a, **kw: {"buy": [["253", "1000"]], "sell": [["253", "1000"]]},
        })()
        # price=253, amount=100 → total=25,300 IDR > 10,000 IDR
        snap = self._snap(price=253.0)
        snap["decision"] = StrategyDecision(
            mode="position_trading",
            action="buy",
            confidence=0.9,
            reason="test",
            target_price=253.0,
            amount=100.0,
            stop_loss=240.0,
            take_profit=270.0,
        )
        outcome = trader.maybe_execute(snap)
        self.assertNotIn("order_below_minimum", outcome.get("reason", ""))
        self.assertNotEqual(outcome["status"], "skipped")

    def test_config_min_order_idr_default(self):
        """BotConfig default min_order_idr must be 30,000 (bot safety floor above the 10,000 IDR exchange minimum)."""
        config = BotConfig(api_key=None)
        self.assertEqual(config.min_order_idr, 30_000.0)

    def test_config_min_order_idr_validation(self):
        """min_order_idr must be positive; zero or negative must raise ValueError."""
        cfg_zero = BotConfig(api_key=None, min_order_idr=0.0)
        with self.assertRaises(ValueError):
            cfg_zero._validate()
        cfg_neg = BotConfig(api_key=None, min_order_idr=-1.0)
        with self.assertRaises(ValueError):
            cfg_neg._validate()

    def test_pixel_idr_scenario(self):
        """Reproduce the exact pixel_idr scenario from the bug report."""
        # pixel_idr price=253, initial capital=100K IDR.
        # With the bot buying ~395 coins, total ≈ 99K IDR >> 10K.
        # Bug was that small amounts from staged entry could be below minimum.
        config = BotConfig(api_key=None, min_order_idr=10_000, dry_run=True,
                           initial_capital=100_000)
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["253", "500"]], "sell": [["253", "500"]]},
        })()
        snap = {
            "pair": "pixel_idr",
            "price": 253.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="position_trading",
                action="buy",
                confidence=0.577,
                reason="position_trading",
                target_price=253.0,
                amount=395.26,   # ≈ 100,000 / 253
                stop_loss=240.0,
                take_profit=270.0,
            ),
        }
        outcome = trader.maybe_execute(snap)
        # The full order is 395 × 253 ≈ 99K IDR >> 10K, should NOT be skipped
        self.assertNotIn("order_below_minimum", outcome.get("reason", ""))


class PumpProtectionTest(unittest.TestCase):
    """Tests for pump protection in _record_price / _is_pumped / maybe_execute."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader(self, pump_pct: float = 0.05, lookback: float = 60.0) -> Trader:
        config = BotConfig(
            api_key=None,
            pump_protection_pct=pump_pct,
            pump_lookback_seconds=lookback,
            dry_run=True,
        )
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {
                "buy": [["100", "10"]], "sell": [["100.1", "10"]],
            },
        })()
        return trader

    def test_is_pumped_returns_false_with_no_history(self):
        trader = self._trader()
        self.assertFalse(trader._is_pumped("btc_idr", 200.0))

    def test_is_pumped_returns_false_when_disabled(self):
        trader = self._trader(pump_pct=0.0)
        trader._price_history = {"btc_idr": [(0.0, 100.0)]}
        self.assertFalse(trader._is_pumped("btc_idr", 200.0))

    def test_is_pumped_true_on_large_rise(self):
        trader = self._trader(pump_pct=0.05)
        trader._price_history = {"btc_idr": [(0.0, 100.0)]}  # inject old price manually
        self.assertTrue(trader._is_pumped("btc_idr", 106.0))   # +6% > 5% threshold

    def test_is_pumped_false_on_small_rise(self):
        trader = self._trader(pump_pct=0.05)
        trader._price_history = {"btc_idr": [(0.0, 100.0)]}
        self.assertFalse(trader._is_pumped("btc_idr", 104.0))  # +4% < 5% threshold

    def test_record_price_populates_history(self):
        trader = self._trader(pump_pct=0.05)
        self.assertEqual(trader._price_history, {})
        trader._record_price("btc_idr", 100.0)
        self.assertIn("btc_idr", trader._price_history)
        self.assertEqual(len(trader._price_history["btc_idr"]), 1)
        self.assertAlmostEqual(trader._price_history["btc_idr"][0][1], 100.0)

    def test_record_price_noop_when_disabled(self):
        trader = self._trader(pump_pct=0.0)
        trader._record_price("btc_idr", 100.0)
        self.assertEqual(trader._price_history, {})

    def test_pump_blocks_buy_in_maybe_execute(self):
        """A pumped price should cause maybe_execute to skip the buy."""
        trader = self._trader(pump_pct=0.05)
        # Inject a historic price well below the current price to simulate pump
        import time as _time
        trader._price_history = {"btc_idr": [(_time.time() - 30, 80.0)]}  # 30s ago @ 80
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0))  # +25% pump
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("pump_detected", outcome["reason"])

    def test_pump_does_not_block_sell(self):
        """Pump protection only applies to buy orders."""
        import time as _time
        trader = self._trader(pump_pct=0.05)
        trader._price_history = {"btc_idr": [(_time.time() - 30, 80.0)]}
        trader.tracker.record_trade("buy", 80.0, 0.1)
        outcome = trader.maybe_execute(_make_buy_snap(price=100.0, action="sell"))
        self.assertNotIn("pump_detected", outcome.get("reason", ""))

    def test_pump_history_isolated_per_pair(self):
        """Prices from different pairs must not cross-contaminate the pump check."""
        import time as _time
        trader = self._trader(pump_pct=0.05)
        # Inject a very low price for a different pair — must not affect btc_idr check
        trader._price_history = {"eth_idr": [(_time.time() - 10, 1.0)]}
        # btc_idr has no history → should not be flagged as pumped
        self.assertFalse(trader._is_pumped("btc_idr", 1_500_000_000.0))


class FakePumpDetectionTest(unittest.TestCase):
    """Tests for _is_fake_pump and its integration in maybe_execute."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader(self, pump_pct: float = 0.05, reversal_pct: float = 0.03,
                lookback: float = 60.0) -> Trader:
        config = BotConfig(
            api_key=None,
            pump_protection_pct=pump_pct,
            pump_lookback_seconds=lookback,
            fake_pump_reversal_pct=reversal_pct,
            dry_run=True,
        )
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {
                "buy": [["100", "10"]], "sell": [["100.1", "10"]],
            },
        })()
        return trader

    def test_is_fake_pump_returns_false_with_no_history(self):
        trader = self._trader()
        self.assertFalse(trader._is_fake_pump("btc_idr", 100.0))

    def test_is_fake_pump_returns_false_when_disabled(self):
        """Fake-pump check is off when reversal_pct = 0."""
        import time as _time
        trader = self._trader(reversal_pct=0.0)
        trader._price_history = {"btc_idr": [
            (_time.time() - 20, 100.0),
            (_time.time() - 10, 110.0),  # peak: +10%
            (_time.time() - 1,  105.0),  # current: -4.5% from peak
        ]}
        self.assertFalse(trader._is_fake_pump("btc_idr", 105.0))

    def test_is_fake_pump_true_spike_then_dump(self):
        """Detect spike (+10%) then dump (-5% from peak) → fake pump."""
        import time as _time
        trader = self._trader(pump_pct=0.05, reversal_pct=0.03)
        trader._price_history = {"btc_idr": [
            (_time.time() - 25, 100.0),   # baseline
            (_time.time() - 15, 110.0),   # peak (+10%) — spike ≥ 5%
            (_time.time() - 5,  104.0),   # dump: (110-104)/110 ≈ 5.5% ≥ 3%
        ]}
        self.assertTrue(trader._is_fake_pump("btc_idr", 104.0))

    def test_is_fake_pump_false_no_spike(self):
        """No spike (rise < pump_pct) → no fake pump even if price drops."""
        import time as _time
        trader = self._trader(pump_pct=0.05, reversal_pct=0.03)
        trader._price_history = {"btc_idr": [
            (_time.time() - 20, 100.0),   # baseline
            (_time.time() - 10, 102.0),   # mild rise +2% < 5% threshold
            (_time.time() - 1,   99.0),   # drop, but no real spike
        ]}
        self.assertFalse(trader._is_fake_pump("btc_idr", 99.0))

    def test_is_fake_pump_false_spike_no_reversal(self):
        """Spike present but price hasn't reversed enough → not yet fake pump."""
        import time as _time
        trader = self._trader(pump_pct=0.05, reversal_pct=0.03)
        trader._price_history = {"btc_idr": [
            (_time.time() - 20, 100.0),   # baseline
            (_time.time() - 10, 110.0),   # peak +10%
            (_time.time() - 1,  109.5),   # only -0.5% from peak (< 3% reversal)
        ]}
        self.assertFalse(trader._is_fake_pump("btc_idr", 109.5))

    def test_is_fake_pump_requires_two_data_points(self):
        """Single entry in buffer → not enough data → returns False."""
        import time as _time
        trader = self._trader()
        trader._price_history = {"btc_idr": [(_time.time() - 5, 100.0)]}
        self.assertFalse(trader._is_fake_pump("btc_idr", 90.0))

    def test_is_fake_pump_isolated_per_pair(self):
        """Fake-pump check must not cross-contaminate pairs."""
        import time as _time
        trader = self._trader()
        # eth_idr had a spike+dump — btc_idr has no history
        trader._price_history = {"eth_idr": [
            (_time.time() - 20, 100.0),
            (_time.time() - 10, 120.0),
            (_time.time() - 1,  110.0),
        ]}
        self.assertFalse(trader._is_fake_pump("btc_idr", 110.0))

    def test_fake_pump_blocks_buy_in_maybe_execute(self):
        """maybe_execute must skip buy when a fake pump is detected."""
        import time as _time
        trader = self._trader(pump_pct=0.05, reversal_pct=0.03)
        # Inject spike+dump pattern into price buffer
        trader._price_history = {"btc_idr": [
            (_time.time() - 25, 100.0),   # baseline
            (_time.time() - 15, 110.0),   # spike +10%
            (_time.time() - 5,  104.0),   # dump ~5.5% from peak
        ]}
        outcome = trader.maybe_execute(_make_buy_snap(price=104.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("fake_pump_detected", outcome["reason"])

    def test_fake_pump_does_not_block_sell(self):
        """Fake-pump guard only applies to buy orders."""
        import time as _time
        trader = self._trader(pump_pct=0.05, reversal_pct=0.03)
        trader._price_history = {"btc_idr": [
            (_time.time() - 25, 100.0),
            (_time.time() - 15, 110.0),
            (_time.time() - 5,  104.0),
        ]}
        trader.tracker.record_trade("buy", 100.0, 0.1)
        outcome = trader.maybe_execute(_make_buy_snap(price=104.0, action="sell"))
        self.assertNotIn("fake_pump_detected", outcome.get("reason", ""))


class EvaluateDynamicTpTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _snap(self, price=110.0, trend_strength=0.5, imbalance=0.2, rsi=60.0):
        from bot.analysis import TrendResult, OrderbookInsight, VolatilityStats, MomentumIndicators
        trend = TrendResult(direction="up", strength=trend_strength, fast_ma=100.0, slow_ma=95.0)
        ob = OrderbookInsight(spread_pct=0.001, bid_volume=10.0, ask_volume=8.0, imbalance=imbalance)
        indicators = MomentumIndicators(rsi=rsi, macd=1.0, macd_signal=0.5, macd_hist=0.5, bb_upper=115.0, bb_mid=105.0, bb_lower=95.0)
        return {
            "pair": "btc_idr",
            "price": price,
            "trend": trend,
            "orderbook": ob,
            "volatility": VolatilityStats(volatility=0.01, avg_volume=100.0),
            "indicators": indicators,
        }

    def test_no_dynamic_tp_configured_returns_target_profit(self):
        config = BotConfig(api_key=None, trailing_tp_pct=0.0)
        trader = Trader(config)
        result = trader.evaluate_dynamic_tp(self._snap())
        self.assertEqual(result, "target_profit_reached")

    def test_trailing_tp_activates_and_holds(self):
        config = BotConfig(api_key=None, trailing_tp_pct=0.02, multi_position_enabled=False)
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        # Price at 110 — trailing TP not yet triggered (just activated)
        result = trader.evaluate_dynamic_tp(self._snap(price=110.0))
        self.assertIsNone(result)
        # Trailing floor should be set at 110 * 0.98 = 107.8
        self.assertAlmostEqual(trader.tracker.trailing_tp_stop, 107.8)

    def test_trailing_tp_triggered_returns_correct_reason(self):
        config = BotConfig(api_key=None, trailing_tp_pct=0.02, multi_position_enabled=False)
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        trader.tracker.activate_trailing_tp(120.0, 0.02)  # floor = 117.6
        # Price falls below floor
        result = trader.evaluate_dynamic_tp(self._snap(price=116.0))
        self.assertEqual(result, "trailing_tp_triggered")

    def test_conditional_tp_holds_when_conditions_met(self):
        config = BotConfig(
            api_key=None,
            conditional_tp_min_trend_strength=0.3,
            conditional_tp_max_rsi=80.0,
        )
        trader = Trader(config)
        # Trend strength 0.5 > 0.3 and RSI 60 < 80 → hold
        result = trader.evaluate_dynamic_tp(self._snap(trend_strength=0.5, rsi=60.0))
        self.assertIsNone(result)

    def test_conditional_tp_closes_when_rsi_overbought(self):
        config = BotConfig(
            api_key=None,
            conditional_tp_max_rsi=70.0,
        )
        trader = Trader(config)
        # RSI 75 >= 70 → overbought → close
        result = trader.evaluate_dynamic_tp(self._snap(rsi=75.0))
        self.assertEqual(result, "target_profit_reached")

    def test_conditional_tp_closes_when_trend_weak(self):
        config = BotConfig(
            api_key=None,
            conditional_tp_min_trend_strength=0.5,
        )
        trader = Trader(config)
        # Trend strength 0.2 < 0.5 → close
        result = trader.evaluate_dynamic_tp(self._snap(trend_strength=0.2))
        self.assertEqual(result, "target_profit_reached")

    def test_conditional_tp_closes_when_ob_imbalance_low(self):
        config = BotConfig(
            api_key=None,
            conditional_tp_min_ob_imbalance=0.15,
        )
        trader = Trader(config)
        # Imbalance 0.05 < 0.15 → sell pressure → close
        result = trader.evaluate_dynamic_tp(self._snap(imbalance=0.05))
        self.assertEqual(result, "target_profit_reached")

    def test_effective_capital_used_in_position_sizing(self):
        """After a profitable trade, effective_capital > initial_capital → larger position."""
        from bot.strategies import make_trade_decision
        from bot.analysis import TrendResult, OrderbookInsight, VolatilityStats, MomentumIndicators
        config = BotConfig(api_key=None, risk_per_trade=0.01, initial_capital=100_000.0,
                           min_order_idr=0)  # disable min-order floor for this sizing test
        tracker_base = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker_rich = PortfolioTracker(100_000.0, 0.2, 0.1)
        # Simulate rich tracker having 50k profit buffer
        tracker_rich.record_trade("buy", 100.0, 1.0)
        tracker_rich.record_trade("sell", 150_000.0, 1.0)  # +50k profit

        trend = TrendResult(direction="up", strength=0.6, fast_ma=100.0, slow_ma=90.0)
        ob = OrderbookInsight(spread_pct=0.001, bid_volume=10.0, ask_volume=7.0, imbalance=0.3)
        vol = VolatilityStats(volatility=0.01, avg_volume=100.0)
        ind = MomentumIndicators(rsi=55.0, macd=1.0, macd_signal=0.5, macd_hist=0.5, bb_upper=110.0, bb_mid=100.0, bb_lower=90.0)

        dec_base = make_trade_decision(trend, ob, vol, 100.0, config,
                                       effective_capital=tracker_base.effective_capital())
        dec_rich = make_trade_decision(trend, ob, vol, 100.0, config,
                                       effective_capital=tracker_rich.effective_capital())

        # Rich tracker has larger effective capital → bigger position size
        self.assertGreater(dec_rich.amount, dec_base.amount)


class TrailingTpFloorAdvancementTest(unittest.TestCase):
    """Regression test: trailing TP floor must advance on each monitoring tick."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_floor_advances_when_price_rises(self):
        """After activation the trailing TP floor must rise with the price."""
        config = BotConfig(api_key=None, trailing_tp_pct=0.01)
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 900.0)

        # Activate floor at price=125 → floor=123.75
        tracker.activate_trailing_tp(125.0, 0.01)
        floor_after_125 = tracker.trailing_tp_stop
        self.assertAlmostEqual(floor_after_125, 123.75)

        # Price rises to 130: simulate the main.py per-tick advancement.
        # The guard no longer requires tp_activated — the floor advances every
        # cycle unconditionally (tp_activated is always True after first call).
        if config.trailing_tp_pct > 0:
            tracker.activate_trailing_tp(130.0, config.trailing_tp_pct)

        floor_after_130 = tracker.trailing_tp_stop
        self.assertAlmostEqual(floor_after_130, 128.7, places=2)
        self.assertGreater(floor_after_130, floor_after_125)

    def test_floor_activates_from_entry_without_tp_target(self):
        """Trailing TP floor must be set from the first holding cycle.

        The main.py loop no longer requires tp_activated to be True before
        calling activate_trailing_tp.  This means trail-TP is always shown
        in the holding display (not '—') and the bot adapts to market
        conditions rather than waiting for a fixed profit target.
        """
        config = BotConfig(api_key=None, trailing_tp_pct=0.02)
        tracker = PortfolioTracker(100_000.0, 0.10, 0.05)
        tracker.record_trade("buy", 100.0, 900.0)

        # Simulate the FIRST holding cycle — no prior activate_trailing_tp call,
        # tp_activated starts False.  The new main.py behaviour (no guard) means
        # activate_trailing_tp is called unconditionally.
        self.assertFalse(tracker.tp_activated)
        if config.trailing_tp_pct > 0:
            tracker.activate_trailing_tp(100.0, config.trailing_tp_pct)

        # Floor should be set immediately at 100 * (1 - 0.02) = 98.0
        self.assertIsNotNone(tracker.trailing_tp_stop)
        self.assertAlmostEqual(tracker.trailing_tp_stop, 98.0)
        self.assertTrue(tracker.tp_activated)

    def test_trailing_tp_exits_on_retrace_before_fixed_tp(self):
        """Bot must exit via trailing TP even when the fixed TP target was not hit.

        When trailing TP is activated from entry (no fixed-TP gate), a retrace
        of more than trailing_tp_pct from the peak triggers an exit regardless
        of whether the fixed profit target was reached.
        """
        # target_profit_pct=0.50 (50%) — an unrealistically high fixed TP
        tracker = PortfolioTracker(100_000.0, 0.50, 0.10)
        tracker.record_trade("buy", 100.0, 900.0)

        # Price rises to 110 → floor at 110 * 0.98 = 107.8
        tracker.activate_trailing_tp(110.0, 0.02)
        self.assertAlmostEqual(tracker.trailing_tp_stop, 107.8)

        # Price retraces to 107 — fixed TP (150) not hit, but floor is breached
        reason = tracker.stop_reason(107.0)
        self.assertEqual(reason, "trailing_tp_triggered")

    def test_floor_does_not_fall_when_price_drops(self):
        """Trailing TP floor must never fall when the price retraces."""
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 900.0)
        tracker.activate_trailing_tp(130.0, 0.01)  # floor=128.7
        peak_floor = tracker.trailing_tp_stop

        # Price drops to 129 — floor must not decrease
        tracker.activate_trailing_tp(129.0, 0.01)
        self.assertAlmostEqual(tracker.trailing_tp_stop, peak_floor)

    def test_floor_triggers_exit_after_advancement(self):
        """After advancing the floor, a price drop below the NEW floor triggers exit."""
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 900.0)

        # Activate at 125 → floor=123.75
        tracker.activate_trailing_tp(125.0, 0.01)
        # Advance to 130 → floor=128.7
        tracker.activate_trailing_tp(130.0, 0.01)

        # Price drops to 128 (above old floor 123.75, but below new floor 128.7)
        reason = tracker.stop_reason(128.0)
        self.assertEqual(reason, "trailing_tp_triggered")

    def test_stop_reason_still_none_above_advanced_floor(self):
        """When price is above the advanced floor, stop_reason returns None (hold)."""
        tracker = PortfolioTracker(100_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 900.0)
        tracker.activate_trailing_tp(130.0, 0.01)  # floor=128.7

        # At 129 (above floor 128.7), equity > target → None (hold)
        reason = tracker.stop_reason(129.0)
        self.assertIsNone(reason)


class ScanAndChooseUnexpectedExceptionTest(unittest.TestCase):
    """Regression test: unexpected exception types must not escape scan_and_choose().

    Before the fix, a KeyError/AttributeError/TypeError raised inside
    _analyze_with_retry() would propagate through scan_and_choose() and all the
    way out of main(), crashing the process at line 892.  After the fix, such
    exceptions are caught per-pair (added to failed_pairs) and the scan cycle
    raises the standard RuntimeError("No pairs could be analyzed") instead.
    """

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader(self):
        from bot.config import BotConfig
        return Trader(BotConfig(api_key=None, dry_run=True))

    def test_keyerror_caught_per_pair(self):
        import unittest.mock as mock
        trader = self._trader()
        with mock.patch.object(trader, "_analyze_with_retry", side_effect=KeyError("missing_key")):
            with self.assertRaises(RuntimeError) as ctx:
                trader.scan_and_choose()
        self.assertIn("No pairs could be analyzed", str(ctx.exception))

    def test_attribute_error_caught_per_pair(self):
        import unittest.mock as mock
        trader = self._trader()
        with mock.patch.object(trader, "_analyze_with_retry", side_effect=AttributeError("attr")):
            with self.assertRaises(RuntimeError):
                trader.scan_and_choose()

    def test_type_error_caught_per_pair(self):
        import unittest.mock as mock
        trader = self._trader()
        with mock.patch.object(trader, "_analyze_with_retry", side_effect=TypeError("type")):
            with self.assertRaises(RuntimeError):
                trader.scan_and_choose()

    def test_index_error_caught_per_pair(self):
        import unittest.mock as mock
        trader = self._trader()
        with mock.patch.object(trader, "_analyze_with_retry", side_effect=IndexError("index")):
            with self.assertRaises(RuntimeError):
                trader.scan_and_choose()


class PairCooldownTraderTest(unittest.TestCase):
    """Tests for the per-pair trade cooldown guard."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _make_snapshot(self, pair="eth_idr", action="buy", conf=0.9):
        return {
            "pair": pair,
            "price": 5_000_000.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action=action,
                confidence=conf,
                reason="test",
                target_price=5_000_000.0,
                amount=0.001,
                stop_loss=4_500_000.0,
                take_profit=5_500_000.0,
            ),
        }

    def test_pair_cooldown_blocks_buy_within_window(self):
        import time
        config = BotConfig(api_key=None, pair_cooldown_seconds=3600, dry_run=True)
        trader = Trader(config)
        # Simulate that this pair was just traded
        trader._pair_last_trade["eth_idr"] = time.time()
        outcome = trader.maybe_execute(self._make_snapshot(pair="eth_idr", action="buy"))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("pair_cooldown", outcome["reason"])

    def test_pair_cooldown_allows_buy_after_window(self):
        import time
        config = BotConfig(api_key=None, pair_cooldown_seconds=5, dry_run=True)
        trader = Trader(config)
        # Trade was 10s ago, window = 5s → should be allowed
        trader._pair_last_trade["eth_idr"] = time.time() - 10
        # Attach a dummy depth client so the buy isn't blocked elsewhere
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["4999999", "1"]], "sell": [["5000001", "1"]]},
        })()
        outcome = trader.maybe_execute(self._make_snapshot(pair="eth_idr", action="buy"))
        self.assertNotIn("pair_cooldown", outcome.get("reason", ""))

    def test_pair_cooldown_does_not_block_different_pair(self):
        import time
        config = BotConfig(api_key=None, pair_cooldown_seconds=3600, dry_run=True)
        trader = Trader(config)
        # Only eth_idr is in cooldown
        trader._pair_last_trade["eth_idr"] = time.time()
        # btc_idr should NOT be blocked
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["4999999", "1"]], "sell": [["5000001", "1"]]},
        })()
        outcome = trader.maybe_execute(self._make_snapshot(pair="btc_idr", action="buy"))
        self.assertNotIn("pair_cooldown", outcome.get("reason", ""))

    def test_pair_cooldown_disabled_when_zero(self):
        import time
        config = BotConfig(api_key=None, pair_cooldown_seconds=0, dry_run=True)
        trader = Trader(config)
        # Set trade timestamp to "just now" — should not block because feature is off
        trader._pair_last_trade["eth_idr"] = time.time()
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["4999999", "1"]], "sell": [["5000001", "1"]]},
        })()
        outcome = trader.maybe_execute(self._make_snapshot(pair="eth_idr", action="buy"))
        self.assertNotIn("pair_cooldown", outcome.get("reason", ""))

    def test_persist_after_trade_records_cooldown_timestamp(self):
        import time
        config = BotConfig(api_key=None, pair_cooldown_seconds=60, dry_run=True)
        trader = Trader(config)
        before = time.time()
        trader._persist_after_trade("sol_idr")
        after = time.time()
        self.assertIn("sol_idr", trader._pair_last_trade)
        self.assertGreaterEqual(trader._pair_last_trade["sol_idr"], before)
        self.assertLessEqual(trader._pair_last_trade["sol_idr"], after)

    def test_persist_after_trade_does_not_record_when_disabled(self):
        config = BotConfig(api_key=None, pair_cooldown_seconds=0, dry_run=True)
        trader = Trader(config)
        trader._persist_after_trade("sol_idr")
        self.assertNotIn("sol_idr", trader._pair_last_trade)

    def test_pair_last_trade_pruned_after_cooldown_expires(self):
        """_persist_after_trade must remove entries older than pair_cooldown_seconds."""
        import time
        config = BotConfig(api_key=None, pair_cooldown_seconds=60, dry_run=True)
        trader = Trader(config)
        # Pre-populate with an expired entry (well past the cooldown window)
        trader._pair_last_trade["expired_idr"] = time.time() - 120
        trader._persist_after_trade("new_idr")
        # The expired entry must have been pruned
        self.assertNotIn("expired_idr", trader._pair_last_trade)
        # The new entry must still be present
        self.assertIn("new_idr", trader._pair_last_trade)


class ZeroAmountBuySkipTest(unittest.TestCase):
    """Bug fix: bot must NOT report PLACED/simulated when all staged steps are below min_order_idr."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @staticmethod
    def _dummy_client():
        """Return a minimal fake client that satisfies depth checks."""
        return type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["2699999", "1"]], "sell": [["2700001", "1"]]},
        })()

    def _make_snapshot(self, pair="cast_idr", price=2_700_000.0, conf=0.353):
        return {
            "pair": pair,
            "price": price,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="buy",
                confidence=conf,
                reason="test",
                target_price=price,
                # Amount small enough that step_amount * price < min_order_idr
                # With price=2_700_000 and amount=0.000003 → Rp8.1 < min Rp15000
                amount=0.000003,
                stop_loss=price * 0.95,
                take_profit=price * 1.05,
            ),
        }

    def test_dry_run_all_steps_below_min_returns_skipped(self):
        """Dry-run: if the total order value is below min_order_idr, status must be 'skipped'."""
        config = BotConfig(
            api_key=None, dry_run=True,
            min_order_idr=15000,
            # Use capital too small to bump amount up to minimum.
            initial_capital=10.0,
            # disable RSI / resistance / cooldown filters so the only skip reason
            # is the min_order check
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
        )
        trader = Trader(config)
        trader.client = self._dummy_client()
        snap = self._make_snapshot()
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        # Either the pre-staged check or the post-loop guard should fire
        self.assertTrue(
            "order_below_minimum" in outcome["reason"]
            or "all_steps_below_min_order" in outcome["reason"],
            f"Unexpected reason: {outcome['reason']}",
        )

    def test_dry_run_portfolio_unchanged_after_zero_amount_skip(self):
        """Portfolio cash must be unchanged (no coins bought) when skipped."""
        config = BotConfig(
            api_key=None, dry_run=True,
            min_order_idr=15000,
            initial_capital=10.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
        )
        trader = Trader(config)
        trader.client = self._dummy_client()
        initial_cash = trader.tracker.cash
        snap = self._make_snapshot()
        trader.maybe_execute(snap)
        self.assertEqual(trader.tracker.cash, initial_cash)
        self.assertEqual(trader.tracker.base_position, 0.0)

    def test_pair_cooldown_not_recorded_after_zero_amount_skip(self):
        """Pair cooldown must NOT be set when the order was never actually placed."""
        config = BotConfig(
            api_key=None, dry_run=True,
            min_order_idr=15000,
            initial_capital=10.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            pair_cooldown_seconds=300.0,
            min_confidence=0.0,
        )
        trader = Trader(config)
        trader.client = self._dummy_client()
        snap = self._make_snapshot(pair="cast_idr")
        trader.maybe_execute(snap)
        # _persist_after_trade should not have been called → no cooldown recorded
        self.assertNotIn("cast_idr", trader._pair_last_trade)

    def test_all_staged_steps_individually_below_min_after_split(self):
        """When total passes pre-check but individual staged splits would be
        below min, the staging must collapse to a single step so the trade
        executes instead of being needlessly skipped."""
        from bot.analysis import VolatilityStats
        # min_order_idr = 30000, price = 100
        # decision.amount = 400 → effective_amount capped at min(400, cash/100)
        # default cash=1_000_000 → max_affordable=10000 → effective_amount=400
        # total IDR = 400 × 100 = 40000 > 30000 (passes pre-check)
        # staged fractions with vol=0.015, conf=0.5: [0.6, 0.4]
        # step1 = 240 × 100 = 24000 < 30000, step2 = 160 × 100 = 16000 < 30000
        # → collapse to single step [400] → 40000 ≥ 30000 → executes
        config = BotConfig(
            api_key=None, dry_run=True,
            min_order_idr=30000,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            max_slippage_pct=0.05,   # generous slippage to avoid early skip
        )
        trader = Trader(config)
        # Depth prices close to 100 to avoid slippage rejection
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["99", "1"]], "sell": [["101", "1"]]},
        })()
        snap = {
            "pair": "split_idr",
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": VolatilityStats(volatility=0.015, avg_volume=1000.0),
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="buy",
                confidence=0.5,
                reason="test",
                target_price=100.0,
                amount=400.0,
                stop_loss=90.0,
                take_profit=110.0,
            ),
        }
        outcome = trader.maybe_execute(snap)
        # The staged entry collapses to a single step, so the trade executes.
        self.assertEqual(outcome["status"], "simulated")
        self.assertEqual(len(outcome["executed_steps"]), 1,
                         "collapsed staging should produce exactly 1 step")


class StagedCollapseTests(unittest.TestCase):
    """Tests for _collapse_staged_if_needed and staged entry min-order logic."""

    def test_collapse_when_any_step_below_min(self):
        """Staged amounts collapse to single step when a step is below minimum."""
        staged = [1.5, 0.9, 0.6]  # at price 100 → [150, 90, 60] IDR
        result = Trader._collapse_staged_if_needed(staged, 3.0, 100.0, 100.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 3.0)

    def test_no_collapse_when_all_above_min(self):
        """When every step is above minimum, staging is preserved."""
        staged = [5.0, 3.0, 2.0]  # at price 100 → [500, 300, 200] IDR
        result = Trader._collapse_staged_if_needed(staged, 10.0, 100.0, 100.0)
        self.assertEqual(result, staged)

    def test_no_collapse_single_step(self):
        """Single-step staging is always returned as-is."""
        staged = [3.0]
        result = Trader._collapse_staged_if_needed(staged, 3.0, 100.0, 100.0)
        self.assertEqual(result, [3.0])

    def test_no_collapse_min_order_zero(self):
        """When min_order_idr is 0, no collapse occurs."""
        staged = [0.01, 0.005, 0.003]
        result = Trader._collapse_staged_if_needed(staged, 0.018, 100.0, 0.0)
        self.assertEqual(result, staged)

    def test_collapse_empty(self):
        """Empty staged list returns empty."""
        result = Trader._collapse_staged_if_needed([], 0.0, 100.0, 30000.0)
        self.assertEqual(result, [])

    def test_staged_entry_collapse_prevents_skip_dry_run(self):
        """A borderline-sized order that would fail in multiple stages
        should succeed as a single step after collapse (dry-run mode)."""
        config = BotConfig(
            api_key=None, dry_run=True,
            min_order_idr=30000,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            max_slippage_pct=0.05,
        )
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": [["4999", "1"]], "sell": [["5001", "1"]]},
        })()
        # price=5000, amount=10 → total=50,000 IDR > 30K
        # staged [0.5, 0.3, 0.2] = [25K, 15K, 10K] — all below 30K
        # collapse → single step [50K] → executes
        from bot.analysis import VolatilityStats
        snap = {
            "pair": "borderline_idr",
            "price": 5000.0,
            "trend": None,
            "orderbook": None,
            "volatility": VolatilityStats(volatility=0.03, avg_volume=1000.0),
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="buy",
                confidence=0.5,
                reason="test-borderline",
                target_price=5000.0,
                amount=10.0,
                stop_loss=4500.0,
                take_profit=5500.0,
            ),
        }
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "simulated")
        self.assertEqual(len(outcome["executed_steps"]), 1)


class WhaleTrackingTests(unittest.TestCase):
    class _Client:
        def get_pairs(self) -> list:
            return [{"name": "btc_idr"}]

        def get_summaries(self) -> dict:
            return {}

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_whale_events_are_tracked_and_restored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "bot_state.json"
            config = BotConfig(api_key=None, dry_run=True, state_path=state_path, multi_position_enabled=False)
            trader = Trader(config, client=self._Client())

            whale = WhaleActivity(detected=True, side="bid", ratio=6.0)
            trader.tracker.base_position = 1.0  # ensure state persists
            trader._record_whale_event("btc_idr", whale, price=100.0)
            trader._save_state("btc_idr")

            fresh_config = BotConfig(api_key=None, dry_run=True, state_path=state_path, multi_position_enabled=False)
            fresh = Trader(fresh_config, client=self._Client())
            events = fresh.whale_events()

            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["pair"], "btc_idr")
            self.assertEqual(event["side"], "bid")
            self.assertAlmostEqual(event["ratio"], 6.0, places=3)


class ForceSellDustTests(unittest.TestCase):
    """Tests for the minimum-order / dust handling in force_sell."""

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_force_sell_clears_dust_when_below_min_coin(self) -> None:
        """force_sell returns dust_cleared when actual balance < per-pair min_coin."""

        class _LiveClient(GuardedTrader._Client):
            def get_account_info(self):
                # Exchange shows only 100 WTEC — way below min 3333.33
                return {"return": {"balance": {"wtec": "100", "idr": "459125"}}}

            def open_orders(self, pair: str):
                return {"return": {"orders": []}}

            def get_pair_min_order(self, pair: str):
                return {"min_coin": 3333.33, "min_idr": 0.0}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        trader.tracker.record_trade("buy", 3.0, 13625.0)

        decision = StrategyDecision(
            mode="swing_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=3.0,
            amount=13625.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "wtec_idr", "price": 3.0, "decision": decision}
        outcome = trader.force_sell(snapshot)

        # Should return dust_cleared — not raise, not place a sell order
        self.assertEqual(outcome["status"], "dust_cleared")
        # Position should be cleared
        self.assertEqual(trader.tracker.base_position, 0.0)

    def test_force_sell_handles_minimum_order_api_error(self) -> None:
        """force_sell must catch 'Minimum order' API error and clear position as dust."""

        class _LiveClient(GuardedTrader._Client):
            def get_account_info(self):
                return {"return": {"balance": {"wtec": "2000", "idr": "459125"}}}

            def open_orders(self, pair: str):
                return {"return": {"orders": []}}

            def create_order(self, pair, order_type, price, amount):
                raise RuntimeError(
                    "API error: {'success': 0, 'error': 'Minimum order 3333.33333333 WTEC', 'error_code': ''}"
                )

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            # Keep min_order_idr low so the IDR check doesn't trigger first
            min_order_idr=1.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        trader.tracker.record_trade("buy", 3.0, 13625.0)

        decision = StrategyDecision(
            mode="swing_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=3.0,
            amount=13625.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "wtec_idr", "price": 3.0, "decision": decision}
        # Should not raise; should be handled gracefully
        outcome = trader.force_sell(snapshot)
        self.assertEqual(outcome["status"], "dust_cleared")
        self.assertEqual(trader.tracker.base_position, 0.0)

    def test_force_sell_does_not_clear_when_amount_meets_minimum(self) -> None:
        """Minimum-order error with amount >= min must bubble up instead of clearing dust."""

        class _LiveClient(GuardedTrader._Client):
            def get_account_info(self):
                return {"return": {"balance": {"wtec": "3000", "idr": "459125"}}}

            def open_orders(self, pair: str):
                return {"return": {"orders": []}}

            def get_pair_min_order(self, pair: str):
                # min_coin is well below the amount to sell
                return {"min_coin": 0.1, "min_idr": 0.0}

            def create_order(self, pair, order_type, price, amount):
                raise RuntimeError(
                    "API error: {'success': 0, 'error': 'Minimum order 0.1 WTEC', 'error_code': ''}"
                )

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            min_order_idr=1.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        trader.tracker.record_trade("buy", 1.0, 3000.0)  # position = 3000 WTEC @ 1 IDR

        decision = StrategyDecision(
            mode="swing_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=1.0,
            amount=3000.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "wtec_idr", "price": 1.0, "decision": decision}

        with self.assertRaises(RuntimeError):
            trader.force_sell(snapshot)
        # Position should remain intact because we did not clear dust
        self.assertAlmostEqual(trader.tracker.base_position, 3000.0)

    def test_force_sell_uses_idr_minimum_not_coin_amount(self) -> None:
        """Ensure force_sell validates IDR value so sells proceed when rupiah minimum is met even if coin amount is below a coin threshold."""

        class _LiveClient(GuardedTrader._Client):
            def __init__(self) -> None:
                self.placed: list[tuple[str, str, float, float]] = []

            def get_account_info(self):
                return {"return": {"balance": {"btc": "1.0", "idr": "1000000"}}}

            def open_orders(self, pair: str):
                return {"return": {"orders": []}}

            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["50000", "1"]]}

            def get_pair_min_order(self, pair: str):
                # Coin minimum is above our amount, but IDR minimum is satisfied.
                return {"min_coin": 5.0, "min_idr": 10_000.0}

            def create_order(self, pair, order_type, price, amount):
                self.placed.append((pair, order_type, price, amount))
                return {"order_id": "123"}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            min_order_idr=30_000.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        trader.tracker.record_trade("buy", 50_000.0, 1.0)  # position = 1 BTC @ 50,000 IDR

        decision = StrategyDecision(
            mode="swing_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=50_000.0,
            amount=1.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "btc_idr", "price": 50_000.0, "decision": decision}

        outcome = trader.force_sell(snapshot)
        self.assertEqual(outcome["status"], "force_sold")
        self.assertEqual(len(trader.client.placed), 1)
        self.assertAlmostEqual(trader.tracker.base_position, 0.0)

    def test_force_sell_clears_when_parsed_coin_min_implies_higher_idr_min(self) -> None:
        """Parsed coin minimum should be converted to IDR and enforced to clear dust."""

        class _LiveClient(GuardedTrader._Client):
            def get_account_info(self):
                return {"return": {"balance": {"arkm": "1.0", "idr": "1000000"}}}

            def open_orders(self, pair: str):
                return {"return": {"orders": []}}

            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["10000", "1"]]}

            def create_order(self, pair, order_type, price, amount):
                raise RuntimeError(
                    "API error: {'success': 0, 'error': 'Minimum order 5.0 ARKM', 'error_code': ''}"
                )

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            min_order_idr=30_000.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        trader.tracker.record_trade("buy", 10_000.0, 1.0)  # 1 ARKM @ 10,000 IDR

        decision = StrategyDecision(
            mode="swing_trading",
            action="sell",
            confidence=0.9,
            reason="exit",
            target_price=10_000.0,
            amount=1.0,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "arkm_idr", "price": 10_000.0, "decision": decision}

        outcome = trader.force_sell(snapshot)
        self.assertEqual(outcome["status"], "dust_cleared")
        self.assertAlmostEqual(trader.tracker.base_position, 0.0)


    def test_force_sell_clears_dust_idr_below_config_min_without_api_call(self) -> None:
        """When sell IDR value is below config min_order_idr, force_sell must clear
        dust without making an API call — even when per-pair min is not cached."""

        class _LiveClient(GuardedTrader._Client):
            order_placed = False

            def get_account_info(self):
                return {"return": {"balance": {"perp": "0.1", "idr": "1000000"}}}

            def open_orders(self, pair: str):
                return {"return": {"orders": []}}

            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["14210", "1"]]}

            def create_order(self, pair, order_type, price, amount):
                _LiveClient.order_placed = True
                raise RuntimeError("Should not reach create_order")

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=500_000.0,
            min_order_idr=30_000.0,
            multi_position_enabled=False,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()
        # Position: 0.1 PERP @ 14,210 IDR → value ≈ Rp1421 < Rp30,000
        trader.tracker.record_trade("buy", 14_210.0, 0.1)

        decision = StrategyDecision(
            mode="day_trading",
            action="sell",
            confidence=0.9,
            reason="trailing_tp_triggered",
            target_price=14_210.0,
            amount=0.1,
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "perp_idr", "price": 14_210.0, "decision": decision}

        outcome = trader.force_sell(snapshot)
        self.assertEqual(outcome["status"], "dust_cleared")
        self.assertAlmostEqual(trader.tracker.base_position, 0.0)
        # Must not have attempted the API call
        self.assertFalse(_LiveClient.order_placed)


class RiskHoldCancellationTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_hold_cancels_open_buy_orders_and_returns_placeholder_cash(self) -> None:
        class _LiveClient(GuardedTrader._Client):
            def __init__(self) -> None:
                self.cancelled: list[str] = []
                self.invalidated = False

            def open_orders(self, pair: str):
                return {"return": {"orders": [{"order_id": "111", "type": "buy"}, {"order_id": "222", "type": "sell"}]}}

            def cancel_order(self, pair: str, order_id: str, order_type: str | None = None):
                self.cancelled.append(str(order_id))
                return {"success": 1}

            def invalidate_open_orders_cache(self, pair: str | None = None):
                self.invalidated = True

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            initial_capital=1_000_000.0,
            multi_position_enabled=True,
            multi_position_max=2,
        )
        trader = GuardedTrader(config)
        trader.client = _LiveClient()

        initial_cash = trader.multi_manager.cash
        tracker = trader.multi_manager.allocate_capital("pixel_idr")
        # Allocation reduces the shared cash pool until the position is cancelled/closed.
        self.assertAlmostEqual(trader.multi_manager.cash, initial_cash - tracker.cash, places=2)

        snapshot = {
            "pair": "pixel_idr",
            "price": 255.0,
            "decision": StrategyDecision(
                mode="swing_trading",
                action="hold",
                confidence=0.0,
                reason="risk_management",
                target_price=255.0,
                amount=0.0,
                stop_loss=None,
                take_profit=None,
            ),
        }

        outcome = trader.maybe_execute(snapshot)

        self.assertEqual(outcome["status"], "hold")
        # Only the BUY order should be cancelled.
        self.assertEqual(trader.client.cancelled, ["111"])
        self.assertTrue(trader.client.invalidated)
        # Placeholder tracker should be released so the bot can allocate elsewhere.
        self.assertIsNone(trader.multi_manager.get_tracker("pixel_idr"))
        self.assertAlmostEqual(trader.multi_manager.cash, initial_cash, places=2)


class GridMinOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_grid_skips_orders_below_minimum(self) -> None:
        placed: list[tuple[str, float, float]] = []

        class _GridClient(GuardedTrader._Client):
            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                placed.append((order_type, price, amount))
                return {"success": 1}

        config = BotConfig(api_key="key", api_secret="secret", dry_run=False, grid_enabled=True, min_order_idr=30_000.0)
        trader = GuardedTrader(config)
        trader.client = _GridClient()

        # Two grid orders: one below Rp30k, one above.
        plan = GridPlan(
            anchor_price=1000.0,
            buy_orders=[GridOrder("buy", 1000.0, 10.0)],   # 10k IDR → below min
            sell_orders=[GridOrder("sell", 1000.0, 40.0)],  # 40k IDR → valid
        )
        snapshot = {
            "pair": "btc_idr",
            "price": 1000.0,
            "decision": StrategyDecision(
                mode="grid",
                action="grid",
                confidence=1.0,
                reason="grid_test",
                target_price=1000.0,
                amount=0.0,
                stop_loss=None,
                take_profit=None,
            ),
            "grid_plan": plan,
        }

        outcome = trader.maybe_execute(snapshot)

        self.assertEqual(outcome["status"], "grid_placed")
        # Only the order above the minimum should be placed.
        self.assertEqual(placed, [("sell", 1000.0, 40.0)])

    def test_grid_all_below_minimum_skips(self) -> None:
        class _GridClient(GuardedTrader._Client):
            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                raise AssertionError("create_order should not be called when all orders below minimum")

        config = BotConfig(api_key="key", api_secret="secret", dry_run=False, grid_enabled=True, min_order_idr=30_000.0)
        trader = GuardedTrader(config)
        trader.client = _GridClient()

        plan = GridPlan(
            anchor_price=500.0,
            buy_orders=[GridOrder("buy", 500.0, 20.0)],   # 10k
            sell_orders=[GridOrder("sell", 500.0, 10.0)],  # 5k
        )
        snapshot = {
            "pair": "eth_idr",
            "price": 500.0,
            "decision": StrategyDecision(
                mode="grid",
                action="grid",
                confidence=1.0,
                reason="grid_test",
                target_price=500.0,
                amount=0.0,
                stop_loss=None,
                take_profit=None,
            ),
            "grid_plan": plan,
        }

        outcome = trader.maybe_execute(snapshot)

        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("all_grid_orders_below_min_order", outcome["reason"])


class BuyPendingFillTests(unittest.TestCase):
    def test_buy_with_zero_receive_does_not_open_position(self) -> None:
        """A buy order that isn't filled (receive_<coin>=0) must not mark the bot as holding."""

        class _LiveClient:
            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["224", "10"]], "sell": [["225", "10"]]}

            def get_summaries(self) -> dict:
                return {}

            def get_pair_min_order(self, pair: str) -> Dict[str, float]:
                return {"min_coin": 0.0, "min_idr": 0.0}

            def load_pair_min_orders(self) -> None:
                pass

            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                self.pair = pair
                self.order_type = order_type
                self.amount = amount
                coin = pair.split("_")[0]
                return {"success": 1, "return": {f"receive_{coin}": "0", "order_id": "123"}}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            multi_position_enabled=False,
            max_slippage_pct=1.0,  # allow buy path without slippage skip
            min_order_idr=1,  # keep test focused on pending-fill handling
        )
        trader = Trader(config, client=_LiveClient())

        decision = StrategyDecision(
            mode="scalping",
            action="buy",
            confidence=0.9,
            reason="entry",
            target_price=224.0,
            amount=100.0,
            stop_loss=None,
            take_profit=None,
        )
        depth = {"buy": [["224", "10"]], "sell": [["225", "10"]]}
        snapshot = {"pair": "pixel_idr", "price": 224.0, "decision": decision, "depth": depth, "orderbook": None}

        outcome = trader.maybe_execute(snapshot)

        self.assertEqual(outcome["status"], "placed")
        self.assertAlmostEqual(outcome["amount"], 0.0)
        self.assertAlmostEqual(trader.tracker.base_position, 0.0)
        self.assertTrue(outcome["executed_steps"][0].get("pending"))

    def test_buy_zero_receive_but_balance_increases_records_position(self) -> None:
        """If receive_<coin>=0 but live balance increases, treat buy as filled."""

        class _LiveClient:
            def __init__(self) -> None:
                self.balance_calls = 0

            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["300", "10"]], "sell": [["301", "10"]]}

            def get_summaries(self) -> dict:
                return {}

            def get_pair_min_order(self, pair: str) -> Dict[str, float]:
                return {"min_coin": 0.0, "min_idr": 0.0}

            def load_pair_min_orders(self) -> None:
                pass

            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                coin = pair.split("_")[0]
                return {"success": 1, "return": {f"receive_{coin}": "0", "order_id": "123"}}

            def get_account_info(self):
                self.balance_calls += 1
                # Exchange reflects the filled amount even though receive_<coin> was 0
                return {"return": {"balance": {"pixel": "100", "idr": "900000"}}}

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            multi_position_enabled=False,
            max_slippage_pct=1.0,
        )
        trader = Trader(config, client=_LiveClient())

        decision = StrategyDecision(
            mode="scalping",
            action="buy",
            confidence=0.9,
            reason="entry",
            target_price=300.0,
            amount=100.0,
            stop_loss=None,
            take_profit=None,
        )
        depth = {"buy": [["300", "10"]], "sell": [["301", "10"]]}
        snapshot = {"pair": "pixel_idr", "price": 300.0, "decision": decision, "depth": depth, "orderbook": None}

        outcome = trader.maybe_execute(snapshot)

        self.assertEqual(outcome["status"], "placed")
        self.assertAlmostEqual(outcome["amount"], 100.0)
        self.assertAlmostEqual(trader.tracker.base_position, 100.0)
        self.assertEqual(trader.tracker.trade_count, 1)
        self.assertIsNone(outcome["executed_steps"][0].get("pending"))
        # Fallback should have queried the live balance
        self.assertGreater(trader.client.balance_calls, 0)

    def test_buy_zero_receive_retries_more_aggressively(self) -> None:
        """When a buy is unfilled, the bot must cancel and retry at a higher price."""

        class _LiveClient:
            def __init__(self) -> None:
                self.calls: list[float] = []
                self.cancelled: tuple[str, str] | None = None

            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["24.5", "10"]], "sell": [["25", "10"]]}

            def get_summaries(self) -> dict:
                return {}

            def get_pair_min_order(self, pair: str) -> Dict[str, float]:
                return {"min_coin": 0.0, "min_idr": 0.0}

            def load_pair_min_orders(self) -> None:
                pass

            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                self.calls.append(price)
                coin = pair.split("_")[0]
                if len(self.calls) == 1:
                    return {"success": 1, "return": {f"receive_{coin}": "0", "order_id": "111"}}
                return {"success": 1, "return": {f"receive_{coin}": str(amount), "order_id": "222"}}

            def cancel_order(self, pair: str, order_id: str, order_type: str):
                self.cancelled = (order_id, order_type)

            def invalidate_account_info_cache(self) -> None:
                pass

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            multi_position_enabled=False,
            max_slippage_pct=1.0,
            entry_aggressiveness_pct=0.0,  # force first price to match ask
            entry_retry_aggressiveness_pct=0.01,  # bump retry by 1%
            min_order_idr=1,
            min_buy_price_idr=0,
            min_coin_price_idr=0,
        )
        client = _LiveClient()
        trader = Trader(config, client=client)

        decision = StrategyDecision(
            mode="scalping",
            action="buy",
            confidence=0.9,
            reason="entry",
            target_price=25.0,
            amount=100.0,
            stop_loss=None,
            take_profit=None,
        )
        depth = {"buy": [["24.5", "10"]], "sell": [["25", "10"]]}
        snapshot = {"pair": "pixel_idr", "price": 25.0, "decision": decision, "depth": depth, "orderbook": None}

        outcome = trader.maybe_execute(snapshot)

        self.assertEqual(outcome["status"], "placed")
        self.assertAlmostEqual(trader.tracker.base_position, 100.0)
        self.assertGreater(len(client.calls), 1)
        self.assertGreater(client.calls[1], client.calls[0])
        self.assertIsNotNone(client.cancelled)
        self.assertIsNone(outcome["executed_steps"][0].get("pending"))

    def test_buy_zero_receive_retry_with_format_price_tuple(self) -> None:
        """Retry must unpack format_price (price, precision) tuple correctly."""

        class _LiveClient:
            def __init__(self) -> None:
                self.calls: list[float] = []
                self.cancelled: tuple[str, str] | None = None

            def get_depth(self, pair: str, count: int = 5):
                return {"buy": [["24.5", "10"]], "sell": [["25", "10"]]}

            def get_summaries(self) -> dict:
                return {}

            def get_pair_min_order(self, pair: str) -> Dict[str, float]:
                return {"min_coin": 0.0, "min_idr": 0.0}

            def load_pair_min_orders(self) -> None:
                pass

            def create_order(self, pair: str, order_type: str, price: float, amount: float):
                self.calls.append(price)
                coin = pair.split("_")[0]
                if len(self.calls) == 1:
                    return {"success": 1, "return": {f"receive_{coin}": "0", "order_id": "111"}}
                return {"success": 1, "return": {f"receive_{coin}": str(amount), "order_id": "222"}}

            def cancel_order(self, pair: str, order_id: str, order_type: str):
                self.cancelled = (order_id, order_type)

            def invalidate_account_info_cache(self) -> None:
                pass

            def format_price(self, pair: str, price: float) -> tuple[float, int]:
                """Return (rounded_price, precision) tuple like IndodaxClient."""
                return (round(price, 2), 2)

        config = BotConfig(
            api_key="key",
            api_secret="secret",
            dry_run=False,
            multi_position_enabled=False,
            max_slippage_pct=1.0,
            entry_aggressiveness_pct=0.0,
            entry_retry_aggressiveness_pct=0.01,
            min_order_idr=1,
            min_buy_price_idr=0,
            min_coin_price_idr=0,
        )
        client = _LiveClient()
        trader = Trader(config, client=client)

        decision = StrategyDecision(
            mode="scalping",
            action="buy",
            confidence=0.9,
            reason="entry",
            target_price=25.0,
            amount=100.0,
            stop_loss=None,
            take_profit=None,
        )
        depth = {"buy": [["24.5", "10"]], "sell": [["25", "10"]]}
        snapshot = {"pair": "pixel_idr", "price": 25.0, "decision": decision, "depth": depth, "orderbook": None}

        # This would raise TypeError before the fix
        outcome = trader.maybe_execute(snapshot)

        self.assertEqual(outcome["status"], "placed")
        self.assertGreater(len(client.calls), 1)
        # retry_price must be a float (not tuple) and higher than initial price
        self.assertIsInstance(client.calls[1], float)
        self.assertGreater(client.calls[1], client.calls[0])


class RugPullFilterTests(unittest.TestCase):
    """Tests for the rug-pull / dead-coin filter in analyze_market."""

    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_detect_rug_pull_drop(self) -> None:
        """detect_rug_pull_risk flags pair with >50% 24h drop."""
        from bot.analysis import detect_rug_pull_risk

        ticker = {"ticker": {"high": "100", "last": "30", "vol_idr": "5000000"}}
        result = detect_rug_pull_risk(ticker, max_drop_24h_pct=0.50)
        self.assertTrue(result.detected)
        self.assertIn("24h_drop", result.reason)
        self.assertGreater(result.drop_24h_pct, 0.5)

    def test_detect_rug_pull_no_drop(self) -> None:
        """detect_rug_pull_risk does not flag normal pair."""
        from bot.analysis import detect_rug_pull_risk

        ticker = {"ticker": {"high": "100", "last": "92", "vol_idr": "5000000"}}
        result = detect_rug_pull_risk(ticker, max_drop_24h_pct=0.50)
        self.assertFalse(result.detected)

    def test_detect_dead_coin_volume(self) -> None:
        """detect_rug_pull_risk flags dead coin with near-zero volume."""
        from bot.analysis import detect_rug_pull_risk

        ticker = {"ticker": {"high": "10", "last": "9.5", "vol_idr": "500"}}
        result = detect_rug_pull_risk(ticker, min_volume_24h_idr=1_000_000)
        self.assertTrue(result.detected)
        self.assertIn("dead_coin_volume", result.reason)

    def test_analyze_market_skips_rug_pull_pair(self) -> None:
        """analyze_market returns hold with rug_pull_risk when coin has crashed >50%."""
        from bot.analysis import RugPullRisk

        class _RugClient(GuardedTrader._Client):
            def get_ticker(self, pair: str):
                # 70% drop from high
                return {"ticker": {"high": "100", "last": "30", "vol_idr": "5000000"}}

            def get_depth(self, pair: str, count: int = 50):
                return {"buy": [["30", "1000"]], "sell": [["31", "1000"]]}

            def get_trades(self, pair: str, count: int = 200):
                return []

        config = BotConfig(
            api_key=None,
            rug_pull_max_drop_24h_pct=0.50,
        )
        trader = GuardedTrader(config)
        trader.client = _RugClient()
        trader._multi_feed = type(
            "_Feed", (), {
                "has_snapshot": False,
                "is_seeded": False,
                "get_ticker": lambda self, p: None,
                "get_depth": lambda self, p: None,
                "get_trades": lambda self, p: None,
            }
        )()

        snap = trader.analyze_market("rug_idr")
        self.assertEqual(snap["decision"].action, "hold")
        self.assertIn("rug_pull_risk", snap["decision"].reason)
        rug = snap.get("rug_pull_risk")
        self.assertIsNotNone(rug)
        self.assertTrue(rug.detected)

    def test_analyze_market_rug_pull_includes_market_data(self) -> None:
        """analyze_market rug-pull snapshot must include orderbook/volatility/levels
        so that the log can display spread, imbalance, vol and support/resistance
        instead of N/A placeholders."""

        class _RugClient(GuardedTrader._Client):
            def get_ticker(self, pair: str):
                return {"ticker": {"high": "100", "last": "30", "vol_idr": "5000000"}}

            def get_depth(self, pair: str, count: int = 50):
                return {"buy": [["30", "1000"]], "sell": [["31", "500"]]}

            def get_trades(self, pair: str, count: int = 200):
                return []

        config = BotConfig(
            api_key=None,
            rug_pull_max_drop_24h_pct=0.50,
        )
        trader = GuardedTrader(config)
        trader.client = _RugClient()
        trader._multi_feed = type(
            "_Feed", (), {
                "has_snapshot": False,
                "is_seeded": False,
                "get_ticker": lambda self, p: None,
                "get_depth": lambda self, p: None,
                "get_trades": lambda self, p: None,
            }
        )()

        snap = trader.analyze_market("rug_idr")
        self.assertEqual(snap["decision"].action, "hold")
        # orderbook, volatility and levels must be present (not None / missing)
        self.assertIn("orderbook", snap)
        self.assertIsNotNone(snap["orderbook"])
        self.assertIn("volatility", snap)
        self.assertIsNotNone(snap["volatility"])
        self.assertIn("levels", snap)
        self.assertIsNotNone(snap["levels"])
        # spread and imbalance should be finite numbers (not NaN)
        import math
        ob = snap["orderbook"]
        self.assertFalse(math.isnan(ob.spread_pct), "spread_pct should not be NaN")
        self.assertFalse(math.isnan(ob.imbalance), "imbalance should not be NaN")


class PairMinOrderCacheTests(unittest.TestCase):
    """Tests for the per-pair minimum order cache (load_pair_min_orders / get_pair_min_order)."""

    def test_load_pair_min_orders_populates_cache(self) -> None:
        """load_pair_min_orders must populate _pair_min_order from /api/pairs response."""
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}

        class _MockSession:
            def get(self, url, **kwargs):
                import json

                class _Resp:
                    def raise_for_status(self): pass
                    def json(self):
                        return [
                            {"id": "btcidr", "trade_min_base_currency": "0.0001", "trade_min_traded_currency": "10000"},
                            {"id": "wtecidr", "trade_min_base_currency": "3333.33333333", "trade_min_traded_currency": "10000"},
                        ]
                return _Resp()

        client.session = _MockSession()
        client.base_url = "https://indodax.com"
        client.timeout = 10
        client.load_pair_min_orders()

        self.assertIn("btcidr", client._pair_min_order)
        self.assertAlmostEqual(client._pair_min_order["btcidr"]["min_coin"], 0.0001)
        self.assertAlmostEqual(client._pair_min_order["wtecidr"]["min_coin"], 3333.33333333, places=5)

    def test_get_pair_min_order_returns_zero_for_unknown(self) -> None:
        """get_pair_min_order must return zeros for unknown pairs."""
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}
        result = client.get_pair_min_order("unknown_idr")
        self.assertEqual(result["min_coin"], 0.0)
        self.assertEqual(result["min_idr"], 0.0)

    def test_parse_minimum_order_error(self) -> None:
        """parse_minimum_order_error must extract the minimum amount from error text."""
        from bot.indodax_client import IndodaxClient

        msg = "API error: {'success': 0, 'error': 'Minimum order 3333.33333333 WTEC', 'error_code': ''}"
        result = IndodaxClient.parse_minimum_order_error(msg)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 3333.33333333, places=5)

    def test_parse_minimum_order_error_no_match(self) -> None:
        """parse_minimum_order_error must return None for non-matching errors."""
        from bot.indodax_client import IndodaxClient

        result = IndodaxClient.parse_minimum_order_error("Some other error")
        self.assertIsNone(result)

    def test_parse_minimum_order_error_with_is(self) -> None:
        """parse_minimum_order_error must handle 'Minimum order is X COIN' format."""
        from bot.indodax_client import IndodaxClient

        msg = "API error: {'success': 0, 'error': 'Minimum order is 37.03703703 CJL.', 'error_code': ''}"
        result = IndodaxClient.parse_minimum_order_error(msg)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 37.03703703, places=5)


class TimeBasedExitTest(unittest.TestCase):
    """Tests for the time-based exit feature (MAX_HOLD_SECONDS)."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _trader(self, max_hold_seconds: float, max_hold_profit_pct: float = 0.01) -> Trader:
        config = BotConfig(
            api_key=None,
            dry_run=True,
            max_hold_seconds=max_hold_seconds,
            max_hold_profit_pct=max_hold_profit_pct,
            multi_position_enabled=False,
        )
        trader = Trader(config)
        trader.client = type("_C", (), {
            "get_depth": lambda self, *a, **kw: {
                "buy": [["100", "10"]], "sell": [["100.05", "10"]],
            },
        })()
        return trader

    def test_time_exit_triggers_when_stagnant(self):
        """Position held too long below profit threshold must be force-sold."""
        import time as _time
        trader = self._trader(max_hold_seconds=1, max_hold_profit_pct=0.05)
        # Open a position at avg_cost=100 → current price=100 → profit=0% < 5%
        trader.tracker.record_trade("buy", 100.0, 0.5)
        # Backdate the open time so the position appears stale
        trader.tracker.position_open_time = _time.time() - 10  # 10s > 1s limit
        snap = _make_buy_snap(price=100.0, action="hold")
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "time_exit")
        self.assertIn("max_hold_seconds", outcome["reason"])

    def test_time_exit_does_not_trigger_when_profitable(self):
        """Position above the profit threshold must NOT be force-sold by time exit."""
        import time as _time
        trader = self._trader(max_hold_seconds=1, max_hold_profit_pct=0.01)
        # Buy at 90, current price=100 → profit ≈ 11% > 1% threshold
        trader.tracker.record_trade("buy", 90.0, 0.5)
        trader.tracker.position_open_time = _time.time() - 10
        snap = _make_buy_snap(price=100.0, action="hold")
        outcome = trader.maybe_execute(snap)
        self.assertNotEqual(outcome["status"], "time_exit")

    def test_time_exit_does_not_trigger_before_limit(self):
        """Position held less than max_hold_seconds must not be time-exited."""
        trader = self._trader(max_hold_seconds=3600, max_hold_profit_pct=0.05)
        # Open position just now — well within the hold limit
        trader.tracker.record_trade("buy", 100.0, 0.5)
        snap = _make_buy_snap(price=100.0, action="hold")
        outcome = trader.maybe_execute(snap)
        self.assertNotEqual(outcome["status"], "time_exit")

    def test_time_exit_disabled_when_zero(self):
        """When max_hold_seconds=0, time-based exit must be disabled."""
        import time as _time
        trader = self._trader(max_hold_seconds=0)
        trader.tracker.record_trade("buy", 100.0, 0.5)
        trader.tracker.position_open_time = _time.time() - 86400  # 1 day old
        snap = _make_buy_snap(price=100.0, action="hold")
        outcome = trader.maybe_execute(snap)
        self.assertNotEqual(outcome["status"], "time_exit")

    def test_time_exit_no_position_no_trigger(self):
        """Time exit must not trigger when no position is held."""
        import time as _time
        trader = self._trader(max_hold_seconds=1)
        # No buy recorded — base_position is 0
        snap = _make_buy_snap(price=100.0, action="buy")
        outcome = trader.maybe_execute(snap)
        self.assertNotEqual(outcome["status"], "time_exit")


class PositionOpenTimeTest(unittest.TestCase):
    """Tests for position_open_time tracking in PortfolioTracker."""

    def test_position_open_time_set_on_first_buy(self):
        """position_open_time must be recorded when first buy is placed."""
        import time as _time
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        before = _time.time()
        tracker.record_trade("buy", 100.0, 1.0)
        after = _time.time()
        self.assertGreaterEqual(tracker.position_open_time, before)
        self.assertLessEqual(tracker.position_open_time, after)

    def test_position_open_time_not_reset_on_staged_buy(self):
        """A second buy (staged entry) must NOT overwrite position_open_time."""
        import time as _time
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 1.0)
        first_open_time = tracker.position_open_time
        _time.sleep(0.01)
        tracker.record_trade("buy", 101.0, 0.5)
        self.assertEqual(tracker.position_open_time, first_open_time)

    def test_position_open_time_cleared_on_full_close(self):
        """position_open_time must be reset to 0 when position is fully sold."""
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 1.0)
        self.assertGreater(tracker.position_open_time, 0.0)
        tracker.record_trade("sell", 110.0, 1.0)
        self.assertEqual(tracker.position_open_time, 0.0)

    def test_position_open_time_not_cleared_on_partial_sell(self):
        """position_open_time must persist after a partial sell."""
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 2.0)
        open_time = tracker.position_open_time
        tracker.record_trade("sell", 110.0, 1.0)  # partial sell
        self.assertEqual(tracker.position_open_time, open_time)

    def test_position_hold_seconds_returns_zero_when_no_position(self):
        """position_hold_seconds must be 0.0 when no position is held."""
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        self.assertEqual(tracker.position_hold_seconds, 0.0)

    def test_position_hold_seconds_positive_while_holding(self):
        """position_hold_seconds must be > 0 immediately after a buy."""
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 1.0)
        self.assertGreater(tracker.position_hold_seconds, 0.0)

    def test_position_open_time_serialized_and_restored(self):
        """position_open_time must survive a to_state / load_state round-trip."""
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 1.0)
        original = tracker.position_open_time
        state = tracker.to_state()
        restored = PortfolioTracker(500_000.0, 0.2, 0.1)
        restored.load_state(state)
        self.assertEqual(restored.position_open_time, original)


class MomentumExitTest(unittest.TestCase):
    """Tests for Trader.check_momentum_exit()."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _snap(self, price=110.0, imbalance=0.0):
        from bot.analysis import TrendResult, OrderbookInsight, VolatilityStats
        trend = TrendResult(direction="up", strength=0.05, fast_ma=105.0, slow_ma=100.0)
        ob = OrderbookInsight(spread_pct=0.001, bid_volume=500.0, ask_volume=500.0, imbalance=imbalance)
        return {
            "pair": "btc_idr",
            "price": price,
            "trend": trend,
            "orderbook": ob,
            "volatility": VolatilityStats(volatility=0.01, avg_volume=100.0),
        }

    def test_disabled_when_thresholds_zero(self):
        config = BotConfig(api_key=None, momentum_exit_ob_threshold=0.0, momentum_exit_min_profit_pct=0.0)
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        # Even with very negative imbalance, exit should not trigger
        self.assertFalse(trader.check_momentum_exit(self._snap(price=103.0, imbalance=-0.9)))

    def test_exit_triggers_when_imbalance_drops_and_profit_sufficient(self):
        config = BotConfig(
            api_key=None,
            momentum_exit_ob_threshold=0.0,
            momentum_exit_min_profit_pct=0.02,
            multi_position_enabled=False,
        )
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        # Price is 3% above cost, imbalance = -0.2 (below threshold 0.0)
        self.assertTrue(trader.check_momentum_exit(self._snap(price=103.0, imbalance=-0.2)))

    def test_no_exit_when_profit_too_low(self):
        config = BotConfig(
            api_key=None,
            momentum_exit_ob_threshold=0.0,
            momentum_exit_min_profit_pct=0.05,
        )
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        # Profit is only 2% (below 5% threshold), should not exit
        self.assertFalse(trader.check_momentum_exit(self._snap(price=102.0, imbalance=-0.3)))

    def test_no_exit_when_imbalance_still_bullish(self):
        config = BotConfig(
            api_key=None,
            momentum_exit_ob_threshold=0.0,
            momentum_exit_min_profit_pct=0.02,
        )
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        # Imbalance = 0.2 (still above threshold 0.0), should not exit
        self.assertFalse(trader.check_momentum_exit(self._snap(price=103.0, imbalance=0.2)))

    def test_no_exit_when_no_position(self):
        config = BotConfig(
            api_key=None,
            momentum_exit_ob_threshold=0.0,
            momentum_exit_min_profit_pct=0.01,
        )
        trader = Trader(config)
        # No buy recorded; avg_cost = 0
        self.assertFalse(trader.check_momentum_exit(self._snap(price=103.0, imbalance=-0.3)))


class PostEntryDumpExitTest(unittest.TestCase):
    """Tests for Trader.check_post_entry_dump()."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_exit_triggers_on_fresh_dump_within_window(self):
        config = BotConfig(
            api_key=None,
            post_entry_dump_pct=0.02,
            post_entry_dump_window_seconds=120,
        )
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        self.assertTrue(trader.check_post_entry_dump(trader.tracker, 95.0))

    def test_exit_ignored_when_window_passed(self):
        config = BotConfig(
            api_key=None,
            post_entry_dump_pct=0.02,
            post_entry_dump_window_seconds=60,
        )
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        # Age the position beyond the window
        trader.tracker.position_open_time = time.time() - 120
        self.assertFalse(trader.check_post_entry_dump(trader.tracker, 90.0))

    def test_exit_disabled_when_threshold_zero(self):
        config = BotConfig(api_key=None, post_entry_dump_pct=0.0)
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        self.assertFalse(trader.check_post_entry_dump(trader.tracker, 90.0))


class PartialTp2And3TrackingTest(unittest.TestCase):
    """Tests for partial_tp2_taken and partial_tp3_taken state management."""

    def test_tp2_taken_reset_on_new_buy(self):
        from bot.tracking import PortfolioTracker
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 1.0)
        tracker.partial_tp2_taken = True
        tracker.record_trade("sell", 110.0, 1.0)  # full close
        tracker.record_trade("buy", 105.0, 1.0)   # new position
        self.assertFalse(tracker.partial_tp2_taken)

    def test_tp3_taken_reset_on_new_buy(self):
        from bot.tracking import PortfolioTracker
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 1.0)
        tracker.partial_tp3_taken = True
        tracker.record_trade("sell", 110.0, 1.0)
        tracker.record_trade("buy", 105.0, 1.0)
        self.assertFalse(tracker.partial_tp3_taken)

    def test_tp2_and_tp3_persist_via_state_round_trip(self):
        from bot.tracking import PortfolioTracker
        tracker = PortfolioTracker(500_000.0, 0.2, 0.1)
        tracker.record_trade("buy", 100.0, 1.0)
        tracker.partial_tp2_taken = True
        tracker.partial_tp3_taken = True
        state = tracker.to_state()
        restored = PortfolioTracker(500_000.0, 0.2, 0.1)
        restored.load_state(state)
        self.assertTrue(restored.partial_tp2_taken)
        self.assertTrue(restored.partial_tp3_taken)


class MultiPositionManagerTest(unittest.TestCase):
    """Tests for MultiPositionManager in tracking.py."""

    def _make_manager(self, capital=3_000_000.0, max_pos=3):
        from bot.tracking import MultiPositionManager
        return MultiPositionManager(
            initial_capital=capital,
            max_positions=max_pos,
            target_profit_pct=0.2,
            max_loss_pct=0.1,
        )

    def test_initial_state(self):
        mgr = self._make_manager(3_000_000.0, 3)
        self.assertEqual(mgr.cash, 3_000_000.0)
        self.assertEqual(mgr.position_count(), 0)
        self.assertTrue(mgr.can_open_position())

    def test_allocate_capital_splits_cash_evenly(self):
        mgr = self._make_manager(3_000_000.0, 3)
        t1 = mgr.allocate_capital("btc_idr")
        # First slot: 3M/3 = 1M
        self.assertAlmostEqual(t1.cash, 1_000_000.0)
        self.assertAlmostEqual(mgr.cash, 2_000_000.0)

        t2 = mgr.allocate_capital("eth_idr")
        # Second slot: 2M/2 = 1M
        self.assertAlmostEqual(t2.cash, 1_000_000.0)
        self.assertAlmostEqual(mgr.cash, 1_000_000.0)

    def test_has_position_false_before_buy(self):
        mgr = self._make_manager()
        mgr.allocate_capital("btc_idr")
        self.assertFalse(mgr.has_position("btc_idr"))

    def test_has_position_true_after_buy(self):
        mgr = self._make_manager()
        t = mgr.allocate_capital("btc_idr")
        t.record_trade("buy", 500_000.0, 1.0)
        self.assertTrue(mgr.has_position("btc_idr"))

    def test_at_max_positions(self):
        mgr = self._make_manager(2_000_000.0, 2)
        t1 = mgr.allocate_capital("btc_idr")
        t1.record_trade("buy", 500_000.0, 1.0)
        t2 = mgr.allocate_capital("eth_idr")
        t2.record_trade("buy", 300_000.0, 1.0)
        self.assertFalse(mgr.can_open_position())
        self.assertEqual(mgr.position_count(), 2)

    def test_return_position_cash_restores_pool(self):
        mgr = self._make_manager(3_000_000.0, 3)
        t = mgr.allocate_capital("btc_idr")
        t.record_trade("buy", 500_000.0, 2.0)   # cost = 1M
        t.record_trade("sell", 550_000.0, 2.0)  # proceeds = 1.1M (+0.1M profit)
        mgr.return_position_cash("btc_idr")
        # Pool receives tracker.cash = initial_slice + profit
        self.assertAlmostEqual(mgr.cash, 3_000_000.0 + 100_000.0, delta=1.0)

    def test_active_positions_only_returns_held(self):
        mgr = self._make_manager()
        t1 = mgr.allocate_capital("btc_idr")  # no buy recorded
        t2 = mgr.allocate_capital("eth_idr")
        t2.record_trade("buy", 300_000.0, 1.0)
        active = mgr.active_positions
        self.assertNotIn("btc_idr", active)
        self.assertIn("eth_idr", active)

    def test_duplicate_allocate_returns_existing_tracker(self):
        mgr = self._make_manager()
        t1 = mgr.allocate_capital("btc_idr")
        t2 = mgr.allocate_capital("btc_idr")
        self.assertIs(t1, t2)

    def test_total_equity_sums_pool_and_positions(self):
        mgr = self._make_manager(3_000_000.0, 3)
        t = mgr.allocate_capital("btc_idr")
        t.record_trade("buy", 1_000_000.0, 1.0)
        # pool = 3M - 1M = 2M, tracker.cash = 0, position = 1 BTC at 1.2M
        equity = mgr.total_equity({"btc_idr": 1_200_000.0})
        self.assertAlmostEqual(equity, 2_000_000.0 + 1_200_000.0, delta=1.0)


class MultiPositionTraderTest(unittest.TestCase):
    """Tests for Trader multi-position routing via maybe_execute and force_sell."""

    class _StubClient:
        """Minimal stub that provides all methods called by maybe_execute."""

        _pair_min_order: Dict[str, Any] = {}

        def get_depth(self, pair: str, count: int = 5) -> Dict[str, Any]:
            # Return empty depth so maybe_execute falls back to snapshot price,
            # preventing slippage guards from triggering.
            return {"buy": [], "sell": []}

        def get_summaries(self) -> dict:
            return {}

        def get_pair_min_order(self, pair: str) -> Dict[str, float]:
            return {"min_coin": 0.0, "min_idr": 0.0}

        def load_pair_min_orders(self) -> None:
            pass

    def _make_trader(self, max_pos=3):
        config = BotConfig(
            api_key=None,
            dry_run=True,
            multi_position_enabled=True,
            multi_position_max=max_pos,
            initial_capital=3_000_000.0,
            min_order_idr=10_000.0,
            pair_cooldown_seconds=0,
            trailing_stop_pct=0.0,
            trailing_tp_pct=0.0,
        )
        trader = Trader(config, client=self._StubClient())
        return trader

    def _buy_snapshot(self, pair="btc_idr", price=1_000_000.0, amount=1.0):
        return {
            "pair": pair,
            "price": price,
            "decision": StrategyDecision(
                action="buy",
                amount=amount,
                confidence=0.9,
                reason="test",
                mode="test",
                target_price=price,
                stop_loss=None,
                take_profit=None,
            ),
            "orderbook": None,
            "volatility": None,
            "indicators": None,
            "trend": None,
        }

    def _sell_snapshot(self, pair="btc_idr", price=1_100_000.0, amount=1.0):
        return {
            "pair": pair,
            "price": price,
            "decision": StrategyDecision(
                action="sell",
                amount=amount,
                confidence=0.9,
                reason="test",
                mode="test",
                target_price=price,
                stop_loss=None,
                take_profit=None,
            ),
            "orderbook": None,
            "volatility": None,
            "indicators": None,
            "trend": None,
        }

    def test_multi_position_enabled_creates_manager(self):
        trader = self._make_trader()
        self.assertIsNotNone(trader.multi_manager)

    def test_buy_creates_per_pair_tracker(self):
        trader = self._make_trader()
        outcome = trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        self.assertIn(outcome["status"], ("simulated", "placed"))
        self.assertTrue(trader.multi_manager.has_position("btc_idr"))

    def test_two_buys_on_different_pairs(self):
        trader = self._make_trader(max_pos=3)
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        trader.maybe_execute(self._buy_snapshot("eth_idr", 500_000.0, 1.0))
        self.assertEqual(trader.multi_manager.position_count(), 2)

    def test_max_positions_blocks_third_buy(self):
        trader = self._make_trader(max_pos=2)
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        trader.maybe_execute(self._buy_snapshot("eth_idr", 500_000.0, 1.0))
        outcome = trader.maybe_execute(self._buy_snapshot("xrp_idr", 100_000.0, 1.0))
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("max_open_positions", outcome["reason"])

    def test_at_max_positions_reflects_multi_manager(self):
        trader = self._make_trader(max_pos=2)
        self.assertFalse(trader.at_max_positions())
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        self.assertFalse(trader.at_max_positions())
        trader.maybe_execute(self._buy_snapshot("eth_idr", 500_000.0, 1.0))
        self.assertTrue(trader.at_max_positions())

    def test_sell_returns_cash_to_pool(self):
        trader = self._make_trader(max_pos=2)
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        initial_cash = trader.multi_manager.cash
        trader.maybe_execute(self._sell_snapshot("btc_idr", 1_100_000.0, 1.0))
        # Position should be closed; cash should increase
        self.assertFalse(trader.multi_manager.has_position("btc_idr"))
        self.assertGreater(trader.multi_manager.cash, initial_cash)

    def test_force_sell_returns_cash_to_pool(self):
        trader = self._make_trader(max_pos=2)
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        snap = self._sell_snapshot("btc_idr", 1_050_000.0, 1.0)
        trader.force_sell(snap)
        self.assertFalse(trader.multi_manager.has_position("btc_idr"))

    def test_active_positions_reflects_holdings(self):
        trader = self._make_trader(max_pos=3)
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        trader.maybe_execute(self._buy_snapshot("eth_idr", 500_000.0, 1.0))
        active = trader.active_positions
        self.assertIn("btc_idr", active)
        self.assertIn("eth_idr", active)
        self.assertEqual(len(active), 2)

    def test_per_pair_trackers_are_independent(self):
        trader = self._make_trader(max_pos=3)
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        trader.maybe_execute(self._buy_snapshot("eth_idr", 500_000.0, 2.0))
        btc_tracker = trader.multi_manager.get_tracker("btc_idr")
        eth_tracker = trader.multi_manager.get_tracker("eth_idr")
        self.assertIsNot(btc_tracker, eth_tracker)
        # avg_cost is filled at reference_price (top ask ≈ snapshot price)
        self.assertGreater(btc_tracker.avg_cost, 0)
        self.assertGreater(eth_tracker.avg_cost, 0)
        self.assertNotAlmostEqual(btc_tracker.avg_cost, eth_tracker.avg_cost, delta=100_000)

    def test_single_position_mode_unchanged(self):
        """Ensure single-position mode behavior is unaffected when explicitly disabled."""
        config = BotConfig(api_key=None, dry_run=True, initial_capital=1_000_000.0, multi_position_enabled=False)
        trader = Trader(config)
        self.assertIsNone(trader.multi_manager)
        trader.tracker.record_trade("buy", 500_000.0, 1.0)
        self.assertEqual(len(trader.active_positions), 1)
        self.assertTrue(trader.at_max_positions())

    def test_portfolio_snapshot_multi_no_positions_shows_initial_capital(self):
        """portfolio_snapshot shows total cash (initial capital) when no positions held."""
        trader = self._make_trader(max_pos=3)
        snap = trader.portfolio_snapshot("btc_idr", 1_000_000.0)
        # No positions yet — equity should equal the full initial capital, not zero.
        self.assertAlmostEqual(snap["equity"], 3_000_000.0, delta=1.0)
        self.assertAlmostEqual(snap["cash"], 3_000_000.0, delta=1.0)
        self.assertEqual(snap["base_position"], 0.0)
        self.assertEqual(snap["realized_pnl"], 0.0)
        self.assertEqual(snap["principal"], 3_000_000.0)

    def test_portfolio_snapshot_multi_with_position_reflects_holdings(self):
        """portfolio_snapshot aggregates equity when at least one position is open."""
        trader = self._make_trader(max_pos=3)
        # Open a position
        trader.maybe_execute(self._buy_snapshot("btc_idr", 1_000_000.0, 1.0))
        snap = trader.portfolio_snapshot("btc_idr", 1_050_000.0)
        # Equity = remaining cash + btc position at current price
        self.assertGreater(snap["equity"], 0)
        self.assertAlmostEqual(snap["principal"], 3_000_000.0, delta=1.0)
        # Profit buffer should be ≥ 0
        self.assertGreaterEqual(snap["profit_buffer"], 0.0)

    def test_portfolio_snapshot_single_mode_delegates_to_tracker(self):
        """In single-position mode, portfolio_snapshot returns tracker.as_dict()."""
        config = BotConfig(api_key=None, dry_run=True, initial_capital=1_000_000.0, multi_position_enabled=False)
        trader = Trader(config)
        snap = trader.portfolio_snapshot("btc_idr", 1_000_000.0)
        expected = trader.tracker.as_dict(1_000_000.0)
        self.assertEqual(snap["equity"], expected["equity"])
        self.assertEqual(snap["cash"], expected["cash"])

    def test_portfolio_snapshot_multi_target_and_floor_not_zero(self):
        """In multi-position mode target_equity and min_equity must reflect config, not 0."""
        trader = self._make_trader(max_pos=3)
        snap = trader.portfolio_snapshot("btc_idr", 1_000_000.0)
        initial = 3_000_000.0
        # Default target_profit_pct=20% → target = 3_600_000
        self.assertGreater(snap["target_equity"], initial)
        # Default max_loss_pct=10% → floor = 2_700_000
        self.assertLess(snap["min_equity"], initial)
        self.assertGreater(snap["min_equity"], 0.0)


class AllocateCapitalMinOrderTests(unittest.TestCase):
    """Tests that allocate_capital respects min_order_idr to prevent
    per-position capital from falling below the exchange minimum."""

    def _make_manager(self, capital: float, max_pos: int):
        from bot.tracking import MultiPositionManager
        return MultiPositionManager(
            initial_capital=capital,
            max_positions=max_pos,
            target_profit_pct=0.2,
            max_loss_pct=0.1,
        )

    def test_allocate_reduces_slots_when_capital_too_thin(self):
        """With 1M IDR and 55 positions, per-slot = ~18K < 30K.
        min_order_idr=30K should reduce effective slots to 33 so each gets ~30K."""
        mgr = self._make_manager(1_000_000.0, 55)
        t1 = mgr.allocate_capital("ponke_idr", min_order_idr=30_000.0)
        # max_slots = int(1_000_000 / 30_000) = 33
        # capital = 1_000_000 / 33 ≈ 30303
        self.assertGreaterEqual(t1.cash, 30_000.0,
                                "Per-position capital must be >= min_order_idr")

    def test_allocate_without_min_order_can_be_below_minimum(self):
        """Without min_order_idr guard, 1M / 55 ≈ 18K which is below 30K."""
        mgr = self._make_manager(1_000_000.0, 55)
        t1 = mgr.allocate_capital("ponke_idr")  # no min_order_idr
        self.assertLess(t1.cash, 30_000.0,
                        "Without guard, per-position capital may be below 30K")

    def test_allocate_with_min_order_successive_slots(self):
        """Successive allocations all respect the minimum when using the guard."""
        mgr = self._make_manager(1_000_000.0, 55)
        for i in range(5):
            t = mgr.allocate_capital(f"pair{i}_idr", min_order_idr=30_000.0)
            self.assertGreaterEqual(t.cash, 30_000.0,
                                    f"Position {i} capital must be >= 30K")

    def test_allocate_small_capital_single_slot(self):
        """When capital is only 50K and min is 30K, only 1 slot should be used."""
        mgr = self._make_manager(50_000.0, 10)
        t1 = mgr.allocate_capital("btc_idr", min_order_idr=30_000.0)
        # max_slots = int(50_000/30_000) = 1 → capital = 50_000
        self.assertAlmostEqual(t1.cash, 50_000.0)

    def test_capital_per_new_position_consistent_with_allocate(self):
        """capital_per_new_position and allocate_capital should give similar amounts
        when both use the same min_order_idr."""
        mgr = self._make_manager(1_000_000.0, 55)
        suggested = mgr.capital_per_new_position(min_order_idr=30_000.0)
        t1 = mgr.allocate_capital("btc_idr", min_order_idr=30_000.0)
        self.assertAlmostEqual(suggested, t1.cash, delta=1.0)


class SellRetryExitProtectionTests(unittest.TestCase):
    """Tests for sell-side retry logic (exit protection)."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_sell_retry_at_lower_price_on_unfilled(self):
        """When a sell order is placed but not filled, the bot should cancel
        and retry at a lower (more aggressive) price."""
        created = []
        cancelled = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                call_count = len(created)
                if call_count == 1:
                    # First sell order: not filled (returns order_id but spend_ponke=0)
                    return {"success": 1, "return": {"order_id": "99", "spend_ponke": 0}}
                else:
                    # Second sell order: filled (returns spend_ponke > 0)
                    return {"success": 1, "return": {"order_id": "100", "spend_ponke": amount, "receive_idr": price * amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "100000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = BotConfig(
            api_key="test_key",
            api_secret="test_secret",
            dry_run=False,
            initial_capital=1_000_000.0,
            multi_position_enabled=False,
            min_order_idr=30_000.0,
            max_slippage_pct=0.05,
            entry_aggressiveness_pct=0.001,
            entry_retry_aggressiveness_pct=0.002,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
        )
        trader = Trader(config)
        trader.client = _Client()
        # Set up a position so sell is valid
        trader.tracker.record_trade("buy", 100.0, 500.0)

        snap = {
            "pair": "ponke_idr",
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="sell",
                confidence=0.9,
                reason="exit",
                target_price=100.0,
                amount=500.0,
                stop_loss=None,
                take_profit=None,
            ),
        }
        outcome = trader.maybe_execute(snap)
        # The sell should have been retried
        self.assertGreaterEqual(len(created), 2,
                                "Sell order should have been retried at aggressive price")
        # The first order should have been cancelled
        self.assertEqual(len(cancelled), 1)
        self.assertEqual(cancelled[0], "99")
        # The retry price should be LOWER than original (more aggressive for sell)
        self.assertLess(created[1]["price"], created[0]["price"])


class ChaseAlgorithmTests(unittest.TestCase):
    """Tests for the professional order execution chase algorithm."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _make_config(self, **overrides):
        defaults = dict(
            api_key="test_key",
            api_secret="test_secret",
            dry_run=False,
            initial_capital=1_000_000.0,
            multi_position_enabled=False,
            min_order_idr=30_000.0,
            max_slippage_pct=0.05,
            entry_aggressiveness_pct=0.001,
            entry_retry_aggressiveness_pct=0.002,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            chase_max_retries=3,
            order_timeout_to_market=True,
        )
        defaults.update(overrides)
        return BotConfig(**defaults)

    def _make_snap(self, pair, action, price, amount, **extra):
        snap = {
            "pair": pair,
            "price": price,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action=action,
                confidence=0.9,
                reason="test",
                target_price=price,
                amount=amount,
                stop_loss=None,
                take_profit=None,
            ),
        }
        snap.update(extra)
        return snap

    # ── Chase Algorithm (Buy) ─────────────────────────────────────────────

    def test_buy_chase_multiple_retries(self):
        """Buy chase should retry up to chase_max_retries times before giving up."""
        created = []
        cancelled = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                # Never fill
                return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": 0}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        # Initial order + 3 chase retries = 4 orders total
        self.assertEqual(len(created), 4, f"Expected 1 initial + 3 chase = 4 orders, got {len(created)}")
        # All 4 orders should have been cancelled (3 in chase + initial has no cancel because it's inside chase)
        self.assertGreaterEqual(len(cancelled), 3)
        # Each retry price should be higher than the previous (aggressive)
        for i in range(1, len(created)):
            self.assertGreaterEqual(created[i]["price"], created[0]["price"],
                                    f"Chase retry {i} price should be >= initial price")

    def test_buy_chase_fills_on_second_retry(self):
        """When a buy fills on the 2nd chase retry, no further retries are made."""
        created = []
        cancelled = []
        _balance = [0.0]  # track simulated balance

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) <= 2:
                    return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": 0}}
                else:
                    _balance[0] = amount  # simulate fill
                    return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": str(_balance[0]), "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        # 1 initial + 2 chase retries (fills on 2nd retry) = 3
        self.assertEqual(len(created), 3, f"Expected 3 orders (fill on 2nd retry), got {len(created)}")
        self.assertEqual(outcome["status"], "placed")

    def test_buy_timeout_to_market_order(self):
        """After all chase retries fail, convert to market order if enabled."""
        created = []
        cancelled = []
        _balance = [0.0]

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) <= 4:
                    # First 4: 1 initial + 3 chase retries, none fill
                    return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": 0}}
                else:
                    # The 5th order is the market-price conversion → fill
                    _balance[0] = amount
                    return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": str(_balance[0]), "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=True)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        # 1 initial + 3 chase + 1 market = 5 orders
        self.assertEqual(len(created), 5, f"Expected 5 orders (market fallback), got {len(created)}")
        self.assertEqual(outcome["status"], "placed")
        # The market order should be at the maximum allowed price
        allowed_max = 100.0 * (1 + 0.05)  # price * (1 + max_slippage_pct)
        self.assertAlmostEqual(created[-1]["price"], allowed_max, places=1)

    # ── Chase Algorithm (Sell / Exit Protection) ──────────────────────────

    def test_sell_chase_multiple_retries(self):
        """Sell chase should retry up to chase_max_retries times."""
        created = []
        cancelled = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) <= 3:
                    return {"success": 1, "return": {"order_id": str(len(created)), "spend_ponke": 0}}
                else:
                    return {"success": 1, "return": {"order_id": str(len(created)), "spend_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "100000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3)
        trader = Trader(config)
        trader.client = _Client()
        trader.tracker.record_trade("buy", 100.0, 500.0)

        snap = self._make_snap("ponke_idr", "sell", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        # 1 initial + 3 chase retries = 4 (fills on last)
        self.assertGreaterEqual(len(created), 4, f"Expected >= 4 sell orders, got {len(created)}")
        # Chase retries should have LOWER prices (more aggressive sell)
        for i in range(2, len(created)):
            self.assertLessEqual(created[i]["price"], created[1]["price"])

    def test_sell_timeout_to_market_order(self):
        """After all sell chase retries fail, convert to market-price sell."""
        created = []
        cancelled = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) <= 4:
                    return {"success": 1, "return": {"order_id": str(len(created)), "spend_ponke": 0}}
                else:
                    return {"success": 1, "return": {"order_id": str(len(created)), "spend_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "100000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=True)
        trader = Trader(config)
        trader.client = _Client()
        trader.tracker.record_trade("buy", 100.0, 500.0)

        snap = self._make_snap("ponke_idr", "sell", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        # 1 initial + 3 chase + 1 market = 5 orders
        self.assertEqual(len(created), 5, f"Expected 5 sell orders (market fallback), got {len(created)}")
        # The market order should be at the minimum allowed price
        allowed_min = 100.0 * (1 - 0.05)  # price * (1 - max_slippage_pct)
        self.assertAlmostEqual(created[-1]["price"], allowed_min, places=1)

    # ── Adaptive Limit Order (Orderbook Re-read) ─────────────────────────

    def test_buy_chase_re_reads_orderbook(self):
        """Each chase retry should re-read the orderbook for adaptive pricing."""
        depth_calls = []
        created = []
        cancelled = []
        _ask_prices = iter([101.0, 102.0, 103.0, 104.0, 105.0])
        _balance = [0.0]

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                try:
                    ask = next(_ask_prices)
                except StopIteration:
                    ask = 105.0
                depth_calls.append(ask)
                return {"buy": [["100.0", "1"]], "sell": [[str(ask), "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) <= 2:
                    return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": 0}}
                else:
                    _balance[0] = amount
                    return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": str(_balance[0]), "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        # get_depth should be called multiple times (1 initial + N chase retries)
        self.assertGreaterEqual(len(depth_calls), 2,
                                "Chase should re-read orderbook for adaptive pricing")

    # ── Smart Entry Buffer ────────────────────────────────────────────────

    def test_smart_entry_buffer_skips_when_price_moved(self):
        """Smart entry buffer should skip buy if ask moves above allowed_max."""
        depth_calls = [0]

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                depth_calls[0] += 1
                if depth_calls[0] == 1:
                    # Initial depth read
                    return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}
                else:
                    # Buffer re-read: ask has moved way up (beyond allowed_max)
                    return {"buy": [["100.0", "1"]], "sell": [["200.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                return {"success": 1, "return": {"order_id": "1", "receive_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(smart_entry_buffer_enabled=True, max_slippage_pct=0.05)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("smart_entry_buffer", outcome["reason"])

    def test_smart_entry_buffer_proceeds_when_price_stable(self):
        """Smart entry buffer proceeds normally when price hasn't moved much."""
        created = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "500", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(smart_entry_buffer_enabled=True, max_slippage_pct=0.05)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        self.assertEqual(outcome["status"], "placed")
        self.assertGreaterEqual(len(created), 1)

    # ── Config validation ─────────────────────────────────────────────────

    def test_chase_max_retries_config(self):
        """chase_max_retries must be non-negative."""
        cfg = BotConfig(api_key=None, chase_max_retries=-1)
        with self.assertRaises(ValueError):
            cfg._validate()

    def test_chase_max_retries_zero_uses_single_retry(self):
        """chase_max_retries=0 should fall back to single retry behaviour."""
        created = []
        cancelled = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": 0}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=0, order_timeout_to_market=False)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        # 1 initial + 1 chase retry (min=1) = 2 orders
        self.assertEqual(len(created), 2, "chase_max_retries=0 should still do 1 retry")

    # ── Partial Fill Management ───────────────────────────────────────────

    def test_buy_partial_fill_records_filled_portion(self):
        """When a buy order is partially filled, record the filled amount
        and chase the remaining unfilled portion."""
        created = []
        cancelled = []
        _balance = [0.0]

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) == 1:
                    # Partial fill: 40% of requested
                    filled = amount * 0.4
                    _balance[0] += filled
                    return {"success": 1, "return": {"order_id": "1", "receive_ponke": filled}}
                else:
                    # Subsequent orders fill completely
                    _balance[0] += amount
                    return {"success": 1, "return": {"order_id": str(len(created)), "receive_ponke": amount}}

            def cancel_order(self, pair, order_id, order_type=None):
                cancelled.append(order_id)
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": str(_balance[0]), "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("ponke_idr", "buy", 100.0, 500.0)
        outcome = trader.maybe_execute(snap)

        self.assertEqual(outcome["status"], "placed")
        # Should have the partial fill step plus chase completion
        partial_steps = [s for s in outcome["executed_steps"] if s.get("partial")]
        self.assertGreaterEqual(len(partial_steps), 1, "Should record partial fill step")

    def test_buy_chase_skips_below_min_order_after_partial_fill(self):
        """Chase retry should not place an order when remaining amount is below min_order_idr."""
        created = []
        _balance = [0.0]

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["270.0", "1"]], "sell": [["270.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                # First call: simulate partial fill via balance bump
                if len(created) == 1:
                    _balance[0] = 100.0  # partial fill of 100 out of ~111
                    return {"success": 1, "return": {"order_id": "1", "receive_cjl": 0}}
                # Should NOT reach here — chase should skip below-min remaining
                return {"success": 1, "return": {"order_id": str(len(created)), "receive_cjl": 0}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"cjl": str(_balance[0]), "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        # min_order_idr=30000, price=270 → min amount ≈ 111.11
        # After partial fill of 100, remaining ≈ 11.11 → 11.11 * 270 = 3000 < 30000
        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False, min_order_idr=30_000.0)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("cjl_idr", "buy", 270.0, 111.11)
        outcome = trader.maybe_execute(snap)

        # Only the initial order should be placed; chase retries should be skipped
        self.assertEqual(len(created), 1, f"Expected 1 order (chase skipped due to min), got {len(created)}")

    def test_sell_chase_skips_below_min_order_after_partial_fill(self):
        """Sell chase retry should skip when remaining amount is below min_order_idr."""
        created = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["270.0", "1"]], "sell": [["270.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) == 1:
                    # Partial fill: sell 140 out of 150, leaving remainder of 10
                    return {"success": 1, "return": {"order_id": "1", "spend_cjl": 140.0, "receive_idr": 37800}}
                return {"success": 1, "return": {"order_id": str(len(created)), "spend_cjl": 0}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"cjl": "10", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        # price=270, amount=150 → order_value=40500 > 30000 (passes initial check)
        # After partial fill of 140, remainder=10 → 10*270=2700 < 30000 (below min)
        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False, min_order_idr=30_000.0)
        trader = Trader(config)
        trader.client = _Client()
        trader.tracker.record_trade("buy", 270.0, 150.0)

        snap = self._make_snap("cjl_idr", "sell", 270.0, 150.0)
        outcome = trader.maybe_execute(snap)

        # Only initial order; chase should skip due to remaining below min
        self.assertEqual(len(created), 1, f"Expected 1 order (sell chase skipped due to min), got {len(created)}")

    def test_buy_chase_market_fallback_skips_below_min(self):
        """Market order fallback after chase exhaustion should skip when below min_order_idr."""
        created = []
        _balance = [0.0]

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["270.0", "1"]], "sell": [["270.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) == 1:
                    _balance[0] = 100.0
                    return {"success": 1, "return": {"order_id": "1", "receive_cjl": 0}}
                return {"success": 1, "return": {"order_id": str(len(created)), "receive_cjl": 0}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"cjl": str(_balance[0]), "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=True, min_order_idr=30_000.0)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("cjl_idr", "buy", 270.0, 111.11)
        outcome = trader.maybe_execute(snap)

        # Only initial order should be placed; chase + market fallback should skip
        self.assertEqual(len(created), 1, f"Expected 1 order (market fallback skipped due to min), got {len(created)}")

    def test_buy_chase_uses_per_pair_min_when_config_min_is_zero(self):
        """Chase should respect per-pair exchange minimum even when config.min_order_idr is 0."""
        created = []
        _balance = [0.0]

        class _Client:
            _pair_min_order = {"cjl_idr": {"min_coin": 37.0, "min_idr": 10_000.0}}

            def get_depth(self, pair, count=5):
                return {"buy": [["270.0", "1"]], "sell": [["270.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) == 1:
                    _balance[0] = 100.0  # partial fill of 100 out of ~111
                    return {"success": 1, "return": {"order_id": "1", "receive_cjl": 0}}
                return {"success": 1, "return": {"order_id": str(len(created)), "receive_cjl": 0}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"cjl": str(_balance[0]), "idr": "1000000"}}}

            def get_pair_min_order(self, pair):
                return self._pair_min_order.get(pair.lower(), {"min_coin": 0.0, "min_idr": 0.0})

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        # min_order_idr=0 (unconfigured), but exchange pair minimum is 10000 IDR / 37 CJL
        # After partial fill of 100, remaining ~11.11 → below pair min of 37 CJL
        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False, min_order_idr=0.0)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("cjl_idr", "buy", 270.0, 111.11)
        outcome = trader.maybe_execute(snap)

        # Only the initial order; chase should skip due to per-pair min
        self.assertEqual(len(created), 1, f"Expected 1 order (chase skipped per-pair min), got {len(created)}")

    def test_buy_chase_catches_runtime_minimum_order_error(self):
        """Chase should catch RuntimeError with 'Minimum order' and break gracefully."""
        created = []
        _balance = [0.0]

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["270.0", "1"]], "sell": [["270.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) == 1:
                    _balance[0] = 100.0
                    return {"success": 1, "return": {"order_id": "1", "receive_cjl": 0}}
                # Exchange rejects the chase retry
                raise RuntimeError("API error: {'success': 0, 'error': 'Minimum order is 37.03703703 CJL.', 'error_code': ''}")

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"cjl": str(_balance[0]), "idr": "1000000"}}}

            def get_pair_min_order(self, pair):
                return self._pair_min_order.get(pair.lower(), {"min_coin": 0.0, "min_idr": 0.0})

            @staticmethod
            def parse_minimum_order_error(msg):
                import re
                m = re.search(r"Minimum order\s+(?:is\s+)?([\d.]+)", msg, re.IGNORECASE)
                return float(m.group(1)) if m else None

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        # min_order_idr=0 (no pre-check catches it), exchange rejects at API level
        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False, min_order_idr=0.0)
        trader = Trader(config)
        trader.client = _Client()

        snap = self._make_snap("cjl_idr", "buy", 270.0, 111.11)
        # Should NOT raise — the chase loop catches the RuntimeError gracefully
        outcome = trader.maybe_execute(snap)

        # Initial order + 1 failed chase attempt
        self.assertEqual(len(created), 2)
        # Cache should be updated with the parsed minimum
        cached = trader.client._pair_min_order.get("cjl_idr", {})
        self.assertGreater(cached.get("min_coin", 0.0), 0.0)

    def test_sell_chase_catches_runtime_minimum_order_error(self):
        """Sell chase should catch RuntimeError with 'Minimum order' and break gracefully."""
        created = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["270.0", "1"]], "sell": [["270.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"pair": pair, "action": action, "price": price, "amount": amount})
                if len(created) == 1:
                    return {"success": 1, "return": {"order_id": "1", "spend_cjl": 140.0, "receive_idr": 37800}}
                raise RuntimeError("API error: {'success': 0, 'error': 'Minimum order is 37.03703703 CJL.', 'error_code': ''}")

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"cjl": "10", "idr": "1000000"}}}

            def get_pair_min_order(self, pair):
                return self._pair_min_order.get(pair.lower(), {"min_coin": 0.0, "min_idr": 0.0})

            @staticmethod
            def parse_minimum_order_error(msg):
                import re
                m = re.search(r"Minimum order\s+(?:is\s+)?([\d.]+)", msg, re.IGNORECASE)
                return float(m.group(1)) if m else None

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=3, order_timeout_to_market=False, min_order_idr=0.0)
        trader = Trader(config)
        trader.client = _Client()
        trader.tracker.record_trade("buy", 270.0, 150.0)

        snap = self._make_snap("cjl_idr", "sell", 270.0, 150.0)
        # Should NOT raise
        outcome = trader.maybe_execute(snap)

        self.assertEqual(len(created), 2)
        cached = trader.client._pair_min_order.get("cjl_idr", {})
        self.assertGreater(cached.get("min_coin", 0.0), 0.0)


class ResumePendingBuyTests(unittest.TestCase):
    """Tests for the pending buy resume mechanism."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _make_config(self, **overrides):
        defaults = dict(
            api_key="test_key",
            api_secret="test_secret",
            dry_run=False,
            initial_capital=1_000_000.0,
            multi_position_enabled=False,
            min_order_idr=30_000.0,
            max_slippage_pct=0.05,
            entry_aggressiveness_pct=0.001,
            entry_retry_aggressiveness_pct=0.002,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            chase_max_retries=3,
            order_timeout_to_market=True,
        )
        defaults.update(overrides)
        return BotConfig(**defaults)

    def _make_snap(self, pair, price, **extra):
        from bot.strategies import StrategyDecision
        snap = {
            "pair": pair,
            "price": price,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="hold",
                confidence=0.9,
                reason="test",
                target_price=price,
                amount=0.0,
                stop_loss=None,
                take_profit=None,
            ),
        }
        snap.update(extra)
        return snap

    def test_chase_exhaustion_marks_pending_buy(self):
        """When all chase retries fail, tracker.has_pending_buy must be True."""

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                coin = pair.split("_")[0]
                return {"success": 1, "return": {f"receive_{coin}": 0, "order_id": "123"}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=2, order_timeout_to_market=False)
        trader = Trader(config)
        trader.client = _Client()

        from bot.strategies import StrategyDecision
        snap = {
            "pair": "ponke_idr",
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="buy",
                confidence=0.9,
                reason="test",
                target_price=100.0,
                amount=500.0,
                stop_loss=None,
                take_profit=None,
            ),
        }
        outcome = trader.maybe_execute(snap)

        self.assertAlmostEqual(trader.tracker.base_position, 0.0)
        self.assertTrue(trader.tracker.has_pending_buy,
                        "Tracker must be marked as pending-buy after chase exhaustion")
        self.assertEqual(len(trader.tracker.pending_orders), 1)

    def test_resume_pending_buy_fills(self):
        """resume_pending_buy should fill a pending order and clear the flag."""

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                coin = pair.split("_")[0]
                return {"success": 1, "return": {f"receive_{coin}": str(amount), "order_id": "456"}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "500", "idr": "500000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config()
        trader = Trader(config)
        trader.client = _Client()

        # Simulate pending buy state
        trader.tracker.mark_pending_buy(500.0, 100.0)
        self.assertTrue(trader.tracker.has_pending_buy)

        snap = self._make_snap("ponke_idr", 101.0)
        result = trader.resume_pending_buy(snap)

        self.assertEqual(result["status"], "resumed")
        self.assertEqual(result["action"], "buy")
        self.assertAlmostEqual(result["amount"], 500.0)
        self.assertAlmostEqual(trader.tracker.base_position, 500.0)
        self.assertFalse(trader.tracker.has_pending_buy,
                         "Pending flag should be cleared after successful fill")

    def test_resume_pending_buy_still_unfilled(self):
        """resume_pending_buy should update price and keep pending if still unfilled."""

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                coin = pair.split("_")[0]
                return {"success": 1, "return": {f"receive_{coin}": "0", "order_id": "789"}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "0", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config()
        trader = Trader(config)
        trader.client = _Client()

        trader.tracker.mark_pending_buy(500.0, 100.0)
        snap = self._make_snap("ponke_idr", 101.0)
        result = trader.resume_pending_buy(snap)

        self.assertEqual(result["status"], "pending")
        self.assertTrue(trader.tracker.has_pending_buy,
                        "Pending flag should remain when still unfilled")
        self.assertAlmostEqual(trader.tracker.base_position, 0.0)

    def test_resume_pending_buy_no_pending(self):
        """resume_pending_buy should return no_pending when nothing pending."""
        config = self._make_config()
        trader = Trader(config)
        snap = self._make_snap("ponke_idr", 101.0)
        result = trader.resume_pending_buy(snap)
        self.assertEqual(result["status"], "no_pending")

    def test_resume_pending_buy_dry_run(self):
        """In dry_run mode, resume_pending_buy should simulate the fill."""
        config = self._make_config(dry_run=True)
        trader = Trader(config)

        trader.tracker.mark_pending_buy(200.0, 50.0)
        snap = self._make_snap("ponke_idr", 55.0)
        result = trader.resume_pending_buy(snap)

        self.assertEqual(result["status"], "resumed")
        self.assertAlmostEqual(trader.tracker.base_position, 200.0)
        self.assertFalse(trader.tracker.has_pending_buy)

    def test_resume_pending_buy_below_min_order_cancels(self):
        """When amount × price < min_order_idr, cancel the pending buy."""

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["10.0", "1"]], "sell": [["11.0", "1"]]}

            def get_pair_min_order(self, pair):
                return {"min_idr": 50000.0, "min_coin": 0.0}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(min_order_idr=50000.0)
        trader = Trader(config)
        trader.client = _Client()

        # amount=2, price≈11 → notional≈22 < 50000 → should cancel
        trader.tracker.mark_pending_buy(2.0, 10.0)
        snap = self._make_snap("ponke_idr", 11.0)
        result = trader.resume_pending_buy(snap)

        self.assertEqual(result["status"], "cancelled_below_min")
        self.assertFalse(trader.tracker.has_pending_buy,
                         "Pending buy should be cancelled when below minimum")

    def test_multi_position_pending_buy_appears_in_active_positions(self):
        """In multi-position mode, a pending buy should appear in active_positions."""
        config = self._make_config(
            multi_position_enabled=True,
            multi_position_max=3,
        )
        trader = Trader(config)

        # Allocate and mark pending
        tracker = trader.multi_manager.allocate_capital("perp_idr", min_order_idr=30000)
        tracker.mark_pending_buy(100.0, 732.0)

        active = trader.active_positions
        self.assertIn("perp_idr", active,
                       "Pair with pending buy should appear in active_positions")

    def test_multi_position_both_filled_both_in_active(self):
        """Both filled pairs must appear in active_positions."""
        config = self._make_config(
            multi_position_enabled=True,
            multi_position_max=3,
        )
        trader = Trader(config)

        t1 = trader.multi_manager.allocate_capital("perp_idr", min_order_idr=30000)
        t1.record_trade("buy", 732.0, 100.0)

        t2 = trader.multi_manager.allocate_capital("ponke_idr", min_order_idr=30000)
        t2.record_trade("buy", 50.0, 1000.0)

        active = trader.active_positions
        self.assertIn("perp_idr", active, "First filled pair must be in active_positions")
        self.assertIn("ponke_idr", active, "Second filled pair must be in active_positions")
        self.assertEqual(len(active), 2)

    def test_sell_unfilled_after_chase_does_not_record_phantom_sell(self):
        """When a sell order is not filled after chase, base_position must remain > 0."""
        created = []

        class _Client:
            _pair_min_order = {}

            def get_depth(self, pair, count=5):
                return {"buy": [["100.0", "1"]], "sell": [["101.0", "1"]]}

            def create_order(self, pair, action, price, amount):
                created.append({"action": action, "price": price, "amount": amount})
                coin = pair.split("_")[0]
                return {"success": 1, "return": {f"spend_{coin}": 0, "order_id": str(len(created))}}

            def cancel_order(self, pair, order_id, order_type=None):
                return {"success": 1}

            def get_account_info(self):
                return {"return": {"balance": {"ponke": "500", "idr": "1000000"}}}

            def invalidate_account_info_cache(self):
                pass

            def open_orders(self, pair):
                return {"return": {"orders": {}}}

        config = self._make_config(chase_max_retries=2, order_timeout_to_market=False)
        trader = Trader(config)
        trader.client = _Client()

        # Pre-buy position so we can sell
        trader.tracker.record_trade("buy", 100.0, 500.0)
        self.assertAlmostEqual(trader.tracker.base_position, 500.0)

        from bot.strategies import StrategyDecision
        snap = {
            "pair": "ponke_idr",
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "indicators": None,
            "insufficient_data": False,
            "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="sell",
                confidence=0.9,
                reason="test",
                target_price=100.0,
                amount=500.0,
                stop_loss=None,
                take_profit=None,
            ),
        }
        outcome = trader.maybe_execute(snap)

        # Position must remain intact — no phantom sell
        self.assertGreater(trader.tracker.base_position, 0.0,
                           "Position must remain when sell was not filled on exchange")
        # The step should be marked as pending
        pending_steps = [s for s in outcome.get("executed_steps", []) if s.get("pending")]
        self.assertTrue(len(pending_steps) > 0,
                        "Unfilled sell step should be marked as pending")


class AdaptiveLimitOrderTests(unittest.TestCase):
    """Tests for the adaptive limit order feature."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @staticmethod
    def _dummy_client(bids=None, asks=None):
        bids = bids or [["2699999", "1"]]
        asks = asks or [["2700001", "1"]]
        return type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": bids, "sell": asks},
        })()

    def _make_snap(self, pair="test_idr", price=2_700_000.0, action="buy", amount=0.5):
        return {
            "pair": pair,
            "price": price,
            "trend": None, "orderbook": None, "volatility": None, "levels": None,
            "indicators": None, "insufficient_data": False, "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading", action=action, confidence=0.9,
                reason="test", target_price=price, amount=amount,
                stop_loss=price * 0.95, take_profit=price * 1.05,
            ),
        }

    def test_adaptive_buy_uses_bid_side(self):
        """With adaptive_order_enabled, buy reference_price is based on best_bid."""
        config = BotConfig(
            api_key=None, dry_run=True,
            adaptive_order_enabled=True,
            entry_aggressiveness_pct=0.001,
            max_slippage_pct=0.05,
            min_order_idr=1.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            smart_entry_buffer_enabled=False,
        )
        trader = Trader(config)
        trader.client = self._dummy_client(
            bids=[["100000", "10"]],
            asks=[["100100", "10"]],
        )
        snap = self._make_snap(price=100_050.0, amount=0.5)
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "simulated")
        # Price should be based on best_bid (100000) * (1 + 0.001) = 100100,
        # NOT on best_ask (100100) * (1 + 0.001) = 100200.1
        self.assertAlmostEqual(outcome["price"], 100000 * 1.001, places=0)

    def test_adaptive_disabled_uses_ask_side(self):
        """With adaptive_order_enabled=False, buy uses best_ask."""
        config = BotConfig(
            api_key=None, dry_run=True,
            adaptive_order_enabled=False,
            entry_aggressiveness_pct=0.001,
            max_slippage_pct=0.05,
            min_order_idr=1.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            smart_entry_buffer_enabled=False,
        )
        trader = Trader(config)
        trader.client = self._dummy_client(
            bids=[["100000", "10"]],
            asks=[["100100", "10"]],
        )
        snap = self._make_snap(price=100_050.0, amount=0.5)
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "simulated")
        # Price should be based on best_ask (100100) * (1 + 0.001)
        self.assertAlmostEqual(outcome["price"], 100100 * 1.001, places=0)

    def test_adaptive_sell_uses_ask_side(self):
        """With adaptive_order_enabled, sell reference_price is based on best_ask."""
        config = BotConfig(
            api_key=None, dry_run=True,
            adaptive_order_enabled=True,
            entry_aggressiveness_pct=0.0,
            max_slippage_pct=0.05,
            min_order_idr=1.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            smart_entry_buffer_enabled=False,
            multi_position_enabled=False,
            target_profit_pct=99.0,
            max_loss_pct=99.0,
        )
        trader = Trader(config)
        trader.client = self._dummy_client(
            bids=[["100000", "10"]],
            asks=[["100100", "10"]],
        )
        # Set up a sell position on the main tracker
        trader.tracker.cash = 0
        trader.tracker.base_position = 1.0
        trader.tracker.avg_cost = 100000.0
        snap = self._make_snap(price=100_050.0, action="sell", amount=0.5)
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "simulated")
        # Sell price based on best_ask (100100) * (1 - 0) = 100100
        self.assertAlmostEqual(outcome["price"], 100100.0, places=0)


class LiquidityCheckTests(unittest.TestCase):
    """Tests for the orderbook liquidity check before entry."""

    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @staticmethod
    def _dummy_client(bids=None, asks=None):
        bids = bids or [["100000", "1"]]
        asks = asks or [["100100", "1"]]
        return type("_C", (), {
            "get_depth": lambda self, *a, **kw: {"buy": bids, "sell": asks},
        })()

    def _make_snap(self, pair="test_idr", price=100_050.0, action="buy", amount=0.5):
        return {
            "pair": pair,
            "price": price,
            "trend": None, "orderbook": None, "volatility": None, "levels": None,
            "indicators": None, "insufficient_data": False, "grid_plan": None,
            "decision": StrategyDecision(
                mode="day_trading", action=action, confidence=0.9,
                reason="test", target_price=price, amount=amount,
                stop_loss=price * 0.95, take_profit=price * 1.05,
            ),
        }

    def test_liquidity_check_skips_thin_market(self):
        """Trades should be skipped when orderbook volume < min_orderbook_volume_idr."""
        config = BotConfig(
            api_key=None, dry_run=True,
            min_orderbook_volume_idr=500_000.0,
            min_order_idr=1.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            smart_entry_buffer_enabled=False,
        )
        trader = Trader(config)
        # bid: 100000 * 1 = 100000 IDR; ask: 100100 * 1 = 100100 IDR
        # total = 200100 < 500000 → should skip
        trader.client = self._dummy_client(
            bids=[["100000", "1"]],
            asks=[["100100", "1"]],
        )
        snap = self._make_snap()
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("liquidity_too_thin", outcome["reason"])

    def test_liquidity_check_passes_with_sufficient_volume(self):
        """Trades should proceed when orderbook volume >= min_orderbook_volume_idr."""
        config = BotConfig(
            api_key=None, dry_run=True,
            min_orderbook_volume_idr=100_000.0,
            min_order_idr=1.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            smart_entry_buffer_enabled=False,
        )
        trader = Trader(config)
        # bid: 100000 * 1 = 100000 IDR; ask: 100100 * 1 = 100100 IDR
        # total = 200100 >= 100000 → should proceed
        trader.client = self._dummy_client(
            bids=[["100000", "1"]],
            asks=[["100100", "1"]],
        )
        snap = self._make_snap()
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "simulated")

    def test_liquidity_check_disabled_by_default(self):
        """When min_orderbook_volume_idr=0 (default), the check is disabled."""
        config = BotConfig(
            api_key=None, dry_run=True,
            min_orderbook_volume_idr=0.0,
            min_order_idr=1.0,
            pair_cooldown_seconds=0.0,
            min_confidence=0.0,
            buy_max_rsi=0.0,
            buy_max_resistance_proximity_pct=0.0,
            smart_entry_buffer_enabled=False,
        )
        trader = Trader(config)
        # Even with 1 IDR of volume, should proceed when check is disabled
        trader.client = self._dummy_client(
            bids=[["1", "1"]],
            asks=[["2", "1"]],
        )
        snap = self._make_snap(price=1.5, amount=1.0)
        outcome = trader.maybe_execute(snap)
        # Should not be skipped for liquidity reasons
        self.assertNotEqual(outcome.get("reason", ""), "liquidity_too_thin")
