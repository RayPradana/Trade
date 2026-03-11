import logging
import unittest
from typing import Dict, Any

import requests

from bot.config import BotConfig
from bot.strategies import StrategyDecision
from bot.trader import Trader
from bot.tracking import PortfolioTracker


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

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
        key = pair or self.config.pair
        return self._snapshots[key]


class GuardedTrader(Trader):
    """Trader with stubbed client to test balance guards."""

    class _Client:
        def get_depth(self, pair: str, count: int = 5) -> Dict[str, Any]:
            return {"buy": [["100", "1"]], "sell": [["100.05", "1"]]}

        def get_summaries(self) -> dict:
            return {}

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

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
                    confidence=0.6,
                    reason="ok",
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
                    confidence=0.9,
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

    def test_force_sell_liquidates_entire_position(self) -> None:
        config = BotConfig(
            api_key=None,
            dry_run=True,
            initial_capital=1000.0,
            max_loss_pct=0.9,
            target_profit_pct=2.0,
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

        def fake_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
        """After MAX_SCAN_RETRIES 429s, _analyze_with_retry must re-raise the last error."""
        config = BotConfig(api_key=None, scan_request_delay=0.0)
        trader = GuardedTrader(config)
        sleep_calls: list[float] = []

        def always_429(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
            resp = requests.Response()
            resp.status_code = 429
            raise requests.HTTPError(response=resp)

        trader.analyze_market = always_429  # type: ignore[method-assign]
        import unittest.mock as mock
        with mock.patch("bot.trader.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            with self.assertRaises(requests.HTTPError):
                trader._analyze_with_retry("btc_idr")
        # Should have slept exactly MAX_SCAN_RETRIES times with increasing delays
        self.assertEqual(len(sleep_calls), trader._MAX_SCAN_RETRIES)
        for i, delay in enumerate(sleep_calls):
            expected = min(trader._SCAN_BACKOFF_BASE * (2 ** i), trader._SCAN_BACKOFF_MAX)
            self.assertAlmostEqual(delay, expected)

    def test_scan_request_delay_is_honoured(self) -> None:
        """scan_and_choose must call time.sleep(scan_request_delay) before each pair."""
        config = BotConfig(api_key=None, scan_request_delay=0.5)
        snapshots = {
            "a_idr": {
                "pair": "a_idr", "price": 1.0, "trend": None, "orderbook": None,
                "volatility": None, "levels": None,
                "decision": StrategyDecision(
                    mode="scalping", action="buy", confidence=0.7, reason="ok",
                    target_price=1, amount=10, stop_loss=0.99, take_profit=1.01,
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
        # sleep should be called once per pair (2 pairs)
        self.assertEqual(mock_sleep.call_count, 2)
        for call_args in mock_sleep.call_args_list:
            self.assertEqual(call_args, mock.call(0.5))

    def test_scan_and_choose_uses_summaries_to_prefetch_tickers(self) -> None:
        """scan_and_choose should call get_summaries() once and pass ticker data to analyze_market."""
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
            trader.scan_and_choose()

        # Both pairs must have been analyzed with a non-None prefetched ticker
        self.assertEqual(len(received_prefetched), 2)
        pairs_seen = {r[0] for r in received_prefetched}
        self.assertIn("btc_idr", pairs_seen)
        self.assertIn("eth_idr", pairs_seen)
        for pair_name, ticker in received_prefetched:
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

        def fake_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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


if __name__ == "__main__":
    unittest.main()
