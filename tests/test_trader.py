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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None) -> Dict[str, Any]:
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
        feed._apply_ws_message_for_pair("a_idr", {"last": "100", "vol_idr": "100"})
        feed._apply_ws_message_for_pair("b_idr", {"last": "100", "vol_idr": "50"})
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

        def tracking_analyze(pair=None, prefetched_ticker=None):
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
        config = BotConfig(api_key=None, max_exposure_per_coin_pct=0.05, dry_run=True)
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
        config = BotConfig(api_key=None, partial_tp_fraction=0.5, dry_run=True)
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
        config = BotConfig(api_key=None, dry_run=True)
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
        config = BotConfig(api_key=None, re_entry_cooldown_seconds=3600, dry_run=True)
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
        config = BotConfig(api_key=None, trailing_tp_pct=0.02)
        trader = Trader(config)
        trader.tracker.record_trade("buy", 100.0, 1.0)
        # Price at 110 — trailing TP not yet triggered (just activated)
        result = trader.evaluate_dynamic_tp(self._snap(price=110.0))
        self.assertIsNone(result)
        # Trailing floor should be set at 110 * 0.98 = 107.8
        self.assertAlmostEqual(trader.tracker.trailing_tp_stop, 107.8)

    def test_trailing_tp_triggered_returns_correct_reason(self):
        config = BotConfig(api_key=None, trailing_tp_pct=0.02)
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
        config = BotConfig(api_key=None, risk_per_trade=0.01, initial_capital=100_000.0)
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
        # This replicates the code added to main.py's position monitoring loop
        # (the bug fix: advance floor before evaluating stop_reason each tick).
        if config.trailing_tp_pct > 0 and tracker.tp_activated:
            tracker.activate_trailing_tp(130.0, config.trailing_tp_pct)

        floor_after_130 = tracker.trailing_tp_stop
        self.assertAlmostEqual(floor_after_130, 128.7, places=2)
        self.assertGreater(floor_after_130, floor_after_125)

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
