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

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

    def analyze_market(self, pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def fake_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def always_429(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def fake_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def recording_analyze(pair: str | None = None, prefetched_ticker: Dict[str, Any] | None = None, skip_depth: bool = False) -> Dict[str, Any]:
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

        def tracking_analyze(pair=None, prefetched_ticker=None, skip_depth=False):
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


def _make_buy_snap(price: float = 100.0, action: str = "buy") -> Dict[str, Any]:
    """Helper to create a minimal buy snapshot for maybe_execute tests."""
    return {
        "pair": "btc_idr",
        "price": price,
        "trend": None,
        "orderbook": None,
        "volatility": None,
        "levels": None,
        "indicators": None,
        "insufficient_data": False,
        "grid_plan": None,
        "decision": StrategyDecision(
            mode="scalping",
            action=action,
            confidence=0.9,
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
        """An order whose total IDR value is below min_order_idr must be skipped."""
        config = BotConfig(api_key=None, min_order_idr=10_000, dry_run=True,
                           initial_capital=100_000)
        trader = Trader(config)
        trader.client = type("_C", (), {
            # bid=ask=253 so slippage guard passes
            "get_depth": lambda self, *a, **kw: {"buy": [["253", "100"]], "sell": [["253", "100"]]},
        })()
        # price=253, amount=1 → total=253 IDR < 10,000 IDR
        snap = self._snap(price=253.0)
        outcome = trader.maybe_execute(snap)
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("order_below_minimum", outcome["reason"])

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
        """BotConfig default min_order_idr must be 15,000 (raised from 10k to avoid tiny fees)."""
        config = BotConfig(api_key=None)
        self.assertEqual(config.min_order_idr, 15_000.0)

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
        """When total passes pre-check but every staged split is below min, must skip."""
        from bot.analysis import VolatilityStats
        # min_order_idr = 30000, price = 100
        # decision.amount = 400 → effective_amount capped at min(400, cash/100)
        # default cash=1_000_000 → max_affordable=10000 → effective_amount=400
        # total IDR = 400 × 100 = 40000 > 30000 (passes pre-check)
        # staged fractions with vol=0.015, conf=0.5: [0.6, 0.4]
        # step1 = 240 × 100 = 24000 < 30000, step2 = 160 × 100 = 16000 < 30000
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
        self.assertEqual(outcome["status"], "skipped")
        self.assertIn("all_steps_below_min_order", outcome["reason"])
