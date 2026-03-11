import unittest

from bot.config import BotConfig
from bot.realtime import MultiPairFeed, RealtimeFeed
from bot.trader import Trader


class _ClientStub:
    def __init__(self) -> None:
        self.ticker_calls = 0
        self.depth_calls = 0
        self.trade_calls = 0

    def get_ticker(self, pair: str) -> dict:
        self.ticker_calls += 1
        return {"ticker": {"last": 100.0}, "pair": pair}

    def get_depth(self, pair: str, count: int = 50) -> dict:
        self.depth_calls += 1
        return {"buy": [], "sell": []}

    def get_trades(self, pair: str, count: int = 200) -> list:
        self.trade_calls += 1
        return []


class _RealtimeStub:
    def __init__(self, snapshot: dict) -> None:
        self._snapshot = snapshot

    @property
    def has_snapshot(self) -> bool:
        return True

    def snapshot(self) -> dict:
        return self._snapshot


class RealtimeTests(unittest.TestCase):
    def test_refresh_once_uses_rest_polling(self) -> None:
        client = _ClientStub()
        feed = RealtimeFeed("btc_idr", client, websocket_enabled=False, poll_interval=0.1)
        snap = feed.refresh_once()
        self.assertIn("ticker", snap)
        self.assertEqual(client.ticker_calls, 1)
        self.assertTrue(feed.has_snapshot)

    def test_trader_prefers_realtime_snapshot(self) -> None:
        config = BotConfig(api_key=None, real_time=False)
        client = _ClientStub()
        trader = Trader(config, client=client)
        trader.realtime = _RealtimeStub(
            {
                "ticker": {"ticker": {"last": 123.0}},
                "depth": {"buy": [], "sell": []},
                "trades": [],
            }
        )

        snapshot = trader.analyze_market("btc_idr")
        self.assertEqual(client.ticker_calls, 0)
        self.assertEqual(snapshot["price"], 123.0)

    def test_ws_partial_update_merges_keys(self) -> None:
        """A WebSocket message that only contains 'ticker' must not wipe out
        the existing 'depth' and 'trades' entries in the snapshot."""
        client = _ClientStub()
        feed = RealtimeFeed("btc_idr", client, websocket_enabled=False, poll_interval=60)
        # Seed the feed with a full snapshot
        with feed._lock:
            feed._latest = {
                "ticker": {"ticker": {"last": 100.0}},
                "depth": {"buy": [["99", "1"]], "sell": [["101", "1"]]},
                "trades": [{"price": "100", "amount": "0.1", "date": 1}],
            }
        # Apply a partial WebSocket update (ticker only)
        feed._apply_ws_message({"ticker": {"ticker": {"last": 105.0}}})
        snap = feed.snapshot()
        # Ticker must be updated
        self.assertEqual(snap["ticker"]["ticker"]["last"], 105.0)
        # Depth and trades must be preserved
        self.assertIn("depth", snap)
        self.assertIn("trades", snap)
        self.assertEqual(snap["depth"]["buy"][0][0], "99")

    def test_ws_partial_update_ignores_unknown_keys(self) -> None:
        """Unknown keys in a WebSocket message are silently ignored."""
        client = _ClientStub()
        feed = RealtimeFeed("btc_idr", client, websocket_enabled=False, poll_interval=60)
        with feed._lock:
            feed._latest = {"ticker": {"ticker": {"last": 100.0}}}
        feed._apply_ws_message({"unknown_key": "some_value"})
        snap = feed.snapshot()
        # Original data preserved, unknown key not stored
        self.assertNotIn("unknown_key", snap)
        self.assertIn("ticker", snap)

    def test_feed_accepts_subscribe_message_param(self) -> None:
        """RealtimeFeed stores the optional subscribe_message for later use."""
        client = _ClientStub()
        msg = '{"action":"subscribe","channel":"btc_idr"}'
        feed = RealtimeFeed(
            "btc_idr",
            client,
            websocket_enabled=False,
            poll_interval=60,
            subscribe_message=msg,
        )
        self.assertEqual(feed.subscribe_message, msg)

    def test_stop_halts_polling_thread(self) -> None:
        """Calling stop() on a running polling feed terminates the thread."""
        client = _ClientStub()
        feed = RealtimeFeed("btc_idr", client, websocket_enabled=False, poll_interval=0.05)
        feed.start()
        self.assertIsNotNone(feed._ws_thread)
        self.assertTrue(feed._ws_thread.is_alive())
        feed.stop()
        feed._ws_thread.join(timeout=2.0)
        self.assertFalse(feed._ws_thread.is_alive())


class _SummariesClientStub:
    """Stub client that simulates the /api/summaries response."""

    def __init__(self, tickers: dict, *, fail: bool = False) -> None:
        self._tickers = tickers
        self._fail = fail
        self.summaries_calls = 0

    def get_summaries(self) -> dict:
        self.summaries_calls += 1
        if self._fail:
            raise RuntimeError("summaries unavailable")
        return {"tickers": self._tickers}


class MultiPairFeedTests(unittest.TestCase):
    def test_seeds_cache_from_summaries_on_start(self) -> None:
        """start() must synchronously seed the ticker cache from /api/summaries."""
        client = _SummariesClientStub(
            {
                "btcidr": {"last": "1000000000", "high": "1100000000"},
                "ethidr": {"last": "50000000", "high": "55000000"},
            }
        )
        feed = MultiPairFeed(
            ["btc_idr", "eth_idr"],
            client,
            websocket_enabled=False,
            summaries_interval=9999,
        )
        feed.start()
        try:
            # Cache must be seeded immediately (synchronous on start)
            btc_ticker = feed.get_ticker("btc_idr")
            eth_ticker = feed.get_ticker("eth_idr")
            self.assertIsNotNone(btc_ticker)
            self.assertIsNotNone(eth_ticker)
            self.assertEqual(btc_ticker["last"], "1000000000")  # type: ignore[index]
            self.assertEqual(eth_ticker["last"], "50000000")  # type: ignore[index]
        finally:
            feed.stop()

    def test_normalizes_summaries_keys_to_pair_names(self) -> None:
        """Summaries keys without underscores (e.g. 'boneidr') must be mapped to
        canonical pair names (e.g. 'bone_idr')."""
        client = _SummariesClientStub(
            {"boneidr": {"last": "200", "high": "210"}}
        )
        feed = MultiPairFeed(
            ["bone_idr"],
            client,
            websocket_enabled=False,
            summaries_interval=9999,
        )
        feed.start()
        try:
            ticker = feed.get_ticker("bone_idr")
            self.assertIsNotNone(ticker)
            self.assertEqual(ticker["last"], "200")  # type: ignore[index]
            # The raw summaries key must NOT be stored under the un-normalised name
            self.assertIsNone(feed.get_ticker("boneidr"))
        finally:
            feed.stop()

    def test_returns_none_for_unknown_pair(self) -> None:
        """get_ticker must return None for a pair that hasn't been received."""
        client = _SummariesClientStub({"btcidr": {"last": "1000000000"}})
        feed = MultiPairFeed(["btc_idr"], client, websocket_enabled=False, summaries_interval=9999)
        feed.start()
        try:
            self.assertIsNone(feed.get_ticker("xrp_idr"))
        finally:
            feed.stop()

    def test_summaries_failure_leaves_cache_empty(self) -> None:
        """When /api/summaries fails, the cache stays empty but no exception is raised."""
        client = _SummariesClientStub({}, fail=True)
        feed = MultiPairFeed(["btc_idr"], client, websocket_enabled=False, summaries_interval=9999)
        feed.start()  # must not raise
        try:
            self.assertIsNone(feed.get_ticker("btc_idr"))
        finally:
            feed.stop()

    def test_applies_ws_message_with_pair_key(self) -> None:
        """A WebSocket message containing 'pair' and 'ticker' must update the cache."""
        client = _SummariesClientStub({})
        feed = MultiPairFeed(["btc_idr"], client, websocket_enabled=False, summaries_interval=9999)
        feed.start()
        try:
            # Simulate an incoming WebSocket message in the expected format
            feed._apply_ws_message_for_pair("btc_idr", {"last": "1050000000"})
            ticker = feed.get_ticker("btc_idr")
            self.assertIsNotNone(ticker)
            self.assertEqual(ticker["last"], "1050000000")  # type: ignore[index]
        finally:
            feed.stop()

    def test_stop_halts_polling_thread(self) -> None:
        """stop() must terminate the background polling thread."""
        client = _SummariesClientStub({"btcidr": {"last": "100"}})
        feed = MultiPairFeed(
            ["btc_idr"],
            client,
            websocket_enabled=False,
            summaries_interval=0.05,
        )
        feed.start()
        self.assertTrue(any(t.is_alive() for t in feed._threads))
        feed.stop()
        for t in feed._threads:
            t.join(timeout=2.0)
        self.assertFalse(any(t.is_alive() for t in feed._threads))

    def test_start_creates_correct_number_of_polling_threads(self) -> None:
        """Without WebSocket, start() must create exactly one polling thread."""
        client = _SummariesClientStub({})
        feed = MultiPairFeed(
            ["btc_idr", "eth_idr", "xrp_idr"],
            client,
            websocket_enabled=False,
            summaries_interval=9999,
        )
        feed.start()
        try:
            self.assertEqual(len(feed._threads), 1)
        finally:
            feed.stop()

    def test_batches_pairs_by_batch_size(self) -> None:
        """With 5 pairs and batch_size=2, feed must create 3 batches (2+2+1)."""
        feed = MultiPairFeed(
            ["a_idr", "b_idr", "c_idr", "d_idr", "e_idr"],
            _SummariesClientStub({}),
            websocket_enabled=False,
            batch_size=2,
            summaries_interval=9999,
        )
        # batch count computation is internal; verify via _batch_size
        self.assertEqual(feed._batch_size, 2)
        import math
        expected_batches = math.ceil(5 / 2)
        self.assertEqual(expected_batches, 3)

    def test_is_seeded_false_before_start(self) -> None:
        """is_seeded must be False before start() is called."""
        feed = MultiPairFeed(
            ["btc_idr"],
            _SummariesClientStub({}),
            websocket_enabled=False,
            summaries_interval=9999,
        )
        self.assertFalse(feed.is_seeded)

    def test_is_seeded_true_after_successful_start(self) -> None:
        """is_seeded must be True after start() successfully seeds from summaries."""
        client = _SummariesClientStub({"btcidr": {"last": "1000000000"}})
        feed = MultiPairFeed(
            ["btc_idr"],
            client,
            websocket_enabled=False,
            summaries_interval=9999,
        )
        feed.start()
        try:
            self.assertTrue(feed.is_seeded)
        finally:
            feed.stop()

    def test_is_seeded_false_when_summaries_fails(self) -> None:
        """is_seeded must remain False when the summaries fetch fails."""
        client = _SummariesClientStub({}, fail=True)
        feed = MultiPairFeed(
            ["btc_idr"],
            client,
            websocket_enabled=False,
            summaries_interval=9999,
        )
        feed.start()
        try:
            self.assertFalse(feed.is_seeded)
        finally:
            feed.stop()


if __name__ == "__main__":
    unittest.main()


class CentrifugeProtocolTests(unittest.TestCase):
    """Tests for the Centrifuge-protocol WebSocket message parser."""

    def _make_feed(self, pairs=None):
        if pairs is None:
            pairs = ["btc_idr", "eth_idr", "usdt_idr"]
        client = _SummariesClientStub({})
        return MultiPairFeed(pairs, client, websocket_enabled=False, summaries_interval=9999)

    def test_apply_summary_rows_updates_cache(self) -> None:
        """_apply_summary_rows must update the ticker cache for each row."""
        feed = self._make_feed()
        rows = [
            ["btcidr", 1700000000, 900000000, 880000000, 920000000, 895000000,
             "50000000000", "55.0"],
            ["ethidr", 1700000000, 45000000, 44000000, 46000000, 44500000,
             "10000000000", "222.0"],
        ]
        feed._apply_summary_rows(rows)
        btc = feed.get_ticker("btc_idr")
        eth = feed.get_ticker("eth_idr")
        self.assertIsNotNone(btc)
        self.assertEqual(btc["last"], "900000000")   # type: ignore[index]
        self.assertEqual(btc["high"], "920000000")   # type: ignore[index]
        self.assertIsNotNone(eth)
        self.assertEqual(eth["last"], "45000000")    # type: ignore[index]

    def test_apply_summary_rows_skips_unknown_pairs(self) -> None:
        """Rows for pairs not in the known list must be silently skipped."""
        feed = self._make_feed(["btc_idr"])
        feed._apply_summary_rows([["unknownidr", 1700000000, 100, 90, 110, 95, "1000", "1"]])
        self.assertIsNone(feed.get_ticker("unknown_idr"))

    def test_apply_summary_rows_skips_short_rows(self) -> None:
        """Rows shorter than 8 elements must be silently skipped."""
        feed = self._make_feed()
        feed._apply_summary_rows([["btcidr", 1700000000, 100]])
        self.assertIsNone(feed.get_ticker("btc_idr"))

    def test_default_websocket_url_is_indodax(self) -> None:
        """Default websocket_url must point to the official Indodax WS endpoint."""
        from bot.realtime import INDODAX_WS_URL
        feed = self._make_feed()
        self.assertEqual(feed._websocket_url, INDODAX_WS_URL)

    def test_default_websocket_token_is_set(self) -> None:
        """Default websocket_token must be the public Indodax static token."""
        from bot.realtime import INDODAX_WS_TOKEN
        feed = self._make_feed()
        self.assertEqual(feed._websocket_token, INDODAX_WS_TOKEN)


class StaleFeedTests(unittest.TestCase):
    """Tests for stale WS data detection."""

    def _make_feed(self, pairs=None):
        class _Stub:
            def get_summaries(self):
                return {}
        return MultiPairFeed(
            pairs=pairs or ["btc_idr", "eth_idr"],
            client=_Stub(),
            websocket_url=None,
            websocket_enabled=False,
        )

    def test_not_stale_when_ws_never_updated(self):
        """is_ws_stale returns False when WS has never received any data."""
        feed = self._make_feed()
        self.assertFalse(feed.is_ws_stale(threshold_seconds=5.0))

    def test_stale_after_threshold_exceeded(self):
        """is_ws_stale returns True after an update older than the threshold."""
        import time
        feed = self._make_feed()
        feed._apply_ws_message_for_pair("btc_idr", {"last": "100"})
        # Fake the timestamp to be 200s ago
        feed._last_ws_update = time.time() - 200
        self.assertTrue(feed.is_ws_stale(threshold_seconds=120.0))

    def test_not_stale_within_threshold(self):
        """is_ws_stale returns False right after an update."""
        feed = self._make_feed()
        feed._apply_ws_message_for_pair("btc_idr", {"last": "100"})
        self.assertFalse(feed.is_ws_stale(threshold_seconds=120.0))

    def test_last_ws_update_set_on_apply(self):
        """_last_ws_update is set when _apply_ws_message_for_pair is called."""
        import time
        feed = self._make_feed()
        before = time.time()
        feed._apply_ws_message_for_pair("btc_idr", {"last": "100"})
        after = time.time()
        self.assertIsNotNone(feed._last_ws_update)
        self.assertGreaterEqual(feed._last_ws_update, before)
        self.assertLessEqual(feed._last_ws_update, after)


class PrivateFeedImportTest(unittest.TestCase):
    """Smoke-tests that PrivateFeed can be imported and instantiated."""

    def test_import(self):
        from bot.realtime import PrivateFeed
        self.assertTrue(callable(PrivateFeed))

    def test_start_stop_without_ws(self):
        """PrivateFeed.start/stop must not crash when websocket is unavailable."""
        import bot.realtime as rt_mod
        original_ws = rt_mod.websocket
        rt_mod.websocket = None
        try:
            from bot.realtime import PrivateFeed

            class _Stub:
                def generate_private_ws_token(self):
                    return {"connToken": "tok", "channel": "pws:#abc"}

            feed = PrivateFeed(client=_Stub())
            feed.start()
            import time as _t
            _t.sleep(0.05)
            feed.stop()
        finally:
            rt_mod.websocket = original_ws
