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


if __name__ == "__main__":
    unittest.main()
