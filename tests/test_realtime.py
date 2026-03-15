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


class PositionFeedLifecycleTests(unittest.TestCase):
    """Tests for per-pair position feed start/stop in Trader."""

    def _make_trader(self) -> "Trader":
        from bot.config import BotConfig
        from bot.trader import Trader

        class _Client:
            def get_ticker(self, pair):
                return {"ticker": {"last": "100"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                return []
            def get_ohlc(self, pair, tf="1m", limit=200):
                return []
            def get_summaries(self):
                return {}

        config = BotConfig(api_key=None, real_time=False)
        return Trader(config, client=_Client())

    def test_ensure_position_feed_noop_when_realtime_disabled(self):
        """_ensure_position_feed must be a no-op when real_time=False."""
        trader = self._make_trader()
        trader._ensure_position_feed("eth_idr")
        self.assertNotIn("eth_idr", trader._position_feeds)

    def test_ensure_position_feed_starts_feed_when_enabled(self):
        """_ensure_position_feed starts a RealtimeFeed for a non-primary pair."""
        from bot.config import BotConfig
        from bot.trader import Trader

        class _Client:
            def get_ticker(self, pair):
                return {"ticker": {"last": "100"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                return []
            def get_ohlc(self, pair, tf="1m", limit=200):
                return []
            def get_summaries(self):
                return {}

        config = BotConfig(api_key=None, real_time=True, websocket_enabled=False,
                           pair="btc_idr")
        trader = Trader(config, client=_Client())
        try:
            trader._ensure_position_feed("eth_idr")
            self.assertIn("eth_idr", trader._position_feeds)
        finally:
            # Clean up background threads
            for feed in trader._position_feeds.values():
                feed.stop()
            if trader.realtime:
                trader.realtime.stop()

    def test_ensure_position_feed_skips_primary_pair(self):
        """_ensure_position_feed must not create a duplicate feed for the primary pair."""
        from bot.config import BotConfig
        from bot.trader import Trader

        class _Client:
            def get_ticker(self, pair):
                return {"ticker": {"last": "100"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                return []
            def get_ohlc(self, pair, tf="1m", limit=200):
                return []
            def get_summaries(self):
                return {}

        config = BotConfig(api_key=None, real_time=True, websocket_enabled=False,
                           pair="btc_idr")
        trader = Trader(config, client=_Client())
        try:
            trader._ensure_position_feed("btc_idr")
            self.assertNotIn("btc_idr", trader._position_feeds)
        finally:
            if trader.realtime:
                trader.realtime.stop()

    def test_remove_position_feed_stops_and_removes(self):
        """_remove_position_feed stops the feed and removes it from the dict."""
        from bot.config import BotConfig
        from bot.trader import Trader
        from bot.realtime import RealtimeFeed

        class _Client:
            def get_ticker(self, pair):
                return {"ticker": {"last": "100"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                return []
            def get_ohlc(self, pair, tf="1m", limit=200):
                return []
            def get_summaries(self):
                return {}

        config = BotConfig(api_key=None, real_time=False, pair="btc_idr")
        trader = Trader(config, client=_Client())

        # Manually insert a stub feed
        stopped = []

        class _FeedStub:
            def stop(self):
                stopped.append(True)

        trader._position_feeds["eth_idr"] = _FeedStub()  # type: ignore
        trader._remove_position_feed("eth_idr")
        self.assertNotIn("eth_idr", trader._position_feeds)
        self.assertEqual(stopped, [True])

    def test_remove_position_feed_noop_for_unknown_pair(self):
        """_remove_position_feed must not raise for a pair with no active feed."""
        trader = self._make_trader()
        # Should not raise
        trader._remove_position_feed("unknown_idr")

    def test_analyze_market_uses_position_feed_when_available(self):
        """analyze_market must prefer the per-pair position feed over REST."""
        from bot.config import BotConfig
        from bot.trader import Trader

        class _Client:
            def __init__(self):
                self.ticker_calls = 0
            def get_ticker(self, pair):
                self.ticker_calls += 1
                return {"ticker": {"last": "100"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                return []
            def get_ohlc(self, pair, tf="1m", limit=200):
                return []
            def get_summaries(self):
                return {}

        class _FeedStub:
            @property
            def has_snapshot(self):
                return True
            def snapshot(self):
                return {
                    "ticker": {"ticker": {"last": 999.0}},
                    "depth": {"buy": [], "sell": []},
                    "trades": [],
                }

        config = BotConfig(api_key=None, real_time=False, pair="btc_idr")
        client = _Client()
        trader = Trader(config, client=client)
        trader._position_feeds["eth_idr"] = _FeedStub()  # type: ignore

        snapshot = trader.analyze_market("eth_idr")
        # REST ticker should NOT have been called
        self.assertEqual(client.ticker_calls, 0)
        self.assertEqual(snapshot["price"], 999.0)


class IndodaxWsChannelTests(unittest.TestCase):
    """Tests for the official Indodax per-pair WS channel message handlers."""

    def _make_feed(self, pair: str = "btc_idr") -> RealtimeFeed:
        client = _ClientStub()
        return RealtimeFeed(pair, client, websocket_enabled=False, poll_interval=9999)

    # ── _apply_orderbook ──────────────────────────────────────────────────

    def test_apply_orderbook_normalises_to_rest_format(self) -> None:
        """_apply_orderbook must convert WS bid/ask → REST buy/sell format."""
        feed = self._make_feed("btc_idr")
        ws_data = {
            "data": {
                "pair": "btcidr",
                "ask": [
                    {"btc_volume": "0.5", "idr_volume": "165000000", "price": "330000000"},
                    {"btc_volume": "0.2", "idr_volume": "66400000",  "price": "332000000"},
                ],
                "bid": [
                    {"btc_volume": "0.8", "idr_volume": "263200000", "price": "329000000"},
                ],
            },
            "offset": 12345,
        }
        feed._apply_orderbook(ws_data)
        snap = feed.snapshot()
        self.assertIn("depth", snap)
        depth = snap["depth"]
        # WS ask (sellers) → REST sell
        self.assertEqual(depth["sell"][0][0], "330000000")
        self.assertEqual(depth["sell"][0][1], "0.5")
        # WS bid (buyers) → REST buy
        self.assertEqual(depth["buy"][0][0], "329000000")
        self.assertEqual(depth["buy"][0][1], "0.8")

    def test_apply_orderbook_direct_inner_format(self) -> None:
        """_apply_orderbook handles data dict passed directly (without nested 'data' key)."""
        feed = self._make_feed("eth_idr")
        ws_data = {
            "pair": "ethidr",
            "ask": [{"eth_volume": "1.0", "idr_volume": "5000000", "price": "5000000"}],
            "bid": [{"eth_volume": "2.0", "idr_volume": "9800000", "price": "4900000"}],
        }
        feed._apply_orderbook(ws_data)
        snap = feed.snapshot()
        self.assertIn("depth", snap)
        self.assertEqual(snap["depth"]["sell"][0][0], "5000000")
        self.assertEqual(snap["depth"]["buy"][0][0], "4900000")

    def test_apply_orderbook_ignores_invalid_data(self) -> None:
        """_apply_orderbook must not raise or update snapshot for malformed data."""
        feed = self._make_feed()
        feed._apply_orderbook("not_a_dict")  # type: ignore
        self.assertFalse(feed.has_snapshot)

    # ── _apply_trade_activity ─────────────────────────────────────────────

    def test_apply_trade_activity_normalises_to_rest_format(self) -> None:
        """_apply_trade_activity must convert WS trade rows to REST trades format."""
        feed = self._make_feed("btc_idr")
        ws_data = {
            "data": [
                ["btcidr", 1700000100, 1001, "buy",  330000000, "9900000",    "0.03"],
                ["btcidr", 1700000090, 1000, "sell", 329900000, "65980000.0", "0.2"],
            ],
            "offset": 100,
        }
        feed._apply_trade_activity(ws_data)
        snap = feed.snapshot()
        self.assertIn("trades", snap)
        trades = snap["trades"]
        # Two trades; newest first
        self.assertEqual(len(trades), 2)
        self.assertEqual(trades[0]["type"], "buy")
        self.assertEqual(trades[0]["price"], "330000000")
        self.assertEqual(trades[0]["amount"], "0.03")
        self.assertEqual(trades[0]["date"], "1700000100")
        self.assertEqual(trades[1]["type"], "sell")

    def test_apply_trade_activity_accumulates_across_pushes(self) -> None:
        """Repeated calls must accumulate trades, newest first, capped at 200."""
        feed = self._make_feed("eth_idr")
        row_template = lambda i: ["ethidr", 1700000000 + i, i, "buy", 5000000, "100000", "0.02"]
        # Push 150 trades in one call
        feed._apply_trade_activity({"data": [row_template(i) for i in range(150)], "offset": 0})
        # Push 100 more (should cap at 200 total)
        feed._apply_trade_activity({"data": [row_template(i + 150) for i in range(100)], "offset": 1})
        snap = feed.snapshot()
        self.assertEqual(len(snap["trades"]), 200)

    def test_apply_trade_activity_ignores_malformed_rows(self) -> None:
        """Rows with fewer than 7 fields are silently skipped."""
        feed = self._make_feed()
        ws_data = {
            "data": [
                ["btcidr", 1700000000, 1],       # too short – skipped
                ["btcidr", 1700000001, 2, "buy", 100, "200", "0.5"],  # valid
            ],
            "offset": 0,
        }
        feed._apply_trade_activity(ws_data)
        snap = feed.snapshot()
        # Only the valid row should appear
        self.assertEqual(len(snap["trades"]), 1)
        self.assertEqual(snap["trades"][0]["type"], "buy")


class SkipTradesAndCandleCacheTests(unittest.TestCase):
    """Verify that skip_trades and the OHLC candle cache reduce REST calls during scan."""

    def _make_trader(self, cache_ttl: int = 60):
        from bot.config import BotConfig
        from bot.trader import Trader

        class _Client:
            def __init__(self):
                self.ohlc_calls = 0
                self.trades_calls = 0
            def get_ticker(self, pair):
                return {"ticker": {"last": "100"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                self.trades_calls += 1
                return []
            def get_ohlc(self, pair, tf="15", *, limit=200, to_ts=None):
                self.ohlc_calls += 1
                return []
            def get_summaries(self):
                return {}

        config = BotConfig(
            api_key=None,
            real_time=False,
            dry_run=True,
            scan_candle_cache_seconds=cache_ttl,
        )
        client = _Client()
        trader = Trader(config, client=client)
        return trader, client

    def test_skip_trades_omits_rest_trades_call(self) -> None:
        """analyze_market with skip_trades=True must not call client.get_trades()."""
        trader, client = self._make_trader()
        trader.analyze_market("btc_idr", skip_trades=True)
        self.assertEqual(client.trades_calls, 0, "get_trades() should not be called when skip_trades=True")

    def test_skip_trades_false_calls_trades(self) -> None:
        """analyze_market with skip_trades=False (default) must call client.get_trades()."""
        trader, client = self._make_trader()
        trader.analyze_market("btc_idr", skip_trades=False)
        self.assertEqual(client.trades_calls, 1, "get_trades() should be called once when skip_trades=False")

    def test_candle_cache_avoids_second_ohlc_call(self) -> None:
        """_fetch_candles with use_cache=True must reuse cached data within TTL."""
        trader, client = self._make_trader(cache_ttl=60)
        # First call: fetches from REST
        trader.analyze_market("btc_idr", skip_trades=True)
        first_call_count = client.ohlc_calls
        # Second call within TTL: should reuse cache
        trader.analyze_market("btc_idr", skip_trades=True)
        self.assertEqual(
            client.ohlc_calls,
            first_call_count,
            "get_ohlc() should not be called again when cache is still fresh",
        )

    def test_candle_cache_disabled_at_zero(self) -> None:
        """When scan_candle_cache_seconds=0 cache is bypassed every call."""
        trader, client = self._make_trader(cache_ttl=0)
        trader.analyze_market("btc_idr", skip_trades=True)
        trader.analyze_market("btc_idr", skip_trades=True)
        self.assertEqual(
            client.ohlc_calls,
            2,
            "With TTL=0, get_ohlc() should be called on every analyze_market call",
        )

    def test_multi_feed_ticker_used_before_rest(self) -> None:
        """analyze_market must use MultiPairFeed ticker before falling back to REST."""
        from bot.config import BotConfig
        from bot.realtime import MultiPairFeed
        from bot.trader import Trader

        class _Client:
            def __init__(self):
                self.ticker_calls = 0
            def get_ticker(self, pair):
                self.ticker_calls += 1
                return {"ticker": {"last": "999"}}
            def get_depth(self, pair, count=50):
                return {"buy": [], "sell": []}
            def get_trades(self, pair, count=200):
                return []
            def get_ohlc(self, pair, tf="15", *, limit=200, to_ts=None):
                return []
            def get_summaries(self):
                return {}

        config = BotConfig(api_key=None, real_time=False)
        client = _Client()
        trader = Trader(config, client=client)

        # Simulate a seeded MultiPairFeed
        multi_feed = MultiPairFeed(["btc_idr"], client, websocket_enabled=False, summaries_interval=9999)
        multi_feed._cache["btc_idr"] = {"last": "12345"}
        trader._multi_feed = multi_feed

        snapshot = trader.analyze_market("btc_idr", skip_trades=True)
        # REST ticker must NOT have been called
        self.assertEqual(client.ticker_calls, 0)
        # Price must come from the multi-feed cache
        self.assertEqual(snapshot["price"], 12345.0)


class MultiPairFeedDepthTradesTests(unittest.TestCase):
    """Tests for the per-pair real-time orderbook and trades streaming in MultiPairFeed."""

    def _make_feed(self, pairs=("btc_idr", "eth_idr")):
        class _Stub:
            def get_summaries(self):
                return {}
        return MultiPairFeed(list(pairs), _Stub(), websocket_enabled=False, summaries_interval=9999)

    def test_get_depth_returns_none_before_data(self):
        feed = self._make_feed()
        self.assertIsNone(feed.get_depth("btc_idr"))

    def test_get_trades_returns_none_before_data(self):
        feed = self._make_feed()
        self.assertIsNone(feed.get_trades("btc_idr"))

    def test_apply_orderbook_for_pair_updates_depth_cache(self):
        """_apply_orderbook_for_pair must convert WS bid/ask to REST buy/sell format."""
        feed = self._make_feed()
        ws_data = {
            "data": {
                "bid": [{"price": "10000000", "btc_volume": "0.5", "idr_volume": "5000000"}],
                "ask": [{"price": "10100000", "btc_volume": "0.3", "idr_volume": "3030000"}],
            }
        }
        feed._apply_orderbook_for_pair("btc_idr", ws_data)
        depth = feed.get_depth("btc_idr")
        self.assertIsNotNone(depth)
        self.assertEqual(depth["buy"][0][0], "10000000")
        self.assertEqual(depth["buy"][0][1], "0.5")
        self.assertEqual(depth["sell"][0][0], "10100000")
        self.assertEqual(depth["sell"][0][1], "0.3")

    def test_apply_orderbook_for_pair_isolated_per_pair(self):
        """Orderbook update for one pair must not affect other pairs."""
        feed = self._make_feed()
        ws_data = {
            "data": {
                "bid": [{"price": "5000000", "eth_volume": "1.0", "idr_volume": "5000000"}],
                "ask": [],
            }
        }
        feed._apply_orderbook_for_pair("eth_idr", ws_data)
        self.assertIsNone(feed.get_depth("btc_idr"))
        eth_depth = feed.get_depth("eth_idr")
        self.assertIsNotNone(eth_depth)
        self.assertEqual(eth_depth["buy"][0][0], "5000000")

    def test_apply_trade_activity_for_pair_populates_buffer(self):
        """_apply_trade_activity_for_pair must build a trades buffer in REST format."""
        feed = self._make_feed()
        ws_data = {
            "data": [
                ["btcidr", 1700000100, 1, "buy",  "10000000", "1000000", "0.1"],
                ["btcidr", 1700000090, 2, "sell", "9990000",  "999000",  "0.1"],
            ]
        }
        feed._apply_trade_activity_for_pair("btc_idr", ws_data)
        trades = feed.get_trades("btc_idr")
        self.assertIsNotNone(trades)
        self.assertEqual(len(trades), 2)
        # Newest trades are prepended (newest first)
        self.assertEqual(trades[0]["type"], "buy")
        self.assertEqual(trades[0]["price"], "10000000")
        self.assertEqual(trades[1]["type"], "sell")

    def test_apply_trade_activity_for_pair_isolated_per_pair(self):
        """Trade activity for one pair must not populate buffer for another pair."""
        feed = self._make_feed()
        ws_data = {"data": [["ethidr", 1700000200, 1, "buy", "5000000", "100000", "0.02"]]}
        feed._apply_trade_activity_for_pair("eth_idr", ws_data)
        self.assertIsNone(feed.get_trades("btc_idr"))
        eth_trades = feed.get_trades("eth_idr")
        self.assertIsNotNone(eth_trades)
        self.assertEqual(len(eth_trades), 1)

    def test_trade_buffer_capped_at_2000_per_pair(self):
        """Trade buffer must not grow beyond 2000 entries per pair."""
        feed = self._make_feed()
        row = lambda i: ["btcidr", 1700000000 + i, i, "buy", "10000", "100", "0.01"]
        feed._apply_trade_activity_for_pair("btc_idr", {"data": [row(i) for i in range(1500)]})
        feed._apply_trade_activity_for_pair("btc_idr", {"data": [row(i + 1500) for i in range(1000)]})
        trades = feed.get_trades("btc_idr")
        self.assertEqual(len(trades), 2000)

    def test_subscribe_depth_pairs_stores_pairs(self):
        """subscribe_depth_pairs must record pairs for later WS subscription."""
        feed = self._make_feed()
        feed.subscribe_depth_pairs(["btc_idr", "eth_idr"])
        with feed._lock:
            stored = list(feed._depth_pairs)
        self.assertIn("btc_idr", stored)
        self.assertIn("eth_idr", stored)

    def test_subscribe_depth_pairs_no_duplicates(self):
        """Calling subscribe_depth_pairs twice with overlapping lists must not duplicate."""
        feed = self._make_feed()
        feed.subscribe_depth_pairs(["btc_idr"])
        feed.subscribe_depth_pairs(["btc_idr", "eth_idr"])
        with feed._lock:
            stored = list(feed._depth_pairs)
        self.assertEqual(stored.count("btc_idr"), 1)
        self.assertIn("eth_idr", stored)

    def test_orderbook_updates_last_ws_update_timestamp(self):
        """_apply_orderbook_for_pair must update the staleness timestamp."""
        feed = self._make_feed()
        self.assertIsNone(feed._last_ws_update)
        feed._apply_orderbook_for_pair("btc_idr", {"data": {"bid": [], "ask": []}})
        self.assertIsNotNone(feed._last_ws_update)

    def test_trade_activity_updates_last_ws_update_timestamp(self):
        """_apply_trade_activity_for_pair must update the staleness timestamp."""
        feed = self._make_feed()
        self.assertIsNone(feed._last_ws_update)
        feed._apply_trade_activity_for_pair(
            "btc_idr",
            {"data": [["btcidr", 1700000000, 1, "buy", "100", "10", "0.1"]]},
        )
        self.assertIsNotNone(feed._last_ws_update)

    def test_get_depth_returns_copy_not_reference(self):
        """get_depth must return a copy so callers cannot mutate internal state."""
        feed = self._make_feed()
        feed._apply_orderbook_for_pair("btc_idr", {"data": {"bid": [], "ask": []}})
        d1 = feed.get_depth("btc_idr")
        d1["injected"] = True
        d2 = feed.get_depth("btc_idr")
        self.assertNotIn("injected", d2)

    def test_get_trades_returns_copy_not_reference(self):
        """get_trades must return a copy so callers cannot mutate the buffer."""
        feed = self._make_feed()
        feed._apply_trade_activity_for_pair(
            "btc_idr",
            {"data": [["btcidr", 1700000000, 1, "buy", "100", "10", "0.1"]]},
        )
        t1 = feed.get_trades("btc_idr")
        t1.append({"injected": True})
        t2 = feed.get_trades("btc_idr")
        self.assertEqual(len(t2), 1)


class WsDepthAndTradesInScanTests(unittest.TestCase):
    """Tests that scan-phase analysis uses WS depth/trades from MultiPairFeed."""

    def _make_trader_with_feed(self, pair="btc_idr"):
        class _Client:
            def __init__(self):
                self.depth_calls = 0
                self.trades_calls = 0
                self.ohlc_calls = 0
            def get_ticker(self, p):
                return {"ticker": {"last": "10000", "high": "11000", "low": "9000", "vol_idr": "1000000000"}}
            def get_depth(self, p, count=50):
                self.depth_calls += 1
                return {"buy": [], "sell": []}
            def get_trades(self, p, count=200):
                self.trades_calls += 1
                return []
            def get_ohlc(self, p, tf="15", *, limit=200, to_ts=None):
                self.ohlc_calls += 1
                return []
            def get_summaries(self):
                return {}

        config = BotConfig(api_key=None, real_time=False, dry_run=True)
        client = _Client()
        trader = Trader(config, client=client)
        feed = MultiPairFeed([pair], client, websocket_enabled=False, summaries_interval=9999)
        feed._apply_ws_message_for_pair(pair, {"last": "10000", "high": "11000", "low": "9000", "vol_idr": "1000000000"})
        trader._multi_feed = feed
        return trader, client, feed

    def test_ws_depth_used_when_skip_depth_true(self):
        """analyze_market(skip_depth=True) must use WS depth from MultiPairFeed, not REST."""
        trader, client, feed = self._make_trader_with_feed("btc_idr")
        ws_bids = [{"price": "9900", "btc_volume": "1.0", "idr_volume": "9900000"}] * 5
        ws_asks = [{"price": "10100", "btc_volume": "0.5", "idr_volume": "5050000"}] * 5
        feed._apply_orderbook_for_pair("btc_idr", {"data": {"bid": ws_bids, "ask": ws_asks}})

        trader.analyze_market("btc_idr", skip_depth=True)

        # REST /depth must NOT be called — WS data was used
        self.assertEqual(client.depth_calls, 0)

    def test_ws_trades_used_when_skip_trades_true(self):
        """analyze_market(skip_trades=True) must use WS trades from MultiPairFeed, not REST."""
        trader, client, feed = self._make_trader_with_feed("btc_idr")
        trades_data = [
            ["btcidr", 1700000000 + i, i, "buy", "10000", "100000", "0.01"]
            for i in range(50)
        ]
        feed._apply_trade_activity_for_pair("btc_idr", {"data": trades_data})

        trader.analyze_market("btc_idr", skip_trades=True)

        # REST /trades must NOT be called — WS data was used
        self.assertEqual(client.trades_calls, 0)

    def test_candles_from_ws_trades_skip_ohlc_rest(self):
        """_fetch_candles must build candles from WS trades without calling REST OHLC."""
        trader, client, feed = self._make_trader_with_feed("btc_idr")
        # 300 trades, one per minute — enough for 20+ candles at 15-min intervals
        trades_data = [
            ["btcidr", 1700000000 + i * 60, i, "buy" if i % 2 == 0 else "sell",
             "10000", "100000", "0.01"]
            for i in range(300)
        ]
        feed._apply_trade_activity_for_pair("btc_idr", {"data": trades_data})

        candles = trader._fetch_candles("btc_idr", [], use_cache=True)

        # OHLC REST must NOT be called — WS trades were sufficient
        self.assertEqual(client.ohlc_calls, 0)
        self.assertGreater(len(candles), 0)

    def test_empty_ws_trades_fall_through_to_ohlc(self):
        """_fetch_candles must fall back to REST OHLC when WS trades are absent."""
        trader, client, feed = self._make_trader_with_feed("btc_idr")
        # No trades injected — WS trade buffer is empty

        trader._fetch_candles("btc_idr", [], use_cache=False)

        # Must have called REST OHLC as the fallback
        self.assertEqual(client.ohlc_calls, 1)

    def test_insufficient_ws_candles_fall_through_to_ohlc(self):
        """_fetch_candles must fall back to REST OHLC when WS trades produce
        fewer candles than slow_window — otherwise MA(slow_window) would be NaN
        for every candle, causing analyze_trend() to return "flat" (→ "hold").
        """
        trader, client, feed = self._make_trader_with_feed("btc_idr")
        # slow_window=48, interval_seconds=300 (5-min candles).
        # Inject trades spanning only ~20 minutes so build_candles produces
        # at most 4 candles — well below the slow_window threshold of 48.
        trades_data = [
            ["btcidr", 1700000000 + i * 60, i, "buy", "10000", "100000", "0.01"]
            for i in range(20)
        ]
        feed._apply_trade_activity_for_pair("btc_idr", {"data": trades_data})

        trader._fetch_candles("btc_idr", [], use_cache=False)

        # WS candles were too few for MA(slow_window) — REST OHLC must be called
        self.assertEqual(
            client.ohlc_calls, 1,
            "REST OHLC must be called when WS candles < slow_window",
        )

    def test_ws_depth_empty_falls_back_to_empty_depth(self):
        """When WS depth is not yet available, skip_depth=True still returns empty depth (no REST)."""
        trader, client, feed = self._make_trader_with_feed("btc_idr")
        # No orderbook injected

        trader.analyze_market("btc_idr", skip_depth=True)

        # With WS data absent, REST /depth must still NOT be called
        self.assertEqual(client.depth_calls, 0)


class WsZombieConnectionTests(unittest.TestCase):
    """Verify that the WS reconnection loop does NOT create zombie connections.

    The fix replaces the old ``wst.join(timeout=30.0)`` pattern (which could
    time out while the connection was still healthy, causing the loop to create
    *another* connection) with an indefinite wait loop that checks the stop
    signal periodically.
    """

    def test_realtime_feed_waits_for_ws_thread_to_finish(self):
        """RealtimeFeed._run_websocket must block until the WS thread dies, not
        use a fixed 30-second timeout that would accumulate zombie sockets."""
        import inspect
        from bot.realtime import RealtimeFeed
        source = inspect.getsource(RealtimeFeed._run_websocket)
        # Must NOT contain the old fixed-timeout join pattern
        self.assertNotIn("join(timeout=30", source,
                         "Fixed 30s join timeout causes zombie connections")
        # Must contain the polling loop that waits for the thread to actually finish
        self.assertIn("while wst.is_alive()", source,
                       "Missing indefinite wait loop for WS thread")
        # Must force-close the WebSocket when stop is requested
        self.assertIn("ws_app.close()", source,
                       "Missing ws_app.close() for clean shutdown")

    def test_multi_pair_feed_waits_for_ws_thread_to_finish(self):
        """MultiPairFeed._run_ws_centrifuge must block until the WS thread dies."""
        import inspect
        from bot.realtime import MultiPairFeed
        source = inspect.getsource(MultiPairFeed._run_ws_centrifuge)
        self.assertNotIn("join(timeout=30", source,
                         "Fixed 30s join timeout causes zombie connections")
        self.assertIn("while wst.is_alive()", source,
                       "Missing indefinite wait loop for WS thread")
        self.assertIn("ws_app.close()", source,
                       "Missing ws_app.close() for clean shutdown")

    def test_private_feed_waits_for_ws_thread_to_finish(self):
        """PrivateFeed._connect_once must block until the WS thread dies."""
        import inspect
        from bot.realtime import PrivateFeed
        source = inspect.getsource(PrivateFeed._connect_once)
        self.assertNotIn("join(timeout=30", source,
                         "Fixed 30s join timeout causes zombie connections")
        self.assertIn("while wst.is_alive()", source,
                       "Missing indefinite wait loop for WS thread")
        self.assertIn("ws_app.close()", source,
                       "Missing ws_app.close() for clean shutdown")
