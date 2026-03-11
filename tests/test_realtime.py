import unittest

from bot.config import BotConfig
from bot.realtime import RealtimeFeed
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
        config = BotConfig(api_key=None, real_time=False, auto_resume=False)
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


if __name__ == "__main__":
    unittest.main()
