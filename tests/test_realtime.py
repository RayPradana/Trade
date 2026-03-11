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


if __name__ == "__main__":
    unittest.main()
