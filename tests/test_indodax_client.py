import time
import unittest

from bot.indodax_client import IndodaxClient


class _DummyClient(IndodaxClient):
    """Minimal client that short-circuits network calls for testing."""

    def __init__(self) -> None:
        super().__init__(
            api_key="key",
            api_secret="secret",
            enable_queue=False,
            enable_request_scheduler=False,
        )
        self.last_params = None
        self.last_method = None

    def _enqueue_private(self, method, params=None):
        self.last_method = method
        self.last_params = params or {}
        return self.last_params

    def _get(self, path, params=None):
        # Avoid real HTTP calls during tests; return empty payloads.
        return {}


class IndodaxClientPrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _DummyClient()

    def test_idr_price_defaults_to_integer_when_no_increment_cache(self):
        resp = self.client.create_order("dupe_idr", "buy", 146.1234, 100)
        self.assertEqual(resp["price"], "146")
        # IDR total should use the rounded price
        self.assertEqual(resp["idr"], "14600")

    def test_non_idr_price_keeps_eight_decimals(self):
        resp = self.client.create_order("btc_usdt", "sell", 12345.67890123, 0.1)
        self.assertEqual(resp["price"], "12345.67890123")
        self.assertEqual(resp["amount"], "0.10000000")
        self.assertEqual(resp["btc"], "0.10000000")

    def test_price_uses_increment_when_cached(self):
        self.client._price_increments = {"dupe_idr": "0.01"}
        self.client._price_increments_expires = time.time() + 3600
        resp = self.client.create_order("dupe_idr", "buy", 146.1234, 10)
        self.assertEqual(resp["price"], "146.12")
        self.assertEqual(resp["idr"], "1461")

    def test_client_order_id_and_time_in_force_pass_through(self):
        resp = self.client.create_order(
            "btc_usdt",
            "buy",
            100.0,
            1.0,
            client_order_id="clientx-123",
            time_in_force="GTC",
        )
        self.assertEqual(resp["client_order_id"], "clientx-123")
        self.assertEqual(resp["time_in_force"], "GTC")


if __name__ == "__main__":
    unittest.main()
