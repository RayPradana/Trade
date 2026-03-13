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


class IndodaxClientPrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _DummyClient()

    def test_idr_price_rounded_to_two_decimals(self):
        resp = self.client.create_order("dupe_idr", "buy", 146.1234, 100)
        self.assertEqual(resp["price"], "146.12000000")
        # IDR total should use the rounded price
        self.assertEqual(resp["idr"], "14612.00000000")

    def test_non_idr_price_keeps_eight_decimals(self):
        resp = self.client.create_order("btc_usdt", "sell", 12345.67890123, 0.1)
        self.assertEqual(resp["price"], "12345.67890123")
        self.assertEqual(resp["amount"], "0.10000000")


if __name__ == "__main__":
    unittest.main()
