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

    def _post_private(self, method, params=None):
        self.last_method = method
        self.last_params = params or {}
        return self.last_params

    def _get(self, path, params=None):
        # Avoid real HTTP calls during tests; return empty payloads.
        return {}

    def _get_signed(self, path, params=None):
        self.last_method = path
        self.last_params = params or {}
        return self.last_params


class IndodaxClientPrecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _DummyClient()

    def test_idr_price_defaults_to_integer_when_no_increment_cache(self):
        resp = self.client.create_order("dupe_idr", "buy", 146.1234, 100)
        self.assertEqual(resp["price"], "146")
        # IDR total should use the rounded price
        self.assertEqual(resp["idr"], "14600")
        # IDR buys should also include base coin quantity
        self.assertEqual(resp["dupe"], "100.00000000")
        self.assertEqual(resp["amount"], "100.00000000")

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


class IndodaxClientNewEndpointsTest(unittest.TestCase):
    """Tests for newly added Indodax API endpoint wrappers."""

    def setUp(self) -> None:
        self.client = _DummyClient()

    def test_cancel_by_client_order_id(self):
        self.client.cancel_by_client_order_id("clientx-abc123")
        self.assertEqual(self.client.last_method, "cancelByClientOrderId")
        self.assertEqual(self.client.last_params["client_order_id"], "clientx-abc123")

    def test_get_order(self):
        self.client.get_order("btc_idr", "59639504")
        self.assertEqual(self.client.last_method, "getOrder")
        self.assertEqual(self.client.last_params["pair"], "btc_idr")
        self.assertEqual(self.client.last_params["order_id"], "59639504")

    def test_get_order_by_client_id(self):
        self.client.get_order_by_client_id("clientx-xyz")
        self.assertEqual(self.client.last_method, "getOrderByClientOrderId")
        self.assertEqual(self.client.last_params["client_order_id"], "clientx-xyz")

    def test_order_history(self):
        self.client.order_history("btc_idr", count=50)
        self.assertEqual(self.client.last_method, "orderHistory")
        self.assertEqual(self.client.last_params["pair"], "btc_idr")
        self.assertEqual(self.client.last_params["count"], 50)

    def test_order_history_with_from_id(self):
        self.client.order_history("btc_idr", from_id=100)
        self.assertEqual(self.client.last_params["from"], 100)

    def test_withdraw_fee(self):
        self.client.withdraw_fee("eth", network="erc20")
        self.assertEqual(self.client.last_method, "withdrawFee")
        self.assertEqual(self.client.last_params["currency"], "eth")
        self.assertEqual(self.client.last_params["network"], "erc20")

    def test_withdraw_coin(self):
        self.client.withdraw_coin(
            "btc", "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            0.001, "req-001", network="btc",
        )
        self.assertEqual(self.client.last_method, "withdrawCoin")
        self.assertEqual(self.client.last_params["currency"], "btc")
        self.assertEqual(self.client.last_params["withdraw_amount"], "0.00100000")
        self.assertEqual(self.client.last_params["request_id"], "req-001")

    def test_trans_history(self):
        self.client.trans_history()
        self.assertEqual(self.client.last_method, "transHistory")

    def test_get_order_history_v2(self):
        self.client.get_order_history_v2("btcidr", start_time=1000, limit=50)
        self.assertEqual(self.client.last_method, "/api/v2/order/histories")
        self.assertEqual(self.client.last_params["symbol"], "btcidr")
        self.assertEqual(self.client.last_params["startTime"], 1000)
        self.assertEqual(self.client.last_params["limit"], 50)

    def test_get_trade_history_v2(self):
        self.client.get_trade_history_v2("ethidr", order_id="aaveidr-limit-3568")
        self.assertEqual(self.client.last_method, "/api/v2/myTrades")
        self.assertEqual(self.client.last_params["symbol"], "ethidr")
        self.assertEqual(self.client.last_params["orderId"], "aaveidr-limit-3568")

    def test_handle_v2_response_error(self):
        """_handle_v2_response should raise on V2 error format."""
        class _Resp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"code": 1109, "error": "Invalid Symbol"}
        with self.assertRaises(RuntimeError) as ctx:
            IndodaxClient._handle_v2_response(_Resp())
        self.assertIn("1109", str(ctx.exception))
        self.assertIn("Invalid Symbol", str(ctx.exception))

    def test_handle_v2_response_success(self):
        """_handle_v2_response should return data on success."""
        class _Resp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return {"data": [{"orderId": "123"}]}
        result = IndodaxClient._handle_v2_response(_Resp())
        self.assertEqual(result["data"][0]["orderId"], "123")


if __name__ == "__main__":
    unittest.main()
