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

    def test_limit_buy_idr_pair_sends_coin_amount_only(self):
        """Per API docs: limit buy must send coin amount, NOT idr."""
        resp = self.client.create_order("dupe_idr", "buy", 146.1234, 100)
        self.assertEqual(resp["price"], "146")
        # Limit buy on IDR pair: coin amount only, no idr field.
        self.assertEqual(resp["dupe"], "100.00000000")
        self.assertNotIn("idr", resp)
        self.assertNotIn("amount", resp)

    def test_market_buy_idr_pair_sends_idr_only(self):
        """Per API docs: market buy only supports idr amount."""
        resp = self.client.create_order("dupe_idr", "buy", 146.1234, 100, order_kind="market")
        self.assertEqual(resp["idr"], "14600")
        self.assertNotIn("dupe", resp)
        self.assertNotIn("amount", resp)

    def test_non_idr_price_keeps_eight_decimals(self):
        resp = self.client.create_order("btc_usdt", "sell", 12345.67890123, 0.1)
        self.assertEqual(resp["price"], "12345.67890123")
        self.assertEqual(resp["btc"], "0.10000000")
        self.assertNotIn("amount", resp)

    def test_sell_idr_pair_sends_coin_amount_only(self):
        """Per API docs: sell must send coin amount, not idr."""
        resp = self.client.create_order("btc_idr", "sell", 500000000, 0.001)
        self.assertEqual(resp["btc"], "0.00100000")
        self.assertNotIn("idr", resp)
        self.assertNotIn("amount", resp)

    def test_price_uses_increment_when_cached(self):
        self.client._price_increments = {"dupe_idr": "0.01"}
        self.client._price_increments_expires = time.time() + 3600
        # Limit buy: coin amount, no idr
        resp = self.client.create_order("dupe_idr", "buy", 146.1234, 10)
        self.assertEqual(resp["price"], "146.12")
        self.assertEqual(resp["dupe"], "10.00000000")
        self.assertNotIn("idr", resp)

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

    def test_integer_amount_precision_for_whole_number_coins(self):
        """Coins with integer trade_min_traded_currency must format amount without decimals."""
        # Simulate a pair that requires integer amounts (e.g. low-price token)
        self.client._amount_precisions["token_idr"] = 0
        resp = self.client.create_order("token_idr", "sell", 5, 150.7)
        self.assertEqual(resp["token"], "151")
        self.assertNotIn("amount", resp)

    def test_amount_precision_default_eight_decimals(self):
        """Without cached precision, amounts default to 8 decimal places."""
        resp = self.client.create_order("btc_idr", "sell", 500000000, 0.00123456)
        self.assertEqual(resp["btc"], "0.00123456")

    def test_explicit_idr_override_still_works(self):
        """Passing idr= explicitly should still send idr (backward compat)."""
        resp = self.client.create_order("btc_idr", "buy", 500000000, 0.001, idr=500000)
        self.assertEqual(resp["idr"], "500000")
        self.assertNotIn("btc", resp)


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


class IndodaxClientAmountPrecisionTest(unittest.TestCase):
    """Tests for per-pair amount precision (format_amount, load_pair_min_orders)."""

    def test_format_amount_defaults_to_eight_decimals(self):
        """Without cached precision, format_amount uses 8 decimal places."""
        client = _DummyClient()
        amt, prec = client.format_amount("btc_idr", 0.001234567890)
        self.assertEqual(prec, 8)
        self.assertAlmostEqual(amt, 0.00123457, places=8)

    def test_format_amount_integer_precision(self):
        """Coins with precision 0 must return integer amounts."""
        client = _DummyClient()
        client._amount_precisions["token_idr"] = 0
        amt, prec = client.format_amount("token_idr", 150.7)
        self.assertEqual(prec, 0)
        self.assertEqual(amt, 151.0)

    def test_load_pair_min_orders_derives_amount_precision(self):
        """load_pair_min_orders should set precision 0 for integer min coins."""
        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}
        client._amount_precisions = {}

        # Per Indodax /api/pairs docs:
        #   trade_min_base_currency = minimum IDR value
        #   trade_min_traded_currency = minimum COIN amount
        pairs_data = [
            {
                "id": "btcidr",
                "ticker_id": "btc_idr",
                "trade_min_base_currency": "50000",
                "trade_min_traded_currency": "0.0001",
            },
            {
                "id": "tokenidr",
                "ticker_id": "token_idr",
                "trade_min_base_currency": "10000",
                "trade_min_traded_currency": "1",
            },
            {
                "id": "dogeidr",
                "ticker_id": "doge_idr",
                "trade_min_base_currency": "5000",
                "trade_min_traded_currency": "100",
            },
        ]
        client.load_pair_min_orders(pairs_data)

        # BTC: trade_min_traded_currency=0.0001 (fractional) → 8 decimals
        self.assertEqual(client._amount_precisions["btcidr"], 8)
        self.assertEqual(client._amount_precisions["btc_idr"], 8)

        # TOKEN: trade_min_traded_currency=1 (integer) → 0 decimals
        self.assertEqual(client._amount_precisions["tokenidr"], 0)
        self.assertEqual(client._amount_precisions["token_idr"], 0)

        # DOGE: trade_min_traded_currency=100 (integer) → 0 decimals
        self.assertEqual(client._amount_precisions["dogeidr"], 0)
        self.assertEqual(client._amount_precisions["doge_idr"], 0)

    def test_load_pair_min_orders_stores_under_both_key_formats(self):
        """Cache should be accessible via both 'btcidr' and 'btc_idr' keys."""
        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}
        client._amount_precisions = {}

        pairs_data = [
            {
                "id": "btcidr",
                "ticker_id": "btc_idr",
                "trade_min_base_currency": "50000",
                "trade_min_traded_currency": "0.0001",
            },
        ]
        client.load_pair_min_orders(pairs_data)

        # Both key formats should exist in the cache
        self.assertIn("btcidr", client._pair_min_order)
        self.assertIn("btc_idr", client._pair_min_order)
        # trade_min_base_currency → min_idr (minimum IDR/base amount)
        self.assertAlmostEqual(client._pair_min_order["btc_idr"]["min_idr"], 50000.0)
        # Amount precision should also be stored under both keys
        self.assertEqual(client._amount_precisions["btc_idr"], 8)
        self.assertEqual(client._amount_precisions["btcidr"], 8)

    def test_create_order_sell_integer_amount_coin(self):
        """Sell order for integer-amount coin must not have decimals."""
        client = _DummyClient()
        client._amount_precisions["doge_idr"] = 0
        resp = client.create_order("doge_idr", "sell", 5, 1500)
        self.assertEqual(resp["doge"], "1500")
        self.assertNotIn(".", resp["doge"])

    def test_create_order_buy_integer_amount_coin(self):
        """Limit buy for integer-amount coin must not have decimals."""
        client = _DummyClient()
        client._amount_precisions["doge_idr"] = 0
        resp = client.create_order("doge_idr", "buy", 5, 1500)
        self.assertEqual(resp["doge"], "1500")
        self.assertNotIn("idr", resp)

    def test_format_amount_fallback_no_underscore(self):
        """format_amount should find precision via no-underscore key fallback."""
        client = _DummyClient()
        # Only store under the id format (no underscore).
        client._amount_precisions["dogeidr"] = 0
        # Lookup with underscore format should still find it.
        amt, prec = client.format_amount("doge_idr", 150.7)
        self.assertEqual(prec, 0)
        self.assertEqual(amt, 151.0)

    def test_load_pair_min_orders_constructed_key_from_currencies(self):
        """Cache should use traded_currency + base_currency as fallback key."""
        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}
        client._amount_precisions = {}

        # Pair without ticker_id — only id + traded/base currency available.
        pairs_data = [
            {
                "id": "tokenidr",
                "traded_currency": "token",
                "base_currency": "idr",
                "trade_min_base_currency": "10000",
                "trade_min_traded_currency": "1",
            },
        ]
        client.load_pair_min_orders(pairs_data)

        # Should be cached under "tokenidr" (from id) and "token_idr" (constructed).
        self.assertIn("tokenidr", client._amount_precisions)
        self.assertIn("token_idr", client._amount_precisions)
        self.assertEqual(client._amount_precisions["token_idr"], 0)
        self.assertIn("token_idr", client._pair_min_order)

    def test_create_order_sell_btc_param_integer_coin(self):
        """Sell with explicit btc= param for integer coin must not have decimals."""
        client = _DummyClient()
        client._amount_precisions["doge_idr"] = 0
        resp = client.create_order("doge_idr", "sell", 5, 999, btc=1500.7)
        self.assertEqual(resp["doge"], "1501")
        self.assertNotIn(".", resp["doge"])

    def test_create_order_buy_btc_param_integer_coin(self):
        """Limit buy with explicit btc= param for integer coin must not have decimals."""
        client = _DummyClient()
        client._amount_precisions["doge_idr"] = 0
        resp = client.create_order("doge_idr", "buy", 5, 999, btc=1500.3)
        self.assertEqual(resp["doge"], "1500")
        self.assertNotIn(".", resp["doge"])

    def test_create_order_fractional_amount_no_decimal_error(self):
        """Amount with many decimals should be formatted to pair precision."""
        client = _DummyClient()
        client._amount_precisions["doge_idr"] = 0
        # Amount from float arithmetic with many decimal places.
        resp = client.create_order("doge_idr", "sell", 5, 150.999999999999)
        self.assertNotIn(".", resp["doge"])


class DecimalRetryTest(unittest.TestCase):
    """Test auto-retry when exchange returns 'amount can't be in decimal.'."""

    def test_create_order_retries_on_decimal_error(self):
        """create_order must retry with integer amount on 'decimal' API error."""
        client = _DummyClient()
        # Simulate precision cache missing for doge_idr → defaults to 8 decimals
        # First _enqueue_private call raises the error; second succeeds.
        call_count = {"n": 0}
        original_enqueue = client._enqueue_private

        def _fake_enqueue(method, params=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError(
                    "API error: {'success': 0, 'error': \"amount can't be in decimal.\"}"
                )
            # Return params on retry so we can inspect them.
            return params

        client._enqueue_private = _fake_enqueue
        resp = client.create_order("doge_idr", "sell", 5, 150.7)
        # Retry must have sent integer amount (no decimal).
        self.assertEqual(resp["doge"], "151")
        self.assertNotIn(".", resp["doge"])
        self.assertEqual(call_count["n"], 2)

    def test_create_order_updates_precision_cache_on_decimal_error(self):
        """After retrying, the precision cache must be updated to 0."""
        client = _DummyClient()
        call_count = {"n": 0}

        def _fake_enqueue(method, params=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError(
                    "API error: {'success': 0, 'error': \"amount can't be in decimal.\"}"
                )
            return params

        client._enqueue_private = _fake_enqueue
        client.create_order("doge_idr", "sell", 5, 150.7)
        # Both key formats must be cached as precision 0.
        self.assertEqual(client._amount_precisions.get("doge_idr"), 0)
        self.assertEqual(client._amount_precisions.get("dogeidr"), 0)

    def test_create_order_buy_retries_on_decimal_error(self):
        """Limit buy must also retry with integer amount on decimal error."""
        client = _DummyClient()
        call_count = {"n": 0}

        def _fake_enqueue(method, params=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError(
                    "API error: {'success': 0, 'error': \"amount can't be in decimal.\"}"
                )
            return params

        client._enqueue_private = _fake_enqueue
        resp = client.create_order("doge_idr", "buy", 5, 150.7)
        self.assertEqual(resp["doge"], "151")
        self.assertNotIn(".", resp["doge"])

    def test_create_order_non_decimal_error_not_retried(self):
        """Non-decimal RuntimeErrors must propagate without retry."""
        client = _DummyClient()

        def _fake_enqueue(method, params=None):
            raise RuntimeError("API error: some other error")

        client._enqueue_private = _fake_enqueue
        with self.assertRaises(RuntimeError) as ctx:
            client.create_order("doge_idr", "sell", 5, 150)
        self.assertIn("some other error", str(ctx.exception))


class DecimalPrecisionDetectionTest(unittest.TestCase):
    """Test Decimal-based precision detection in load_pair_min_orders."""

    def _make_client(self):
        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}
        client._amount_precisions = {}
        return client

    def test_string_integer_min_gives_precision_zero(self):
        """trade_min_traded_currency='100' → precision 0."""
        client = self._make_client()
        client.load_pair_min_orders([{
            "id": "dogeidr", "ticker_id": "doge_idr",
            "trade_min_base_currency": "5000",
            "trade_min_traded_currency": "100",
        }])
        self.assertEqual(client._amount_precisions["doge_idr"], 0)

    def test_string_fractional_min_gives_precision_eight(self):
        """trade_min_traded_currency='0.0001' → precision 8."""
        client = self._make_client()
        client.load_pair_min_orders([{
            "id": "btcidr", "ticker_id": "btc_idr",
            "trade_min_base_currency": "50000",
            "trade_min_traded_currency": "0.0001",
        }])
        self.assertEqual(client._amount_precisions["btc_idr"], 8)

    def test_numeric_integer_min_gives_precision_zero(self):
        """trade_min_traded_currency=100 (int) → precision 0."""
        client = self._make_client()
        client.load_pair_min_orders([{
            "id": "dogeidr", "ticker_id": "doge_idr",
            "trade_min_base_currency": 5000,
            "trade_min_traded_currency": 100,
        }])
        self.assertEqual(client._amount_precisions["doge_idr"], 0)

    def test_numeric_float_integer_min_gives_precision_zero(self):
        """trade_min_traded_currency=100.0 (float) → precision 0 via Decimal.normalize()."""
        client = self._make_client()
        client.load_pair_min_orders([{
            "id": "dogeidr", "ticker_id": "doge_idr",
            "trade_min_base_currency": 5000,
            "trade_min_traded_currency": 100.0,
        }])
        self.assertEqual(client._amount_precisions["doge_idr"], 0)

    def test_zero_min_defaults_to_eight(self):
        """trade_min_traded_currency=0 → precision 8 (unknown, safe default)."""
        client = self._make_client()
        client.load_pair_min_orders([{
            "id": "unknownidr", "ticker_id": "unknown_idr",
            "trade_min_base_currency": "5000",
            "trade_min_traded_currency": "0",
        }])
        self.assertEqual(client._amount_precisions["unknown_idr"], 8)

    def test_missing_min_defaults_to_eight(self):
        """Missing trade_min_traded_currency → precision 8."""
        client = self._make_client()
        client.load_pair_min_orders([{
            "id": "unknownidr", "ticker_id": "unknown_idr",
            "trade_min_base_currency": "5000",
        }])
        self.assertEqual(client._amount_precisions["unknown_idr"], 8)

    def test_string_one_min_gives_precision_zero(self):
        """trade_min_traded_currency='1' → precision 0."""
        client = self._make_client()
        client.load_pair_min_orders([{
            "id": "tokenidr", "ticker_id": "token_idr",
            "trade_min_base_currency": "10000",
            "trade_min_traded_currency": "1",
        }])
        self.assertEqual(client._amount_precisions["token_idr"], 0)


class FormatAmountAutoRefreshTest(unittest.TestCase):
    """Test that format_amount auto-refreshes cache on miss."""

    def test_format_amount_refreshes_on_cache_miss(self):
        """format_amount should try to refresh pair data when precision unknown."""
        client = _DummyClient()
        # Start with empty precision cache.
        client._amount_precisions = {}
        # The _DummyClient._get returns {} which won't populate the cache,
        # but _maybe_refresh_pair_min_orders should be called.
        refreshed = {"called": False}
        original_refresh = client._maybe_refresh_pair_min_orders

        def _track_refresh():
            refreshed["called"] = True
            # Simulate loading — set precision for doge_idr to 0.
            client._amount_precisions["doge_idr"] = 0
            client._amount_precisions["dogeidr"] = 0

        client._maybe_refresh_pair_min_orders = _track_refresh
        amt, prec = client.format_amount("doge_idr", 150.7)
        self.assertTrue(refreshed["called"])
        self.assertEqual(prec, 0)
        self.assertEqual(amt, 151.0)

    def test_format_amount_no_refresh_when_cached(self):
        """format_amount must not refresh when precision is already cached."""
        client = _DummyClient()
        client._amount_precisions["doge_idr"] = 0
        refreshed = {"called": False}

        def _track_refresh():
            refreshed["called"] = True

        client._maybe_refresh_pair_min_orders = _track_refresh
        amt, prec = client.format_amount("doge_idr", 150.7)
        self.assertFalse(refreshed["called"])
        self.assertEqual(prec, 0)


if __name__ == "__main__":
    unittest.main()
