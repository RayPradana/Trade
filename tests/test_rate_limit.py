import unittest

from bot.rate_limit import RateLimitedOrderQueue


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class RateLimitQueueTests(unittest.TestCase):
    def test_respects_min_interval(self) -> None:
        clock = _FakeClock()
        queue = RateLimitedOrderQueue(min_interval=1.0, time_provider=clock.time, sleeper=clock.sleep)
        executed_at: list[float] = []

        def task(label: str) -> str:
            executed_at.append(clock.time())
            return label

        res1 = queue.submit(task, "first")
        res2 = queue.submit(task, "second")
        self.assertEqual(res1.result(timeout=1.0), "first")
        self.assertEqual(res2.result(timeout=1.0), "second")
        self.assertGreaterEqual(executed_at[1] - executed_at[0], 1.0)
        queue.stop()


if __name__ == "__main__":
    unittest.main()


class IndodaxClientCacheTests(unittest.TestCase):
    """Tests for TTL-based caches in IndodaxClient."""

    def _make_client(self, account_ttl=30.0, open_orders_ttl=15.0):
        from bot.indodax_client import IndodaxClient
        client = IndodaxClient(api_key="k", api_secret="s", enable_queue=False)
        client.configure_caches(account_info_ttl=account_ttl, open_orders_ttl=open_orders_ttl)
        return client

    def test_account_info_cache_returns_cached_value(self):
        import time as _time
        client = self._make_client(account_ttl=30.0)
        calls = []

        def _mock_post(method, params=None):
            calls.append(method)
            return {"success": 1, "return": {"balance": {"idr": "1000000"}}}

        client._post_private = _mock_post
        r1 = client.get_account_info()
        r2 = client.get_account_info()
        # Second call should be served from cache (only 1 real call)
        self.assertEqual(len(calls), 1)
        self.assertIs(r1, r2)

    def test_account_info_cache_expires(self):
        client = self._make_client(account_ttl=1.0)  # 1-second TTL
        calls = []

        def _mock_post(method, params=None):
            calls.append(method)
            return {"success": 1, "return": {}}

        client._post_private = _mock_post
        client.get_account_info()
        # Force expiry by backdating the expiry time
        client._account_info_expires = 0.0
        client.get_account_info()
        self.assertEqual(len(calls), 2)

    def test_account_info_cache_disabled_when_ttl_zero(self):
        client = self._make_client(account_ttl=0.0)
        calls = []

        def _mock_post(method, params=None):
            calls.append(method)
            return {"success": 1, "return": {}}

        client._post_private = _mock_post
        client.get_account_info()
        client.get_account_info()
        # No caching → 2 real calls
        self.assertEqual(len(calls), 2)

    def test_invalidate_account_info_cache_forces_fresh_fetch(self):
        client = self._make_client(account_ttl=60.0)
        calls = []

        def _mock_post(method, params=None):
            calls.append(method)
            return {"success": 1, "return": {}}

        client._post_private = _mock_post
        client.get_account_info()
        self.assertEqual(len(calls), 1)
        client.invalidate_account_info_cache()
        client.get_account_info()
        self.assertEqual(len(calls), 2)

    def test_open_orders_cache_per_pair(self):
        client = self._make_client(open_orders_ttl=15.0)
        calls = []

        def _mock_post(method, params=None):
            calls.append((method, (params or {}).get("pair")))
            return {"success": 1, "return": {"orders": []}}

        client._post_private = _mock_post
        client.open_orders("btc_idr")
        client.open_orders("btc_idr")   # cached
        client.open_orders("eth_idr")   # different pair, not cached
        self.assertEqual(len(calls), 2)
        pairs = [c[1] for c in calls]
        self.assertIn("btc_idr", pairs)
        self.assertIn("eth_idr", pairs)

    def test_invalidate_open_orders_cache_single_pair(self):
        client = self._make_client(open_orders_ttl=60.0)
        calls = []

        def _mock_post(method, params=None):
            calls.append((method, (params or {}).get("pair")))
            return {"success": 1, "return": {"orders": []}}

        client._post_private = _mock_post
        client.open_orders("btc_idr")
        client.open_orders("eth_idr")
        client.invalidate_open_orders_cache("btc_idr")
        client.open_orders("btc_idr")   # invalidated → re-fetch
        client.open_orders("eth_idr")   # still cached
        btc_calls = [c for c in calls if c[1] == "btc_idr"]
        self.assertEqual(len(btc_calls), 2)

    def test_open_orders_cache_disabled_when_ttl_zero(self):
        client = self._make_client(open_orders_ttl=0.0)
        calls = []

        def _mock_post(method, params=None):
            calls.append(method)
            return {"success": 1, "return": {"orders": []}}

        client._post_private = _mock_post
        client.open_orders("btc_idr")
        client.open_orders("btc_idr")
        self.assertEqual(len(calls), 2)


class LoadPairMinOrdersFromDataTests(unittest.TestCase):
    """Tests for load_pair_min_orders() with pre-fetched pairs data."""

    def _make_pairs_data(self):
        return [
            {"id": "btcidr", "trade_min_base_currency": "0.0001", "trade_min_traded_currency": "10000"},
            {"id": "ethidr", "trade_min_base_currency": "0.001", "trade_min_traded_currency": "5000"},
        ]

    def test_load_from_data_skips_rest_call(self):
        """Passing pairs_info avoids an /api/pairs REST call."""
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}
        client.session = None  # Would explode if used
        client.base_url = "https://indodax.com"
        client.timeout = 10

        get_pairs_called = []

        def _no_rest(*args, **kwargs):  # pragma: no cover
            get_pairs_called.append(True)
            raise RuntimeError("REST call should not happen")

        client.get_pairs = _no_rest

        client.load_pair_min_orders(self._make_pairs_data())

        self.assertEqual(get_pairs_called, [], "get_pairs() must not be called when data is provided")
        self.assertIn("btcidr", client._pair_min_order)
        self.assertAlmostEqual(client._pair_min_order["btcidr"]["min_coin"], 0.0001)
        self.assertIn("ethidr", client._pair_min_order)

    def test_load_from_data_populates_cache_correctly(self):
        """Pre-fetched data is parsed identically to a live /api/pairs call."""
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}
        client.session = None
        client.base_url = "https://indodax.com"
        client.timeout = 10

        pairs_data = self._make_pairs_data()
        client.load_pair_min_orders(pairs_data)

        self.assertEqual(len(client._pair_min_order), 2)
        self.assertAlmostEqual(client._pair_min_order["ethidr"]["min_idr"], 5000.0)

    def test_load_without_args_still_calls_rest(self):
        """Calling load_pair_min_orders() without args still fetches /api/pairs."""
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient.__new__(IndodaxClient)
        client._pair_min_order = {}

        rest_calls = []

        def _mock_get_pairs():
            rest_calls.append(True)
            return self._make_pairs_data()

        client.get_pairs = _mock_get_pairs
        client.load_pair_min_orders()

        self.assertEqual(len(rest_calls), 1)
        self.assertIn("btcidr", client._pair_min_order)

    def test_is_pair_min_order_cache_stale_fresh_after_load(self):
        """Cache must not be stale immediately after a successful load."""
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient(api_key="k", api_secret="s", enable_queue=False)
        # Inject data without a real REST call
        client.load_pair_min_orders(self._make_pairs_data())
        self.assertFalse(
            client.is_pair_min_order_cache_stale(),
            "Cache should be fresh right after loading",
        )

    def test_is_pair_min_order_cache_stale_before_first_load(self):
        """Cache must be stale before any load (expiry timestamp=0)."""
        from bot.indodax_client import IndodaxClient

        client = IndodaxClient(api_key="k", api_secret="s", enable_queue=False)
        self.assertTrue(
            client.is_pair_min_order_cache_stale(),
            "Cache should be stale before any load",
        )


class NoDuplicatePairsCallTest(unittest.TestCase):
    """scan_and_choose() must not call /api/pairs twice on the first cycle."""

    def setUp(self):
        import logging
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        import logging
        logging.disable(logging.NOTSET)

    def test_get_pairs_called_once_on_first_scan(self):
        """On the first scan_and_choose() call, /api/pairs must be fetched only once."""
        from bot.config import BotConfig
        from bot.trader import Trader
        from bot.strategies import StrategyDecision

        pairs_data = [
            {"id": "btcidr", "trade_min_base_currency": "0.0001", "trade_min_traded_currency": "10000"},
            {"id": "ethidr", "trade_min_base_currency": "0.001", "trade_min_traded_currency": "5000"},
        ]
        get_pairs_call_count = []

        class _Client:
            _pair_min_order = {}
            load_min_order_calls = []

            def get_pairs(self):
                get_pairs_call_count.append(1)
                # Return data with both "name" field (for _all_pairs) and
                # min-order fields (for the cache).
                return [
                    {"name": "btc_idr", "id": "btcidr",
                     "trade_min_base_currency": "0.0001", "trade_min_traded_currency": "10000"},
                    {"name": "eth_idr", "id": "ethidr",
                     "trade_min_base_currency": "0.001", "trade_min_traded_currency": "5000"},
                ]

            def get_summaries(self):
                return {}

            def load_pair_min_orders(self, pairs_info=None):
                _Client.load_min_order_calls.append(pairs_info is not None)
                # Populate the cache so _ensure_pair_min_order_cache is a no-op
                for p in (pairs_info or []):
                    pair_id = (p.get("id") or "").lower()
                    self._pair_min_order[pair_id] = {"min_coin": 0.0, "min_idr": 0.0}

        config = BotConfig(
            api_key=None,
            pair="btc_idr",
            pair_min_order_cache_enabled=True,
        )

        snapshot = {
            "pair": "btc_idr",
            "price": 100.0,
            "trend": None,
            "orderbook": None,
            "volatility": None,
            "levels": None,
            "decision": StrategyDecision(
                mode="day_trading",
                action="hold",
                confidence=0.5,
                reason="test",
                target_price=100.0,
                amount=0.0,
                stop_loss=None,
                take_profit=None,
            ),
        }

        client_instance = _Client()

        class _Trader(Trader):
            def analyze_market(self, pair=None, prefetched_ticker=None, skip_depth=False, skip_trades=False):
                return snapshot

        trader = _Trader(config, client=client_instance)
        trader.scan_and_choose()

        self.assertEqual(sum(get_pairs_call_count), 1,
                         "get_pairs() should be called exactly once on the first cycle")
        # Also verify load_pair_min_orders was called with pre-fetched data
        self.assertTrue(
            _Client.load_min_order_calls,
            "load_pair_min_orders() should have been called with pre-fetched data",
        )
        self.assertTrue(
            _Client.load_min_order_calls[0],
            "load_pair_min_orders() should receive the pre-fetched pairs_data (not None)",
        )
