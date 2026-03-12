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
