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
