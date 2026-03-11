from __future__ import annotations

import time
from queue import Queue
from threading import Event, Thread
from typing import Any, Callable, Optional, Tuple


class QueuedResult:
    def __init__(self) -> None:
        self._event = Event()
        self._value: Any = None
        self._exc: Optional[BaseException] = None

    def set_result(self, value: Any) -> None:
        self._value = value
        self._event.set()

    def set_exception(self, exc: BaseException) -> None:
        self._exc = exc
        self._event.set()

    def result(self, timeout: Optional[float] = None) -> Any:
        completed = self._event.wait(timeout)
        if not completed:
            raise TimeoutError("Queued task did not complete within timeout")
        if self._exc:
            raise self._exc
        return self._value


class RateLimitedOrderQueue:
    """Simple worker queue enforcing a minimum interval between order requests."""

    def __init__(
        self,
        min_interval: float = 0.25,
        *,
        time_provider: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = max(0.0, float(min_interval))
        self._time = time_provider
        self._sleep = sleeper
        self._queue: "Queue[Optional[Tuple[Callable[..., Any], tuple, dict, QueuedResult]]]" = Queue()
        self._last_exec: Optional[float] = None
        self._stop = Event()
        self._worker: Optional[Thread] = None

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> QueuedResult:
        result = QueuedResult()
        self.start()
        self._queue.put((func, args, kwargs, result))
        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                break
            func, args, kwargs, result = item
            now = self._time()
            if self._last_exec is not None:
                elapsed = now - self._last_exec
                wait_for = self.min_interval - elapsed
                if wait_for > 0:
                    self._sleep(wait_for)
            try:
                value = func(*args, **kwargs)
                result.set_result(value)
            except BaseException as exc:  # pragma: no cover - propagate via result
                result.set_exception(exc)
            self._last_exec = self._time()
