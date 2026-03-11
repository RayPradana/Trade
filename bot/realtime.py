from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, Optional

try:  # optional dependency
    import websocket  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    websocket = None  # type: ignore

logger = logging.getLogger(__name__)


class RealtimeFeed:
    """Realtime market data provider.

    Attempts WebSocket streaming when available; otherwise falls back to REST polling.
    """

    def __init__(
        self,
        pair: str,
        client: Any,
        *,
        websocket_url: Optional[str] = None,
        poll_interval: float = 1.0,
        websocket_enabled: bool = True,
    ) -> None:
        self.pair = pair
        self.client = client
        self.websocket_url = websocket_url
        self.poll_interval = max(0.2, poll_interval)
        self.websocket_enabled = websocket_enabled

        self._latest: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ws_thread: Optional[threading.Thread] = None

    @property
    def has_snapshot(self) -> bool:
        with self._lock:
            return bool(self._latest)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def start(self) -> None:
        if self._ws_thread and self._ws_thread.is_alive():
            return
        self._stop.clear()
        self._ws_thread = threading.Thread(target=self._run, daemon=True)
        self._ws_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=1.0)

    def refresh_once(self) -> Dict[str, Any]:
        """Trigger a single REST refresh (useful for tests)."""
        data = self._pull_rest()
        with self._lock:
            self._latest = data
        return data

    def _run(self) -> None:
        if self.websocket_enabled and websocket and self.websocket_url:
            try:
                self._run_websocket()
                return
            except Exception:  # pragma: no cover - best-effort fallback
                logger.warning("WebSocket stream failed, falling back to REST", exc_info=True)
        self._run_polling()

    def _run_polling(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._latest = self._pull_rest()
            except Exception:
                logger.debug("Realtime polling error", exc_info=True)
            self._stop.wait(self.poll_interval)

    def _pull_rest(self) -> Dict[str, Any]:
        ticker = self.client.get_ticker(self.pair)
        depth = self.client.get_depth(self.pair, count=50)
        trades = self.client.get_trades(self.pair, count=200)
        return {"ticker": ticker, "depth": depth, "trades": trades}

    def _run_websocket(self) -> None:
        assert websocket is not None  # for type checkers
        assert self.websocket_url

        def _on_message(_: Any, message: str) -> None:
            try:
                parsed = json.loads(message)
                if not isinstance(parsed, dict):
                    return
                # Expect payload contains these keys; otherwise ignore.
                if {"ticker", "depth", "trades"} & set(parsed.keys()):
                    with self._lock:
                        self._latest = parsed
            except Exception:
                logger.debug("Failed to parse websocket message", exc_info=True)

        ws = websocket.WebSocketApp(
            self.websocket_url,
            on_message=_on_message,
            on_error=lambda *_: logger.debug("WebSocket error", exc_info=True),
            on_close=lambda *_: logger.info("WebSocket closed for %s", self.pair),
        )
        wst = threading.Thread(target=ws.run_forever, kwargs={"ping_interval": 20}, daemon=True)
        wst.start()
        # simple loop to stop when requested; rely on ws thread to receive data
        while not self._stop.is_set():
            time.sleep(0.2)
        try:
            ws.close()
        except Exception:
            logger.debug("Failed closing websocket", exc_info=True)
