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

# Reconnection back-off constants
_WS_BACKOFF_INITIAL = 2.0   # seconds before first retry
_WS_BACKOFF_MAX = 60.0      # ceiling for exponential back-off


class RealtimeFeed:
    """Realtime market data provider.

    Attempts WebSocket streaming when available; otherwise falls back to REST polling.

    WebSocket behaviour
    ------------------
    * Automatically reconnects after a disconnect using exponential back-off
      (2 s → 4 s → … → 60 s, reset on successful connect).
    * Messages are **merged** into the cached snapshot per-key so a ticker-only
      push does not wipe out the cached order-book or trade list.
    * An optional *subscribe_message* (raw JSON string) is sent immediately
      after the connection is established – use this to subscribe to the pair
      stream on servers that require an explicit subscription command.
    """

    def __init__(
        self,
        pair: str,
        client: Any,
        *,
        websocket_url: Optional[str] = None,
        poll_interval: float = 1.0,
        websocket_enabled: bool = True,
        subscribe_message: Optional[str] = None,
    ) -> None:
        self.pair = pair
        self.client = client
        self.websocket_url = websocket_url
        self.poll_interval = max(0.2, poll_interval)
        self.websocket_enabled = websocket_enabled
        self.subscribe_message = subscribe_message

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

    # ------------------------------------------------------------------
    # WebSocket helpers
    # ------------------------------------------------------------------

    def _apply_ws_message(self, parsed: Dict[str, Any]) -> None:
        """Merge a parsed WebSocket payload into ``_latest``.

        Only the keys present in *parsed* are updated; other keys in the
        existing snapshot are kept intact.  This prevents a ticker-only push
        from discarding the cached order-book or trade list.
        """
        update: Dict[str, Any] = {}
        for key in ("ticker", "depth", "trades"):
            if key in parsed:
                update[key] = parsed[key]
        if update:
            with self._lock:
                self._latest.update(update)

    def _run_websocket(self) -> None:
        assert websocket is not None  # for type checkers
        assert self.websocket_url

        backoff = _WS_BACKOFF_INITIAL

        while not self._stop.is_set():
            # Each outer iteration represents one connection attempt.
            _connected = threading.Event()

            def _on_open(ws: Any) -> None:
                nonlocal backoff
                backoff = _WS_BACKOFF_INITIAL  # reset on successful connect
                _connected.set()
                logger.info("WebSocket connected for %s", self.pair)
                if self.subscribe_message:
                    try:
                        ws.send(self.subscribe_message)
                        logger.debug("Sent WebSocket subscription: %s", self.subscribe_message)
                    except Exception:
                        logger.debug("Failed to send WebSocket subscription", exc_info=True)

            def _on_message(_: Any, message: str) -> None:
                try:
                    parsed = json.loads(message)
                    if isinstance(parsed, dict):
                        self._apply_ws_message(parsed)
                except Exception:
                    logger.debug("Failed to parse WebSocket message", exc_info=True)

            def _on_error(_: Any, error: Any) -> None:
                logger.debug("WebSocket error: %s", error)

            def _on_close(_: Any, code: Any, msg: Any) -> None:
                logger.info("WebSocket closed for %s (code=%s)", self.pair, code)

            ws = websocket.WebSocketApp(
                self.websocket_url,
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )

            # run_forever blocks until the connection closes or an error occurs.
            wst = threading.Thread(
                target=ws.run_forever,
                kwargs={"ping_interval": 20, "ping_timeout": 10},
                daemon=True,
            )
            wst.start()
            # Wait for the WS thread to finish (disconnect / error).
            # Use a generous timeout so a hung thread doesn't block stop() forever.
            wst.join(timeout=30.0)
            if wst.is_alive():
                logger.debug("WebSocket thread did not finish within timeout; continuing …")

            if self._stop.is_set():
                break  # clean shutdown – do not reconnect

            logger.warning(
                "WebSocket disconnected for %s; reconnecting in %.0fs …",
                self.pair,
                backoff,
            )
            self._stop.wait(backoff)       # interruptible sleep – honours stop()
            backoff = min(backoff * 2, _WS_BACKOFF_MAX)
