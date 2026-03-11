from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

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


class MultiPairFeed:
    """Persistent multi-pair ticker cache for scan loops.

    Maintains an in-memory ticker cache keyed by *pair name* (e.g. ``btc_idr``)
    so that :meth:`get_ticker` can serve any pair without an individual REST
    call.

    Data sources (in priority order)
    ---------------------------------
    1. **WebSocket batch streams** – when *websocket_enabled* is ``True``,
       *websocket_url* is set, and the ``websocket-client`` package is
       installed, pairs are split into batches of at most *batch_size* and one
       WebSocket connection is opened per batch.  Each connection sends a
       single ``{"action": "subscribe", "channel": [pairs …]}`` message.
    2. **REST summaries polling** – when WebSocket is disabled or unavailable
       a background thread calls ``/api/summaries`` every *summaries_interval*
       seconds.

    In both cases the feed performs an **initial synchronous summaries fetch**
    when :meth:`start` is called so that the cache is seeded immediately
    (before any background thread has a chance to run).

    Key normalisation
    -----------------
    Indodax ``/api/summaries`` returns ticker keys **without** underscores
    (e.g. ``btcidr``).  The feed maps those keys back to the canonical pair
    name (``btc_idr``) using the list of pairs supplied at construction time,
    so any pair present in the list is always stored under its canonical name
    regardless of how the API spells the key.
    """

    def __init__(
        self,
        pairs: List[str],
        client: Any,
        *,
        websocket_url: Optional[str] = None,
        websocket_enabled: bool = True,
        batch_size: int = 100,
        summaries_interval: float = 60.0,
    ) -> None:
        self._pairs = list(pairs)
        self._client = client
        self._websocket_url = websocket_url
        self._websocket_enabled = websocket_enabled
        self._batch_size = max(1, batch_size)
        self._summaries_interval = max(10.0, summaries_interval)

        # Ticker cache: canonical pair name → raw ticker dict
        self._cache: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

        # Pre-build normalised-key → pair lookup for O(1) reverse mapping.
        # e.g. "btcidr" → "btc_idr"
        self._key_to_pair: Dict[str, str] = {
            p.replace("_", ""): p for p in self._pairs
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ticker(self, pair: str) -> Optional[Dict[str, Any]]:
        """Return the latest cached ticker for *pair*, or ``None``."""
        with self._lock:
            return self._cache.get(pair)

    def start(self) -> None:
        """Seed the cache and start background refresh threads."""
        self._stop.clear()
        # Synchronous seed so the cache is warm before the first scan cycle.
        self._update_from_summaries()

        if self._websocket_enabled and websocket and self._websocket_url:
            for i in range(0, len(self._pairs), self._batch_size):
                batch = self._pairs[i : i + self._batch_size]
                t = threading.Thread(
                    target=self._run_ws_batch,
                    args=(batch,),
                    daemon=True,
                    name=f"MultiPairFeed-ws-{i // self._batch_size}",
                )
                t.start()
                self._threads.append(t)
        else:
            t = threading.Thread(
                target=self._run_summaries_polling,
                daemon=True,
                name="MultiPairFeed-poll",
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        """Stop all background threads."""
        self._stop.set()
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)
        self._threads.clear()

    # ------------------------------------------------------------------
    # REST summaries polling
    # ------------------------------------------------------------------

    def _update_from_summaries(self) -> None:
        """Fetch /api/summaries and merge tickers into the cache."""
        try:
            summaries = self._client.get_summaries()
            raw = summaries.get("tickers", {}) if isinstance(summaries, dict) else {}
            updates: Dict[str, Any] = {}
            for key, data in raw.items():
                # Map by normalised key (e.g. "btcidr" → "btc_idr")
                pair_name = self._key_to_pair.get(str(key).lower().replace("_", ""))
                if pair_name:
                    updates[pair_name] = data
            if updates:
                with self._lock:
                    self._cache.update(updates)
                logger.debug("MultiPairFeed: cached %d tickers from summaries", len(updates))
        except Exception:
            logger.debug("MultiPairFeed: summaries fetch failed", exc_info=True)

    def _run_summaries_polling(self) -> None:
        """Periodically refresh ticker cache from /api/summaries (skip initial – already done in start)."""
        while not self._stop.wait(self._summaries_interval):
            self._update_from_summaries()

    # ------------------------------------------------------------------
    # WebSocket batch streaming
    # ------------------------------------------------------------------

    def _run_ws_batch(self, batch: List[str]) -> None:
        """Manage a single WebSocket connection covering *batch* pairs."""
        assert websocket is not None
        assert self._websocket_url

        subscribe_msg = json.dumps({"action": "subscribe", "channel": batch})
        backoff = _WS_BACKOFF_INITIAL

        while not self._stop.is_set():
            def _on_open(ws: Any) -> None:
                nonlocal backoff
                backoff = _WS_BACKOFF_INITIAL
                logger.info(
                    "MultiPairFeed WS connected (batch of %d, first=%s)",
                    len(batch),
                    batch[0],
                )
                try:
                    ws.send(subscribe_msg)
                    logger.debug(
                        "MultiPairFeed WS subscribed: %s … (+%d more)",
                        batch[0],
                        len(batch) - 1,
                    )
                except Exception:
                    logger.debug("MultiPairFeed WS subscribe failed", exc_info=True)

            def _on_message(_: Any, message: str) -> None:
                try:
                    parsed = json.loads(message)
                    if not isinstance(parsed, dict):
                        return
                    # Accept common Indodax WebSocket message shapes:
                    #   {"pair": "btc_idr", "ticker": {"last": "100", …}}
                    #   {"channel": "btcidr", "data": {"last": "100", …}}
                    raw_pair = (
                        parsed.get("pair")
                        or parsed.get("channel")
                        or parsed.get("symbol")
                    )
                    if not raw_pair:
                        return
                    # Normalise to canonical pair name
                    normalised = str(raw_pair).lower().replace("-", "_")
                    pair_name = self._key_to_pair.get(
                        normalised.replace("_", ""), normalised
                    )
                    ticker_data = (
                        parsed.get("ticker")
                        or parsed.get("data")
                    )
                    if ticker_data and isinstance(ticker_data, dict):
                        with self._lock:
                            self._cache[pair_name] = ticker_data
                except Exception:
                    logger.debug("MultiPairFeed WS message parse error", exc_info=True)

            def _on_error(_: Any, error: Any) -> None:
                logger.debug(
                    "MultiPairFeed WS error (batch first=%s): %s", batch[0], error
                )

            def _on_close(_: Any, code: Any, msg: Any) -> None:
                logger.info(
                    "MultiPairFeed WS closed (batch first=%s, code=%s)", batch[0], code
                )

            ws_app = websocket.WebSocketApp(
                self._websocket_url,
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            wst = threading.Thread(
                target=ws_app.run_forever,
                kwargs={"ping_interval": 20, "ping_timeout": 10},
                daemon=True,
            )
            wst.start()
            wst.join(timeout=30.0)
            if wst.is_alive():
                logger.debug(
                    "MultiPairFeed WS thread did not finish within timeout (batch first=%s)",
                    batch[0],
                )

            if self._stop.is_set():
                break

            logger.warning(
                "MultiPairFeed WS disconnected (batch first=%s); reconnecting in %.0fs …",
                batch[0],
                backoff,
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2, _WS_BACKOFF_MAX)
