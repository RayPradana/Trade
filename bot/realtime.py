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

# Official Indodax market-data WebSocket endpoint and public static token.
# These are published in the official API docs:
# https://github.com/btcid/indodax-official-api-docs/blob/master/Marketdata-websocket.md
# The token is intentionally public (shared by all clients). It contains an
# exp claim in the far future (~2031) and must NOT be confused with the
# user-specific private WS token obtained from /api/private_ws/v1/generate_token.
INDODAX_WS_URL = "wss://ws3.indodax.com/ws/"
INDODAX_WS_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJleHAiOjE5NDY2MTg0MTV9"
    ".UR1lBM6Eqh0yWz-PVirw1uPCxe60FdchR8eNVdsskeo"
)


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
            # suppress_origin=True removes the Origin header which Indodax WS
            # servers require to be absent (otherwise returns 403 Forbidden).
            wst = threading.Thread(
                target=ws.run_forever,
                kwargs={"ping_interval": 20, "ping_timeout": 10, "suppress_origin": True},
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
        websocket_url: Optional[str] = INDODAX_WS_URL,
        websocket_enabled: bool = True,
        websocket_token: Optional[str] = INDODAX_WS_TOKEN,
        batch_size: int = 100,
        summaries_interval: float = 60.0,
    ) -> None:
        self._pairs = list(pairs)
        self._client = client
        self._websocket_url = websocket_url
        self._websocket_enabled = websocket_enabled
        self._websocket_token = websocket_token
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
        # Timestamp of the last WebSocket message that updated the cache.
        # Used by :meth:`is_ws_stale` to detect a silent disconnect.
        self._last_ws_update: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_ticker(self, pair: str) -> Optional[Dict[str, Any]]:
        """Return the latest cached ticker for *pair*, or ``None``."""
        with self._lock:
            return self._cache.get(pair)

    @property
    def is_seeded(self) -> bool:
        """``True`` once the cache has been populated by at least one successful summaries fetch."""
        with self._lock:
            return bool(self._cache)

    def is_ws_stale(self, threshold_seconds: float = 120.0) -> bool:
        """Return ``True`` when no WS ticker update has been received within *threshold_seconds*.

        Returns ``False`` when the WebSocket feed was never started (REST-only mode).
        """
        with self._lock:
            if self._last_ws_update is None:
                return False  # WS never received data — not considered stale
            return (time.time() - self._last_ws_update) > threshold_seconds

    def _apply_ws_message_for_pair(self, pair: str, ticker: Dict[str, Any]) -> None:
        """Update the cache for *pair* with *ticker* data.

        This is used internally by the WebSocket message handler and exposed
        as a test helper so unit tests can inject ticker data without needing
        a live WebSocket connection.
        """
        with self._lock:
            self._cache[pair] = ticker
            self._last_ws_update = time.time()

    def start(self) -> None:
        """Seed the cache and start background refresh threads."""
        self._stop.clear()
        # Synchronous seed so the cache is warm before the first scan cycle.
        self._update_from_summaries()

        if self._websocket_enabled and websocket and self._websocket_url:
            # Use a single WebSocket connection to the official Indodax market
            # data endpoint.  The Centrifuge protocol used by Indodax covers all
            # pairs with one "market:summary-24h" subscription, so no batching is
            # required.
            t = threading.Thread(
                target=self._run_ws_centrifuge,
                daemon=True,
                name="MultiPairFeed-ws",
            )
            t.start()
            self._threads.append(t)
            # Always run a companion REST-polling thread even when WebSocket is
            # active.  It detects WS stale conditions and keeps the cache fresh
            # when the WS connection is silent for longer than the summaries
            # interval.  The polling interval is deliberately slower than the
            # public API rate limit (180 req/min) to avoid contributing to 429s.
            poll_t = threading.Thread(
                target=self._run_stale_aware_polling,
                daemon=True,
                name="MultiPairFeed-stale-poll",
            )
            poll_t.start()
            self._threads.append(poll_t)
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
        """Fetch /api/summaries and merge tickers into the cache.

        Indodax returns ticker keys *without* underscores (e.g. ``btcidr``).
        The ``_key_to_pair`` lookup maps each such key back to the canonical
        pair name with an underscore (e.g. ``btc_idr``), so every pair present
        in the known-pairs list is stored under its canonical name.
        """
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
        """Periodically refresh ticker cache from /api/summaries.

        When the WebSocket feed is enabled but has gone stale the polling
        interval is halved so the REST fallback keeps the cache fresh.
        """
        while not self._stop.wait(self._summaries_interval):
            self._update_from_summaries()

    def _run_stale_aware_polling(self) -> None:
        """Companion REST poller that activates when the WebSocket feed goes stale.

        Polls ``/api/summaries`` at the normal interval *and* triggers an
        immediate refresh whenever the WS has been silent for longer than
        ``_summaries_interval`` seconds (a sign of a silent disconnect).
        """
        while not self._stop.wait(self._summaries_interval):
            stale = self.is_ws_stale(self._summaries_interval)
            if stale:
                logger.warning(
                    "MultiPairFeed: WS data is stale (no update in ≥%.0fs) — "
                    "refreshing from REST summaries",
                    self._summaries_interval,
                )
            self._update_from_summaries()

    # ------------------------------------------------------------------
    # WebSocket streaming (Indodax Centrifuge protocol)
    # ------------------------------------------------------------------

    def _run_ws_centrifuge(self) -> None:
        """Manage a single WebSocket connection to the Indodax market-data server.

        Protocol (Centrifuge-based, documented at
        https://github.com/btcid/indodax-official-api-docs/blob/master/Marketdata-websocket.md):

        1. Connect to ``wss://ws3.indodax.com/ws/`` with ``suppress_origin=True``.
        2. Authenticate with the static public token:
           ``{"params": {"token": "…"}, "id": 1}``
        3. Subscribe to ``market:summary-24h`` for all-pairs ticker snapshots:
           ``{"method": 1, "params": {"channel": "market:summary-24h"}, "id": 2}``
        4. Parse incoming push messages:
           ``{"result": {"channel": "market:summary-24h",
                         "data": {"data": [["btcidr", ts, last, low, high, prev,
                                             idr_vol, coin_vol], …]}}}``
        """
        assert websocket is not None
        assert self._websocket_url

        token = self._websocket_token or INDODAX_WS_TOKEN
        auth_msg = json.dumps({"params": {"token": token}, "id": 1})
        subscribe_msg = json.dumps(
            {"method": 1, "params": {"channel": "market:summary-24h"}, "id": 2}
        )
        backoff = _WS_BACKOFF_INITIAL

        while not self._stop.is_set():
            _authenticated = threading.Event()

            def _on_open(ws: Any) -> None:
                nonlocal backoff
                backoff = _WS_BACKOFF_INITIAL
                logger.info("MultiPairFeed WS connected to %s", self._websocket_url)
                try:
                    ws.send(auth_msg)
                except Exception:
                    logger.debug("MultiPairFeed WS auth send failed", exc_info=True)

            def _on_message(_ws: Any, message: str) -> None:
                try:
                    parsed = json.loads(message)
                    if not isinstance(parsed, dict):
                        return

                    # ── Auth response: {"id": 1, "result": {"client": …}} ────
                    if parsed.get("id") == 1 and "result" in parsed:
                        _authenticated.set()
                        try:
                            _ws.send(subscribe_msg)
                            logger.debug(
                                "MultiPairFeed WS subscribed to market:summary-24h"
                            )
                        except Exception:
                            logger.debug(
                                "MultiPairFeed WS subscribe send failed", exc_info=True
                            )
                        return

                    # ── Push: {"result": {"channel": "market:summary-24h",
                    #                      "data": {"data": [[pair, ts, last, …], …]}}}
                    result = parsed.get("result") or parsed.get("push", {}).get("pub", {})
                    if not isinstance(result, dict):
                        return
                    channel = result.get("channel", "")
                    data_wrapper = result.get("data", {})
                    if not isinstance(data_wrapper, dict):
                        return
                    rows = data_wrapper.get("data", [])
                    if not isinstance(rows, list):
                        return

                    if "market:summary-24h" in channel:
                        self._apply_summary_rows(rows)
                except Exception:
                    logger.debug("MultiPairFeed WS message parse error", exc_info=True)

            def _on_error(_: Any, error: Any) -> None:
                logger.debug("MultiPairFeed WS error: %s", error)

            def _on_close(_: Any, code: Any, msg: Any) -> None:
                logger.info("MultiPairFeed WS closed (code=%s)", code)

            ws_app = websocket.WebSocketApp(
                self._websocket_url,
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            # suppress_origin=True is required by Indodax – without it the server
            # returns 403 Forbidden due to Origin header mismatch.
            wst = threading.Thread(
                target=ws_app.run_forever,
                kwargs={"ping_interval": 20, "ping_timeout": 10, "suppress_origin": True},
                daemon=True,
            )
            wst.start()
            wst.join(timeout=30.0)
            if wst.is_alive():
                logger.debug("MultiPairFeed WS thread did not finish within timeout")

            if self._stop.is_set():
                break

            logger.warning(
                "MultiPairFeed WS disconnected; reconnecting in %.0fs …", backoff
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2, _WS_BACKOFF_MAX)

    def _apply_summary_rows(self, rows: List[Any]) -> None:
        """Update the ticker cache from ``market:summary-24h`` push data.

        Each row is an array:
        ``[pair_key, timestamp, last, low, high, prev_price, idr_volume, coin_volume]``
        where *pair_key* is the pair without underscore (e.g. ``"btcidr"``).
        """
        updates: Dict[str, Any] = {}
        for row in rows:
            if not isinstance(row, list) or len(row) < 8:
                continue
            try:
                raw_key = str(row[0]).lower().replace("_", "")
                pair_name = self._key_to_pair.get(raw_key)
                if pair_name is None:
                    continue
                updates[pair_name] = {
                    "last": str(row[2]),
                    "low": str(row[3]),
                    "high": str(row[4]),
                    "vol_idr": str(row[6]),
                    "server_time": int(row[1]),
                }
            except (IndexError, TypeError, ValueError):
                continue
        if updates:
            with self._lock:
                self._cache.update(updates)
            logger.debug(
                "MultiPairFeed WS: updated %d tickers from market:summary-24h",
                len(updates),
            )



# ---------------------------------------------------------------------------
# Private WebSocket feed (order updates)
# ---------------------------------------------------------------------------

_PRIVATE_WS_URL = "wss://pws.indodax.com/ws/?cf_ws_frame_ping_pong=true"


class PrivateFeed:
    """Real-time private order-update feed using the Indodax Private WebSocket.

    Connects to ``wss://pws.indodax.com/ws/`` and subscribes to the user's
    private channel to receive order-fill and status-change events.

    Usage::

        def on_order(event: dict) -> None:
            print(event)  # {"eventType": "order_update", "order": {...}}

        feed = PrivateFeed(client=trader.client, on_order_update=on_order)
        feed.start()   # non-blocking; runs in a daemon thread
        …
        feed.stop()

    The *on_order_update* callback is called from the WS reader thread with
    each ``order_update`` event dict from the ``pub.data`` list.  Keep it
    fast and non-blocking.
    """

    def __init__(
        self,
        client: Any,
        on_order_update: Optional[Any] = None,
        *,
        ws_url: str = _PRIVATE_WS_URL,
    ) -> None:
        self._client = client
        self._on_order_update = on_order_update
        self._ws_url = ws_url
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the private feed in a background daemon thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="PrivateFeed-ws",
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        backoff = _WS_BACKOFF_INITIAL
        while not self._stop.is_set():
            try:
                token_info = self._client.generate_private_ws_token()
                conn_token = token_info["connToken"]
                channel = token_info["channel"]
            except Exception as exc:
                logger.warning("PrivateFeed: could not obtain WS token: %s", exc)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, _WS_BACKOFF_MAX)
                continue

            backoff = _WS_BACKOFF_INITIAL  # reset on successful token fetch
            self._connect_once(conn_token, channel)

            if self._stop.is_set():
                break
            logger.warning(
                "PrivateFeed: disconnected; reconnecting in %.0fs …", backoff
            )
            self._stop.wait(backoff)
            backoff = min(backoff * 2, _WS_BACKOFF_MAX)

    def _connect_once(self, conn_token: str, channel: str) -> None:
        """Open one WS connection, authenticate, subscribe and block until close."""
        if websocket is None:
            logger.warning("PrivateFeed: websocket-client not installed; cannot connect")
            return

        auth_msg = json.dumps({"connect": {"token": conn_token}, "id": 1})
        subscribe_msg = json.dumps({"subscribe": {"channel": channel}, "id": 2})

        def _on_open(ws: Any) -> None:
            logger.info("PrivateFeed: WS connected")
            try:
                ws.send(auth_msg)
            except Exception:
                logger.debug("PrivateFeed: auth send failed", exc_info=True)

        def _on_message(_ws: Any, message: str) -> None:
            try:
                parsed = json.loads(message)
                if not isinstance(parsed, dict):
                    return

                # Auth confirmation → send subscribe
                if parsed.get("id") == 1 and "connect" in parsed:
                    try:
                        _ws.send(subscribe_msg)
                        logger.debug("PrivateFeed: subscribed to %s", channel)
                    except Exception:
                        logger.debug("PrivateFeed: subscribe send failed", exc_info=True)
                    return

                # Order update push
                # Format: {"push":{"channel":"pws:#...","pub":{"data":[{"eventType":"order_update","order":{...}}]}}}
                push = parsed.get("push", {})
                pub = push.get("pub", {})
                data_list = pub.get("data", [])
                if not isinstance(data_list, list):
                    return
                for event in data_list:
                    if isinstance(event, dict) and event.get("eventType") == "order_update":
                        if self._on_order_update:
                            try:
                                self._on_order_update(event)
                            except Exception:
                                logger.debug(
                                    "PrivateFeed: on_order_update callback error",
                                    exc_info=True,
                                )
            except Exception:
                logger.debug("PrivateFeed: message parse error", exc_info=True)

        def _on_error(_: Any, error: Any) -> None:
            logger.debug("PrivateFeed: WS error: %s", error)

        def _on_close(_: Any, code: Any, msg: Any) -> None:
            logger.info("PrivateFeed: WS closed (code=%s)", code)

        ws_app = websocket.WebSocketApp(
            self._ws_url,
            on_open=_on_open,
            on_message=_on_message,
            on_error=_on_error,
            on_close=_on_close,
        )
        wst = threading.Thread(
            target=ws_app.run_forever,
            kwargs={"ping_interval": 20, "ping_timeout": 10, "suppress_origin": True},
            daemon=True,
        )
        wst.start()
        wst.join(timeout=30.0)
        if wst.is_alive():
            logger.debug("PrivateFeed: WS thread did not finish within timeout")
