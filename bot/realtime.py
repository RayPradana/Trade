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
        # Rolling trade buffer for WS-sourced trades (newest first, max 200).
        # Persists across reconnects so analysis always has recent history.
        self._ws_trades_buf: List[Dict[str, Any]] = []
        self._ws_trades_lock = threading.Lock()

    @property
    def has_snapshot(self) -> bool:
        with self._lock:
            return bool(self._latest)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def start(self) -> None:
        with self._lock:
            if self._ws_thread and self._ws_thread.is_alive():
                return
            self._stop.clear()
            self._ws_thread = threading.Thread(target=self._run, daemon=True)
            self._ws_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._ws_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

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

    def _apply_orderbook(self, data: Dict[str, Any]) -> None:
        """Normalize a ``market:order-book-{pair}`` push into the REST depth format.

        The WS push contains a ``data`` wrapper with ``bid`` (buyers) and ``ask``
        (sellers) lists.  Each entry is a dict with ``price``, ``idr_volume``, and
        a coin-specific volume field (e.g. ``btc_volume``).

        The REST format expected by :func:`~bot.analysis.analyze_orderbook` is::

            {"buy": [["price", "coin_vol"], …], "sell": [["price", "coin_vol"], …]}
        """
        if not isinstance(data, dict):
            return
        inner = data.get("data", data)
        if not isinstance(inner, dict):
            return

        # Derive the coin name from the pair: "btc_idr" → "btc"
        coin = self.pair.split("_")[0] if "_" in self.pair else self.pair[:-3]
        vol_key = f"{coin}_volume"

        def _convert(orders: Any) -> List[List[str]]:
            result: List[List[str]] = []
            if not isinstance(orders, list):
                return result
            for o in orders:
                if not isinstance(o, dict):
                    continue
                price = str(o.get("price", "0"))
                # Try the coin-specific volume key first, then any remaining key
                # that is not "price" or "idr_volume" (covers edge cases).
                vol = str(
                    o.get(vol_key)
                    or next(
                        (v for k, v in o.items() if k not in ("price", "idr_volume")),
                        "0",
                    )
                )
                result.append([price, vol])
            return result

        buy_orders = _convert(inner.get("bid", []))
        sell_orders = _convert(inner.get("ask", []))

        with self._lock:
            self._latest["depth"] = {"buy": buy_orders, "sell": sell_orders}

    def _apply_trade_activity(self, data: Dict[str, Any]) -> None:
        """Accumulate ``market:trade-activity-{pair}`` push events into the trade buffer.

        Each row in the push payload is an array::

            [pair, timestamp, sequence, side, price, idr_volume, coin_volume]

        Rows are normalized to the REST trades format::

            {"date": "…", "price": "…", "amount": "…", "type": "buy"|"sell"}

        and prepended to a rolling buffer (newest first, max 200 entries) that
        is stored in ``_latest["trades"]``.
        """
        rows = data.get("data", [])
        if not isinstance(rows, list):
            return

        new_trades: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 7:
                continue
            try:
                new_trades.append(
                    {
                        "date": str(int(row[1])),
                        "price": str(row[4]),
                        "amount": str(row[6]),
                        "type": str(row[3]),  # "buy" or "sell"
                    }
                )
            except (IndexError, TypeError, ValueError):
                continue

        if not new_trades:
            return

        with self._ws_trades_lock:
            # Prepend newest trades (newest first to match REST API order)
            self._ws_trades_buf = new_trades + self._ws_trades_buf
            # Keep only the most recent 200 trades
            if len(self._ws_trades_buf) > 200:
                self._ws_trades_buf = self._ws_trades_buf[:200]
            trades_snapshot = list(self._ws_trades_buf)

        with self._lock:
            self._latest["trades"] = trades_snapshot

    def _run_websocket(self) -> None:
        """Connect to Indodax market-data WS and subscribe to per-pair channels.

        Uses the official Centrifuge-based protocol documented at:
        https://github.com/btcid/indodax-official-api-docs/blob/master/Marketdata-websocket.md

        Subscribes to:
        * ``market:order-book-{pair}``   – real-time orderbook depth
        * ``market:trade-activity-{pair}`` – real-time trade executions

        If *subscribe_message* is set (legacy / custom WS endpoint), falls back
        to the old generic behaviour of sending that message verbatim.
        """
        assert websocket is not None
        assert self.websocket_url

        pair_nodash = self.pair.replace("_", "")
        ob_channel = f"market:order-book-{pair_nodash}"
        ta_channel = f"market:trade-activity-{pair_nodash}"

        auth_msg = json.dumps({"params": {"token": INDODAX_WS_TOKEN}, "id": 1})
        subscribe_msgs = [
            json.dumps({"method": 1, "params": {"channel": ob_channel}, "id": 2}),
            json.dumps({"method": 1, "params": {"channel": ta_channel}, "id": 3}),
        ]

        backoff = _WS_BACKOFF_INITIAL

        while not self._stop.is_set():
            _authenticated = threading.Event()

            def _on_open(ws: Any) -> None:
                nonlocal backoff
                backoff = _WS_BACKOFF_INITIAL
                logger.info("RealtimeFeed WS connected for %s", self.pair)
                try:
                    if self.subscribe_message:
                        # Legacy / custom endpoint: send the configured message verbatim.
                        ws.send(self.subscribe_message)
                        logger.debug("Sent custom subscribe message for %s", self.pair)
                    else:
                        # Official Indodax Centrifuge protocol: authenticate first.
                        ws.send(auth_msg)
                        logger.debug("RealtimeFeed WS auth sent for %s", self.pair)
                except Exception:
                    logger.debug("RealtimeFeed WS open handler failed", exc_info=True)

            def _on_message(_ws: Any, message: str) -> None:
                try:
                    parsed = json.loads(message)
                    if not isinstance(parsed, dict):
                        return

                    if self.subscribe_message:
                        # Legacy path: apply message as-is
                        self._apply_ws_message(parsed)
                        return

                    # ── Auth response: {"id": 1, "result": {…}} ────────────
                    if parsed.get("id") == 1 and "result" in parsed:
                        _authenticated.set()
                        for sub_msg in subscribe_msgs:
                            try:
                                _ws.send(sub_msg)
                            except Exception:
                                logger.debug(
                                    "RealtimeFeed WS subscribe send failed", exc_info=True
                                )
                        logger.debug(
                            "RealtimeFeed WS subscribed to %s and %s",
                            ob_channel,
                            ta_channel,
                        )
                        return

                    # ── Channel push ────────────────────────────────────────
                    result = (
                        parsed.get("result")
                        or parsed.get("push", {}).get("pub", {})
                    )
                    if not isinstance(result, dict):
                        return
                    channel = result.get("channel", "")
                    data = result.get("data", {})
                    if not isinstance(data, dict):
                        return

                    if channel == ob_channel:
                        self._apply_orderbook(data)
                    elif channel == ta_channel:
                        self._apply_trade_activity(data)

                except Exception:
                    logger.debug("RealtimeFeed WS message error", exc_info=True)

            def _on_error(_: Any, error: Any) -> None:
                logger.debug("RealtimeFeed WS error for %s: %s", self.pair, error)

            def _on_close(_: Any, code: Any, msg: Any) -> None:
                logger.info("RealtimeFeed WS closed for %s (code=%s)", self.pair, code)

            ws_app = websocket.WebSocketApp(
                self.websocket_url,
                on_open=_on_open,
                on_message=_on_message,
                on_error=_on_error,
                on_close=_on_close,
            )
            # suppress_origin=True is required by Indodax – the server returns
            # 403 Forbidden when an Origin header is present.
            wst = threading.Thread(
                target=ws_app.run_forever,
                kwargs={"ping_interval": 20, "ping_timeout": 10, "suppress_origin": True},
                daemon=True,
            )
            wst.start()
            wst.join(timeout=30.0)
            if wst.is_alive():
                logger.debug("RealtimeFeed WS thread did not finish within timeout")

            if self._stop.is_set():
                break

            logger.warning(
                "RealtimeFeed WS disconnected for %s; reconnecting in %.0fs …",
                self.pair,
                backoff,
            )
            self._stop.wait(backoff)
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

        # ── Per-pair real-time orderbook + trades via WS ──────────────────
        # Populated by market:order-book-{pair} and market:trade-activity-{pair}
        # channels subscribed for the active watchlist.  Keyed by canonical
        # pair name (e.g. "btc_idr").
        self._depth_cache: Dict[str, Dict[str, Any]] = {}
        # Rolling trade buffer per pair, newest first, max 2000 entries.
        # Large enough to build 20+ candles for technical indicators without
        # any REST OHLC call.
        self._trades_buf: Dict[str, List[Dict[str, Any]]] = {}
        # Pairs currently subscribed to depth/trades channels.
        self._depth_pairs: List[str] = []
        # Reference to the currently active WebSocket app so that
        # subscribe_depth_pairs() can send subscription messages at any time.
        self._ws_active: Optional[Any] = None
        self._ws_active_lock = threading.Lock()
        # Monotonically increasing id counter for subscription messages.
        # Starts at 100 to avoid conflicts with auth (id=1) and summary (id=2).
        self._sub_id_counter: int = 100

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

    def get_depth(self, pair: str) -> Optional[Dict[str, Any]]:
        """Return the latest real-time orderbook for *pair*, or ``None``.

        Populated by the ``market:order-book-{pair}`` WebSocket channel.
        Returns ``None`` when the channel has not yet delivered data for this pair.
        Returns a shallow copy so callers cannot mutate the internal cache.
        """
        with self._lock:
            depth = self._depth_cache.get(pair)
            if depth is None:
                return None
            return dict(depth)

    def get_trades(self, pair: str) -> Optional[List[Dict[str, Any]]]:
        """Return the rolling real-time trade buffer for *pair*, or ``None``.

        Populated by the ``market:trade-activity-{pair}`` WebSocket channel.
        Newest trades are first; up to 2000 entries per pair.
        Returns ``None`` when no WS trade data is available yet for this pair.
        """
        with self._lock:
            buf = self._trades_buf.get(pair)
            return list(buf) if buf else None

    def subscribe_depth_pairs(self, pairs: List[str]) -> None:
        """Subscribe to real-time orderbook + trades channels for *pairs*.

        Sends ``market:order-book-{pair}`` and ``market:trade-activity-{pair}``
        subscription messages over the active WebSocket connection for each
        pair not yet subscribed.  New subscriptions are remembered so they are
        automatically re-sent after any reconnect.

        Safe to call from any thread at any time (even before the WS connects).
        """
        with self._lock:
            new_pairs = [p for p in pairs if p not in self._depth_pairs]
            self._depth_pairs = list(dict.fromkeys(self._depth_pairs + new_pairs))

        if not new_pairs:
            return

        with self._ws_active_lock:
            ws = self._ws_active
        if ws is not None:
            self._send_depth_subscriptions(ws, new_pairs)

    def _send_depth_subscriptions(self, ws: Any, pairs: List[str]) -> None:
        """Send WS subscribe messages for *pairs* orderbook + trades channels."""
        for pair in pairs:
            pair_nodash = pair.replace("_", "")
            for channel_type in ("order-book", "trade-activity"):
                channel = f"market:{channel_type}-{pair_nodash}"
                with self._lock:
                    sub_id = self._sub_id_counter
                    self._sub_id_counter += 1
                msg = json.dumps(
                    {"method": 1, "params": {"channel": channel}, "id": sub_id}
                )
                try:
                    ws.send(msg)
                    logger.debug("MultiPairFeed: subscribed to %s", channel)
                except Exception:
                    logger.debug(
                        "MultiPairFeed: failed to subscribe to %s",
                        channel,
                        exc_info=True,
                    )

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
                with self._ws_active_lock:
                    self._ws_active = ws
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
                        # Re-subscribe to all watchlist depth/trades channels on
                        # (re-)connect so real-time orderbook and trade data resume
                        # immediately after any disconnect.
                        with self._lock:
                            depth_pairs = list(self._depth_pairs)
                        if depth_pairs:
                            self._send_depth_subscriptions(_ws, depth_pairs)
                        return

                    # ── Push: {"result": {"channel": "…", "data": {…}}} ─────
                    result = parsed.get("result") or parsed.get("push", {}).get("pub", {})
                    if not isinstance(result, dict):
                        return
                    channel = result.get("channel", "")
                    data_wrapper = result.get("data", {})
                    if not isinstance(data_wrapper, dict):
                        return

                    if "market:summary-24h" in channel:
                        rows = data_wrapper.get("data", [])
                        if isinstance(rows, list):
                            self._apply_summary_rows(rows)

                    elif "market:order-book-" in channel:
                        pair_nodash = channel.split("market:order-book-", 1)[-1]
                        pair_name = self._key_to_pair.get(pair_nodash.lower())
                        if pair_name:
                            self._apply_orderbook_for_pair(pair_name, data_wrapper)

                    elif "market:trade-activity-" in channel:
                        pair_nodash = channel.split("market:trade-activity-", 1)[-1]
                        pair_name = self._key_to_pair.get(pair_nodash.lower())
                        if pair_name:
                            self._apply_trade_activity_for_pair(pair_name, data_wrapper)

                except Exception:
                    logger.debug("MultiPairFeed WS message parse error", exc_info=True)

            def _on_error(_: Any, error: Any) -> None:
                logger.debug("MultiPairFeed WS error: %s", error)

            def _on_close(_: Any, code: Any, msg: Any) -> None:
                logger.info("MultiPairFeed WS closed (code=%s)", code)
                with self._ws_active_lock:
                    self._ws_active = None

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

    def _apply_orderbook_for_pair(self, pair: str, data: Dict[str, Any]) -> None:
        """Normalize a ``market:order-book-{pair}`` push into the REST depth format.

        The WS push contains a ``data`` wrapper with ``bid`` (buyers) and ``ask``
        (sellers) lists.  Each entry is a dict with ``price``, ``idr_volume``, and
        a coin-specific volume field (e.g. ``btc_volume``).

        The REST format expected by :func:`~bot.analysis.analyze_orderbook` is::

            {"buy": [["price", "coin_vol"], …], "sell": [["price", "coin_vol"], …]}
        """
        if not isinstance(data, dict):
            return
        inner = data.get("data", data)
        if not isinstance(inner, dict):
            return

        # Derive the coin name from the pair: "btc_idr" → "btc"
        coin = pair.split("_")[0] if "_" in pair else pair[:-3]
        vol_key = f"{coin}_volume"

        def _convert(orders: Any) -> List[List[str]]:
            result: List[List[str]] = []
            if not isinstance(orders, list):
                return result
            for o in orders:
                if not isinstance(o, dict):
                    continue
                price = str(o.get("price", "0"))
                vol = str(
                    o.get(vol_key)
                    or next(
                        (v for k, v in o.items() if k not in ("price", "idr_volume")),
                        "0",
                    )
                )
                result.append([price, vol])
            return result

        buy_orders = _convert(inner.get("bid", []))
        sell_orders = _convert(inner.get("ask", []))

        with self._lock:
            self._depth_cache[pair] = {"buy": buy_orders, "sell": sell_orders}
            self._last_ws_update = time.time()

    def _apply_trade_activity_for_pair(self, pair: str, data: Dict[str, Any]) -> None:
        """Accumulate ``market:trade-activity-{pair}`` push events into the per-pair buffer.

        Each row in the push payload is an array::

            [pair, timestamp, sequence, side, price, idr_volume, coin_volume]

        Rows are normalized to the REST trades format::

            {"date": "…", "price": "…", "amount": "…", "type": "buy"|"sell"}

        and prepended to a rolling buffer (newest first, max 2000 entries).
        The larger buffer (vs the 200-entry per-pair RealtimeFeed buffer) ensures
        enough trade history to build 20+ candles for technical indicators without
        any REST call.
        """
        rows = data.get("data", [])
        if not isinstance(rows, list):
            return

        new_trades: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, list) or len(row) < 7:
                continue
            try:
                new_trades.append(
                    {
                        "date": str(int(row[1])),
                        "price": str(row[4]),
                        "amount": str(row[6]),
                        "type": str(row[3]),  # "buy" or "sell"
                    }
                )
            except (IndexError, TypeError, ValueError):
                continue

        if not new_trades:
            return

        with self._lock:
            buf = self._trades_buf.get(pair, [])
            buf = new_trades + buf
            if len(buf) > 2000:
                buf = buf[:2000]
            self._trades_buf[pair] = buf
            self._last_ws_update = time.time()



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
