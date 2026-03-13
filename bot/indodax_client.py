from __future__ import annotations

import hashlib
import hmac
import logging
import re
import threading
import time
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional, Tuple

import requests

from .rate_limit import ApiRequestScheduler, RateLimitedOrderQueue
from .analysis import _TF_SECONDS as _OHLC_TF_SECONDS  # shared timeframe → seconds map

logger = logging.getLogger(__name__)


class IndodaxClient:
    """Lightweight Indodax API wrapper supporting public and private endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        session: Optional[requests.Session] = None,
        base_url: str = "https://indodax.com",
        timeout: int = 15,
        order_queue: Optional[RateLimitedOrderQueue] = None,
        order_min_interval: float = 0.25,
        enable_queue: bool = True,
        *,
        public_min_interval: float = 0.15,
        request_scheduler: Optional[ApiRequestScheduler] = None,
        request_min_interval: float = 0.2,
        enable_request_scheduler: bool = True,
        public_time_provider=None,
        public_sleeper=None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.order_queue = (
            order_queue
            if enable_queue
            else None
        )
        if self.order_queue is None and enable_queue:
            self.order_queue = RateLimitedOrderQueue(min_interval=order_min_interval)
        # Generic REST scheduler (public + private) to smooth out bursts
        self._request_scheduler = request_scheduler if (enable_request_scheduler and request_scheduler) else None
        if self._request_scheduler is None and enable_request_scheduler:
            self._request_scheduler = ApiRequestScheduler(min_interval=request_min_interval)
        # Public REST rate-limit (serialize calls across threads).
        self.public_min_interval = max(0.0, public_min_interval)
        self._public_time = public_time_provider or time.monotonic
        self._public_sleep = public_sleeper or time.sleep
        self._public_lock = threading.Lock()
        self._last_public_request: float = 0.0
        # Per-pair minimum order cache:
        #   keys are pair names (e.g. "btc_idr")
        #   values are dicts with "min_coin" (min base currency amount)
        #   and "min_idr" (min IDR value).
        self._pair_min_order: Dict[str, Dict[str, float]] = {}
        # TTL for the pair-minimum-order cache.  Defaults to 3600 s (1 hour).
        # Indodax rarely changes minimum order requirements, but a TTL ensures
        # the cache is eventually refreshed when the bot runs for multiple days.
        self._pair_min_order_cache_ttl: float = 3600.0
        self._pair_min_order_expires: float = 0.0
        # Price increment cache (tick size per pair) from /api/price_increments.
        # keys are pair names (e.g. "btc_idr"), values are Decimal-compatible strings.
        self._price_increments: Dict[str, str] = {}
        self._price_increments_ttl: float = 3600.0
        self._price_increments_expires: float = 0.0
        # ── Private API response caches ───────────────────────────────────────
        # TTL-based in-memory caches for expensive private REST endpoints.
        # Each entry is (cached_value, expiry_timestamp).
        self._account_info_cache_ttl: float = 30.0
        self._account_info_cached: Optional[Dict[str, Any]] = None
        self._account_info_expires: float = 0.0
        self._open_orders_cache_ttl: float = 15.0
        # per-pair cache: {pair: (data, expiry)}
        self._open_orders_cache: Dict[str, tuple] = {}

    def configure_caches(
        self,
        account_info_ttl: float = 30.0,
        open_orders_ttl: float = 15.0,
    ) -> None:
        """Configure TTL (seconds) for private API response caches.

        Call after construction to override the defaults.  Pass ``0`` to
        disable a specific cache (always fetch live data).

        :param account_info_ttl: TTL for :meth:`get_account_info` responses.
        :param open_orders_ttl:  TTL for :meth:`open_orders` responses per pair.
        """
        self._account_info_cache_ttl = max(0.0, account_info_ttl)
        self._open_orders_cache_ttl = max(0.0, open_orders_ttl)

    def invalidate_account_info_cache(self) -> None:
        """Force the next :meth:`get_account_info` call to fetch live data."""
        self._account_info_expires = 0.0
        self._account_info_cached = None

    def invalidate_open_orders_cache(self, pair: Optional[str] = None) -> None:
        """Force the next :meth:`open_orders` call to fetch live data.

        :param pair: Invalidate only this pair's cache.  Pass ``None`` to
                     invalidate all cached open-order responses.
        """
        if pair is None:
            self._open_orders_cache.clear()
        else:
            self._open_orders_cache.pop(pair, None)

    # -------------------- public API -------------------- #
    def get_pairs(self) -> List[Dict[str, Any]]:
        return self._get("/api/pairs")

    def load_pair_min_orders(self, pairs_info: Optional[List[Dict[str, Any]]] = None) -> None:
        """Fetch exchange pair list and cache per-pair minimum order sizes.

        The Indodax ``/api/pairs`` endpoint returns a list of pair objects,
        each containing ``trade_min_base_currency`` (minimum coin amount) and
        ``trade_min_traded_currency`` (minimum IDR amount).  This method
        populates :attr:`_pair_min_order` for quick lookup before order
        placement so that "Minimum order" errors are prevented proactively.

        :param pairs_info: Optional pre-fetched ``/api/pairs`` response data.
            When provided the cache is populated from this list *without*
            making an additional REST call.  Pass the data that was already
            fetched for another purpose (e.g. building the watchlist) to avoid
            a duplicate ``/api/pairs`` request and reduce 429 risk.

        Safe to call multiple times; the cache is fully replaced on each call.
        """
        if pairs_info is None:
            try:
                pairs_info = self.get_pairs()
            except Exception as exc:
                logger.warning("load_pair_min_orders: failed to fetch /api/pairs: %s", exc)
                return
        if not isinstance(pairs_info, list):
            logger.warning("load_pair_min_orders: unexpected response format")
            return
        loaded = 0
        for info in pairs_info:
            if not isinstance(info, dict):
                continue
            pair_id = (info.get("id") or info.get("pair_id") or "").lower().replace("/", "_")
            if not pair_id:
                continue
            try:
                min_coin = float(info.get("trade_min_base_currency") or 0)
            except (TypeError, ValueError):
                min_coin = 0.0
            try:
                min_idr = float(info.get("trade_min_traded_currency") or 0)
            except (TypeError, ValueError):
                min_idr = 0.0
            self._pair_min_order[pair_id] = {"min_coin": min_coin, "min_idr": min_idr}
            loaded += 1
        logger.info("load_pair_min_orders: cached minimum orders for %d pairs", loaded)
        ttl = getattr(self, "_pair_min_order_cache_ttl", 3600.0)
        self._pair_min_order_expires = time.time() + ttl

    def load_price_increments(self, increments: Optional[Dict[str, Any]] = None) -> None:
        """Fetch price increments (/api/price_increments) and cache tick sizes.

        The API returns a mapping ``{"increments": {"btc_idr": "1000", ...}}``.
        We keep the raw string to preserve the original decimal precision and
        later quantize prices accordingly before sending orders.
        """
        if increments is None:
            try:
                increments = self._get("/api/price_increments")
            except Exception as exc:
                logger.warning("load_price_increments: failed to fetch price increments: %s", exc)
                return
        if not isinstance(increments, dict):
            logger.warning("load_price_increments: unexpected response format")
            return
        inc_map = increments.get("increments") if isinstance(increments.get("increments"), dict) else increments
        if not isinstance(inc_map, dict):
            logger.warning("load_price_increments: missing increments map")
            return
        self._price_increments = {k.lower(): str(v) for k, v in inc_map.items() if v is not None}
        self._price_increments_expires = time.time() + getattr(self, "_price_increments_ttl", 3600.0)

    def is_price_increment_cache_stale(self) -> bool:
        return time.time() >= getattr(self, "_price_increments_expires", 0.0)

    def get_price_increment(self, pair: str) -> Optional[str]:
        """Return cached tick size string for *pair*, or ``None`` if unknown."""
        return self._price_increments.get(pair.lower())

    def get_pair_min_order(self, pair: str) -> Dict[str, float]:
        """Return the cached minimum order sizes for *pair*.

        The cache is considered stale when it has passed its TTL.  The caller
        (Trader._ensure_pair_min_order_cache) is responsible for periodic
        refreshes; this method only returns whatever is currently cached.

        :returns: ``{"min_coin": float, "min_idr": float}`` — both default to
                  0.0 when the pair is not in the cache or the cache is empty.
        """
        return self._pair_min_order.get(pair.lower(), {"min_coin": 0.0, "min_idr": 0.0})

    def is_pair_min_order_cache_stale(self) -> bool:
        """Return ``True`` when the pair minimum-order cache needs refreshing."""
        expires = getattr(self, "_pair_min_order_expires", 0.0)
        return time.time() >= expires

    def get_summaries(self) -> Dict[str, Any]:
        return self._get("/api/summaries")

    def get_ticker(self, pair: str) -> Dict[str, Any]:
        return self._get(f"/api/ticker/{pair}")

    def get_depth(self, pair: str, count: int = 50) -> Dict[str, Any]:
        return self._get(f"/api/depth/{pair}", params={"count": count})

    def get_trades(self, pair: str, count: int = 200) -> List[Dict[str, Any]]:
        return self._get(f"/api/trades/{pair}", params={"count": count})

    def get_ohlc(
        self,
        pair: str,
        tf: str = "15",
        *,
        limit: int = 200,
        to_ts: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch OHLCV candles from ``/tradingview/history_v2``.

        :param pair:  Canonical pair name (e.g. ``btc_idr``).
        :param tf:    Timeframe string accepted by the API: ``"1"``, ``"15"``,
                      ``"30"``, ``"60"``, ``"240"``, ``"1D"``, ``"3D"``, ``"1W"``.
                      Defaults to ``"15"`` (15-minute).
        :param limit: Approximate number of candles to request.
        :param to_ts: End-of-range Unix timestamp.  Defaults to *now*.
        :returns:     List of dicts with keys ``Time``, ``Open``, ``High``,
                      ``Low``, ``Close``, ``Volume``.  Empty list on failure.
        """
        if to_ts is None:
            to_ts = int(time.time())
        tf_seconds = _OHLC_TF_SECONDS.get(tf, 900)
        from_ts = to_ts - limit * tf_seconds
        symbol = pair.replace("_", "").upper()  # "btc_idr" → "BTCIDR"
        result = self._get(
            "/tradingview/history_v2",
            params={"from": from_ts, "to": to_ts, "tf": tf, "symbol": symbol},
        )
        return result if isinstance(result, list) else []

    # -------------------- private API -------------------- #
    def get_account_info(self) -> Dict[str, Any]:
        """Return account info, using a TTL cache to reduce API requests.

        The response is cached for :attr:`_account_info_cache_ttl` seconds.
        Cache is bypassed when ``_account_info_cache_ttl <= 0``.
        After an order is placed the cache should be invalidated via
        :meth:`invalidate_account_info_cache` so the next balance check
        reflects the updated state.
        """
        if self._account_info_cache_ttl > 0:
            now = time.monotonic()
            if self._account_info_cached is not None and now < self._account_info_expires:
                logger.debug("account_info cache hit (expires in %.1fs)", self._account_info_expires - now)
                return self._account_info_cached
            result = self._post_private("getInfo")
            self._account_info_cached = result
            self._account_info_expires = time.monotonic() + self._account_info_cache_ttl
            return result
        return self._post_private("getInfo")

    def open_orders(self, pair: str) -> Dict[str, Any]:
        """Return open orders for *pair*, using a per-pair TTL cache.

        Cached for :attr:`_open_orders_cache_ttl` seconds per pair.
        Bypass when ``_open_orders_cache_ttl <= 0``.
        """
        if self._open_orders_cache_ttl > 0:
            now = time.monotonic()
            cached = self._open_orders_cache.get(pair)
            if cached is not None:
                data, expiry = cached
                if now < expiry:
                    logger.debug("open_orders cache hit for %s (expires in %.1fs)", pair, expiry - now)
                    return data
            result = self._post_private("openOrders", {"pair": pair})
            self._open_orders_cache[pair] = (result, time.monotonic() + self._open_orders_cache_ttl)
            return result
        return self._post_private("openOrders", {"pair": pair})

    def trade_history(self, pair: str, count: int = 50) -> Dict[str, Any]:
        return self._post_private("tradeHistory", {"pair": pair, "count": count})

    def generate_private_ws_token(self) -> Dict[str, str]:
        """Request a Private WebSocket connection token and channel identifier.

        Calls ``POST /api/private_ws/v1/generate_token`` as documented at:
        https://github.com/btcid/indodax-official-api-docs/blob/master/Private-websocket.md

        :returns: ``{"connToken": "...", "channel": "pws:#..."}``
        :raises ValueError: when API key/secret are not set.
        :raises RuntimeError: when the API returns a non-success response.
        """
        if not self.api_key:
            raise ValueError("API key is required for private WebSocket token")
        if not self.api_secret:
            raise ValueError("API secret is required for private WebSocket token")

        request_body = f"client=tapi&tapi_key={self.api_key}"
        sign = hmac.new(
            self.api_secret.encode("utf-8"),
            request_body.encode("utf-8"),
            hashlib.sha512,
        ).hexdigest()
        headers = {
            "Sign": sign,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = self.session.post(
            f"{self.base_url}/api/private_ws/v1/generate_token",
            data=request_body,
            headers=headers,
            timeout=self.timeout,
        )
        data = self._handle_response(response)
        ret = data.get("return") if isinstance(data, dict) else None
        if not ret or not ret.get("connToken"):
            raise RuntimeError(f"Failed to obtain private WS token: {data}")
        return {"connToken": ret["connToken"], "channel": ret["channel"]}

    def _maybe_refresh_price_increments(self) -> None:
        if not self._price_increments or self.is_price_increment_cache_stale():
            self.load_price_increments()

    def format_price(self, pair: str, price: float) -> Tuple[float, int]:
        """Round *price* to the tick size expected by the exchange for *pair*.

        Uses ``/api/price_increments`` when available to quantize prices so
        Indodax does not reject orders with messages like "decimal number for
        price is 3". Returns the rounded price plus the decimal precision used
        for string formatting.
        """
        precision = 8  # default
        quantized_price = price
        used_increment = False

        try:
            self._maybe_refresh_price_increments()
            inc_str = self.get_price_increment(pair)
            if inc_str:
                step = Decimal(str(inc_str))
                if step > 0:
                    precision = max(0, -step.as_tuple().exponent)
                    quantized_price = float(
                        (Decimal(price) / step).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * step
                    )
                    used_increment = True
        except (InvalidOperation, ValueError):
            pass

        # Fallback when increment is unavailable: IDR pairs default to integer prices.
        if not used_increment:
            if pair.endswith("_idr"):
                precision = 0
                quantized_price = round(price)
            else:
                precision = 8
                quantized_price = round(price, precision)

        return quantized_price, precision

    def create_order(
        self,
        pair: str,
        order_type: str,
        price: float,
        amount: float,
        *,
        order_kind: str = "limit",
        client_order_id: Optional[str] = None,
        time_in_force: Optional[str] = None,
        idr: Optional[float] = None,
        btc: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Place a trade order according to Indodax Private REST API docs.

        Supports documented optional parameters:
        - order_type (``order_kind``) : ``limit`` / ``market`` (default ``limit``)
        - client_order_id            : user-supplied id (<= 36 chars, [A-Za-z0-9_-])
        - time_in_force              : currently ``GTC`` or ``MOC`` for limit
        - idr / btc                  : explicit amount fields per API docs
        """
        price, precision = self.format_price(pair, price)
        price_str = f"{price:.{precision}f}"
        payload: Dict[str, Any] = {"pair": pair, "type": order_type, "price": price_str, "order_type": order_kind}
        base_coin = pair.split("_", 1)[0]
        is_idr_pair = pair.endswith("_idr")

        if order_type == "buy":
            if idr is not None:
                payload["idr"] = f"{float(idr):.0f}"
            elif is_idr_pair:
                idr_total = round(price * amount)
                payload["idr"] = f"{idr_total:.0f}"
                coin_amt = amount
                amount_str = f"{coin_amt:.8f}"
                payload[base_coin] = amount_str
                payload["amount"] = amount_str
            else:
                coin_amt = btc if btc is not None else amount
                amount_str = f"{coin_amt:.8f}"
                payload[base_coin] = amount_str
                payload["amount"] = amount_str
        else:
            coin_amt = btc if btc is not None else amount
            amount_str = f"{coin_amt:.8f}"
            payload[base_coin] = amount_str
            payload["amount"] = amount_str

        if client_order_id:
            if len(client_order_id) > 36:
                raise ValueError("client_order_id must be <= 36 characters")
            payload["client_order_id"] = client_order_id
        if time_in_force:
            payload["time_in_force"] = time_in_force

        return self._enqueue_private("trade", payload)

    def cancel_order(self, pair: str, order_id: str, order_type: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"pair": pair, "order_id": order_id}
        if order_type:
            payload["type"] = order_type
        return self._enqueue_private("cancelOrder", payload)

    @staticmethod
    def parse_minimum_order_error(error_message: str) -> Optional[float]:
        """Extract the minimum required coin amount from a 'Minimum order' error.

        Indodax returns messages like ``"Minimum order 3333.33333333 WTEC"``
        when a sell/buy amount is below the exchange floor.  This helper parses
        that message and returns the minimum amount as a float, or ``None`` if
        the message does not match the expected pattern.

        :param error_message: The ``error`` string from the Indodax API response
                              dict, or the full ``str(exc)`` text.
        :returns: Minimum coin amount (float) or ``None``.
        """
        match = re.search(r"Minimum order\s+([\d.]+)", error_message, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    # -------------------- helpers -------------------- #
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        scheduler = getattr(self, "_request_scheduler", None)
        if scheduler is not None:
            return scheduler.submit(self._perform_get, path, params).result()
        return self._perform_get(path, params)

    def _perform_get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        # Serialize public REST calls to avoid hitting Indodax per-IP limits
        # when multiple positions trigger concurrent fetches (depth/trades).
        min_interval = getattr(self, "public_min_interval", 0.0)
        if min_interval > 0:
            lock = getattr(self, "_public_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._public_lock = lock
            time_fn = getattr(self, "_public_time", time.monotonic)
            sleep_fn = getattr(self, "_public_sleep", time.sleep)
            last_ts = getattr(self, "_last_public_request", 0.0)
            with lock:
                now = time_fn()
                wait = last_ts + min_interval - now
                if wait > 0:
                    sleep_fn(wait)
                    now = time_fn()
                self._last_public_request = now
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        return self._handle_response(response)

    def _post_private(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        scheduler = getattr(self, "_request_scheduler", None)
        if scheduler is not None:
            return scheduler.submit(self._perform_post_private, method, params).result()
        return self._perform_post_private(method, params)

    def _perform_post_private(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key:
            raise ValueError("API key is required for private endpoints")
        if not self.api_secret:
            raise ValueError("API secret is required for private endpoints")

        payload: Dict[str, Any] = {"method": method, "nonce": int(time.time() * 1000)}
        if params:
            payload.update({k: v for k, v in params.items() if v is not None})

        encoded = urlencode(payload)
        sign = hmac.new(
            self.api_secret.encode("utf-8"),
            encoded.encode("utf-8"),
            hashlib.sha512,
        ).hexdigest()
        headers = {
            "Key": self.api_key,
            "Sign": sign,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = self.session.post(
            f"{self.base_url}/tapi", data=encoded, headers=headers, timeout=self.timeout
        )
        return self._handle_response(response)

    def _enqueue_private(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.order_queue:
            return self._post_private(method, params)
        return self.order_queue.submit(self._post_private, method, params).result()

    @staticmethod
    def _handle_response(response: requests.Response) -> Any:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"HTTP error: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("Failed to parse JSON response") from exc

        # Private API success flag
        if isinstance(data, dict) and ("success" in data) and not data.get("success"):
            raise RuntimeError(f"API error: {data}")
        return data
