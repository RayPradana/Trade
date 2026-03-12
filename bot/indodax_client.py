from __future__ import annotations

import hashlib
import hmac
import logging
import re
import time
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional

import requests

from .rate_limit import RateLimitedOrderQueue
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
        # Per-pair minimum order cache:
        #   keys are pair names (e.g. "btc_idr")
        #   values are dicts with "min_coin" (min base currency amount)
        #   and "min_idr" (min IDR value).
        self._pair_min_order: Dict[str, Dict[str, float]] = {}

    # -------------------- public API -------------------- #
    def get_pairs(self) -> List[Dict[str, Any]]:
        return self._get("/api/pairs")

    def load_pair_min_orders(self) -> None:
        """Fetch exchange pair list and cache per-pair minimum order sizes.

        The Indodax ``/api/pairs`` endpoint returns a list of pair objects,
        each containing ``trade_min_base_currency`` (minimum coin amount) and
        ``trade_min_traded_currency`` (minimum IDR amount).  This method
        populates :attr:`_pair_min_order` for quick lookup before order
        placement so that "Minimum order" errors are prevented proactively.

        Safe to call multiple times; the cache is fully replaced on each call.
        """
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

    def get_pair_min_order(self, pair: str) -> Dict[str, float]:
        """Return the cached minimum order sizes for *pair*.

        :returns: ``{"min_coin": float, "min_idr": float}`` — both default to
                  0.0 when the pair is not in the cache.
        """
        return self._pair_min_order.get(pair.lower(), {"min_coin": 0.0, "min_idr": 0.0})

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
        return self._post_private("getInfo")

    def open_orders(self, pair: str) -> Dict[str, Any]:
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

    def create_order(self, pair: str, order_type: str, price: float, amount: float) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"pair": pair, "type": order_type, "price": f"{price:.8f}"}
        is_idr_pair = pair.endswith("_idr")
        if order_type == "buy":
            if is_idr_pair:
                payload["idr"] = f"{price * amount:.8f}"
            else:
                payload["amount"] = f"{amount:.8f}"
        else:
            payload["amount"] = f"{amount:.8f}"
        return self._enqueue_private("trade", payload)

    def cancel_order(self, pair: str, order_id: str) -> Dict[str, Any]:
        return self._enqueue_private("cancelOrder", {"pair": pair, "order_id": order_id})

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
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        return self._handle_response(response)

    def _post_private(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
