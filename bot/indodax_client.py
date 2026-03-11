from __future__ import annotations

import hashlib
import hmac
import time
from urllib.parse import urlencode
from typing import Any, Dict, List, Optional

import requests

from .rate_limit import RateLimitedOrderQueue

# Mapping of Indodax OHLC timeframe strings to seconds.
_OHLC_TF_SECONDS: Dict[str, int] = {
    "1": 60,
    "15": 900,
    "30": 1800,
    "60": 3600,
    "240": 14400,
    "1D": 86400,
    "3D": 259200,
    "1W": 604800,
}


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

    # -------------------- public API -------------------- #
    def get_pairs(self) -> List[Dict[str, Any]]:
        return self._get("/api/pairs")

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
