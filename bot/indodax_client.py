from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional

import requests


class IndodaxClient:
    """Lightweight Indodax API wrapper supporting public and private endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        session: Optional[requests.Session] = None,
        base_url: str = "https://indodax.com",
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

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
        return self._post_private("trade", payload)

    def cancel_order(self, pair: str, order_id: str) -> Dict[str, Any]:
        return self._post_private("cancelOrder", {"pair": pair, "order_id": order_id})

    # -------------------- helpers -------------------- #
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        response = self.session.get(url, params=params, timeout=self.timeout)
        return self._handle_response(response)

    def _post_private(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret are required for private endpoints")

        payload: Dict[str, Any] = {"method": method, "nonce": int(time.time() * 1000)}
        if params:
            payload.update({k: v for k, v in params.items() if v is not None})

        encoded = requests.compat.urlencode(payload)
        signature = hmac.new(
            self.api_secret.encode("utf-8"), encoded.encode("utf-8"), hashlib.sha512
        ).hexdigest()

        headers = {
            "Key": self.api_key,
            "Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        response = self.session.post(
            f"{self.base_url}/tapi", data=encoded, headers=headers, timeout=self.timeout
        )
        return self._handle_response(response)

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
