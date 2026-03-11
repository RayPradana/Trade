from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .analysis import analyze_orderbook, analyze_trend, analyze_volatility, build_candles
from .config import BotConfig
from .indodax_client import IndodaxClient
from .strategies import StrategyDecision, make_trade_decision

logger = logging.getLogger(__name__)


class Trader:
    def __init__(self, config: BotConfig, client: Optional[IndodaxClient] = None) -> None:
        self.config = config
        self.client = client or IndodaxClient(config.api_key, config.api_secret)

    def _extract_price(self, ticker: Dict[str, Any]) -> float:
        if "ticker" in ticker:
            return float(ticker["ticker"].get("last") or ticker["ticker"].get("last_price", 0))
        return float(ticker.get("last") or ticker.get("last_price") or 0)

    def analyze_market(self) -> Dict[str, Any]:
        ticker = self.client.get_ticker(self.config.pair)
        depth = self.client.get_depth(self.config.pair, count=200)
        trades = self.client.get_trades(self.config.pair, count=400)

        candles = build_candles(trades, interval_seconds=self.config.interval_seconds, limit=96)
        trend = analyze_trend(candles, self.config.fast_window, self.config.slow_window)
        orderbook = analyze_orderbook(depth)
        vol = analyze_volatility(candles)
        price = self._extract_price(ticker)

        decision = make_trade_decision(trend, orderbook, vol, price, self.config)
        return {
            "price": price,
            "trend": trend,
            "orderbook": orderbook,
            "volatility": vol,
            "decision": decision,
            "candles": candles,
        }

    def maybe_execute(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        decision: StrategyDecision = snapshot["decision"]
        price = snapshot["price"]

        if decision.action == "hold":
            return {"status": "hold", "reason": decision.reason}

        if decision.confidence < self.config.min_confidence:
            return {
                "status": "skipped",
                "reason": f"confidence {decision.confidence} below threshold {self.config.min_confidence}",
            }

        # simple slippage guard using top of book
        depth = self.client.get_depth(self.config.pair, count=5)
        top_bid = float(depth.get("buy", [[0]])[0][0]) if depth.get("buy") else price
        top_ask = float(depth.get("sell", [[0]])[0][0]) if depth.get("sell") else price
        reference_price = top_ask if decision.action == "buy" else top_bid
        allowed_max = price * (1 + self.config.max_slippage_pct)
        allowed_min = price * (1 - self.config.max_slippage_pct)

        if decision.action == "buy" and reference_price > allowed_max:
            return {"status": "skipped", "reason": "slippage too high for buy"}
        if decision.action == "sell" and reference_price < allowed_min:
            return {"status": "skipped", "reason": "slippage too high for sell"}

        if self.config.dry_run:
            logger.info("DRY-RUN %s %s @ %s", decision.action, decision.amount, reference_price)
            return {
                "status": "simulated",
                "action": decision.action,
                "price": reference_price,
                "amount": decision.amount,
                "mode": decision.mode,
                "stop_loss": decision.stop_loss,
                "take_profit": decision.take_profit,
                "confidence": decision.confidence,
                "reason": decision.reason,
            }

        # live trading path
        if self.config.api_key is None or self.config.api_secret is None:
            raise ValueError("API credentials required for live trading")

        order_resp = self.client.create_order(
            self.config.pair, decision.action, reference_price, decision.amount
        )
        return {
            "status": "placed",
            "order": order_resp,
            "action": decision.action,
            "price": reference_price,
            "amount": decision.amount,
            "mode": decision.mode,
        }
