from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class BotConfig:
    api_key: Optional[str]
    api_secret: Optional[str]
    pair: str = "btc_idr"
    base_order_size: float = 0.0001  # in base asset
    risk_per_trade: float = 0.01  # 1% default
    dry_run: bool = True
    min_confidence: float = 0.52
    interval_seconds: int = 300
    fast_window: int = 12
    slow_window: int = 48
    max_slippage_pct: float = 0.001

    @classmethod
    def from_env(cls) -> "BotConfig":
        return cls(
            api_key=os.getenv("INDODAX_KEY"),
            api_secret=os.getenv("INDODAX_SECRET"),
            pair=os.getenv("TRADE_PAIR", "btc_idr").lower(),
            base_order_size=float(os.getenv("BASE_ORDER_SIZE", "0.0001")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
            dry_run=os.getenv("DRY_RUN", "true").lower() in {"1", "true", "yes"},
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.52")),
            interval_seconds=int(os.getenv("INTERVAL_SECONDS", "300")),
            fast_window=int(os.getenv("FAST_WINDOW", "12")),
            slow_window=int(os.getenv("SLOW_WINDOW", "48")),
            max_slippage_pct=float(os.getenv("MAX_SLIPPAGE_PCT", "0.001")),
        )

    def require_auth(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError("INDODAX_KEY and INDODAX_SECRET are required for live trading")
