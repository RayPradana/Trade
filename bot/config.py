from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _load_dotenv(path: Optional[Path] = None) -> None:
    """Load environment variables from a .env file without overriding existing values."""
    dotenv_path = path or Path(__file__).resolve().parent.parent / ".env"
    if not dotenv_path.exists():
        return
    # Escaped quotes inside values are not supported and will be treated as literal characters.
    # Use plain quoting, e.g. KEY='value with spaces'

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        if key in os.environ:
            continue
        cleaned = value.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1]
        os.environ[key] = cleaned


@dataclass
class BotConfig:
    api_key: Optional[str]
    pair: str = "btc_idr"  # Default/fallback pair when automatic scanning yields no candidates
    scan_pairs: Optional[List[str]] = None
    base_order_size: float = 0.0001  # size in base asset (e.g., BTC for btc_idr)
    risk_per_trade: float = 0.01  # 1% default
    dry_run: bool = True
    min_confidence: float = 0.52
    interval_seconds: int = 300
    fast_window: int = 12
    slow_window: int = 48
    max_slippage_pct: float = 0.001
    initial_capital: float = 1_000_000.0  # in quote currency (e.g., IDR)
    target_profit_pct: float = 0.2  # 20%
    max_loss_pct: float = 0.1  # 10%

    @classmethod
    def from_env(cls) -> "BotConfig":
        _load_dotenv()
        pairs_env = os.getenv("TRADE_PAIRS")
        scan_pairs = [p.strip().lower() for p in pairs_env.split(",")] if pairs_env else None
        return cls(
            api_key=os.getenv("INDODAX_KEY"),
            pair=os.getenv("TRADE_PAIR", "btc_idr").lower(),
            scan_pairs=scan_pairs,
            base_order_size=float(os.getenv("BASE_ORDER_SIZE", "0.0001")),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
            dry_run=os.getenv("DRY_RUN", "true").lower() in {"1", "true", "yes"},
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.52")),
            interval_seconds=int(os.getenv("INTERVAL_SECONDS", "300")),
            fast_window=int(os.getenv("FAST_WINDOW", "12")),
            slow_window=int(os.getenv("SLOW_WINDOW", "48")),
            max_slippage_pct=float(os.getenv("MAX_SLIPPAGE_PCT", "0.001")),
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "1000000")),
            target_profit_pct=float(os.getenv("TARGET_PROFIT_PCT", "0.2")),
            max_loss_pct=float(os.getenv("MAX_LOSS_PCT", "0.1")),
        )

    def require_auth(self) -> None:
        if not self.api_key:
            raise ValueError("INDODAX_KEY is required for live trading")
