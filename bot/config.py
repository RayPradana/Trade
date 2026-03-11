from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_dotenv(path: Optional[Path] = None) -> None:
    """Load environment variables from a .env file without overriding existing values."""
    dotenv_path = path or Path(__file__).resolve().parent.parent / ".env"
    if not dotenv_path.exists():
        return
    # Values can be quoted with single or double quotes (e.g., KEY="value with spaces").
    # Escaped quotes inside values are not supported and will be treated as literal characters.

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
    api_secret: Optional[str] = None
    pair: str = "btc_idr"  # Last-resort fallback used when pair discovery via get_pairs() fails entirely
    risk_per_trade: float = 0.01  # 1% default; order size = risk_per_trade * initial_capital / coin_price
    dry_run: bool = True
    run_once: bool = False
    real_time: bool = False
    grid_enabled: bool = False
    grid_levels_per_side: int = 3
    grid_spacing_pct: float = 0.004  # 0.4% spacing
    grid_order_size: Optional[float] = None
    order_queue_enabled: bool = True
    order_min_interval: float = 1.5  # seconds between order requests; higher reduces 429 risk
    scan_request_delay: float = 0.2  # seconds to wait before each per-pair API call during scanning
    trade_count: int = 1000  # recent trades to fetch per pair when building candles from trades
    min_candles: int = 20  # minimum candle count required for reliable indicator computation
    websocket_enabled: bool = True
    websocket_url: Optional[str] = "wss://ws3.indodax.com/ws/"  # Indodax market-data WebSocket
    websocket_subscribe_message: Optional[str] = None  # raw JSON string sent to the server on connect
    websocket_batch_size: int = 100  # max pairs per WebSocket connection for multi-pair feed
    pairs_per_cycle: int = 50  # 0 = scan all pairs every cycle; >0 = rotate N pairs per cycle
    min_confidence: float = 0.52
    interval_seconds: int = 300
    fast_window: int = 12
    slow_window: int = 48
    max_slippage_pct: float = 0.001
    initial_capital: float = 1_000_000.0  # in quote currency (e.g., IDR)
    target_profit_pct: float = 0.2  # 20%
    max_loss_pct: float = 0.1  # 10%
    trailing_stop_pct: float = 0.0  # 0 = disabled; e.g. 0.02 = 2% trailing stop
    staged_entry_steps: int = 3
    position_check_interval_seconds: int = 60  # faster poll when monitoring an open position
    cycle_summary_interval: int = 10  # print a performance summary every N full scan cycles
    trade_mode: str = "continuous"  # "single": one buy→sell cycle then stop; "continuous": 24/7
    state_path: Optional[Path] = None  # None = disabled; set via STATE_PATH env var to enable auto-resume
    state_backup_interval: int = 10  # save a backup copy every N scan cycles (0 = disabled)
    # Minimum 24-h IDR trading volume a pair must have to be analysed.
    # Pairs below this threshold are skipped during multi-pair scanning.
    # Set to 0 to disable the filter (default).
    min_volume_idr: float = 0.0
    # Optional path to a log file.  When set, every log line is written to
    # both stdout/stderr *and* this file.  Rotation is not handled here;
    # use an external log-rotation daemon (logrotate / Docker logging driver)
    # or set LOG_FILE to a new path on each restart.
    log_file: Optional[str] = None
    # Telegram bot integration – when both fields are set the bot sends a
    # message to the specified chat whenever an order is placed/simulated or
    # a portfolio stop fires.
    telegram_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    # WebSocket stale-data threshold (seconds).  If the MultiPairFeed WS has
    # not received any ticker update within this window the feed is considered
    # stale and the REST summaries fallback is triggered immediately.
    ws_stale_threshold: float = 120.0

    @classmethod
    def from_env(cls) -> "BotConfig":
        existing_keys = set(os.environ.keys())
        _load_dotenv()
        real_time = os.getenv("REALTIME_MODE", os.getenv("REAL_TIME", "false")).lower() in {"1", "true", "yes"}
        interval_default = "1" if real_time else "300"
        user_set_interval = "INTERVAL_SECONDS" in existing_keys
        grid_enabled = os.getenv("GRID_ENABLED", "false").lower() in {"1", "true", "yes"}
        websocket_enabled = os.getenv("WEBSOCKET_ENABLED", "true").lower() in {"1", "true", "yes"}
        interval_env = os.getenv("INTERVAL_SECONDS")
        if real_time:
            interval_seconds = int(interval_env) if user_set_interval and interval_env is not None else 1
        else:
            interval_seconds = int(interval_env) if interval_env is not None else int(interval_default)
        cfg = cls(
            api_key=os.getenv("INDODAX_KEY"),
            api_secret=os.getenv("INDODAX_SECRET"),
            risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
            dry_run=os.getenv("DRY_RUN", "true").lower() in {"1", "true", "yes"},
            run_once=os.getenv("RUN_ONCE", "false").lower() in {"1", "true", "yes"},
            real_time=real_time,
            grid_enabled=grid_enabled,
            grid_levels_per_side=int(os.getenv("GRID_LEVELS_PER_SIDE", "3")),
            grid_spacing_pct=float(os.getenv("GRID_SPACING_PCT", "0.004")),
            grid_order_size=float(os.getenv("GRID_ORDER_SIZE")) if os.getenv("GRID_ORDER_SIZE") else None,
            order_queue_enabled=os.getenv("ORDER_QUEUE_ENABLED", "true").lower() in {"1", "true", "yes"},
            order_min_interval=float(os.getenv("ORDER_MIN_INTERVAL", "1.5")),
            scan_request_delay=float(os.getenv("SCAN_REQUEST_DELAY", "0.2")),
            trade_count=int(os.getenv("TRADE_COUNT", "1000")),
            min_candles=int(os.getenv("MIN_CANDLES", "20")),
            websocket_enabled=websocket_enabled,
            websocket_url=os.getenv("WEBSOCKET_URL", "wss://ws3.indodax.com/ws/") or None,
            websocket_subscribe_message=os.getenv("WEBSOCKET_SUBSCRIBE_MESSAGE"),
            websocket_batch_size=int(os.getenv("WEBSOCKET_BATCH_SIZE", "100")),
            pairs_per_cycle=int(os.getenv("PAIRS_PER_CYCLE", "50")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.52")),
            interval_seconds=interval_seconds,
            fast_window=int(os.getenv("FAST_WINDOW", "12")),
            slow_window=int(os.getenv("SLOW_WINDOW", "48")),
            max_slippage_pct=float(os.getenv("MAX_SLIPPAGE_PCT", "0.001")),
            initial_capital=float(os.getenv("INITIAL_CAPITAL", "1000000")),
            target_profit_pct=float(os.getenv("TARGET_PROFIT_PCT", "0.2")),
            max_loss_pct=float(os.getenv("MAX_LOSS_PCT", "0.1")),
            trailing_stop_pct=float(os.getenv("TRAILING_STOP_PCT", "0.0")),
            staged_entry_steps=int(os.getenv("STAGED_ENTRY_STEPS", "3")),
            position_check_interval_seconds=int(os.getenv("POSITION_CHECK_INTERVAL", "60")),
            cycle_summary_interval=int(os.getenv("CYCLE_SUMMARY_INTERVAL", "10")),
            trade_mode=os.getenv("TRADE_MODE", "continuous").lower(),
            state_path=Path(os.getenv("STATE_PATH", "bot_state.json")),
            state_backup_interval=int(os.getenv("STATE_BACKUP_INTERVAL", "10")),
            min_volume_idr=float(os.getenv("MIN_VOLUME_IDR", "0")),
            log_file=os.getenv("LOG_FILE") or None,
            telegram_token=os.getenv("TELEGRAM_TOKEN") or None,
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID") or None,
            ws_stale_threshold=float(os.getenv("WS_STALE_THRESHOLD", "120")),
        )
        cfg._validate()
        return cfg

    def require_auth(self) -> None:
        if not self.api_key:
            raise ValueError("INDODAX_KEY is required for live trading")
        if not self.api_secret:
            raise ValueError("INDODAX_SECRET is required for live trading")

    def _validate(self) -> None:
        if not (0 < self.risk_per_trade <= 0.5):
            raise ValueError("RISK_PER_TRADE must be between 0 and 0.5")
        if self.grid_levels_per_side <= 0:
            raise ValueError("GRID_LEVELS_PER_SIDE must be positive")
        if self.grid_spacing_pct <= 0:
            raise ValueError("GRID_SPACING_PCT must be positive")
        if self.grid_order_size is not None and self.grid_order_size <= 0:
            raise ValueError("GRID_ORDER_SIZE must be positive when set")
        if self.order_min_interval <= 0:
            raise ValueError("ORDER_MIN_INTERVAL must be positive")
        if self.scan_request_delay < 0:
            raise ValueError("SCAN_REQUEST_DELAY must be non-negative")
        if self.trade_count <= 0:
            raise ValueError("TRADE_COUNT must be positive")
        if self.min_candles <= 0:
            raise ValueError("MIN_CANDLES must be positive")
        if self.websocket_batch_size <= 0:
            raise ValueError("WEBSOCKET_BATCH_SIZE must be positive")
        if self.pairs_per_cycle < 0:
            raise ValueError("PAIRS_PER_CYCLE must be non-negative")
        if self.interval_seconds <= 0:
            raise ValueError("INTERVAL_SECONDS must be positive")
        if self.max_slippage_pct < 0:
            raise ValueError("MAX_SLIPPAGE_PCT must be non-negative")
        if self.staged_entry_steps <= 0:
            raise ValueError("STAGED_ENTRY_STEPS must be positive")
        if self.position_check_interval_seconds <= 0:
            raise ValueError("POSITION_CHECK_INTERVAL must be positive")
        if self.cycle_summary_interval <= 0:
            raise ValueError("CYCLE_SUMMARY_INTERVAL must be positive")
        if self.trade_mode not in {"single", "continuous"}:
            raise ValueError("TRADE_MODE must be 'single' or 'continuous'")
        if self.trailing_stop_pct < 0:
            raise ValueError("TRAILING_STOP_PCT must be non-negative")
        if self.state_backup_interval < 0:
            raise ValueError("STATE_BACKUP_INTERVAL must be non-negative")
        if self.min_volume_idr < 0:
            raise ValueError("MIN_VOLUME_IDR must be non-negative")
        if self.ws_stale_threshold <= 0:
            raise ValueError("WS_STALE_THRESHOLD must be positive")
        if not self.dry_run and not self.api_key:
            self.require_auth()
