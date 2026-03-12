from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


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
    pairs_per_cycle: int = 20  # 0 = scan all pairs every cycle; >0 = rotate N pairs per cycle
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
    # Minimum portfolio cash (IDR) required to use multi-step staged entry.
    # When the available cash falls below this threshold the bot collapses to a
    # single-step entry so that the trade is not discarded because individual
    # staged steps fall below the exchange minimum order size.  Set to 0 to
    # always use staged entry regardless of equity.
    staged_entry_min_equity: float = 1_000_000.0
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
    # Multi-timeframe analysis — list of additional OHLC timeframes to fetch
    # per pair (beyond the primary one derived from interval_seconds).  Each
    # entry is a string accepted by the Indodax OHLCV API: "1", "15", "30",
    # "60", "240".  E.g. ["1", "15", "60"] fetches 1-min, 15-min and 1-h candles.
    # Empty list (default) disables multi-timeframe analysis.
    mtf_timeframes: List[str] = field(default_factory=list)
    # Maximum fraction of initial capital that can be exposed to a single
    # coin at any time (0 = no cap).  E.g. 0.3 = max 30% in one coin.
    max_exposure_per_coin_pct: float = 0.0
    # Maximum daily realised loss expressed as a fraction of initial capital.
    # When the daily loss exceeds this threshold the bot stops trading until
    # the next calendar day (UTC).  0 = no cap.
    max_daily_loss_pct: float = 0.0
    # Discord webhook URL for order / error notifications.  When set, the
    # bot posts a plain-text message to this webhook.
    discord_webhook_url: Optional[str] = None
    # Number of scan cycles between dynamic-pair-list refreshes.
    # 0 = disabled (use the static pair list from get_pairs() forever).
    # Default 5: refresh the watchlist every 5 cycles to keep the top-N
    # by volume+volatility without hammering the API.
    dynamic_pairs_refresh_cycles: int = 5
    # Number of top pairs (by composite volume×volatility score) kept in the
    # dynamic watchlist.  Scanning fewer pairs avoids rate limits and focuses
    # the bot on the most active instruments.  Default 20 follows the
    # professional practice of trading only the best 20–30 pairs at a time.
    dynamic_pairs_top_n: int = 20
    # Partial take-profit: when > 0, sell this fraction of the position when
    # price first reaches the TP level, and let the remainder run.
    # E.g. 0.5 = sell half at TP, keep the rest.  0 = sell all (default).
    partial_tp_fraction: float = 0.0
    # Re-entry logic: after a sell, wait at least this many seconds before
    # considering a new buy on the same pair.  0 = no cooldown (default).
    re_entry_cooldown_seconds: float = 0.0
    # Re-entry price dip: price must fall at least this fraction below the
    # last sell price before a re-entry buy is allowed.  0 = no dip required.
    re_entry_dip_pct: float = 0.0
    # Adaptive scanning interval: when enabled, the bot uses a shorter sleep
    # between scan cycles during high-volatility periods.
    adaptive_interval_enabled: bool = False
    # Minimum sleep between cycles (seconds) when adaptive interval is active
    # and volatility is elevated.
    adaptive_interval_min_seconds: int = 30
    # Portfolio-wide risk limit: maximum total position value as a fraction of
    # current equity across all coins combined.  0 = no cap (default).
    # Note: the current architecture tracks a single pair, so this is
    # equivalent to the per-coin exposure cap at the portfolio level.
    max_portfolio_risk_pct: float = 0.0
    # Minimum total order-book depth (sum of top-20 bid and ask levels in IDR)
    # a pair must have before it is analysed.  0 = no filter (default).
    min_liquidity_depth_idr: float = 0.0
    # Profit-buffer drawdown protection: stop new buys when the profit buffer
    # has fallen more than this fraction from its all-time peak.
    # E.g. 0.3 = stop trading if profits drop 30% from their peak.
    # 0 = no protection (default).
    profit_buffer_drawdown_pct: float = 0.0
    # ── Dynamic / trailing take-profit ───────────────────────────────────────
    # When > 0, once price crosses the initial TP target the bot activates a
    # trailing floor this far below the running peak price.  The position is
    # only closed when price falls back through the floor, locking in the best
    # available exit rather than a fixed 20%.
    # E.g. 0.01 = trail 1% below the highest price since TP was hit.
    trailing_tp_pct: float = 0.0
    # Conditional TP: extra market-condition checks before closing at TP.
    # If the conditions below are met the bot holds the position and waits
    # (using the trailing TP if configured, or a bare hold if not).
    # 0 = disabled (default).
    #
    # Hold past TP while trend strength >= this value (0 = disabled).
    conditional_tp_min_trend_strength: float = 0.0
    # Hold past TP while order-book imbalance (bid dominance) >= this value.
    # Positive imbalance means more buy pressure; negative means sell pressure.
    # E.g. 0.1 = hold only while bids dominate by at least 10%.  0 = disabled.
    conditional_tp_min_ob_imbalance: float = 0.0
    # Hold past TP while RSI < this threshold (not yet overbought).
    # E.g. 70 = close only when RSI >= 70.  0 = disabled.
    conditional_tp_max_rsi: float = 0.0
    # ── Sell-wall / orderbook wall guard ─────────────────────────────────────
    # Skip buy when the total ask-side depth (top-20 levels) is at least this
    # many times larger than the total bid-side depth.
    # E.g. 5 = skip if ask volume ≥ 5× bid volume (strong sell-wall pressure).
    # 0 = disabled (default).
    orderbook_wall_threshold: float = 0.0
    # ── Pump protection ───────────────────────────────────────────────────────
    # Skip entry when the price has risen by more than this fraction within the
    # last pump_lookback_seconds.  Prevents FOMO buys after a sharp spike.
    # E.g. 0.05 = skip buy if price is up >5% vs the oldest recorded price in
    # the lookback window.  0 = disabled (default).
    pump_protection_pct: float = 0.0
    # Rolling window (seconds) used to detect a pump.  Default 60 seconds.
    pump_lookback_seconds: float = 60.0
    # ── Spread filter ─────────────────────────────────────────────────────────
    # Skip any trade (buy or sell) when the bid-ask spread exceeds this
    # fraction of the best bid price.
    # E.g. 0.002 = skip when spread > 0.2%.  0 = disabled (default).
    max_spread_pct: float = 0.0
    # ── Smart Entry Engine (SEE) ──────────────────────────────────────────────
    # Three-part entry quality filter:
    #   1. Pre-pump detection — flags early volume accumulation before a pump.
    #   2. Whale pressure reading — net bid/ask whale ratio for direction bias.
    #   3. Fake breakout guard — requires volume to confirm a resistance break.
    # When see_enabled=True, signals boost or penalise confidence before trades.
    see_enabled: bool = False
    # Volume surge ratio: recent_avg / baseline_avg must exceed this to detect
    # a pre-pump accumulation phase.  E.g. 2.0 = 2× average baseline volume.
    see_volume_surge_ratio: float = 2.0
    # Minimum net whale pressure ratio (bid_whale_ratio − ask_whale_ratio) to
    # classify as significant directional whale activity.
    see_whale_pressure_min: float = 2.0
    # Minimum (recent_vol / avg_vol) ratio to treat a resistance breakout as
    # volume-confirmed and genuine.  E.g. 0.7 = 70% of average volume needed.
    see_breakout_volume_min: float = 0.7
    # ── Anti-fake-pump detection ──────────────────────────────────────────────
    # On Indodax, manipulative actors frequently execute a rapid pump followed
    # by an equally rapid dump within ≈20 seconds.  This guard watches the
    # per-pair price history buffer for the two-phase pattern:
    #   Phase 1 – spike: price rose ≥ pump_protection_pct vs the oldest sample.
    #   Phase 2 – reversal: price has since fallen ≥ fake_pump_reversal_pct from
    #             the in-window peak back toward (or below) the spike entry.
    # When both phases are detected the buy is skipped to avoid buying the dump.
    #
    # Requires pump_protection_pct > 0 (uses the same per-pair price buffer).
    # 0 = disabled (default).  E.g. 0.03 = skip when price dropped ≥3% from peak.
    fake_pump_reversal_pct: float = 0.0
    # ── Indodax minimum order value ───────────────────────────────────────────
    # Indodax rejects any order whose total IDR value (price × amount) is below
    # 10,000 IDR.  The bot enforces this limit before submitting to the exchange
    # so that the error is caught gracefully as a "skipped" outcome rather than
    # raising a runtime exception.
    #
    # Default is raised to 15,000 IDR above the exchange minimum to avoid
    # wasting fees on tiny trades and to ensure trades "feel" meaningful.
    # Must be > 0.
    min_order_idr: float = 15_000.0
    # Consecutive loss protection
    max_consecutive_losses: int = 0  # 0=disabled; stop trading after N losing sells in a row
    # Volatility cooldown
    volatility_cooldown_pct: float = 0.0  # 0=disabled; e.g. 0.05 = 5% price spike
    volatility_cooldown_seconds: float = 0.0  # e.g. 300 = 5 min pause after spike
    # Circuit breaker
    circuit_breaker_max_errors: int = 0  # 0=disabled; pause after N consecutive API errors
    circuit_breaker_pause_seconds: float = 300.0  # duration of circuit-breaker pause
    # Balance validation
    balance_check_enabled: bool = False
    # Stale order cancellation
    stale_order_seconds: float = 0.0
    # Strategy auto-disable
    strategy_auto_disable_losses: int = 0
    # Multi-level partial take profit (2nd level)
    partial_tp2_fraction: float = 0.0  # fraction to sell at 2nd TP (0=disabled)
    partial_tp2_target_pct: float = 0.0  # price must rise this % above buy price to trigger 2nd TP
    # Trade journal path (CSV file path; None = disabled)
    journal_path: Optional[str] = None
    # Max open positions across all pairs (0=no limit)
    max_open_positions: int = 0
    # Spread anomaly filter
    spread_anomaly_multiplier: float = 0.0
    # Orderbook absorption detection threshold (0=disabled)
    orderbook_absorption_threshold: float = 0.0
    # Flash dump protection
    flash_dump_pct: float = 0.0
    flash_dump_lookback_seconds: float = 60.0
    # ── Adaptive position sizing ──────────────────────────────────────────────
    # When enabled, risk_per_trade and max_open_positions are automatically
    # adjusted based on current equity (effective_capital), making the bot more
    # aggressive with small capital and more conservative with large capital.
    #
    # Three tiers (small / medium / large) are defined by two equity thresholds:
    #   equity < adaptive_tier1_equity         → Tier 0 (small cap)
    #   adaptive_tier1_equity ≤ equity < adaptive_tier2_equity → Tier 1 (medium cap)
    #   equity ≥ adaptive_tier2_equity          → Tier 2 (large cap)
    #
    # Set ADAPTIVE_SIZING_ENABLED=true to activate.  When disabled (default)
    # the static RISK_PER_TRADE and MAX_OPEN_POSITIONS values are used.
    adaptive_sizing_enabled: bool = False
    # Equity thresholds (IDR)
    adaptive_tier1_equity: float = 2_000_000.0   # below this → Tier 0 (small cap)
    adaptive_tier2_equity: float = 5_000_000.0   # above this → Tier 2 (large cap)
    # Risk per trade for each tier (fraction of equity, e.g. 0.10 = 10%)
    adaptive_tier0_risk: float = 0.10   # small cap: 10%
    adaptive_tier1_risk: float = 0.07   # medium cap: 7%
    adaptive_tier2_risk: float = 0.03   # large cap: 3%
    # Max open positions for each tier
    adaptive_tier0_max_pos: int = 3
    adaptive_tier1_max_pos: int = 4
    adaptive_tier2_max_pos: int = 5
    # ── Per-pair trade cooldown ───────────────────────────────────────────────
    # After a buy or sell is executed for a pair, the pair is put in cooldown for
    # this many seconds.  Any subsequent buy signal on the same pair within the
    # window is skipped with status="skipped", reason="pair_cooldown".
    # 0 = disabled.  Default: 300 (5 minutes).
    pair_cooldown_seconds: float = 300.0
    # ── RSI overbought buy filter ─────────────────────────────────────────────
    # Hard-skip BUY when RSI is at or above this threshold.  Complements the
    # existing soft penalty (conf *= 0.8 above 70) by providing a hard cut-off
    # for extreme overbought conditions (e.g. RSI ≥ 85 after a spike).
    # 0 = disabled.  Default: 85.
    buy_max_rsi: float = 85.0
    # ── Distance-to-resistance buy filter ────────────────────────────────────
    # Hard-skip BUY when the current price is within this fraction of the
    # nearest resistance level.  For example 0.01 = skip if price is < 1%
    # below resistance.  Only active when resistance data is available.
    # 0 = disabled.  Default: 0.01 (1%).
    buy_max_resistance_proximity_pct: float = 0.01

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
            pairs_per_cycle=int(os.getenv("PAIRS_PER_CYCLE", "20")),
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
            staged_entry_min_equity=float(os.getenv("STAGED_ENTRY_MIN_EQUITY", "1000000")),
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
            mtf_timeframes=[
                tf.strip()
                for tf in os.getenv("MTF_TIMEFRAMES", "").split(",")
                if tf.strip()
            ],
            max_exposure_per_coin_pct=float(os.getenv("MAX_EXPOSURE_PER_COIN_PCT", "0")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0")),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL") or None,
            dynamic_pairs_refresh_cycles=int(os.getenv("DYNAMIC_PAIRS_REFRESH_CYCLES", "5")),
            dynamic_pairs_top_n=int(os.getenv("DYNAMIC_PAIRS_TOP_N", "20")),
            partial_tp_fraction=float(os.getenv("PARTIAL_TP_FRACTION", "0")),
            re_entry_cooldown_seconds=float(os.getenv("RE_ENTRY_COOLDOWN_SECONDS", "0")),
            re_entry_dip_pct=float(os.getenv("RE_ENTRY_DIP_PCT", "0")),
            adaptive_interval_enabled=os.getenv("ADAPTIVE_INTERVAL_ENABLED", "false").lower() in {"1", "true", "yes"},
            adaptive_interval_min_seconds=int(os.getenv("ADAPTIVE_INTERVAL_MIN_SECONDS", "30")),
            max_portfolio_risk_pct=float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "0")),
            min_liquidity_depth_idr=float(os.getenv("MIN_LIQUIDITY_DEPTH_IDR", "0")),
            profit_buffer_drawdown_pct=float(os.getenv("PROFIT_BUFFER_DRAWDOWN_PCT", "0")),
            trailing_tp_pct=float(os.getenv("TRAILING_TP_PCT", "0")),
            conditional_tp_min_trend_strength=float(os.getenv("CONDITIONAL_TP_MIN_TREND_STRENGTH", "0")),
            conditional_tp_min_ob_imbalance=float(os.getenv("CONDITIONAL_TP_MIN_OB_IMBALANCE", "0")),
            conditional_tp_max_rsi=float(os.getenv("CONDITIONAL_TP_MAX_RSI", "0")),
            orderbook_wall_threshold=float(os.getenv("ORDERBOOK_WALL_THRESHOLD", "0")),
            pump_protection_pct=float(os.getenv("PUMP_PROTECTION_PCT", "0")),
            pump_lookback_seconds=float(os.getenv("PUMP_LOOKBACK_SECONDS", "60")),
            max_spread_pct=float(os.getenv("MAX_SPREAD_PCT", "0")),
            see_enabled=os.getenv("SEE_ENABLED", "false").lower() in {"1", "true", "yes"},
            see_volume_surge_ratio=float(os.getenv("SEE_VOLUME_SURGE_RATIO", "2.0")),
            see_whale_pressure_min=float(os.getenv("SEE_WHALE_PRESSURE_MIN", "2.0")),
            see_breakout_volume_min=float(os.getenv("SEE_BREAKOUT_VOLUME_MIN", "0.7")),
            fake_pump_reversal_pct=float(os.getenv("FAKE_PUMP_REVERSAL_PCT", "0")),
            min_order_idr=float(os.getenv("MIN_ORDER_IDR", "15000")),
            max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "0")),
            volatility_cooldown_pct=float(os.getenv("VOLATILITY_COOLDOWN_PCT", "0")),
            volatility_cooldown_seconds=float(os.getenv("VOLATILITY_COOLDOWN_SECONDS", "0")),
            circuit_breaker_max_errors=int(os.getenv("CIRCUIT_BREAKER_MAX_ERRORS", "0")),
            circuit_breaker_pause_seconds=float(os.getenv("CIRCUIT_BREAKER_PAUSE_SECONDS", "300")),
            balance_check_enabled=os.getenv("BALANCE_CHECK_ENABLED", "false").lower() in {"1", "true", "yes"},
            stale_order_seconds=float(os.getenv("STALE_ORDER_SECONDS", "0")),
            strategy_auto_disable_losses=int(os.getenv("STRATEGY_AUTO_DISABLE_LOSSES", "0")),
            partial_tp2_fraction=float(os.getenv("PARTIAL_TP2_FRACTION", "0")),
            partial_tp2_target_pct=float(os.getenv("PARTIAL_TP2_TARGET_PCT", "0")),
            journal_path=os.getenv("JOURNAL_PATH") or None,
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "0")),
            spread_anomaly_multiplier=float(os.getenv("SPREAD_ANOMALY_MULTIPLIER", "0")),
            orderbook_absorption_threshold=float(os.getenv("ORDERBOOK_ABSORPTION_THRESHOLD", "0")),
            flash_dump_pct=float(os.getenv("FLASH_DUMP_PCT", "0")),
            flash_dump_lookback_seconds=float(os.getenv("FLASH_DUMP_LOOKBACK_SECONDS", "60")),
            adaptive_sizing_enabled=os.getenv("ADAPTIVE_SIZING_ENABLED", "false").lower() in {"1", "true", "yes"},
            adaptive_tier1_equity=float(os.getenv("ADAPTIVE_TIER1_EQUITY", "2000000")),
            adaptive_tier2_equity=float(os.getenv("ADAPTIVE_TIER2_EQUITY", "5000000")),
            adaptive_tier0_risk=float(os.getenv("ADAPTIVE_TIER0_RISK", "0.10")),
            adaptive_tier1_risk=float(os.getenv("ADAPTIVE_TIER1_RISK", "0.07")),
            adaptive_tier2_risk=float(os.getenv("ADAPTIVE_TIER2_RISK", "0.03")),
            adaptive_tier0_max_pos=int(os.getenv("ADAPTIVE_TIER0_MAX_POS", "3")),
            adaptive_tier1_max_pos=int(os.getenv("ADAPTIVE_TIER1_MAX_POS", "4")),
            adaptive_tier2_max_pos=int(os.getenv("ADAPTIVE_TIER2_MAX_POS", "5")),
            pair_cooldown_seconds=float(os.getenv("PAIR_COOLDOWN_SECONDS", "300")),
            buy_max_rsi=float(os.getenv("BUY_MAX_RSI", "85")),
            buy_max_resistance_proximity_pct=float(os.getenv("BUY_MAX_RESISTANCE_PROXIMITY_PCT", "0.01")),
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
        if self.staged_entry_min_equity < 0:
            raise ValueError("STAGED_ENTRY_MIN_EQUITY must be non-negative")
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
        if self.max_exposure_per_coin_pct < 0:
            raise ValueError("MAX_EXPOSURE_PER_COIN_PCT must be non-negative")
        if self.max_daily_loss_pct < 0:
            raise ValueError("MAX_DAILY_LOSS_PCT must be non-negative")
        if self.dynamic_pairs_refresh_cycles < 0:
            raise ValueError("DYNAMIC_PAIRS_REFRESH_CYCLES must be non-negative")
        if self.dynamic_pairs_top_n < 0:
            raise ValueError("DYNAMIC_PAIRS_TOP_N must be non-negative")
        _valid_tfs = {"1", "15", "30", "60", "240", "1D", "3D", "1W"}
        for tf in self.mtf_timeframes:
            if tf not in _valid_tfs:
                raise ValueError(
                    f"MTF_TIMEFRAMES contains invalid timeframe '{tf}'; "
                    f"valid options: {sorted(_valid_tfs)}"
                )
        if not (0.0 <= self.partial_tp_fraction < 1.0):
            raise ValueError("PARTIAL_TP_FRACTION must be in [0, 1)")
        if self.re_entry_cooldown_seconds < 0:
            raise ValueError("RE_ENTRY_COOLDOWN_SECONDS must be non-negative")
        if self.re_entry_dip_pct < 0:
            raise ValueError("RE_ENTRY_DIP_PCT must be non-negative")
        if self.adaptive_interval_min_seconds <= 0:
            raise ValueError("ADAPTIVE_INTERVAL_MIN_SECONDS must be positive")
        if self.max_portfolio_risk_pct < 0:
            raise ValueError("MAX_PORTFOLIO_RISK_PCT must be non-negative")
        if self.min_liquidity_depth_idr < 0:
            raise ValueError("MIN_LIQUIDITY_DEPTH_IDR must be non-negative")
        if not (0.0 <= self.profit_buffer_drawdown_pct < 1.0):
            raise ValueError("PROFIT_BUFFER_DRAWDOWN_PCT must be in [0, 1)")
        if self.trailing_tp_pct < 0:
            raise ValueError("TRAILING_TP_PCT must be non-negative")
        if self.conditional_tp_max_rsi < 0:
            raise ValueError("CONDITIONAL_TP_MAX_RSI must be non-negative")
        if self.orderbook_wall_threshold < 0:
            raise ValueError("ORDERBOOK_WALL_THRESHOLD must be non-negative")
        if self.pump_protection_pct < 0:
            raise ValueError("PUMP_PROTECTION_PCT must be non-negative")
        if self.pump_lookback_seconds <= 0:
            raise ValueError("PUMP_LOOKBACK_SECONDS must be positive")
        if self.max_spread_pct < 0:
            raise ValueError("MAX_SPREAD_PCT must be non-negative")
        if self.see_volume_surge_ratio <= 0:
            raise ValueError("SEE_VOLUME_SURGE_RATIO must be positive")
        if self.see_whale_pressure_min < 0:
            raise ValueError("SEE_WHALE_PRESSURE_MIN must be non-negative")
        if not (0.0 <= self.see_breakout_volume_min <= 1.0):
            raise ValueError("SEE_BREAKOUT_VOLUME_MIN must be between 0 and 1")
        if self.fake_pump_reversal_pct < 0:
            raise ValueError("FAKE_PUMP_REVERSAL_PCT must be non-negative")
        if self.min_order_idr <= 0:
            raise ValueError("MIN_ORDER_IDR must be positive")
        if self.max_consecutive_losses < 0:
            raise ValueError("MAX_CONSECUTIVE_LOSSES must be non-negative")
        if self.volatility_cooldown_pct < 0:
            raise ValueError("VOLATILITY_COOLDOWN_PCT must be non-negative")
        if self.volatility_cooldown_seconds < 0:
            raise ValueError("VOLATILITY_COOLDOWN_SECONDS must be non-negative")
        if self.circuit_breaker_max_errors < 0:
            raise ValueError("CIRCUIT_BREAKER_MAX_ERRORS must be non-negative")
        if self.circuit_breaker_pause_seconds < 0:
            raise ValueError("CIRCUIT_BREAKER_PAUSE_SECONDS must be non-negative")
        if self.stale_order_seconds < 0:
            raise ValueError("STALE_ORDER_SECONDS must be non-negative")
        if self.strategy_auto_disable_losses < 0:
            raise ValueError("STRATEGY_AUTO_DISABLE_LOSSES must be non-negative")
        if not (0.0 <= self.partial_tp2_fraction < 1.0):
            raise ValueError("PARTIAL_TP2_FRACTION must be in [0, 1)")
        if self.partial_tp2_target_pct < 0:
            raise ValueError("PARTIAL_TP2_TARGET_PCT must be non-negative")
        if self.max_open_positions < 0:
            raise ValueError("MAX_OPEN_POSITIONS must be non-negative")
        if self.spread_anomaly_multiplier < 0:
            raise ValueError("SPREAD_ANOMALY_MULTIPLIER must be non-negative")
        if self.orderbook_absorption_threshold < 0:
            raise ValueError("ORDERBOOK_ABSORPTION_THRESHOLD must be non-negative")
        if self.flash_dump_pct < 0:
            raise ValueError("FLASH_DUMP_PCT must be non-negative")
        if self.flash_dump_lookback_seconds <= 0:
            raise ValueError("FLASH_DUMP_LOOKBACK_SECONDS must be positive")
        if self.adaptive_tier1_equity <= 0:
            raise ValueError("ADAPTIVE_TIER1_EQUITY must be positive")
        if self.adaptive_tier2_equity <= self.adaptive_tier1_equity:
            raise ValueError("ADAPTIVE_TIER2_EQUITY must be greater than ADAPTIVE_TIER1_EQUITY")
        for name, val in (
            ("ADAPTIVE_TIER0_RISK", self.adaptive_tier0_risk),
            ("ADAPTIVE_TIER1_RISK", self.adaptive_tier1_risk),
            ("ADAPTIVE_TIER2_RISK", self.adaptive_tier2_risk),
        ):
            if not (0 < val <= 0.5):
                raise ValueError(f"{name} must be in (0, 0.5]")
        for name, val in (
            ("ADAPTIVE_TIER0_MAX_POS", self.adaptive_tier0_max_pos),
            ("ADAPTIVE_TIER1_MAX_POS", self.adaptive_tier1_max_pos),
            ("ADAPTIVE_TIER2_MAX_POS", self.adaptive_tier2_max_pos),
        ):
            if val < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.pair_cooldown_seconds < 0:
            raise ValueError("PAIR_COOLDOWN_SECONDS must be non-negative")
        if self.buy_max_rsi < 0:
            raise ValueError("BUY_MAX_RSI must be non-negative")
        if self.buy_max_resistance_proximity_pct < 0:
            raise ValueError("BUY_MAX_RESISTANCE_PROXIMITY_PCT must be non-negative")
        if not self.dry_run and not self.api_key:
            self.require_auth()
