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
    order_min_interval: float = 2.0  # seconds between order requests; higher reduces 429 risk
    scan_request_delay: float = 0.2  # seconds to wait before each per-pair API call during scanning
    trade_count: int = 1000  # recent trades to fetch per pair when building candles from trades
    min_candles: int = 20  # minimum candle count required for reliable indicator computation
    websocket_enabled: bool = True
    websocket_url: Optional[str] = "wss://ws3.indodax.com/ws/"  # Indodax market-data WebSocket
    websocket_subscribe_message: Optional[str] = None  # raw JSON string sent to the server on connect
    websocket_batch_size: int = 100  # max pairs per WebSocket connection for multi-pair feed
    pairs_per_cycle: int = 20  # 0 = scan all pairs every cycle; >0 = rotate N pairs per cycle
    min_confidence: float = 0.52
    # ── Confidence-based position sizing ──────────────────────────────────────
    # When enabled, position size is a direct percentage of available capital
    # determined by confidence tier rather than the risk/stop-distance formula.
    # Tier thresholds and percentages:
    #   confidence < tier_skip              → skip trade (return 0)
    #   tier_skip  ≤ confidence < tier_low  → tier_low_pct  of capital
    #   tier_low   ≤ confidence < tier_mid  → tier_mid_pct  of capital
    #   tier_mid   ≤ confidence < tier_high → tier_high_pct of capital
    #   confidence ≥ tier_high              → tier_max_pct  of capital
    confidence_position_sizing_enabled: bool = False
    confidence_tier_skip: float = 0.40   # below this: skip trade
    confidence_tier_low: float = 0.50    # 0.40 – 0.50 → low pct
    confidence_tier_mid: float = 0.65    # 0.50 – 0.65 → mid pct
    confidence_tier_high: float = 0.80   # 0.65 – 0.80 → high pct
    confidence_tier_low_pct: float = 0.10   # 10 % of capital
    confidence_tier_mid_pct: float = 0.15   # 15 % of capital
    confidence_tier_high_pct: float = 0.20  # 20 % of capital
    confidence_tier_max_pct: float = 0.25   # 25 % of capital (> tier_high)
    interval_seconds: int = 300
    fast_window: int = 12
    slow_window: int = 48
    max_slippage_pct: float = 0.001
    initial_capital: float = 1_000_000.0  # in quote currency (e.g., IDR)
    target_profit_pct: float = 0.2  # 20%
    max_loss_pct: float = 0.1  # 10%
    trailing_stop_pct: float = 0.03  # 3% trailing stop (rises with price, never falls)
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
    # ── Top Volume Auto Pair Selector ─────────────────────────────────────────
    # Automatically limits the trading watchlist to the top-N most active pairs
    # ranked by 24-h IDR volume × daily price range (volume×volatility score).
    # This focuses the bot exclusively on liquid, actively moving instruments
    # and avoids slow/illiquid coins like tiny-cap pairs that barely move.
    #
    # dynamic_pairs_refresh_cycles:
    #   Number of scan cycles between watchlist refreshes.
    #   0 = disabled (use static pair list forever).
    #   Default 5: refresh every 5 cycles; pairs are re-ranked by
    #   volume×volatility so the list stays current without hammering the API.
    dynamic_pairs_refresh_cycles: int = 5
    # dynamic_pairs_top_n:
    #   Maximum number of pairs kept in the active watchlist after ranking.
    #   Default 20 follows professional practice (scan only the 15–20 best).
    #   0 = no limit (keep all pairs that pass the filters below).
    dynamic_pairs_top_n: int = 20
    # top_volume_min_volume_idr:
    #   Minimum 24-h IDR trading volume a pair must have to be *included in
    #   the dynamic watchlist* (applied before top-N ranking).  This is a
    #   stricter, watchlist-specific threshold that complements MIN_VOLUME_IDR.
    #   E.g. 300_000_000 = only consider pairs with ≥ 300 M IDR daily volume.
    #   0 = no minimum (default).
    top_volume_min_volume_idr: float = 0.0
    # top_volume_min_price_change_24h_pct:
    #   Minimum absolute 24-h price change a pair must show to be included
    #   in the dynamic watchlist.  Excludes stagnant pairs (e.g. DENT at 4 IDR
    #   that sits unchanged for hours) where technical signals are meaningless.
    #   E.g. 0.005 = skip pairs with < 0.5% 24-h price change.
    #   0 = disabled (default).
    top_volume_min_price_change_24h_pct: float = 0.0
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
    trailing_tp_pct: float = 0.02  # 2% trailing floor below peak after TP activation
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
    # ── Minimum price filter ──────────────────────────────────────────────────
    # Skip any buy when the current coin price is below this IDR amount.
    # Prevents entry into very cheap coins (e.g. DENT at 4 IDR) where the
    # minimum price increment represents a disproportionately large move.
    # E.g. 10.0 = skip coins priced below 10 IDR.  0 = disabled (default).
    min_buy_price_idr: float = 0.0
    # ── Tick-move filter ──────────────────────────────────────────────────────
    # Skip any buy when the minimum possible price increment (tick) represents
    # a fraction of the current price that exceeds this threshold.
    # Derived from the gap between the best and second-best bid in the
    # orderbook; falls back to the bid-ask spread when only one bid level
    # exists.
    # E.g. 0.08 = skip if the smallest tick is > 8% of the price.
    # A coin at 4 IDR where 4→5 = 25% would be skipped when set to 0.08.
    # 0 = disabled (default).
    max_tick_move_pct: float = 0.0
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
    # ── Anti rug-pull / dead coin filter ─────────────────────────────────────
    # Skip BUY signals on pairs that look like rug-pulls or dead coins.
    # Criteria are evaluated from the 24-h ticker before any other analysis.
    #
    # rug_pull_max_drop_24h_pct:
    #   Skip pairs where the 24-h high-to-last drop exceeds this fraction.
    #   E.g. 0.50 = skip if price dropped ≥ 50% from its 24-h high.
    #   0 = disabled (default).
    rug_pull_max_drop_24h_pct: float = 0.0
    # rug_pull_min_volume_idr:
    #   Skip pairs whose 24-h IDR volume is below this amount.
    #   Complements the existing MIN_VOLUME_IDR scan filter by applying
    #   a hard per-pair minimum directly inside the rug-pull check.
    #   0 = disabled (default).
    rug_pull_min_volume_idr: float = 0.0
    # rug_pull_min_trades_24h:
    #   Skip pairs with fewer than this many trades in the last 24 h.
    #   0 = disabled (default).
    rug_pull_min_trades_24h: int = 0
    # ── Per-pair minimum order auto-detection ─────────────────────────────────
    # When enabled, the bot fetches the exchange pair list on startup and caches
    # the per-pair minimum trade amounts (coin and IDR).  Before every order the
    # cached minimum is consulted so tiny amounts never reach the exchange.
    # The cache is refreshed automatically every pair_min_order_refresh_cycles.
    # 0 = never refresh (one-time fetch on startup).
    pair_min_order_cache_enabled: bool = True
    pair_min_order_refresh_cycles: int = 0  # 0 = fetch once, never refresh
    # ── Private API response caches ───────────────────────────────────────────
    # TTL-based in-memory caches for expensive private REST endpoints that are
    # polled repeatedly during normal operation.  Caching reduces unnecessary
    # API requests and protects against Indodax rate limits (HTTP 429).
    #
    # account_info_cache_ttl:
    #   Seconds to cache the response of ``getInfo`` (balance / account details).
    #   The balance is refreshed at most once per this interval; stale data is
    #   served for any call within the TTL window.  0 = disabled (always live).
    #   Default: 30 seconds.
    #
    # open_orders_cache_ttl:
    #   Seconds to cache the response of ``openOrders`` per pair.
    #   0 = disabled (always live).  Default: 15 seconds.
    account_info_cache_ttl: float = 30.0
    open_orders_cache_ttl: float = 15.0
    # ── Time-based exit (anti-stagnation) ─────────────────────────────────────
    # Force-sell a position that has been open longer than this many seconds
    # without reaching the profit threshold below.  Prevents capital from being
    # locked in illiquid / slow-moving coins (e.g. tiny-cap pairs on Indodax).
    #
    # max_hold_seconds:
    #   Maximum number of seconds to hold an open position before the
    #   time-based exit check fires.  0 = disabled (default).
    #   E.g. 1800 = exit after 30 minutes.
    #
    # max_hold_profit_pct:
    #   The time-exit only triggers when the unrealised profit is *below* this
    #   fraction.  Positions that are already profitable beyond this level are
    #   left alone to continue running.
    #   E.g. 0.01 = only force-exit if profit < 1% after max_hold_seconds.
    #   Default: 0.01.
    max_hold_seconds: float = 0.0
    max_hold_profit_pct: float = 0.01
    # ── Adaptive hold time based on volume ────────────────────────────────────
    # When volume_high_threshold_idr > 0, the effective max_hold_seconds is
    # chosen dynamically:
    #   - volume >= threshold  → max_hold_seconds_volume_high (default 5400 = 90 min)
    #   - volume <  threshold  → max_hold_seconds_volume_low  (default 1800 = 30 min)
    # Set volume_high_threshold_idr = 0 to disable adaptive hold time.
    volume_high_threshold_idr: float = 0.0
    max_hold_seconds_volume_high: float = 5400.0  # 90 minutes
    max_hold_seconds_volume_low: float = 1800.0   # 30 minutes
    # ── Multi-position / parallel trading ─────────────────────────────────────
    # Allow the bot to hold up to *multi_position_max* open positions across
    # different pairs simultaneously.  Capital is split evenly among all open
    # slots; each position manages its own PortfolioTracker.
    #
    # multi_position_enabled:
    #   When True the bot can open up to multi_position_max trades in parallel.
    #   False = classic single-position mode (default).
    #
    # multi_position_max:
    #   Maximum number of simultaneous open positions.  E.g. 3 = at most three
    #   different pairs can be held at the same time.  Default: 3.
    multi_position_enabled: bool = False
    multi_position_max: int = 3
    # ── Orderbook imbalance position-size boost ────────────────────────────────
    # Multiply the computed position size by ob_imbalance_size_multiplier when
    # the order-book imbalance exceeds ob_imbalance_boost_threshold.  This lets
    # the bot enter with a larger stake when strong buy pressure is detected
    # (bid volume >> ask volume) — a common precursor to a pump on Indodax.
    #
    # ob_imbalance_boost_threshold:
    #   Minimum imbalance value to trigger the boost.
    #   Imbalance = (bid_vol − ask_vol) / (bid_vol + ask_vol), range −1 … +1.
    #   E.g. 0.50 ≈ bid volume is 3× ask volume.  0 = disabled (default).
    #
    # ob_imbalance_size_multiplier:
    #   Factor by which the position size is multiplied when the threshold is met.
    #   E.g. 2.0 = double the normal entry size on a whale-bid signal.
    ob_imbalance_boost_threshold: float = 0.0
    ob_imbalance_size_multiplier: float = 2.0
    # ── Entry imbalance guard ──────────────────────────────────────────────────
    # Hard-skip BUY when the order-book imbalance is below this threshold
    # (i.e. when sellers dominate the book).
    # Imbalance = (bid_vol − ask_vol) / (bid_vol + ask_vol), range −1 … +1.
    # E.g. −0.1 = skip buy when ask volume exceeds bid volume by ≥ ~11%.
    # 0 = disabled (default). Typical value: −0.15 to block clear seller dominance.
    ob_imbalance_min_entry: float = 0.0
    # ── Trade flow (buy/sell ratio) entry filter ───────────────────────────────
    # Skip BUY when fewer than this fraction of recent trades were buyer-initiated
    # (market orders hitting the ask).  Range 0–1.
    # E.g. 0.45 = skip buy when less than 45% of trades are market buys.
    # 0 = disabled (default). Typical value: 0.45.
    trade_flow_min_buy_ratio: float = 0.0
    # ── Momentum exit (early exit on weakening momentum) ──────────────────────
    # Exit an open profitable position BEFORE reaching the TP target when
    # market momentum fades. Two conditions must BOTH be met to trigger the
    # early exit:
    #
    # momentum_exit_ob_threshold:
    #   Order-book imbalance falls below this level (seller pressure detected).
    #   Range −1 … +1. E.g. 0.0 = exit when book becomes seller-dominant.
    #   0 = disabled (default).
    #
    # momentum_exit_min_profit_pct:
    #   Only trigger the early exit when the unrealised profit is at least this
    #   fraction.  Prevents exiting at a loss on a brief dip.
    #   E.g. 0.01 = require ≥ 1% profit before early exit.
    #   0 = disabled (default; requires momentum_exit_ob_threshold > 0 too).
    momentum_exit_ob_threshold: float = 0.0
    momentum_exit_min_profit_pct: float = 0.0
    # ── Multi-level partial take profit (3rd level) ────────────────────────────
    # Sell a fraction of the position at a third TP target price.
    # partial_tp3_fraction: fraction to sell (0 = disabled).
    # partial_tp3_target_pct: price must rise this % above buy price to trigger.
    partial_tp3_fraction: float = 0.0
    partial_tp3_target_pct: float = 0.0

    # ── Liquidity Sweep Detection ──────────────────────────────────────────────
    # Detect when price sweeps through a key level and quickly reverses.
    # 0 = disabled (default).
    liquidity_sweep_enabled: bool = False
    liquidity_sweep_lookback: int = 10  # candles to look back for the key level
    liquidity_sweep_min_pct: float = 0.01  # minimum sweep size (1% default)
    liquidity_sweep_reversal_pct: float = 0.005  # reversal needed to confirm (0.5%)

    # ── Liquidity Trap Detection ───────────────────────────────────────────────
    # Detect false breakouts that trap buyers/sellers.
    # 0 = disabled (default).
    liquidity_trap_enabled: bool = False
    liquidity_trap_breakout_pct: float = 0.005  # min breakout size to flag
    liquidity_trap_reversal_pct: float = 0.008  # reversal needed to confirm trap

    # ── Liquidity Vacuum Detection ────────────────────────────────────────────
    # Skip buy when there is a large gap (vacuum) in the orderbook above current price.
    # 0 = disabled (default).
    liquidity_vacuum_min_gap_pct: float = 0.0  # min gap fraction to flag (0=disabled)
    liquidity_vacuum_depth_levels: int = 10  # levels to inspect

    # ── Smart Money Footprint ──────────────────────────────────────────────────
    # Detect institutional accumulation/distribution via volume-price divergence.
    # 0 = disabled (default).
    smart_money_enabled: bool = False
    smart_money_volume_factor: float = 3.0  # volume spike multiplier threshold
    smart_money_divergence_lookback: int = 5  # candles for divergence check

    # ── Volume Acceleration ────────────────────────────────────────────────────
    # Detect accelerating volume trend (volume derivative) as a buy amplifier.
    # false = disabled (default).
    volume_accel_enabled: bool = False
    volume_accel_window: int = 5  # candles to compute acceleration over
    volume_accel_min_ratio: float = 1.5  # acceleration ratio threshold

    # ── Micro Trend Detection ──────────────────────────────────────────────────
    # Use a short-window (e.g. last 3 candles) micro trend to filter entries.
    # false = disabled (default).
    micro_trend_enabled: bool = False
    micro_trend_window: int = 3  # candles for micro trend (shorter than fast_window)

    # ── Spread Expansion Detector ─────────────────────────────────────────────
    # Block buy when the current spread is expanding (> multiplier * recent avg).
    # 0 = disabled (default).
    spread_expansion_enabled: bool = False
    spread_expansion_multiplier: float = 2.0  # current > this * recent_avg = expanding
    spread_expansion_window: int = 10  # recent candle count to build spread baseline

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
            order_min_interval=float(os.getenv("ORDER_MIN_INTERVAL", "2.0")),
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
            trailing_stop_pct=float(os.getenv("TRAILING_STOP_PCT", "0.03")),
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
            top_volume_min_volume_idr=float(os.getenv("TOP_VOLUME_MIN_VOLUME_IDR", "0")),
            top_volume_min_price_change_24h_pct=float(os.getenv("TOP_VOLUME_MIN_PRICE_CHANGE_24H_PCT", "0")),
            partial_tp_fraction=float(os.getenv("PARTIAL_TP_FRACTION", "0")),
            re_entry_cooldown_seconds=float(os.getenv("RE_ENTRY_COOLDOWN_SECONDS", "0")),
            re_entry_dip_pct=float(os.getenv("RE_ENTRY_DIP_PCT", "0")),
            adaptive_interval_enabled=os.getenv("ADAPTIVE_INTERVAL_ENABLED", "false").lower() in {"1", "true", "yes"},
            adaptive_interval_min_seconds=int(os.getenv("ADAPTIVE_INTERVAL_MIN_SECONDS", "30")),
            max_portfolio_risk_pct=float(os.getenv("MAX_PORTFOLIO_RISK_PCT", "0")),
            min_liquidity_depth_idr=float(os.getenv("MIN_LIQUIDITY_DEPTH_IDR", "0")),
            profit_buffer_drawdown_pct=float(os.getenv("PROFIT_BUFFER_DRAWDOWN_PCT", "0")),
            trailing_tp_pct=float(os.getenv("TRAILING_TP_PCT", "0.02")),
            conditional_tp_min_trend_strength=float(os.getenv("CONDITIONAL_TP_MIN_TREND_STRENGTH", "0")),
            conditional_tp_min_ob_imbalance=float(os.getenv("CONDITIONAL_TP_MIN_OB_IMBALANCE", "0")),
            conditional_tp_max_rsi=float(os.getenv("CONDITIONAL_TP_MAX_RSI", "0")),
            orderbook_wall_threshold=float(os.getenv("ORDERBOOK_WALL_THRESHOLD", "0")),
            pump_protection_pct=float(os.getenv("PUMP_PROTECTION_PCT", "0")),
            pump_lookback_seconds=float(os.getenv("PUMP_LOOKBACK_SECONDS", "60")),
            max_spread_pct=float(os.getenv("MAX_SPREAD_PCT", "0")),
            min_buy_price_idr=float(os.getenv("MIN_BUY_PRICE_IDR", "0")),
            max_tick_move_pct=float(os.getenv("MAX_TICK_MOVE_PCT", "0")),
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
            rug_pull_max_drop_24h_pct=float(os.getenv("RUG_PULL_MAX_DROP_24H_PCT", "0")),
            rug_pull_min_volume_idr=float(os.getenv("RUG_PULL_MIN_VOLUME_IDR", "0")),
            rug_pull_min_trades_24h=int(os.getenv("RUG_PULL_MIN_TRADES_24H", "0")),
            pair_min_order_cache_enabled=os.getenv("PAIR_MIN_ORDER_CACHE_ENABLED", "true").lower() in {"1", "true", "yes"},
            pair_min_order_refresh_cycles=int(os.getenv("PAIR_MIN_ORDER_REFRESH_CYCLES", "0")),
            account_info_cache_ttl=float(os.getenv("ACCOUNT_INFO_CACHE_TTL", "30")),
            open_orders_cache_ttl=float(os.getenv("OPEN_ORDERS_CACHE_TTL", "15")),
            confidence_position_sizing_enabled=os.getenv("CONFIDENCE_POSITION_SIZING_ENABLED", "false").lower() in {"1", "true", "yes"},
            confidence_tier_skip=float(os.getenv("CONFIDENCE_TIER_SKIP", "0.40")),
            confidence_tier_low=float(os.getenv("CONFIDENCE_TIER_LOW", "0.50")),
            confidence_tier_mid=float(os.getenv("CONFIDENCE_TIER_MID", "0.65")),
            confidence_tier_high=float(os.getenv("CONFIDENCE_TIER_HIGH", "0.80")),
            confidence_tier_low_pct=float(os.getenv("CONFIDENCE_TIER_LOW_PCT", "0.10")),
            confidence_tier_mid_pct=float(os.getenv("CONFIDENCE_TIER_MID_PCT", "0.15")),
            confidence_tier_high_pct=float(os.getenv("CONFIDENCE_TIER_HIGH_PCT", "0.20")),
            confidence_tier_max_pct=float(os.getenv("CONFIDENCE_TIER_MAX_PCT", "0.25")),
            max_hold_seconds=float(os.getenv("MAX_HOLD_SECONDS", "0")),
            max_hold_profit_pct=float(os.getenv("MAX_HOLD_PROFIT_PCT", "0.01")),
            volume_high_threshold_idr=float(os.getenv("VOLUME_HIGH_THRESHOLD_IDR", "0")),
            max_hold_seconds_volume_high=float(os.getenv("MAX_HOLD_SECONDS_VOLUME_HIGH", "5400")),
            max_hold_seconds_volume_low=float(os.getenv("MAX_HOLD_SECONDS_VOLUME_LOW", "1800")),
            multi_position_enabled=os.getenv("MULTI_POSITION_ENABLED", "false").lower() in {"1", "true", "yes"},
            multi_position_max=int(os.getenv("MULTI_POSITION_MAX", "3")),
            ob_imbalance_boost_threshold=float(os.getenv("OB_IMBALANCE_BOOST_THRESHOLD", "0")),
            ob_imbalance_size_multiplier=float(os.getenv("OB_IMBALANCE_SIZE_MULTIPLIER", "2.0")),
            ob_imbalance_min_entry=float(os.getenv("OB_IMBALANCE_MIN_ENTRY", "0")),
            trade_flow_min_buy_ratio=float(os.getenv("TRADE_FLOW_MIN_BUY_RATIO", "0")),
            momentum_exit_ob_threshold=float(os.getenv("MOMENTUM_EXIT_OB_THRESHOLD", "0")),
            momentum_exit_min_profit_pct=float(os.getenv("MOMENTUM_EXIT_MIN_PROFIT_PCT", "0")),
            partial_tp3_fraction=float(os.getenv("PARTIAL_TP3_FRACTION", "0")),
            partial_tp3_target_pct=float(os.getenv("PARTIAL_TP3_TARGET_PCT", "0")),
            liquidity_sweep_enabled=os.getenv("LIQUIDITY_SWEEP_ENABLED", "false").lower() in {"1", "true", "yes"},
            liquidity_sweep_lookback=int(os.getenv("LIQUIDITY_SWEEP_LOOKBACK", "10")),
            liquidity_sweep_min_pct=float(os.getenv("LIQUIDITY_SWEEP_MIN_PCT", "0.01")),
            liquidity_sweep_reversal_pct=float(os.getenv("LIQUIDITY_SWEEP_REVERSAL_PCT", "0.005")),
            liquidity_trap_enabled=os.getenv("LIQUIDITY_TRAP_ENABLED", "false").lower() in {"1", "true", "yes"},
            liquidity_trap_breakout_pct=float(os.getenv("LIQUIDITY_TRAP_BREAKOUT_PCT", "0.005")),
            liquidity_trap_reversal_pct=float(os.getenv("LIQUIDITY_TRAP_REVERSAL_PCT", "0.008")),
            liquidity_vacuum_min_gap_pct=float(os.getenv("LIQUIDITY_VACUUM_MIN_GAP_PCT", "0")),
            liquidity_vacuum_depth_levels=int(os.getenv("LIQUIDITY_VACUUM_DEPTH_LEVELS", "10")),
            smart_money_enabled=os.getenv("SMART_MONEY_ENABLED", "false").lower() in {"1", "true", "yes"},
            smart_money_volume_factor=float(os.getenv("SMART_MONEY_VOLUME_FACTOR", "3.0")),
            smart_money_divergence_lookback=int(os.getenv("SMART_MONEY_DIVERGENCE_LOOKBACK", "5")),
            volume_accel_enabled=os.getenv("VOLUME_ACCEL_ENABLED", "false").lower() in {"1", "true", "yes"},
            volume_accel_window=int(os.getenv("VOLUME_ACCEL_WINDOW", "5")),
            volume_accel_min_ratio=float(os.getenv("VOLUME_ACCEL_MIN_RATIO", "1.5")),
            micro_trend_enabled=os.getenv("MICRO_TREND_ENABLED", "false").lower() in {"1", "true", "yes"},
            micro_trend_window=int(os.getenv("MICRO_TREND_WINDOW", "3")),
            spread_expansion_enabled=os.getenv("SPREAD_EXPANSION_ENABLED", "false").lower() in {"1", "true", "yes"},
            spread_expansion_multiplier=float(os.getenv("SPREAD_EXPANSION_MULTIPLIER", "2.0")),
            spread_expansion_window=int(os.getenv("SPREAD_EXPANSION_WINDOW", "10")),
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
        if self.top_volume_min_volume_idr < 0:
            raise ValueError("TOP_VOLUME_MIN_VOLUME_IDR must be non-negative")
        if self.top_volume_min_price_change_24h_pct < 0:
            raise ValueError("TOP_VOLUME_MIN_PRICE_CHANGE_24H_PCT must be non-negative")
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
        if self.min_buy_price_idr < 0:
            raise ValueError("MIN_BUY_PRICE_IDR must be non-negative")
        if self.max_tick_move_pct < 0:
            raise ValueError("MAX_TICK_MOVE_PCT must be non-negative")
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
        if self.rug_pull_max_drop_24h_pct < 0 or self.rug_pull_max_drop_24h_pct > 1:
            raise ValueError("RUG_PULL_MAX_DROP_24H_PCT must be in [0, 1]")
        if self.rug_pull_min_volume_idr < 0:
            raise ValueError("RUG_PULL_MIN_VOLUME_IDR must be non-negative")
        if self.rug_pull_min_trades_24h < 0:
            raise ValueError("RUG_PULL_MIN_TRADES_24H must be non-negative")
        if self.pair_min_order_refresh_cycles < 0:
            raise ValueError("PAIR_MIN_ORDER_REFRESH_CYCLES must be non-negative")
        if self.account_info_cache_ttl < 0:
            raise ValueError("ACCOUNT_INFO_CACHE_TTL must be non-negative")
        if self.open_orders_cache_ttl < 0:
            raise ValueError("OPEN_ORDERS_CACHE_TTL must be non-negative")
        if self.max_hold_seconds < 0:
            raise ValueError("MAX_HOLD_SECONDS must be non-negative")
        if self.max_hold_profit_pct < 0:
            raise ValueError("MAX_HOLD_PROFIT_PCT must be non-negative")
        if self.volume_high_threshold_idr < 0:
            raise ValueError("VOLUME_HIGH_THRESHOLD_IDR must be non-negative")
        if self.max_hold_seconds_volume_high < 0:
            raise ValueError("MAX_HOLD_SECONDS_VOLUME_HIGH must be non-negative")
        if self.max_hold_seconds_volume_low < 0:
            raise ValueError("MAX_HOLD_SECONDS_VOLUME_LOW must be non-negative")
        if self.multi_position_max < 1:
            raise ValueError("MULTI_POSITION_MAX must be >= 1")
        if self.ob_imbalance_boost_threshold < 0 or self.ob_imbalance_boost_threshold > 1:
            raise ValueError("OB_IMBALANCE_BOOST_THRESHOLD must be in [0, 1]")
        if self.ob_imbalance_size_multiplier <= 0:
            raise ValueError("OB_IMBALANCE_SIZE_MULTIPLIER must be positive")
        if not (-1.0 <= self.ob_imbalance_min_entry <= 1.0):
            raise ValueError("OB_IMBALANCE_MIN_ENTRY must be in [-1, 1]")
        if not (0.0 <= self.trade_flow_min_buy_ratio <= 1.0):
            raise ValueError("TRADE_FLOW_MIN_BUY_RATIO must be in [0, 1]")
        if not (-1.0 <= self.momentum_exit_ob_threshold <= 1.0):
            raise ValueError("MOMENTUM_EXIT_OB_THRESHOLD must be in [-1, 1]")
        if self.momentum_exit_min_profit_pct < 0:
            raise ValueError("MOMENTUM_EXIT_MIN_PROFIT_PCT must be non-negative")
        if not (0.0 <= self.partial_tp3_fraction < 1.0):
            raise ValueError("PARTIAL_TP3_FRACTION must be in [0, 1)")
        if self.partial_tp3_target_pct < 0:
            raise ValueError("PARTIAL_TP3_TARGET_PCT must be non-negative")
        if self.confidence_position_sizing_enabled:
            for name, val in (
                ("CONFIDENCE_TIER_SKIP", self.confidence_tier_skip),
                ("CONFIDENCE_TIER_LOW", self.confidence_tier_low),
                ("CONFIDENCE_TIER_MID", self.confidence_tier_mid),
                ("CONFIDENCE_TIER_HIGH", self.confidence_tier_high),
            ):
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"{name} must be in [0, 1]")
            for name, val in (
                ("CONFIDENCE_TIER_LOW_PCT", self.confidence_tier_low_pct),
                ("CONFIDENCE_TIER_MID_PCT", self.confidence_tier_mid_pct),
                ("CONFIDENCE_TIER_HIGH_PCT", self.confidence_tier_high_pct),
                ("CONFIDENCE_TIER_MAX_PCT", self.confidence_tier_max_pct),
            ):
                if not (0.0 < val <= 1.0):
                    raise ValueError(f"{name} must be in (0, 1]")
        if not self.dry_run and not self.api_key:
            self.require_auth()
