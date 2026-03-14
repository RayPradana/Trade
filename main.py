from __future__ import annotations

import concurrent.futures
import datetime
import logging
import os
import signal
import sys
import threading
import time
from typing import Optional

import requests

from bot.config import BotConfig
from bot.trader import Trader
from bot.journal import TradeJournal
from bot.risk_management import (
    check_circuit_breaker as rm_check_circuit_breaker,
    check_max_drawdown,
    adjust_risk_dynamically,
    check_anomaly_shutdown,
)
from bot.portfolio_management import (
    evaluate_multi_asset_portfolio,
    plan_rebalance,
    assess_diversification,
    compute_correlation_matrix,
)
from bot.ml_models import (
    detect_regime,
    optimize_strategy,
    detect_anomalies as ml_detect_anomalies,
)
from bot.scanning import (
    scan_momentum,
    scan_trends,
)
from bot.execution import (
    monitor_execution_quality,
)
from bot.autonomous import (
    AutonomousTradingState,
    run_autonomous_cycle,
    check_autonomous_health,
    rotate_pairs,
    auto_switch_strategy,
    CrashEvent,
    decide_restart,
    ComponentHealth,
    evaluate_failover,
    ScheduledTask,
    schedule_tasks,
    update_task_after_run,
    PollingConfig,
    adjust_polling_interval,
    diagnose_and_recover,
)

# ── ANSI codes — disabled when stdout is not a terminal ─────────────────
_USE_COLOR = sys.stdout.isatty()

_RESET   = "\033[0m"  if _USE_COLOR else ""
_BOLD    = "\033[1m"  if _USE_COLOR else ""
_DIM     = "\033[2m"  if _USE_COLOR else ""
_CYAN    = "\033[36m" if _USE_COLOR else ""
_GREEN   = "\033[32m" if _USE_COLOR else ""
_YELLOW  = "\033[33m" if _USE_COLOR else ""
_RED     = "\033[31m" if _USE_COLOR else ""
_MAGENTA = "\033[35m" if _USE_COLOR else ""
_BLUE    = "\033[34m" if _USE_COLOR else ""
_WHITE   = "\033[97m" if _USE_COLOR else ""

_LEVEL_COLORS = {
    "DEBUG":    _CYAN,
    "INFO":     _GREEN,
    "WARNING":  _YELLOW,
    "ERROR":    _RED,
    "CRITICAL": _MAGENTA,
}

_ACTION_ICONS = {
    "buy":  "📈",
    "sell": "📉",
    "hold": "⏸️",
    "grid": "🔲",
}

_STATUS_ICONS = {
    "simulated":      "🧪",
    "placed":         "✅",
    "skipped":        "⏭️",
    "hold":           "⏸️",
    "stopped":        "🛑",
    "force_sold":     "📤",
    "grid_simulated": "🔲",
    "grid_placed":    "🔲",
}

# ── Display primitives ───────────────────────────────────────────────────

def _conf_bar(conf: float, width: int = 10) -> str:
    """Unicode block progress-bar for a confidence value (0–1)."""
    filled = int(conf * width)
    bar = "█" * filled + "░" * (width - filled)
    color = _GREEN if conf >= 0.7 else (_YELLOW if conf >= 0.52 else _RED)
    return f"{color}{bar}{_RESET} {_BOLD}{conf:.3f}{_RESET}"


def _pnl_str(value: float) -> str:
    """Colored PnL string: green ▲ for profit, red ▼ for loss."""
    if value > 0:
        return f"{_GREEN}{_BOLD}▲ +{value:,.2f}{_RESET}"
    if value < 0:
        return f"{_RED}{_BOLD}▼ {value:,.2f}{_RESET}"
    return f"{_DIM}  {value:,.2f}{_RESET}"


def _pct_str(value: float) -> str:
    """Colored percentage string."""
    if value > 0:
        return f"{_GREEN}+{value:.2f}%{_RESET}"
    if value < 0:
        return f"{_RED}{value:.2f}%{_RESET}"
    return f" {value:.2f}%"


def _idr(value: float) -> str:
    """Format a number with thousands-separator (IDR style)."""
    return f"Rp {value:>15,.2f}"


def _idr_compact(value: float) -> str:
    """Compact IDR display: billions → 'B', millions → 'M', otherwise full."""
    if value >= 1_000_000_000:
        return f"Rp {value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"Rp {value / 1_000_000:.2f}M"
    return f"Rp {value:,.2f}"


class _ColoredFormatter(logging.Formatter):
    """Formatter that adds ANSI color codes to log level names."""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname, "")
        record = logging.makeLogRecord(record.__dict__)
        record.levelname = f"{color}{record.levelname:8}{_RESET}"
        return super().format(record)


def configure_logging(log_file: Optional[str] = None) -> None:
    """Configure root logger with a coloured console handler.

    When *log_file* is provided an additional plain-text :class:`FileHandler`
    is attached so every log record is written to both the terminal and the
    file simultaneously.  The file is opened in *append* mode so restarting
    the bot does not overwrite previous runs.

    ``force=True`` is passed to :func:`logging.basicConfig` so that the
    function is idempotent and can be called more than once (e.g. once for
    early bootstrap and again after the config is loaded with a log-file path).
    Without ``force=True`` the second call would be a silent no-op and file
    logging would never be activated.
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        _ColoredFormatter(
            fmt=f"{_DIM}%(asctime)s{_RESET} [%(levelname)s] {_CYAN}%(name)s{_RESET}: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handlers: list = [console_handler]

    if log_file:
        try:
            # Ensure parent directory exists
            log_dir = os.path.dirname(os.path.abspath(log_file))
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(
                logging.Formatter(
                    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
            handlers.append(file_handler)
        except OSError as exc:
            # Don't crash if the log file can't be created; just warn.
            logging.basicConfig(level=logging.INFO, handlers=[console_handler], force=True)
            logging.warning("Could not open log file '%s': %s", log_file, exc)
            return

    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True)


# ── Telegram notifications ───────────────────────────────────────────────

def _send_telegram(token: str, chat_id: str, text: str) -> None:
    """Send a plain-text message to a Telegram chat via the Bot API.

    Failures are logged as warnings and never propagated — notifications are
    best-effort and must not interrupt the trading loop.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
        if not resp.ok:
            logging.warning(
                "Telegram notification failed (HTTP %d): %s", resp.status_code, resp.text[:200]
            )
    except Exception as exc:
        logging.warning("Telegram notification error: %s", exc)


def _send_discord(webhook_url: str, text: str) -> None:
    """Send a plain-text message to a Discord channel via an Incoming Webhook.

    Failures are logged as warnings and never propagated.
    """
    try:
        resp = requests.post(
            webhook_url,
            json={"content": text},
            timeout=10,
        )
        if not resp.ok:
            logging.warning(
                "Discord notification failed (HTTP %d): %s", resp.status_code, resp.text[:200]
            )
    except Exception as exc:
        logging.warning("Discord notification error: %s", exc)


def _notify(config: BotConfig, text: str) -> None:
    """Send notifications via Telegram and/or Discord when configured."""
    if config.telegram_token and config.telegram_chat_id:
        _send_telegram(config.telegram_token, config.telegram_chat_id, text)
    if config.discord_webhook_url:
        _send_discord(config.discord_webhook_url, text)

def _separator(label: str = "") -> str:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    width = 68
    if label:
        body = f" {label} "
        ts_tag = f" {ts} "
        dashes = max(4, width - len(body) - len(ts_tag))
        left  = dashes // 2
        right = dashes - left
        return (
            f"{_BOLD}{_BLUE}{'═' * left}{body}{'═' * right}{_RESET}"
            f"{_DIM}{ts_tag}{_RESET}"
        )
    return f"{_DIM}{'─' * width}  {ts}{_RESET}"


# ── Startup banner ────────────────────────────────────────────────────────

def _log_startup_banner(config: BotConfig) -> None:
    """Print a startup banner summarising key configuration."""
    dry_tag  = (f"{_GREEN}✅ YES (simulated){_RESET}"
                if config.dry_run else f"{_RED}🔴 NO  (LIVE){_RESET}")
    resume   = str(config.state_path) if config.state_path else f"{_DIM}disabled{_RESET}"
    sep = f"{_DIM}{'─' * 50}{_RESET}"
    logging.info(sep)
    logging.info("🤖  %s%sINDODAX TRADING BOT%s  %sby RayPradana%s",
                 _BOLD, _WHITE, _RESET, _DIM, _RESET)
    logging.info(sep)
    logging.info("   %-12s %s%-12s%s  │  dry-run  : %s",
                 "mode     :", _BOLD, config.trade_mode.upper(), _RESET, dry_tag)
    logging.info("   %-12s %s  │  pair     : %s%s%s",
                 "capital  :", f"{_BOLD}Rp {config.initial_capital:>18,.0f}{_RESET}",
                 _BOLD, config.pair, _RESET)
    logging.info("   %-12s %s+%.1f%%%s      │  max-loss : %s-%.1f%%%s",
                 "target   :",
                 _GREEN, config.target_profit_pct * 100, _RESET,
                 _RED,   config.max_loss_pct * 100,      _RESET)
    logging.info("   %-12s %s", "resume   :", resume)
    logging.info(sep)


# ── Account info / balance display ───────────────────────────────────────

def _log_account_info(info: dict) -> None:
    """Log the Indodax account balance and account details in a rich tree format.

    *info* is the raw ``getInfo`` response dict from the private API:
    ``{"success": 1, "return": {"balance": {"idr": "...","btc": "...",...}, ...}}``.
    """
    ret = info.get("return") or {}
    balance = ret.get("balance") or {}
    balance_hold = ret.get("balance_hold") or {}
    name  = ret.get("name")  or "—"
    email = ret.get("email") or "—"
    user_id = ret.get("user_id") or "—"
    verification = ret.get("verification_status") or "—"
    server_time_raw = ret.get("server_time")
    server_ts = (
        datetime.datetime.fromtimestamp(int(server_time_raw)).strftime("%Y-%m-%d %H:%M:%S")
        if server_time_raw else "—"
    )

    sep = f"{_DIM}{'─' * 50}{_RESET}"
    logging.info(sep)
    logging.info("💼  %s%sACCOUNT INFO%s  %sIndodax%s",
                 _BOLD, _WHITE, _RESET, _DIM, _RESET)
    logging.info(sep)

    if name and name != "—":
        logging.info("   %-14s %s%s%s", "name     :", _BOLD, name, _RESET)
    if user_id and user_id != "—":
        logging.info("   %-14s %s%s%s", "user id  :", _DIM, user_id, _RESET)
    if email and email != "—":
        logging.info("   %-14s %s%s%s", "email    :", _DIM, email, _RESET)
    if verification and verification != "—":
        verified_icon = "✅" if str(verification).lower() == "verified" else "⚠️"
        logging.info("   %-14s %s %s%s", "status   :", verified_icon, verification, _RESET)
    logging.info("   %-14s %s%s%s", "server   :", _DIM, server_ts, _RESET)
    logging.info(sep)

    # ── IDR cash balance ─────────────────────────────────────────────────
    idr_free = float(balance.get("idr") or "0")
    idr_hold = float(balance_hold.get("idr") or "0")
    idr_total = idr_free + idr_hold
    logging.info("   %s%s%s", _BOLD, "💰  Balances", _RESET)
    logging.info(
        "   %s├─%s %s%-6s%s  free : %s    hold : %s    total : %s%s%s",
        _DIM, _RESET,
        _BOLD, "IDR", _RESET,
        f"{_GREEN}Rp {idr_free:>18,.2f}{_RESET}",
        f"{_YELLOW}Rp {idr_hold:>18,.2f}{_RESET}",
        _BOLD, f"Rp {idr_total:>18,.2f}", _RESET,
    )

    # ── Non-zero coin balances ────────────────────────────────────────────
    coin_rows = []
    for coin, free_str in sorted(balance.items()):
        if coin == "idr":
            continue
        free  = float(free_str or "0")
        held  = float(balance_hold.get(coin) or "0")
        if free <= 0 and held <= 0:
            continue
        coin_rows.append((coin, free, held))

    for i, (coin, free, held) in enumerate(coin_rows):
        is_last = (i == len(coin_rows) - 1)
        connector = "└─" if is_last else "├─"
        total = free + held
        logging.info(
            "   %s%s%s %s%-6s%s  free : %s%.8f%s    hold : %s%.8f%s    total : %s%.8f%s",
            _DIM, connector, _RESET,
            _BOLD, coin.upper(), _RESET,
            _GREEN,  free,  _RESET,
            _YELLOW, held,  _RESET,
            _BOLD,   total, _RESET,
        )

    if not coin_rows:
        logging.info("   %s└─%s %sno open coin positions%s", _DIM, _RESET, _DIM, _RESET)

    logging.info(sep)


def _log_account_info_dry() -> None:
    """Placeholder account display used when running in dry-run mode."""
    sep = f"{_DIM}{'─' * 50}{_RESET}"
    logging.info(sep)
    logging.info(
        "💼  %sACCOUNT INFO%s  %s(dry-run — no API call made)%s",
        _BOLD, _RESET, _DIM, _RESET,
    )
    logging.info(sep)



def _log_signal(snapshot: dict) -> None:
    """Log a buy/sell signal with market data and indicators in tree format."""
    decision = snapshot["decision"]
    ob       = snapshot.get("orderbook")
    levels   = snapshot.get("levels")
    vol      = snapshot.get("volatility")
    ind      = snapshot.get("indicators")
    pair     = snapshot["pair"]
    price    = snapshot["price"]

    action_color = (_GREEN if decision.action == "buy"
                    else _RED if decision.action == "sell" else _YELLOW)
    icon = _ACTION_ICONS.get(decision.action, "❓")

    # ── Header line ──────────────────────────────────────────────────────
    logging.info(
        "%s %s%s %s%s  %s%s%s  ·  %s  ·  conf %s  ·  %s%s%s",
        icon,
        action_color, _BOLD, decision.action.upper(), _RESET,
        _BOLD, pair, _RESET,
        _idr(price),
        _conf_bar(decision.confidence),
        _DIM, decision.mode, _RESET,
    )

    # ── Reason ───────────────────────────────────────────────────────────
    logging.info("   %s├─%s reason  : %s", _DIM, _RESET, decision.reason)

    # ── Market data ──────────────────────────────────────────────────────
    if ob:
        spread_str = f"{ob.spread_pct * 100:.4f}%"
        imb_val    = ob.imbalance
        imb_str    = f"{imb_val:+.3f}"
        imb_color  = _GREEN if imb_val > 0.1 else (_RED if imb_val < -0.1 else _YELLOW)
    else:
        spread_str = "N/A"
        imb_str    = "N/A"
        imb_color  = _DIM
    vol_str = f"{vol.volatility:.4f}" if vol else "N/A"
    logging.info(
        "   %s├─%s market  : spread=%s%s%s  imbalance=%s%s%s  vol=%s%s%s",
        _DIM, _RESET,
        _DIM, spread_str, _RESET,
        imb_color, imb_str, _RESET,
        _DIM, vol_str, _RESET,
    )

    # ── Support / Resistance ─────────────────────────────────────────────
    sup = (f"{_GREEN}{_idr_compact(levels.support)}{_RESET}"
           if (levels and levels.support) else f"{_DIM}N/A{_RESET}")
    res = (f"{_RED}{_idr_compact(levels.resistance)}{_RESET}"
           if (levels and levels.resistance) else f"{_DIM}N/A{_RESET}")
    logging.info(
        "   %s├─%s levels  : support=%s    resistance=%s",
        _DIM, _RESET, sup, res,
    )

    # ── Technical indicators ─────────────────────────────────────────────
    # Show "—" only when indicators are entirely absent (ind is None).
    # When BB bands are zero but RSI/MACD have values, show partial data so the
    # log is always informative, even with few candles available.
    _bb_missing = (
        ind is not None
        and ind.bb_upper == 0.0 and ind.bb_mid == 0.0 and ind.bb_lower == 0.0
    )
    _ind_missing = (
        ind is None
        or (_bb_missing and ind.rsi == 0.0 and ind.macd == 0.0)
    )
    if ind and not _ind_missing:
        rsi_color  = (_GREEN if 40 < ind.rsi < 60
                      else _RED if ind.rsi >= 70 or ind.rsi <= 30 else _YELLOW)
        macd_color = _GREEN if ind.macd > 0 else _RED
        if _bb_missing:
            bb_part = f"BB[{_DIM}N/A{_RESET}]"
        else:
            bb_lo = _idr_compact(ind.bb_lower)
            bb_mi = _idr_compact(ind.bb_mid)
            bb_hi = _idr_compact(ind.bb_upper)
            bb_part = (
                f"BB[{_GREEN}{bb_lo}{_RESET} / {bb_mi} / {_RED}{bb_hi}{_RESET}]"
            )
        logging.info(
            "   %s└─%s indic   : RSI=%s%.1f%s  MACD=%s%+.6f%s  %s",
            _DIM, _RESET,
            rsi_color, ind.rsi, _RESET,
            macd_color, ind.macd, _RESET,
            bb_part,
        )
    else:
        logging.info("   %s└─%s indic   : —  (no candle data)", _DIM, _RESET)


# ── Order result display ─────────────────────────────────────────────────

def _log_outcome(outcome: dict) -> None:
    """Log the result of maybe_execute in tree format."""
    status = outcome.get("status", "—")
    action = outcome.get("action", "—")
    price  = outcome.get("price")
    amount = outcome.get("amount")
    icon   = _STATUS_ICONS.get(status, "❓")

    status_color = (
        _GREEN  if status in ("simulated", "placed")
        else _RED    if status in ("stopped", "force_sold")
        else _YELLOW if status in ("skipped",)
        else _DIM
    )

    price_str  = f"{_BOLD}{_idr(float(price))}{_RESET}" if price  else f"{_DIM}—{_RESET}"
    amount_str = f"{_BOLD}{float(amount):.8f}{_RESET}"   if amount else f"{_DIM}—{_RESET}"
    action_str = action.upper() if action else "—"

    logging.info(
        "%s %s%s%s  ·  %s  ·  %s  ·  @ %s",
        icon,
        status_color, _BOLD + status.upper() + _RESET, _RESET,
        action_str,
        amount_str,
        price_str,
    )

    steps  = outcome.get("executed_steps")
    stop   = outcome.get("stop_loss")
    tp     = outcome.get("take_profit")
    reason = outcome.get("reason")

    lines: list = []
    if steps and len(steps) > 1:
        lines.append(f"steps     : {len(steps)} staged entries")
    if stop:
        lines.append(f"stop-loss : {_RED}{_idr(float(stop))}{_RESET}")
    if tp:
        lines.append(f"take-prof : {_GREEN}{_idr(float(tp))}{_RESET}")
    if reason and status not in ("simulated", "placed"):
        lines.append(f"reason    : {_DIM}{reason}{_RESET}")

    for i, line in enumerate(lines):
        connector = "└─" if i == len(lines) - 1 else "├─"
        logging.info("   %s%s%s %s", _DIM, connector, _RESET, line)


# ── Portfolio display ────────────────────────────────────────────────────

def _log_portfolio(
    portfolio: dict,
    initial_capital: float = 0.0,
    *,
    trailing_stop_enabled: bool | None = None,
    trailing_tp_enabled: bool | None = None,
) -> None:
    """Log a multi-line tree-formatted portfolio summary.

    Uses the in-memory tracker snapshot (no REST calls), so it is safe to call
    frequently without tripping exchange rate limits.

    trailing_stop_enabled / trailing_tp_enabled are optional hints from config
    so we can explain whether a missing trail value is due to the feature being
    disabled (avoid confusion with "—" when users expect a trail).  None means
    we do not know / will not annotate.
    """
    equity   = portfolio["equity"]
    cash     = portfolio["cash"]
    pos      = portfolio["base_position"]
    pnl      = portfolio["realized_pnl"]
    avg_cost = portfolio.get("avg_cost", 0.0)
    trades   = portfolio.get("trade_count", 0)
    win_rate = portfolio.get("win_rate", 0.0)
    trail    = portfolio.get("trailing_stop")
    t_equity = portfolio.get("target_equity", 0.0)
    m_equity = portfolio.get("min_equity", 0.0)
    # Capital management fields
    principal      = portfolio.get("principal", initial_capital)
    profit_buffer  = portfolio.get("profit_buffer", 0.0)
    eff_capital    = portfolio.get("effective_capital", principal)
    pb_drawdown    = portfolio.get("profit_buffer_drawdown", 0.0)
    # Trailing TP
    trailing_tp    = portfolio.get("trailing_tp_stop")

    # unrealized = (equity - cash) - pos * avg_cost   [equity = cash + pos * mark_price]
    unrealized = (equity - cash) - pos * avg_cost if pos > 0 else 0.0

    equity_pct_str = ""
    if initial_capital > 0:
        equity_pct = (equity - initial_capital) / initial_capital * 100
        equity_pct_str = f"  ({_pct_str(equity_pct)} vs initial)"

    logging.info("%s📊 Portfolio%s", _BOLD, _RESET)
    logging.info(
        "   %s├─%s equity   : %s%s",
        _DIM, _RESET,
        f"{_BOLD}{_idr(equity)}{_RESET}",
        equity_pct_str,
    )
    logging.info(
        "   %s├─%s cash     : %s    position : %s%.8f%s coin",
        _DIM, _RESET,
        f"{_idr(cash)}",
        _BOLD, pos, _RESET,
    )
    if pos > 0:
        logging.info(
            "   %s├─%s avg cost : %s    unrealized: %s",
            _DIM, _RESET,
            f"{_idr(avg_cost)}",
            _pnl_str(unrealized),
        )
    logging.info(
        "   %s├─%s real PnL : %s    trades : %s%d%s  win-rate : %s%.0f%%%s",
        _DIM, _RESET,
        _pnl_str(pnl),
        _BOLD, trades, _RESET,
        _GREEN if win_rate >= 0.5 else _RED, win_rate * 100, _RESET,
    )
    # Capital management row: principal / profit buffer / effective capital
    pb_str = (
        f"{_GREEN}{_idr(profit_buffer)}{_RESET}"
        if profit_buffer > 0
        else f"{_DIM}Rp 0{_RESET}"
    )
    pb_dd_str = (
        f"  {_RED}▼{pb_drawdown:.1%}{_RESET}" if pb_drawdown > 0 else ""
    )
    logging.info(
        "   %s├─%s principal: %s    profit buf: %s%s    eff cap: %s",
        _DIM, _RESET,
        _idr(principal),
        pb_str, pb_dd_str,
        f"{_BOLD}{_idr(eff_capital)}{_RESET}",
    )
    _trail_hint = ""
    if trail is None and trailing_stop_enabled is not None:
        _trail_hint = " (disabled)" if not trailing_stop_enabled else " (pending)"
    trail_str  = (
        f"{_YELLOW}{_idr(trail)}{_RESET}"
        if trail is not None else f"{_DIM}—{_RESET}{_trail_hint}"
    )
    _ttp_hint = ""
    if trailing_tp is None and trailing_tp_enabled is not None:
        _ttp_hint = " (disabled)" if not trailing_tp_enabled else " (pending)"
    ttp_str    = (
        f"{_GREEN}{_idr(trailing_tp)}{_RESET}"
        if trailing_tp is not None else f"{_DIM}—{_RESET}{_ttp_hint}"
    )
    target_str = f"{_GREEN}{_idr(t_equity)}{_RESET}"  if t_equity else f"{_DIM}—{_RESET}"
    floor_str  = f"{_RED}{_idr(m_equity)}{_RESET}"    if m_equity else f"{_DIM}—{_RESET}"
    logging.info(
        "   %s└─%s trail    : %s  trail-TP: %s    target : %s    floor : %s",
        _DIM, _RESET, trail_str, ttp_str, target_str, floor_str,
    )


# ── Holding-position status ───────────────────────────────────────────────

def _log_holding(
    pair: str,
    price: float,
    portfolio: dict,
    initial_capital: float = 0.0,
    *,
    trailing_stop_enabled: bool | None = None,
    trailing_tp_enabled: bool | None = None,
) -> None:
    """Log a compact holding-position summary with unrealized PnL."""
    pos      = portfolio["base_position"]
    avg_cost = portfolio.get("avg_cost", 0.0)
    equity   = portfolio["equity"]
    cash     = portfolio["cash"]
    trail    = portfolio.get("trailing_stop")
    t_equity = portfolio.get("target_equity", 0.0)
    trailing_tp_floor = portfolio.get("trailing_tp_stop")

    unrealized = (equity - cash) - pos * avg_cost if pos > 0 else 0.0
    unreal_pct = (unrealized / (pos * avg_cost) * 100) if (pos > 0 and avg_cost > 0) else 0.0

    equity_pct_str = ""
    if initial_capital > 0:
        equity_pct = (equity - initial_capital) / initial_capital * 100
        equity_pct_str = f"  ({_pct_str(equity_pct)})"

    logging.info(
        "⏸️  %sHOLDING%s  %s%s%s  ·  price %s",
        _BOLD, _RESET,
        _BOLD, pair, _RESET,
        f"{_YELLOW}{_idr(price)}{_RESET}",
    )
    logging.info(
        "   %s├─%s position : %s%.8f%s coin  @  avg %s",
        _DIM, _RESET,
        _BOLD, pos, _RESET,
        f"{_idr(avg_cost)}",
    )
    logging.info(
        "   %s├─%s unrealized: %s  (%s%.2f%%%s)",
        _DIM, _RESET,
        _pnl_str(unrealized),
        _GREEN if unreal_pct >= 0 else _RED, unreal_pct, _RESET,
    )
    logging.info(
        "   %s├─%s equity    : %s%s    cash : %s",
        _DIM, _RESET,
        f"{_BOLD}{_idr(equity)}{_RESET}",
        equity_pct_str,
        f"{_idr(cash)}",
    )
    _trail_hint = ""
    if trail is None and trailing_stop_enabled is not None:
        _trail_hint = " (disabled)" if not trailing_stop_enabled else " (pending)"
    trail_str  = (
        f"{_YELLOW}{_idr(trail)}{_RESET}"
        if trail is not None else f"{_DIM}—{_RESET}{_trail_hint}"
    )
    _ttp_hint = ""
    if trailing_tp_floor is None and trailing_tp_enabled is not None:
        _ttp_hint = " (disabled)" if not trailing_tp_enabled else " (pending)"
    ttp_str    = (
        f"{_GREEN}{_idr(trailing_tp_floor)}{_RESET}"
        if trailing_tp_floor is not None else f"{_DIM}—{_RESET}{_ttp_hint}"
    )
    target_str = f"{_GREEN}{_idr(t_equity)}{_RESET}"          if t_equity          else f"{_DIM}—{_RESET}"
    logging.info(
        "   %s└─%s trail     : %s  trail-TP: %s    target : %s",
        _DIM, _RESET, trail_str, ttp_str, target_str,
    )


def main() -> None:
    configure_logging()  # early bootstrap logging before config is available
    config = BotConfig.from_env()
    configure_logging(log_file=config.log_file)  # re-configure with file handler if set

    if not config.dry_run:
        config.require_auth()

    trader = Trader(config)
    journal = TradeJournal(config.journal_path) if config.journal_path else TradeJournal()
    if hasattr(trader, 'set_journal'):
        trader.set_journal(journal)
    _log_startup_banner(config)

    # ── Account info / balance (live mode only) ──────────────────────────
    if not config.dry_run:
        try:
            _log_account_info(trader.client.get_account_info())
        except Exception as _exc:
            logging.warning("⚠️  Could not fetch account info: %s", _exc)
    else:
        _log_account_info_dry()

    # ── Graceful shutdown on SIGTERM (e.g. Docker / systemd stop) ──────────
    _shutdown = threading.Event()

    def _request_shutdown(signum: int, frame: object) -> None:
        logging.info("⏹️  Shutdown signal received — finishing current cycle …")
        _shutdown.set()

    signal.signal(signal.SIGTERM, _request_shutdown)

    pair = config.pair
    cycle = 0
    consecutive_errors = 0
    _max_backoff = 300  # 5 minutes cap
    # Maximum exponent for backoff calculation: 2^_MAX_BACKOFF_EXPONENT = 1024 s,
    # which is above the _max_backoff ceiling so the cap always wins.
    _MAX_BACKOFF_EXPONENT = 10
    scan_cycles = 0  # counts cycles that completed a full scan (for periodic summary)
    snapshot: dict = {}  # most recent scan snapshot (used for adaptive interval)
    # Track whether a position has been entered in this session.
    # Used by TRADE_MODE=single to know when a full buy→sell cycle is complete.
    _entered_position: bool = (
        len(trader.active_positions) > 0
        if getattr(trader, "multi_manager", None) is not None
        else trader.tracker.base_position > 0
    )

    # ── Autonomous trading state ─────────────────────────────────────────────
    _autonomous_state = AutonomousTradingState(
        is_running=True,
        health_status="running",
        max_errors_before_pause=10,
        pause_duration_seconds=60.0,
    )
    _start_time = time.time()
    _crash_history: list = []
    _error_log: list = []

    # ── Task scheduling ──────────────────────────────────────────────────────
    _now_ms = lambda: int(time.time() * 1000)
    _tasks = [
        ScheduledTask(name="health_check", interval_seconds=120.0, priority=1, enabled=True),
        ScheduledTask(name="pair_rotation", interval_seconds=300.0, priority=2, enabled=True),
        ScheduledTask(name="strategy_review", interval_seconds=600.0, priority=3, enabled=True),
        ScheduledTask(name="risk_monitoring", interval_seconds=180.0, priority=2, enabled=True),
        ScheduledTask(name="portfolio_analysis", interval_seconds=600.0, priority=3, enabled=True),
        ScheduledTask(name="ml_regime_check", interval_seconds=300.0, priority=3, enabled=True),
        ScheduledTask(name="cleanup", interval_seconds=900.0, priority=5, enabled=True),
    ]

    # ── Dynamic polling ──────────────────────────────────────────────────────
    _polling_config = PollingConfig(
        base_interval_ms=config.interval_seconds * 1000,
        min_interval_ms=max(1000, getattr(config, "adaptive_interval_min_seconds", 3) * 1000),
        max_interval_ms=60000,
        current_interval_ms=config.interval_seconds * 1000,
        mode="adaptive" if config.adaptive_interval_enabled else "fixed",
    )

    # ── Component health tracking ────────────────────────────────────────────
    _components = [
        ComponentHealth(name="exchange_api", is_healthy=True, last_heartbeat=_now_ms(), priority=0),
        ComponentHealth(name="market_data", is_healthy=True, last_heartbeat=_now_ms(), priority=1),
        ComponentHealth(name="order_engine", is_healthy=True, last_heartbeat=_now_ms(), priority=1),
        ComponentHealth(name="risk_engine", is_healthy=True, last_heartbeat=_now_ms(), priority=1),
        ComponentHealth(name="ml_engine", is_healthy=True, last_heartbeat=_now_ms(), priority=2),
        ComponentHealth(name="portfolio_engine", is_healthy=True, last_heartbeat=_now_ms(), priority=2),
    ]

    # ── Auto-resume: use the pair from saved state if we're resuming ────────
    if trader.restored_pair:
        pair = trader.restored_pair
        logging.info(
            "🔄 %sRESUMING%s  pair=%s%s%s  pos=%.8f",
            _BOLD, _RESET,
            _BOLD, pair, _RESET,
            trader.tracker.base_position,
        )

    while not _shutdown.is_set():
        cycle += 1
        logging.info(_separator(f"Cycle #{cycle}"))

        try:
            # ── Autonomous state tracking ────────────────────────────────────
            _autonomous_state = AutonomousTradingState(
                is_running=True,
                uptime_seconds=time.time() - _start_time,
                total_cycles=_autonomous_state.total_cycles,
                successful_cycles=_autonomous_state.successful_cycles,
                failed_cycles=_autonomous_state.failed_cycles,
                last_cycle_timestamp=_now_ms(),
                current_pair=pair,
                current_strategy=_autonomous_state.current_strategy,
                health_status=_autonomous_state.health_status,
                error_count=consecutive_errors,
                max_errors_before_pause=_autonomous_state.max_errors_before_pause,
                pause_duration_seconds=_autonomous_state.pause_duration_seconds,
            )

            # ── Failover check ───────────────────────────────────────────────
            for _comp in _components:
                _comp.last_heartbeat = _now_ms()
            _failover = evaluate_failover(_components)
            if _failover.failover_triggered:
                logging.warning(
                    "⚠️  Failover triggered: %s (status=%s)",
                    _failover.failed_components,
                    _failover.system_status,
                )
                _recovery = diagnose_and_recover(
                    _error_log,
                    component_health=_components,
                )
                if _recovery.needs_recovery and not _recovery.can_auto_recover:
                    logging.error(
                        "🛑 Manual intervention needed: %s", _recovery.diagnosis,
                    )

            # ── Task scheduling ──────────────────────────────────────────────
            _schedule_result = schedule_tasks(_tasks, current_time=_now_ms())
            for _task_name in _schedule_result.tasks_due:
                _task_idx = next(
                    (i for i, t in enumerate(_tasks) if t.name == _task_name), None,
                )
                if _task_idx is None:
                    continue
                _task = _tasks[_task_idx]
                _task_success = True
                try:
                    if _task_name == "health_check":
                        _health = check_autonomous_health(_autonomous_state)
                        if _health.get("needs_restart"):
                            logging.warning(
                                "🔄 Health check: needs restart (errors=%d)",
                                _health["error_count"],
                            )
                    elif _task_name == "risk_monitoring":
                        # ── Risk management monitoring via risk_management module ──
                        try:
                            _rm_cb = rm_check_circuit_breaker(
                                consecutive_losses=_autonomous_state.failed_cycles,
                                api_errors=consecutive_errors,
                            )
                            if _rm_cb.is_tripped:
                                logging.warning(
                                    "🛑 Risk monitoring: circuit breaker tripped (%s, severity=%s)",
                                    _rm_cb.triggers, _rm_cb.severity,
                                )
                                for _comp in _components:
                                    if _comp.name == "risk_engine":
                                        _comp.is_healthy = False
                            else:
                                for _comp in _components:
                                    if _comp.name == "risk_engine":
                                        _comp.is_healthy = True
                                        _comp.consecutive_failures = 0
                        except Exception as _rm_exc:
                            logging.debug("Risk monitoring check failed: %s", _rm_exc)
                    elif _task_name == "portfolio_analysis":
                        # ── Portfolio management analysis ─────────────────────────
                        try:
                            _pa = trader._get_portfolio_analysis()
                            if _pa.get("portfolio_eval"):
                                _pe = _pa["portfolio_eval"]
                                logging.info(
                                    "📊 Portfolio: %d assets, concentration=%.2f, value=%.2f",
                                    _pe.num_assets, _pe.concentration_score, _pe.total_value,
                                )
                            if _pa.get("diversification"):
                                _div = _pa["diversification"]
                                logging.info(
                                    "📊 Diversification: score=%.2f, effective_assets=%.1f",
                                    _div.score, _div.effective_assets,
                                )
                                for _comp in _components:
                                    if _comp.name == "portfolio_engine":
                                        _comp.is_healthy = True
                                        _comp.consecutive_failures = 0
                        except Exception as _pa_exc:
                            logging.debug("Portfolio analysis failed: %s", _pa_exc)
                    elif _task_name == "ml_regime_check":
                        # ── ML regime detection for strategy optimization ────────
                        try:
                            if snapshot and snapshot.get("candles"):
                                _candles = snapshot["candles"]
                                if len(_candles) >= 10:
                                    _regime = detect_regime(_candles, lookback=min(30, len(_candles)))
                                    logging.info(
                                        "🤖 ML regime: %s (confidence=%.2f)",
                                        _regime.regime, _regime.confidence,
                                    )
                                    for _comp in _components:
                                        if _comp.name == "ml_engine":
                                            _comp.is_healthy = True
                                            _comp.consecutive_failures = 0
                        except Exception as _ml_exc:
                            logging.debug("ML regime check failed: %s", _ml_exc)
                    elif _task_name == "cleanup":
                        trader.cleanup_stale_data()
                except Exception:
                    _task_success = False
                _tasks[_task_idx] = update_task_after_run(
                    _task, success=_task_success, current_time=_now_ms(),
                )

            # ── Position monitoring ──────────────────────────────────────────
            # When the bot is holding one or more positions (from this run or
            # restored via auto-resume) analyse each held pair first.
            # In multi-position mode all active pairs are fetched in parallel so
            # the market-data round-trip for each pair does not block the others.
            # Decision-making and order execution remain sequential to avoid race
            # conditions on shared tracker state.
            _active = list(trader.active_positions.items())
            _still_holding = False
            _has_pending_order = False  # shorter sleep when orders await fill

            # Fetch market snapshots for all held pairs in parallel.
            _held_snapshots: dict = {}
            if len(_active) > 1:
                def _fetch_held(pair_tracker: tuple) -> tuple:
                    _p, _t = pair_tracker
                    return _p, trader.analyze_market(_p)

                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(_active), thread_name_prefix="held-fetch"
                ) as _pool:
                    for _p, _snap in _pool.map(_fetch_held, _active):
                        _held_snapshots[_p] = _snap
            elif _active:
                _p, _t = _active[0]
                _held_snapshots[_p] = trader.analyze_market(_p)

            for held_pair, held_tracker in _active:
                held_snapshot = _held_snapshots.get(held_pair) or trader.analyze_market(held_pair)
                held_price = held_snapshot["price"]

                # ── Resume pending (unfilled) buy orders ──────────────────
                # When a buy was placed but never filled, keep adjusting the
                # price until it IS filled so no order is left hanging.
                # NOTE: We also resume when base_position > 0 — this handles
                # multi-stage buys where stage 1 filled but stage 2 did not.
                if held_tracker.has_pending_buy:
                    logging.info(
                        "🔄 %sRESUME%s  %s%s%s  ·  re-attempting pending buy @ Rp %s",
                        _BOLD, _RESET,
                        _BOLD, held_pair, _RESET,
                        f"{held_price:15,.2f}",
                    )
                    resume_out = trader.resume_pending_buy(held_snapshot)
                    if resume_out.get("status") == "resumed":
                        _entered_position = True
                        portfolio = trader.portfolio_snapshot(held_pair, held_price)
                        _log_portfolio(
                            portfolio,
                            config.initial_capital,
                            trailing_stop_enabled=config.trailing_stop_pct > 0,
                            trailing_tp_enabled=config.trailing_tp_pct > 0,
                        )
                        _notify(
                            config,
                            f"🔄 PENDING BUY FILLED {held_pair} @ Rp {held_price:,.0f}\n"
                            f"Amount: {resume_out.get('amount', 0):.8f}",
                        )
                    elif resume_out.get("status") in ("cancelled_below_min", "cancelled_zero"):
                        logging.info(
                            "❌ Pending buy cancelled for %s: %s",
                            held_pair, resume_out.get("status"),
                        )
                    else:
                        _still_holding = True
                        _has_pending_order = True
                    consecutive_errors = 0
                    continue  # next held pair

                # ── Resume pending (unfilled) sell orders ─────────────────
                # When a sell was placed but never filled, keep adjusting the
                # price until it IS filled so no order is left hanging.
                if getattr(held_tracker, "has_pending_sell", False):
                    logging.info(
                        "🔄 %sRESUME SELL%s  %s%s%s  ·  re-attempting pending sell @ Rp %s",
                        _BOLD, _RESET,
                        _BOLD, held_pair, _RESET,
                        f"{held_price:15,.2f}",
                    )
                    resume_sell_out = trader.resume_pending_sell(held_snapshot)
                    if resume_sell_out.get("status") in ("resumed", "partial"):
                        portfolio = held_tracker.as_dict(held_price)
                        _is_partial = resume_sell_out.get("status") == "partial"
                        logging.info(
                            "   %s├─%s amount   : %s%.8f%s coin  ·  price Rp %s%s",
                            _DIM, _RESET,
                            _BOLD, resume_sell_out.get("amount", 0), _RESET,
                            f"{held_price:15,.2f}",
                            f"  (partial, {resume_sell_out.get('remaining', 0):.8f} remaining)" if _is_partial else "",
                        )
                        _log_portfolio(
                            portfolio,
                            config.initial_capital,
                            trailing_stop_enabled=config.trailing_stop_pct > 0,
                            trailing_tp_enabled=config.trailing_tp_pct > 0,
                        )
                        _status_label = "PARTIAL SELL" if _is_partial else "PENDING SELL FILLED"
                        _notify(
                            config,
                            f"🔄 {_status_label} {held_pair} @ Rp {held_price:,.0f}\n"
                            f"Amount: {resume_sell_out.get('amount', 0):.8f}\n"
                            f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                        )
                        if _is_partial:
                            _still_holding = True
                    elif resume_sell_out.get("status") in ("cancelled_below_min", "cancelled_zero", "no_position"):
                        logging.info(
                            "❌ Pending sell cancelled for %s: %s",
                            held_pair, resume_sell_out.get("status"),
                        )
                    else:
                        _still_holding = True
                        _has_pending_order = True
                    consecutive_errors = 0
                    continue  # next held pair

                if config.trailing_stop_pct > 0:
                    held_tracker.update_trailing_stop(held_price, config.trailing_stop_pct)

                # Update the trailing TP floor on every tick while the
                # position is open.  Activating from the first cycle (not
                # just after the fixed TP target) makes the bot adaptive:
                # the floor rises with the market and the position exits when
                # price retraces more than trailing_tp_pct from its peak,
                # regardless of whether the fixed profit target was reached.
                if config.trailing_tp_pct > 0:
                    held_tracker.activate_trailing_tp(held_price, config.trailing_tp_pct)

                stop_reason = held_tracker.stop_reason(held_price)
                held_decision = held_snapshot["decision"]

                # ── Momentum exit (adaptive early exit) ──────────────────
                if stop_reason is None and trader.check_momentum_exit(held_snapshot):
                    stop_reason = "momentum_exit"

                # ── Immediate dump exit (protect against post-entry dumps) ────
                if stop_reason is None and trader.check_post_entry_dump(held_tracker, held_price):
                    stop_reason = "post_entry_dump"

                # ── Partial take-profit check (level 1) ───────────────────
                if (
                    config.partial_tp_fraction > 0
                    and not held_tracker.partial_tp_taken
                    and held_decision.take_profit is not None
                    and held_price >= held_decision.take_profit
                ):
                    whale = held_snapshot.get("whale")
                    adjusted_tp = held_decision.take_profit
                    if whale and whale.detected and whale.side == "ask":
                        adjusted_tp = held_price * 0.999
                        logging.info(
                            "🐋 Sell wall at TP — adjusting partial-TP target to %.2f",
                            adjusted_tp,
                        )
                    if held_price >= adjusted_tp:
                        ptp_outcome = trader.partial_take_profit(held_snapshot, config.partial_tp_fraction)
                        portfolio = held_tracker.as_dict(held_price)
                        logging.info(
                            "🎯 PARTIAL-TP  %s%s%s  %.0f%% @ Rp %s",
                            _BOLD, held_pair, _RESET,
                            config.partial_tp_fraction * 100,
                            f"{held_price:15,.2f}",
                        )
                        _log_portfolio(
                            portfolio,
                            config.initial_capital,
                            trailing_stop_enabled=config.trailing_stop_pct > 0,
                            trailing_tp_enabled=config.trailing_tp_pct > 0,
                        )
                        _notify(
                            config,
                            f"🎯 PARTIAL-TP {held_pair} @ Rp {held_price:,.0f}\n"
                            f"Fraction: {config.partial_tp_fraction:.0%}\n"
                            f"Amount: {ptp_outcome.get('amount', 0):.8f}\n"
                            f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                        )

                # ── Partial take-profit check (level 2) ───────────────────
                if (
                    config.partial_tp2_fraction > 0
                    and not held_tracker.partial_tp2_taken
                    and held_tracker.avg_cost > 0
                    and config.partial_tp2_target_pct > 0
                    and held_price >= held_tracker.avg_cost * (1 + config.partial_tp2_target_pct)
                ):
                    ptp2_outcome = trader.partial_take_profit(held_snapshot, config.partial_tp2_fraction)
                    held_tracker.partial_tp2_taken = True
                    portfolio = held_tracker.as_dict(held_price)
                    logging.info(
                        "🎯 PARTIAL-TP2  %s%s%s  %.0f%% @ Rp %s  (+%.1f%%)",
                        _BOLD, held_pair, _RESET,
                        config.partial_tp2_fraction * 100,
                        f"{held_price:15,.2f}",
                        config.partial_tp2_target_pct * 100,
                    )
                    _log_portfolio(
                        portfolio,
                        config.initial_capital,
                        trailing_stop_enabled=config.trailing_stop_pct > 0,
                        trailing_tp_enabled=config.trailing_tp_pct > 0,
                    )
                    _notify(
                        config,
                        f"🎯 PARTIAL-TP2 {held_pair} @ Rp {held_price:,.0f}\n"
                        f"Fraction: {config.partial_tp2_fraction:.0%}\n"
                        f"Amount: {ptp2_outcome.get('amount', 0):.8f}\n"
                        f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                    )

                # ── Partial take-profit check (level 3) ───────────────────
                if (
                    config.partial_tp3_fraction > 0
                    and not held_tracker.partial_tp3_taken
                    and held_tracker.avg_cost > 0
                    and config.partial_tp3_target_pct > 0
                    and held_price >= held_tracker.avg_cost * (1 + config.partial_tp3_target_pct)
                ):
                    ptp3_outcome = trader.partial_take_profit(held_snapshot, config.partial_tp3_fraction)
                    held_tracker.partial_tp3_taken = True
                    portfolio = held_tracker.as_dict(held_price)
                    logging.info(
                        "🎯 PARTIAL-TP3  %s%s%s  %.0f%% @ Rp %s  (+%.1f%%)",
                        _BOLD, held_pair, _RESET,
                        config.partial_tp3_fraction * 100,
                        f"{held_price:15,.2f}",
                        config.partial_tp3_target_pct * 100,
                    )
                    _log_portfolio(
                        portfolio,
                        config.initial_capital,
                        trailing_stop_enabled=config.trailing_stop_pct > 0,
                        trailing_tp_enabled=config.trailing_tp_pct > 0,
                    )
                    _notify(
                        config,
                        f"🎯 PARTIAL-TP3 {held_pair} @ Rp {held_price:,.0f}\n"
                        f"Fraction: {config.partial_tp3_fraction:.0%}\n"
                        f"Amount: {ptp3_outcome.get('amount', 0):.8f}\n"
                        f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                    )

                # ── Dynamic TP override ────────────────────────────────────
                if stop_reason == "target_profit_reached":
                    dynamic_reason = trader.evaluate_dynamic_tp(held_snapshot)
                    if dynamic_reason is None:
                        trailing_tp_floor = held_tracker.trailing_tp_stop
                        logging.info(
                            "🚀 %sDYNAMIC-TP%s  %s%s%s  · holding past TP%s",
                            _BOLD, _RESET,
                            _BOLD, held_pair, _RESET,
                            f"  trailing_floor=Rp {trailing_tp_floor:,.2f}" if trailing_tp_floor else "",
                        )
                        portfolio = held_tracker.as_dict(held_price)
                        _log_portfolio(
                            portfolio,
                            config.initial_capital,
                            trailing_stop_enabled=config.trailing_stop_pct > 0,
                            trailing_tp_enabled=config.trailing_tp_pct > 0,
                        )
                        consecutive_errors = 0
                        _still_holding = True
                        continue  # next held pair
                    stop_reason = dynamic_reason

                should_exit = stop_reason is not None or held_decision.action == "sell"

                if should_exit:
                    exit_reason = stop_reason or f"sell signal ({held_decision.reason[:60]})"
                    logging.info(
                        "📤 %sEXIT%s  %s%s%s  ·  %s",
                        _BOLD, _RESET,
                        _BOLD, held_pair, _RESET,
                        exit_reason,
                    )
                    force_outcome = trader.force_sell(held_snapshot)

                    # When force_sell returns pending_sell, the order was not
                    # filled — keep the position active for resume on next cycle.
                    if force_outcome.get("status") == "pending_sell":
                        logging.info(
                            "🔄 %sPENDING SELL%s  %s%s%s  ·  sell not filled — will retry next cycle",
                            _BOLD, _RESET,
                            _BOLD, held_pair, _RESET,
                        )
                        _still_holding = True
                        consecutive_errors = 0
                        continue

                    portfolio = held_tracker.as_dict(held_price)
                    logging.info(
                        "   %s├─%s amount   : %s%.8f%s coin  ·  price Rp %s",
                        _DIM, _RESET,
                        _BOLD, force_outcome.get("amount", 0), _RESET,
                        f"{held_price:15,.2f}",
                    )
                    _log_portfolio(
                        portfolio,
                        config.initial_capital,
                        trailing_stop_enabled=config.trailing_stop_pct > 0,
                        trailing_tp_enabled=config.trailing_tp_pct > 0,
                    )
                    _notify(
                        config,
                        f"📤 EXIT {held_pair} @ Rp {held_price:,.0f}\n"
                        f"Reason: {exit_reason}\n"
                        f"Amount: {force_outcome.get('amount', 0):.8f}\n"
                        f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                    )

                    # ── Re-entry analysis after trail/stop sell ──────────────
                    # When the exit was triggered by a trailing stop or stop
                    # loss, analyse whether the coin still has upside potential.
                    # If so, mark a pending buy at a lower target price so the
                    # bot can re-enter instead of staying flat.
                    if stop_reason in (
                        "trailing_stop", "stop_loss", "trailing_tp_triggered",
                        "momentum_exit", "post_entry_dump",
                    ):
                        try:
                            _reentry = trader.analyze_reentry_opportunity(held_snapshot)
                            if _reentry and _reentry.get("reentry"):
                                _tgt = _reentry["target_price"]
                                logging.info(
                                    "🔄 %sRE-ENTRY OPPORTUNITY%s  %s%s%s  ·  target Rp %s  regime=%s  imbalance=%.2f",
                                    _BOLD, _RESET,
                                    _BOLD, held_pair, _RESET,
                                    f"{_tgt:,.2f}",
                                    _reentry.get("regime", "?"),
                                    _reentry.get("imbalance", 0),
                                )
                                _notify(
                                    config,
                                    f"🔄 RE-ENTRY OPPORTUNITY {held_pair}\n"
                                    f"Target: Rp {_tgt:,.2f}\n"
                                    f"Regime: {_reentry.get('regime', '?')}",
                                )
                            else:
                                logging.info(
                                    "📊 No re-entry opportunity for %s — staying flat",
                                    held_pair,
                                )
                        except Exception as _re_exc:
                            logging.debug("Re-entry analysis failed for %s: %s", held_pair, _re_exc)

                    consecutive_errors = 0
                    if config.trade_mode == "single" and _entered_position:
                        logging.info("✅ %sSingle-trade cycle complete — stopping.%s", _BOLD, _RESET)
                        break
                    if config.run_once:
                        break
                else:
                    _still_holding = True
                    portfolio = held_tracker.as_dict(held_price)
                    _log_holding(
                        held_pair,
                        held_price,
                        portfolio,
                        config.initial_capital,
                        trailing_stop_enabled=config.trailing_stop_pct > 0,
                        trailing_tp_enabled=config.trailing_tp_pct > 0,
                    )
                    consecutive_errors = 0

            # After monitoring all held positions:
            # • Single-position mode: if still holding, sleep and skip scan.
            # • Multi-position mode: if still at capacity, sleep; otherwise fall
            #   through to scan for new opportunities.
            if _still_holding:
                if not config.multi_position_enabled or trader.at_max_positions():
                    if config.run_once:
                        break
                    # Use shorter sleep when orders are pending fill to retry faster
                    _sleep = (
                        config.resume_buy_wait_seconds
                        if _has_pending_order
                        else config.position_check_interval_seconds
                    )
                    time.sleep(_sleep)
                    continue
                # Multi-position with free slots: scan for additional pairs now.
                logging.info(
                    "🔍 Holding %d position(s) — scanning for new pair opportunities …",
                    len(trader.active_positions),
                )

            # ── Scan all pairs and choose the best opportunity ───────────────
            if trader.at_max_positions():
                # All position slots are full; wait before next cycle.
                if config.run_once:
                    break
                time.sleep(config.position_check_interval_seconds)
                continue
            pair, snapshot = trader.scan_and_choose()
            _log_signal(snapshot)
            # ── Log ML & strategy signals if available ────────────────────────
            if snapshot.get("ml_regime"):
                logging.debug(
                    "🤖 ML regime: %s (conf=%.2f)",
                    snapshot["ml_regime"].regime,
                    snapshot["ml_regime"].confidence,
                )
            if snapshot.get("ml_prediction"):
                logging.debug(
                    "🤖 ML prediction: %s (agreement=%.2f)",
                    snapshot["ml_prediction"].predicted_direction,
                    snapshot["ml_prediction"].model_agreement,
                )
            if snapshot.get("adv_trend"):
                logging.debug(
                    "📈 Adv trend: %s (strength=%.2f)",
                    snapshot["adv_trend"].action,
                    snapshot["adv_trend"].strength,
                )
            if snapshot.get("ob_pressure"):
                logging.debug(
                    "📊 OB pressure: %s (pressure=%.2f)",
                    snapshot["ob_pressure"].signal,
                    snapshot["ob_pressure"].pressure,
                )
            outcome = trader.maybe_execute(snapshot)
            _log_outcome(outcome)
            # Immediately compute trailing stops after a buy so the portfolio
            # display shows trail/trail-TP values from the very first cycle.
            if outcome.get("action") == "buy":
                _bought_tracker = trader._active_tracker(snapshot["pair"])
                if _bought_tracker.base_position > 0:
                    if config.trailing_stop_pct > 0:
                        _bought_tracker.update_trailing_stop(snapshot["price"], config.trailing_stop_pct)
                    if config.trailing_tp_pct > 0:
                        _bought_tracker.activate_trailing_tp(snapshot["price"], config.trailing_tp_pct)
            portfolio = trader.portfolio_snapshot(snapshot["pair"], snapshot["price"])
            _log_portfolio(
                portfolio,
                config.initial_capital,
                trailing_stop_enabled=config.trailing_stop_pct > 0,
                trailing_tp_enabled=config.trailing_tp_pct > 0,
            )

            # Telegram notification for actionable outcomes
            _out_action = outcome.get("action", "hold")
            _out_status = outcome.get("status", "")
            if _out_action in ("buy", "sell") and _out_status in ("simulated", "placed"):
                # ── Execution quality monitoring ────────────────────────────
                try:
                    _eq = monitor_execution_quality(
                        order_price=snapshot["price"],
                        fill_price=outcome.get("price", snapshot["price"]),
                        quantity=outcome.get("amount", 0),
                        side=_out_action,
                        latency_ms=outcome.get("latency_ms", 0),
                    )
                    logging.debug(
                        "📊 Execution quality: slippage=%.4f%%, score=%.2f",
                        _eq.slippage_pct * 100,
                        _eq.quality_score,
                    )
                except Exception as _eq_exc:
                    logging.debug("Execution quality monitoring failed: %s", _eq_exc)

                _notify(
                    config,
                    f"{'📈 BUY' if _out_action == 'buy' else '📉 SELL'} "
                    f"{snapshot['pair']} @ Rp {snapshot['price']:,.0f}\n"
                    f"Amount: {outcome.get('amount', 0):.8f}\n"
                    f"Status: {_out_status}\n"
                    f"Equity: Rp {portfolio['equity']:,.2f}",
                )

            consecutive_errors = 0
            scan_cycles += 1

            # ── Update autonomous state on successful cycle ──────────────────
            _autonomous_state = AutonomousTradingState(
                is_running=True,
                uptime_seconds=time.time() - _start_time,
                total_cycles=_autonomous_state.total_cycles + 1,
                successful_cycles=_autonomous_state.successful_cycles + 1,
                failed_cycles=_autonomous_state.failed_cycles,
                last_cycle_timestamp=_now_ms(),
                current_pair=pair,
                current_strategy=_autonomous_state.current_strategy,
                health_status="running",
                error_count=0,
                max_errors_before_pause=_autonomous_state.max_errors_before_pause,
                pause_duration_seconds=_autonomous_state.pause_duration_seconds,
            )
            # Reset error log on successful cycle
            _error_log.clear()

            # Mark that we've entered a position (for single-trade mode)
            if outcome.get("action") == "buy" and trader._active_tracker(snapshot["pair"]).base_position > 0:
                _entered_position = True

            # ── Stop condition: liquidate remaining position and rotate ───────
            if outcome.get("status") == "stopped":
                logging.info(
                    "🛑 %sSTOP%s  %s  — liquidating and rotating …",
                    _BOLD, _RESET, outcome.get("reason", ""),
                )
                _stop_pair = snapshot["pair"]
                if trader._active_tracker(_stop_pair).base_position > 0:
                    force_outcome = trader.force_sell(snapshot)
                    if force_outcome.get("status") == "pending_sell":
                        logging.info(
                            "🔄 %sPENDING SELL%s  %s  — sell not filled, will retry next cycle",
                            _BOLD, _RESET, _stop_pair,
                        )
                    else:
                        logging.info(
                            "   📤 Force-sold : %s%.8f%s coin  ·  Rp %s",
                            _BOLD, force_outcome.get("amount", 0), _RESET,
                            f"{force_outcome.get('price', 0):15,.2f}",
                        )
                # Re-compute portfolio after any liquidation
                portfolio = trader._active_tracker(_stop_pair).as_dict(snapshot["price"])
                logging.info(
                    "   📊 %sRotation%s : pnl=%s  equity=Rp %s  "
                    "trades=%d  win=%.0f%%",
                    _BOLD, _RESET,
                    _pnl_str(portfolio["realized_pnl"]),
                    f"{portfolio['equity']:15,.2f}",
                    portfolio["trade_count"],
                    portfolio["win_rate"] * 100,
                )
                if config.trade_mode == "single" and _entered_position:
                    logging.info("✅ %sSingle-trade cycle complete — stopping.%s", _BOLD, _RESET)
                    break
                if config.run_once:
                    break
                logging.info(_separator())
                time.sleep(config.interval_seconds)
                continue  # find next opportunity instead of halting

        except (requests.RequestException, RuntimeError, ValueError) as _err:
            consecutive_errors += 1
            # ── Classify error for autonomous recovery ───────────────────────
            _etype = "connection" if isinstance(_err, requests.RequestException) else "runtime"
            _error_log.append({"type": _etype, "message": str(_err), "timestamp": _now_ms()})
            _crash_history.append(CrashEvent(
                timestamp=_now_ms(),
                error_type=_etype,
                error_message=str(_err)[:200],
                component="trading_loop",
                recoverable=True,
            ))
            _autonomous_state = AutonomousTradingState(
                is_running=True,
                uptime_seconds=time.time() - _start_time,
                total_cycles=_autonomous_state.total_cycles + 1,
                successful_cycles=_autonomous_state.successful_cycles,
                failed_cycles=_autonomous_state.failed_cycles + 1,
                last_cycle_timestamp=_now_ms(),
                current_pair=pair,
                current_strategy=_autonomous_state.current_strategy,
                health_status="degraded",
                error_count=consecutive_errors,
                max_errors_before_pause=_autonomous_state.max_errors_before_pause,
                pause_duration_seconds=_autonomous_state.pause_duration_seconds,
            )
            # Mark exchange_api as unhealthy on connection errors
            if isinstance(_err, requests.RequestException):
                for _comp in _components:
                    if _comp.name == "exchange_api":
                        _comp.consecutive_failures += 1
                        if _comp.consecutive_failures >= _comp.max_failures:
                            _comp.is_healthy = False
            # ── Dynamic backoff via autonomous restart logic ──────────────────
            _restart_decision = decide_restart(
                _crash_history,
                max_restarts=_autonomous_state.max_errors_before_pause,
                base_delay=float(config.interval_seconds),
            )
            if _restart_decision.should_restart:
                backoff = min(_restart_decision.delay_seconds, _max_backoff)
            else:
                exponent = min(consecutive_errors - 1, _MAX_BACKOFF_EXPONENT)
                backoff = min(config.interval_seconds * (2 ** exponent), _max_backoff)
            logging.exception(
                "⚠️  %sError #%d%s  pair=%s  backing off %.0fs …",
                _BOLD, consecutive_errors, _RESET, pair, backoff,
            )
            if config.run_once:
                logging.info("run_once enabled; exiting after recoverable error")
                break
            time.sleep(backoff)
            continue
        except Exception as _err:  # noqa: BLE001 — broad catch prevents unexpected crash
            # Catch any unexpected exception type (KeyError, AttributeError, TypeError,
            # IndexError, etc.) that is not explicitly in the tuple above.  Without this
            # handler such errors would propagate all the way out of main() and crash the
            # bot process entirely instead of retrying with back-off.
            consecutive_errors += 1
            _error_log.append({"type": "unexpected", "message": str(_err), "timestamp": _now_ms()})
            _crash_history.append(CrashEvent(
                timestamp=_now_ms(),
                error_type=type(_err).__name__,
                error_message=str(_err)[:200],
                component="trading_loop",
                recoverable=True,
            ))
            _autonomous_state = AutonomousTradingState(
                is_running=True,
                uptime_seconds=time.time() - _start_time,
                total_cycles=_autonomous_state.total_cycles + 1,
                successful_cycles=_autonomous_state.successful_cycles,
                failed_cycles=_autonomous_state.failed_cycles + 1,
                last_cycle_timestamp=_now_ms(),
                current_pair=pair,
                current_strategy=_autonomous_state.current_strategy,
                health_status="degraded",
                error_count=consecutive_errors,
                max_errors_before_pause=_autonomous_state.max_errors_before_pause,
                pause_duration_seconds=_autonomous_state.pause_duration_seconds,
            )
            _restart_decision = decide_restart(
                _crash_history,
                max_restarts=_autonomous_state.max_errors_before_pause,
                base_delay=float(config.interval_seconds),
            )
            if _restart_decision.should_restart:
                backoff = min(_restart_decision.delay_seconds, _max_backoff)
            else:
                exponent = min(consecutive_errors - 1, _MAX_BACKOFF_EXPONENT)
                backoff = min(config.interval_seconds * (2 ** exponent), _max_backoff)
            logging.exception(
                "⚠️  Unexpected error #%d  pair=%s  backing off %.0fs …",
                consecutive_errors, pair, backoff,
            )
            if config.run_once:
                logging.info("run_once enabled; exiting after unexpected error")
                break
            time.sleep(backoff)
            continue
        except KeyboardInterrupt:
            logging.info("⏹️  %sBOT STOPPED%s by user", _BOLD, _RESET)
            break

        if config.run_once:
            break

        # Periodic performance summary every N full-scan cycles
        if scan_cycles > 0 and scan_cycles % config.cycle_summary_interval == 0:
            logging.info(
                "📊 %sPeriodic summary%s  scan #%d : pnl=%s  equity=Rp %s  "
                "trades=%d  win=%.0f%%",
                _BOLD, _RESET, scan_cycles,
                _pnl_str(portfolio["realized_pnl"]),
                f"{portfolio['equity']:15,.2f}",
                portfolio["trade_count"],
                portfolio["win_rate"] * 100,
            )
            logging.info("📓 %s", journal.summary_str())

        # Periodic state backup every N scan cycles
        if (
            config.state_path is not None
            and config.state_backup_interval > 0
            and scan_cycles > 0
            and scan_cycles % config.state_backup_interval == 0
        ):
            _backup_path = config.state_path.with_name(
                config.state_path.stem + "_backup" + config.state_path.suffix
            )
            trader.persistence.backup(_backup_path)

        logging.info(_separator())
        # ── Dynamic polling interval ─────────────────────────────────────────
        _vol_snapshot = 0.0
        if snapshot:
            _vol_obj = snapshot.get("volatility")
            if _vol_obj:
                _vol_snapshot = getattr(_vol_obj, "volatility", 0.0)
        _has_open = len(trader.active_positions) > 0
        _polling_config, _polling_adj = adjust_polling_interval(
            _polling_config,
            recent_volatility=_vol_snapshot,
            recent_errors=consecutive_errors,
            has_open_orders=_has_open,
        )
        # Use the shorter of the two adaptive systems (trader's built-in
        # adaptive interval vs autonomous dynamic polling) so the bot reacts
        # to the most responsive signal.
        _sleep_secs = min(
            trader._effective_interval(snapshot),
            max(1, _polling_config.current_interval_ms // 1000),
        )
        if _sleep_secs != config.interval_seconds:
            logging.debug(
                "⚡ Adaptive interval: sleeping %ds (normal=%ds, reason=%s)",
                _sleep_secs, config.interval_seconds, _polling_adj.adjustment_reason,
            )
        time.sleep(_sleep_secs)


if __name__ == "__main__":
    _MAX_RESTARTS = 10
    _process_crash_history: list = []
    _restart_count = 0
    while _restart_count <= _MAX_RESTARTS:
        try:
            main()
            break
        except KeyboardInterrupt:
            break
        except Exception as _exc:
            _restart_count += 1
            _process_crash_history.append(CrashEvent(
                timestamp=int(time.time() * 1000),
                error_type=type(_exc).__name__,
                error_message=str(_exc)[:200],
                component="main_process",
                recoverable=True,
            ))
            _restart_decision = decide_restart(
                _process_crash_history,
                max_restarts=_MAX_RESTARTS,
                base_delay=5.0,
                backoff_factor=2.0,
            )
            if _restart_decision.should_restart:
                _wait = _restart_decision.delay_seconds
            else:
                logging.getLogger(__name__).error(
                    "Bot crashed — max restarts reached, stopping.",
                )
                break
            logging.getLogger(__name__).error(
                "Bot crashed (attempt %d): %s — restarting in %.0fs (%s)",
                _restart_count, _exc, _wait, _restart_decision.reason,
            )
            time.sleep(_wait)
