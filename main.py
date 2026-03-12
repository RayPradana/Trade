from __future__ import annotations

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
    if ind:
        rsi_color  = (_GREEN if 40 < ind.rsi < 60
                      else _RED if ind.rsi >= 70 or ind.rsi <= 30 else _YELLOW)
        macd_color = _GREEN if ind.macd > 0 else _RED
        bb_lo = _idr_compact(ind.bb_lower)
        bb_mi = _idr_compact(ind.bb_mid)
        bb_hi = _idr_compact(ind.bb_upper)
        logging.info(
            "   %s└─%s indic   : RSI=%s%.1f%s  MACD=%s%+.6f%s  BB[%s%s%s / %s / %s%s%s]",
            _DIM, _RESET,
            rsi_color, ind.rsi, _RESET,
            macd_color, ind.macd, _RESET,
            _GREEN, bb_lo, _RESET,
            bb_mi,
            _RED, bb_hi, _RESET,
        )
    else:
        logging.info("   %s└─%s indic   : —", _DIM, _RESET)


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

def _log_portfolio(portfolio: dict, initial_capital: float = 0.0) -> None:
    """Log a multi-line tree-formatted portfolio summary."""
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
    trail_str  = f"{_YELLOW}{_idr(trail)}{_RESET}"    if trail    else f"{_DIM}—{_RESET}"
    ttp_str    = f"{_GREEN}{_idr(trailing_tp)}{_RESET}" if trailing_tp else f"{_DIM}—{_RESET}"
    target_str = f"{_GREEN}{_idr(t_equity)}{_RESET}"  if t_equity else f"{_DIM}—{_RESET}"
    floor_str  = f"{_RED}{_idr(m_equity)}{_RESET}"    if m_equity else f"{_DIM}—{_RESET}"
    logging.info(
        "   %s└─%s trail    : %s  trail-TP: %s    target : %s    floor : %s",
        _DIM, _RESET, trail_str, ttp_str, target_str, floor_str,
    )


# ── Holding-position status ───────────────────────────────────────────────

def _log_holding(pair: str, price: float, portfolio: dict,
                 initial_capital: float = 0.0) -> None:
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
    trail_str  = f"{_YELLOW}{_idr(trail)}{_RESET}"            if trail             else f"{_DIM}—{_RESET}"
    ttp_str    = f"{_GREEN}{_idr(trailing_tp_floor)}{_RESET}"  if trailing_tp_floor else f"{_DIM}—{_RESET}"
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
    _entered_position: bool = trader.tracker.base_position > 0

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
            # ── Position monitoring ──────────────────────────────────────────
            # When the bot is already holding a position (from this run or
            # restored via auto-resume) analyse that pair first.  Decide to
            # exit (sell) or stay in the trade before scanning new pairs.
            if trader.tracker.base_position > 0:
                held_snapshot = trader.analyze_market(pair)
                held_price = held_snapshot["price"]

                if config.trailing_stop_pct > 0:
                    trader.tracker.update_trailing_stop(held_price, config.trailing_stop_pct)

                # Update the trailing TP floor on every tick while the
                # position is open.  Activating from the first cycle (not
                # just after the fixed TP target) makes the bot adaptive:
                # the floor rises with the market and the position exits when
                # price retraces more than trailing_tp_pct from its peak,
                # regardless of whether the fixed profit target was reached.
                if config.trailing_tp_pct > 0:
                    trader.tracker.activate_trailing_tp(held_price, config.trailing_tp_pct)

                stop_reason = trader.tracker.stop_reason(held_price)
                held_decision = held_snapshot["decision"]

                # ── Momentum exit (adaptive early exit) ─────────────────────
                # Exit BEFORE the TP target when momentum weakens (imbalance
                # drops) while the position is still profitable.  This prevents
                # the bot from stubbornly waiting for a fixed target while the
                # market turns bearish.
                if stop_reason is None and trader.check_momentum_exit(held_snapshot):
                    stop_reason = "momentum_exit"

                # ── Partial take-profit check (level 1) ──────────────────────
                # When PARTIAL_TP_FRACTION is set and price has reached the TP
                # level for the first time, sell a fraction and let the rest run.
                if (
                    config.partial_tp_fraction > 0
                    and not trader.tracker.partial_tp_taken
                    and held_decision.take_profit is not None
                    and held_price >= held_decision.take_profit
                ):
                    # Auto-shift TP down when a sell wall blocks the target
                    whale = held_snapshot.get("whale")
                    adjusted_tp = held_decision.take_profit
                    if whale and whale.detected and whale.side == "ask":
                        # Sell wall at TP → take partial profit at current price
                        adjusted_tp = held_price * 0.999  # slight discount to ensure fill
                        logging.info(
                            "🐋 Sell wall at TP — adjusting partial-TP target to %.2f",
                            adjusted_tp,
                        )
                    if held_price >= adjusted_tp:
                        ptp_outcome = trader.partial_take_profit(held_snapshot, config.partial_tp_fraction)
                        portfolio = trader.tracker.as_dict(held_price)
                        logging.info(
                            "🎯 PARTIAL-TP  %s%s%s  %.0f%% @ Rp %s",
                            _BOLD, pair, _RESET,
                            config.partial_tp_fraction * 100,
                            f"{held_price:15,.2f}",
                        )
                        _log_portfolio(portfolio, config.initial_capital)
                        _notify(
                            config,
                            f"🎯 PARTIAL-TP {pair} @ Rp {held_price:,.0f}\n"
                            f"Fraction: {config.partial_tp_fraction:.0%}\n"
                            f"Amount: {ptp_outcome.get('amount', 0):.8f}\n"
                            f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                        )

                # ── Partial take-profit check (level 2) ──────────────────────
                # When PARTIAL_TP2_FRACTION is set and price has risen by
                # PARTIAL_TP2_TARGET_PCT above the entry price, sell a 2nd
                # fraction and let the rest continue running.
                if (
                    config.partial_tp2_fraction > 0
                    and not trader.tracker.partial_tp2_taken
                    and trader.tracker.avg_cost > 0
                    and config.partial_tp2_target_pct > 0
                    and held_price >= trader.tracker.avg_cost * (1 + config.partial_tp2_target_pct)
                ):
                    ptp2_outcome = trader.partial_take_profit(held_snapshot, config.partial_tp2_fraction)
                    trader.tracker.partial_tp2_taken = True
                    portfolio = trader.tracker.as_dict(held_price)
                    logging.info(
                        "🎯 PARTIAL-TP2  %s%s%s  %.0f%% @ Rp %s  (+%.1f%%)",
                        _BOLD, pair, _RESET,
                        config.partial_tp2_fraction * 100,
                        f"{held_price:15,.2f}",
                        config.partial_tp2_target_pct * 100,
                    )
                    _log_portfolio(portfolio, config.initial_capital)
                    _notify(
                        config,
                        f"🎯 PARTIAL-TP2 {pair} @ Rp {held_price:,.0f}\n"
                        f"Fraction: {config.partial_tp2_fraction:.0%}\n"
                        f"Amount: {ptp2_outcome.get('amount', 0):.8f}\n"
                        f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                    )

                # ── Partial take-profit check (level 3) ──────────────────────
                # When PARTIAL_TP3_FRACTION is set and price has risen by
                # PARTIAL_TP3_TARGET_PCT above the entry price, sell a 3rd
                # fraction and let the remainder run.
                if (
                    config.partial_tp3_fraction > 0
                    and not trader.tracker.partial_tp3_taken
                    and trader.tracker.avg_cost > 0
                    and config.partial_tp3_target_pct > 0
                    and held_price >= trader.tracker.avg_cost * (1 + config.partial_tp3_target_pct)
                ):
                    ptp3_outcome = trader.partial_take_profit(held_snapshot, config.partial_tp3_fraction)
                    trader.tracker.partial_tp3_taken = True
                    portfolio = trader.tracker.as_dict(held_price)
                    logging.info(
                        "🎯 PARTIAL-TP3  %s%s%s  %.0f%% @ Rp %s  (+%.1f%%)",
                        _BOLD, pair, _RESET,
                        config.partial_tp3_fraction * 100,
                        f"{held_price:15,.2f}",
                        config.partial_tp3_target_pct * 100,
                    )
                    _log_portfolio(portfolio, config.initial_capital)
                    _notify(
                        config,
                        f"🎯 PARTIAL-TP3 {pair} @ Rp {held_price:,.0f}\n"
                        f"Fraction: {config.partial_tp3_fraction:.0%}\n"
                        f"Amount: {ptp3_outcome.get('amount', 0):.8f}\n"
                        f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                    )

                # Exit if a hard stop fired or the strategy says sell.
                # For target_profit_reached: give dynamic TP a chance to override.
                if stop_reason == "target_profit_reached":
                    dynamic_reason = trader.evaluate_dynamic_tp(held_snapshot)
                    if dynamic_reason is None:
                        # Dynamic TP says hold — log and continue monitoring
                        trailing_tp_floor = trader.tracker.trailing_tp_stop
                        logging.info(
                            "🚀 %sDYNAMIC-TP%s  %s%s%s  · holding past TP%s",
                            _BOLD, _RESET,
                            _BOLD, pair, _RESET,
                            f"  trailing_floor=Rp {trailing_tp_floor:,.2f}" if trailing_tp_floor else "",
                        )
                        portfolio = trader.tracker.as_dict(held_price)
                        _log_portfolio(portfolio, config.initial_capital)
                        consecutive_errors = 0
                        if config.run_once:
                            break
                        time.sleep(config.position_check_interval_seconds)
                        continue
                    stop_reason = dynamic_reason  # use the resolved reason for exit

                should_exit = stop_reason is not None or held_decision.action == "sell"

                if should_exit:
                    exit_reason = stop_reason or f"sell signal ({held_decision.reason[:60]})"
                    logging.info(
                        "📤 %sEXIT%s  %s%s%s  ·  %s",
                        _BOLD, _RESET,
                        _BOLD, pair, _RESET,
                        exit_reason,
                    )
                    force_outcome = trader.force_sell(held_snapshot)
                    portfolio = trader.tracker.as_dict(held_price)
                    logging.info(
                        "   %s├─%s amount   : %s%.8f%s coin  ·  price Rp %s",
                        _DIM, _RESET,
                        _BOLD, force_outcome.get("amount", 0), _RESET,
                        f"{held_price:15,.2f}",
                    )
                    _log_portfolio(portfolio, config.initial_capital)
                    _notify(
                        config,
                        f"📤 EXIT {pair} @ Rp {held_price:,.0f}\n"
                        f"Reason: {exit_reason}\n"
                        f"Amount: {force_outcome.get('amount', 0):.8f}\n"
                        f"PnL: Rp {portfolio['realized_pnl']:,.2f}",
                    )
                    consecutive_errors = 0
                    # single-trade mode: one complete cycle (buy → sell) is enough
                    if config.trade_mode == "single" and _entered_position:
                        logging.info("✅ %sSingle-trade cycle complete — stopping.%s", _BOLD, _RESET)
                        break
                    if config.run_once:
                        break
                    logging.info(_separator())
                    time.sleep(trader._effective_interval(held_snapshot))
                    continue  # scan for next opportunity

                # Still holding – use faster polling interval
                portfolio = trader.tracker.as_dict(held_price)
                _log_holding(pair, held_price, portfolio, config.initial_capital)
                consecutive_errors = 0
                if config.run_once:
                    break
                time.sleep(config.position_check_interval_seconds)
                continue

            # ── Scan all pairs and choose the best opportunity ───────────────
            pair, snapshot = trader.scan_and_choose()
            _log_signal(snapshot)
            outcome = trader.maybe_execute(snapshot)
            _log_outcome(outcome)
            # Immediately compute trailing stops after a buy so the portfolio
            # display shows trail/trail-TP values from the very first cycle.
            if outcome.get("action") == "buy" and trader.tracker.base_position > 0:
                if config.trailing_stop_pct > 0:
                    trader.tracker.update_trailing_stop(snapshot["price"], config.trailing_stop_pct)
                if config.trailing_tp_pct > 0:
                    trader.tracker.activate_trailing_tp(snapshot["price"], config.trailing_tp_pct)
            portfolio = trader.tracker.as_dict(snapshot["price"])
            _log_portfolio(portfolio, config.initial_capital)

            # Telegram notification for actionable outcomes
            _out_action = outcome.get("action", "hold")
            _out_status = outcome.get("status", "")
            if _out_action in ("buy", "sell") and _out_status in ("simulated", "placed"):
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

            # Mark that we've entered a position (for single-trade mode)
            if outcome.get("action") == "buy" and trader.tracker.base_position > 0:
                _entered_position = True

            # ── Stop condition: liquidate remaining position and rotate ───────
            if outcome.get("status") == "stopped":
                logging.info(
                    "🛑 %sSTOP%s  %s  — liquidating and rotating …",
                    _BOLD, _RESET, outcome.get("reason", ""),
                )
                if trader.tracker.base_position > 0:
                    force_outcome = trader.force_sell(snapshot)
                    logging.info(
                        "   📤 Force-sold : %s%.8f%s coin  ·  Rp %s",
                        _BOLD, force_outcome.get("amount", 0), _RESET,
                        f"{force_outcome.get('price', 0):15,.2f}",
                    )
                # Re-compute portfolio after any liquidation
                portfolio = trader.tracker.as_dict(snapshot["price"])
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

        except (requests.RequestException, RuntimeError, ValueError):
            consecutive_errors += 1
            # Cap the exponent so the computed delay doesn't grow beyond _max_backoff.
            # Without this cap the multiplication could produce a very large intermediate
            # value even though min() would ultimately clamp it.
            # 2^_MAX_BACKOFF_EXPONENT = 1024 s, which is well above the _max_backoff ceiling of 300 s.
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
        except Exception:  # noqa: BLE001 — broad catch prevents unexpected crash
            # Catch any unexpected exception type (KeyError, AttributeError, TypeError,
            # IndexError, etc.) that is not explicitly in the tuple above.  Without this
            # handler such errors would propagate all the way out of main() and crash the
            # bot process entirely instead of retrying with back-off.
            consecutive_errors += 1
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
        _sleep_secs = trader._effective_interval(snapshot)
        if _sleep_secs != config.interval_seconds:
            logging.debug("⚡ Adaptive interval: sleeping %ds (normal=%ds)", _sleep_secs, config.interval_seconds)
        time.sleep(_sleep_secs)


if __name__ == "__main__":
    _MAX_RESTARTS = 10
    _RESTART_BACKOFF = [5, 10, 30, 60, 120]
    _restart_count = 0
    while _restart_count <= _MAX_RESTARTS:
        try:
            main()
            break
        except KeyboardInterrupt:
            break
        except Exception as _exc:
            _restart_count += 1
            _wait = _RESTART_BACKOFF[min(_restart_count - 1, len(_RESTART_BACKOFF) - 1)]
            logging.getLogger(__name__).error(
                "Bot crashed (attempt %d): %s — restarting in %ds",
                _restart_count, _exc, _wait,
            )
            time.sleep(_wait)
