from __future__ import annotations

import datetime
import logging
import signal
import sys
import threading
import time

import requests

from bot.config import BotConfig
from bot.trader import Trader

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


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        _ColoredFormatter(
            fmt=f"{_DIM}%(asctime)s{_RESET} [%(levelname)s] {_CYAN}%(name)s{_RESET}: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])


# ── Section separators ───────────────────────────────────────────────────

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


# ── Signal display (BUY/SELL decision) ───────────────────────────────────

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
    spread_pct = (ob.spread_pct * 100) if ob else float("nan")
    imbalance  = ob.imbalance if ob else float("nan")
    volatility = vol.volatility if vol else float("nan")
    imb_color  = _GREEN if imbalance > 0.1 else (_RED if imbalance < -0.1 else _YELLOW)
    logging.info(
        "   %s├─%s market  : spread=%s%.4f%%%s  imbalance=%s%+.3f%s  vol=%s%.4f%s",
        _DIM, _RESET,
        _DIM, spread_pct, _RESET,
        imb_color, imbalance, _RESET,
        _DIM, volatility, _RESET,
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
    trail_str  = f"{_YELLOW}{_idr(trail)}{_RESET}"    if trail    else f"{_DIM}—{_RESET}"
    target_str = f"{_GREEN}{_idr(t_equity)}{_RESET}"  if t_equity else f"{_DIM}—{_RESET}"
    floor_str  = f"{_RED}{_idr(m_equity)}{_RESET}"    if m_equity else f"{_DIM}—{_RESET}"
    logging.info(
        "   %s└─%s trail    : %s    target : %s    floor : %s",
        _DIM, _RESET, trail_str, target_str, floor_str,
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
    trail_str  = f"{_YELLOW}{_idr(trail)}{_RESET}"   if trail    else f"{_DIM}—{_RESET}"
    target_str = f"{_GREEN}{_idr(t_equity)}{_RESET}" if t_equity else f"{_DIM}—{_RESET}"
    logging.info(
        "   %s└─%s trail     : %s    target : %s",
        _DIM, _RESET, trail_str, target_str,
    )


def main() -> None:
    configure_logging()

    config = BotConfig.from_env()
    if not config.dry_run:
        config.require_auth()

    trader = Trader(config)
    _log_startup_banner(config)

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
    scan_cycles = 0  # counts cycles that completed a full scan (for periodic summary)
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

                stop_reason = trader.tracker.stop_reason(held_price)
                held_decision = held_snapshot["decision"]

                # Exit if a hard stop fired or the strategy says sell.
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
                        "   %s├─%s amount   : %s%.8f%s coin  ·  price Rp %15,.2f",
                        _DIM, _RESET,
                        _BOLD, force_outcome.get("amount", 0), _RESET,
                        held_price,
                    )
                    _log_portfolio(portfolio, config.initial_capital)
                    consecutive_errors = 0
                    # single-trade mode: one complete cycle (buy → sell) is enough
                    if config.trade_mode == "single" and _entered_position:
                        logging.info("✅ %sSingle-trade cycle complete — stopping.%s", _BOLD, _RESET)
                        break
                    if config.run_once:
                        break
                    logging.info(_separator())
                    time.sleep(config.interval_seconds)
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
            portfolio = trader.tracker.as_dict(snapshot["price"])
            _log_portfolio(portfolio, config.initial_capital)

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
                        "   📤 Force-sold : %s%.8f%s coin  ·  Rp %15,.2f",
                        _BOLD, force_outcome.get("amount", 0), _RESET,
                        force_outcome.get("price", 0),
                    )
                # Re-compute portfolio after any liquidation
                portfolio = trader.tracker.as_dict(snapshot["price"])
                logging.info(
                    "   📊 %sRotation%s : pnl=%s  equity=Rp %15,.2f  "
                    "trades=%d  win=%.0f%%",
                    _BOLD, _RESET,
                    _pnl_str(portfolio["realized_pnl"]),
                    portfolio["equity"],
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
            exponent = min(consecutive_errors - 1, 10)
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
        except KeyboardInterrupt:
            logging.info("⏹️  %sBOT STOPPED%s by user", _BOLD, _RESET)
            break

        if config.run_once:
            break

        # Periodic performance summary every N full-scan cycles
        if scan_cycles > 0 and scan_cycles % config.cycle_summary_interval == 0:
            logging.info(
                "📊 %sPeriodic summary%s  scan #%d : pnl=%s  equity=Rp %15,.2f  "
                "trades=%d  win=%.0f%%",
                _BOLD, _RESET, scan_cycles,
                _pnl_str(portfolio["realized_pnl"]),
                portfolio["equity"],
                portfolio["trade_count"],
                portfolio["win_rate"] * 100,
            )

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
        time.sleep(config.interval_seconds)


if __name__ == "__main__":
    main()
