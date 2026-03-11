from __future__ import annotations

import logging
import signal
import sys
import threading
import time

import requests

from bot.config import BotConfig
from bot.trader import Trader

# ANSI color codes – automatically disabled when output is not a terminal
_USE_COLOR = sys.stdout.isatty()

_RESET = "\033[0m" if _USE_COLOR else ""
_BOLD = "\033[1m" if _USE_COLOR else ""
_DIM = "\033[2m" if _USE_COLOR else ""
_CYAN = "\033[36m" if _USE_COLOR else ""
_GREEN = "\033[32m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_RED = "\033[31m" if _USE_COLOR else ""
_MAGENTA = "\033[35m" if _USE_COLOR else ""
_BLUE = "\033[34m" if _USE_COLOR else ""

_LEVEL_COLORS = {
    "DEBUG": _CYAN,
    "INFO": _GREEN,
    "WARNING": _YELLOW,
    "ERROR": _RED,
    "CRITICAL": _MAGENTA,
}

_ACTION_ICONS = {
    "buy": "📈",
    "sell": "📉",
    "hold": "⏸️",
    "grid": "🔲",
}

_STATUS_ICONS = {
    "simulated": "🧪",
    "placed": "✅",
    "skipped": "⏭️",
    "hold": "⏸️",
    "stopped": "🛑",
    "force_sold": "📤",
    "grid_simulated": "🔲",
    "grid_placed": "🔲",
}


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


def _separator(label: str = "") -> str:
    width = 72
    if label:
        pad = max(0, (width - len(label) - 2) // 2)
        return f"{_BOLD}{_BLUE}{'─' * pad} {label} {'─' * pad}{_RESET}"
    return f"{_DIM}{'─' * width}{_RESET}"


def _log_portfolio(portfolio: dict, prefix: str = "   portf  ") -> None:
    logging.info(
        "%s: equity=%,.2f  cash=%,.2f  pos=%.8f  pnl=%+,.2f  "
        "trades=%d  win=%.0f%%  trail=%s",
        prefix,
        portfolio["equity"],
        portfolio["cash"],
        portfolio["base_position"],
        portfolio["realized_pnl"],
        portfolio["trade_count"],
        portfolio["win_rate"] * 100,
        f"{portfolio['trailing_stop']:,.2f}" if portfolio.get("trailing_stop") else "—",
    )


def main() -> None:
    configure_logging()

    config = BotConfig.from_env()
    if not config.dry_run:
        config.require_auth()

    trader = Trader(config)
    if trader.restored_state:
        restored_portfolio = trader.restored_state.get("portfolio", {})
        logging.info(
            "🔄 Auto-resume: pair=%s  cash=%s  position=%s",
            trader.restored_state.get("pair", config.pair),
            restored_portfolio.get("cash"),
            restored_portfolio.get("base_position"),
        )

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
                    logging.info("📤 Exiting position on %s — %s", pair, exit_reason)
                    force_outcome = trader.force_sell(held_snapshot)
                    portfolio = trader.tracker.as_dict(held_price)
                    logging.info(
                        "📊 Position closed: amount=%.8f  price=%,.2f",
                        force_outcome.get("amount", 0),
                        held_price,
                    )
                    _log_portfolio(portfolio)
                    consecutive_errors = 0
                    if config.run_once:
                        break
                    logging.info(_separator())
                    time.sleep(config.interval_seconds)
                    continue  # scan for next opportunity

                # Still holding – use faster polling interval
                portfolio = trader.tracker.as_dict(held_price)
                logging.info(
                    "⏸️  Holding %s | price=%,.2f | pos=%.8f | equity=%,.2f | trail=%s",
                    pair,
                    held_price,
                    trader.tracker.base_position,
                    portfolio["equity"],
                    f"{portfolio['trailing_stop']:,.2f}" if portfolio.get("trailing_stop") else "—",
                )
                consecutive_errors = 0
                if config.run_once:
                    break
                time.sleep(config.position_check_interval_seconds)
                continue

            # ── Scan all pairs and choose the best opportunity ───────────────
            pair, snapshot = trader.scan_and_choose()
            summary = snapshot["decision"]
            icon = _ACTION_ICONS.get(summary.action, "❓")
            logging.info(
                "%s %s  pair=%-12s  mode=%-16s  price=%,.2f  conf=%.3f",
                icon,
                summary.action.upper(),
                pair,
                summary.mode,
                snapshot["price"],
                summary.confidence,
            )
            logging.info(
                "   reason : %s",
                summary.reason,
            )
            ob = snapshot.get("orderbook")
            levels = snapshot.get("levels")
            vol = snapshot.get("volatility")
            ind = snapshot.get("indicators")
            logging.info(
                "   market : spread=%.4f%%  imbalance=%+.3f  vol=%.4f  "
                "support=%s  resistance=%s",
                (ob.spread_pct * 100) if ob else float("nan"),
                ob.imbalance if ob else float("nan"),
                vol.volatility if vol else float("nan"),
                f"{levels.support:,.2f}" if levels and levels.support else "N/A",
                f"{levels.resistance:,.2f}" if levels and levels.resistance else "N/A",
            )
            if ind:
                logging.info(
                    "   indic  : RSI=%.1f  MACD=%.6f  BB[%.2f / %.2f / %.2f]",
                    ind.rsi,
                    ind.macd,
                    ind.bb_lower,
                    ind.bb_mid,
                    ind.bb_upper,
                )
            outcome = trader.maybe_execute(snapshot)
            status_icon = _STATUS_ICONS.get(outcome.get("status", ""), "❓")
            logging.info(
                "%s result : status=%-14s  action=%s  amount=%s  price=%s",
                status_icon,
                outcome.get("status"),
                outcome.get("action", "—"),
                outcome.get("amount", "—"),
                outcome.get("price", "—"),
            )
            portfolio = trader.tracker.as_dict(snapshot["price"])
            _log_portfolio(portfolio)

            consecutive_errors = 0
            scan_cycles += 1

            # ── Stop condition: liquidate remaining position and rotate ───────
            if outcome.get("status") == "stopped":
                logging.info(
                    "🛑 Stop condition reached: %s — liquidating and rotating …",
                    outcome.get("reason"),
                )
                if trader.tracker.base_position > 0:
                    force_outcome = trader.force_sell(snapshot)
                    logging.info(
                        "📤 Force-sold: amount=%s  price=%s",
                        force_outcome.get("amount"),
                        force_outcome.get("price"),
                    )
                # Re-compute portfolio after any liquidation
                portfolio = trader.tracker.as_dict(snapshot["price"])
                logging.info(
                    "📊 Rotation summary: pnl=%+,.2f  equity=%,.2f  trades=%d  win=%.0f%%",
                    portfolio["realized_pnl"],
                    portfolio["equity"],
                    portfolio["trade_count"],
                    portfolio["win_rate"] * 100,
                )
                if config.run_once:
                    break
                logging.info(_separator())
                time.sleep(config.interval_seconds)
                continue  # find next opportunity instead of halting

        except (requests.RequestException, RuntimeError, ValueError):
            consecutive_errors += 1
            # Cap the exponent to avoid enormous intermediate values before min()
            exponent = min(consecutive_errors - 1, 10)
            backoff = min(config.interval_seconds * (2 ** exponent), _max_backoff)
            logging.exception(
                "⚠️  Error #%d in bot loop (pair=%s) — backing off %.0fs",
                consecutive_errors,
                pair,
                backoff,
            )
            if config.run_once:
                logging.info("run_once enabled; exiting after recoverable error")
                break
            time.sleep(backoff)
            continue
        except KeyboardInterrupt:
            logging.info("⏹️  Bot stopped by user")
            break

        if config.run_once:
            break

        # Periodic performance summary every N full-scan cycles
        if scan_cycles > 0 and scan_cycles % config.cycle_summary_interval == 0:
            logging.info(
                "📊 Periodic summary (scan #%d): pnl=%+,.2f  equity=%,.2f  "
                "trades=%d  win=%.0f%%",
                scan_cycles,
                portfolio["realized_pnl"],
                portfolio["equity"],
                portfolio["trade_count"],
                portfolio["win_rate"] * 100,
            )

        logging.info(_separator())
        time.sleep(config.interval_seconds)


if __name__ == "__main__":
    main()
