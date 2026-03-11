from __future__ import annotations

import argparse
import logging
import sys
import time

import requests

from bot.config import BotConfig
from bot.trader import Trader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated Indodax trading bot")
    parser.add_argument("--pair", default=None, help="Trading pair, e.g. btc_idr")
    parser.add_argument("--pairs", default=None, help="Comma-separated pairs to scan before trading")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires API keys)")
    parser.add_argument("--interval", type=int, default=None, help="Candle interval seconds")
    parser.add_argument("--once", action="store_true", help="Run a single iteration then exit")
    parser.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence to trade")
    parser.add_argument("--initial-capital", type=float, default=None, help="Starting capital in quote currency")
    parser.add_argument("--target-profit", type=float, default=None, help="Target profit percentage (0.2 = 20%)")
    parser.add_argument("--max-loss", type=float, default=None, help="Max loss percentage (0.1 = 10%)")
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main() -> None:
    args = parse_args()
    configure_logging()

    config = BotConfig.from_env()
    if args.pair:
        config.pair = args.pair
    if args.pairs:
        config.scan_pairs = [p.strip().lower() for p in args.pairs.split(",") if p.strip()]
    if args.interval:
        config.interval_seconds = args.interval
    if args.min_confidence is not None:
        config.min_confidence = args.min_confidence
    if args.initial_capital is not None:
        config.initial_capital = args.initial_capital
    if args.target_profit is not None:
        config.target_profit_pct = args.target_profit
    if args.max_loss is not None:
        config.max_loss_pct = args.max_loss
    if args.live:
        config.dry_run = False
        config.require_auth()

    trader = Trader(config)
    while True:
        try:
            pair, snapshot = trader.scan_and_choose()
            summary = snapshot["decision"]
            logging.info(
                "pair=%s mode=%s action=%s price=%.2f conf=%.2f reason=%s",
                pair,
                summary.mode,
                summary.action,
                snapshot["price"],
                summary.confidence,
                summary.reason,
            )
            ob = snapshot.get("orderbook")
            levels = snapshot.get("levels")
            vol = snapshot.get("volatility")
            logging.info(
                "analytics pair=%s spread=%.4f imbalance=%.2f vol=%.4f support=%s resistance=%s",
                pair,
                ob.spread_pct if ob else float("nan"),
                ob.imbalance if ob else float("nan"),
                vol.volatility if vol else float("nan"),
                getattr(levels, "support", None),
                getattr(levels, "resistance", None),
            )
            outcome = trader.maybe_execute(snapshot)
            logging.info("result=%s", outcome)
            portfolio_price = snapshot["price"]
            logging.info("portfolio=%s", trader.tracker.as_dict(portfolio_price))
            if outcome.get("status") == "stopped":
                break
        except (requests.RequestException, RuntimeError, ValueError):
            logging.exception("Recoverable error in bot loop (pair=%s)", config.pair)
            if args.once:
                raise
        except KeyboardInterrupt:
            logging.info("Stopping bot")
            break

        if args.once:
            break
        time.sleep(config.interval_seconds)


if __name__ == "__main__":
    main()
