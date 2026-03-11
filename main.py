from __future__ import annotations

import argparse
import logging
import sys
import time

from bot.config import BotConfig
from bot.trader import Trader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated Indodax trading bot")
    parser.add_argument("--pair", default=None, help="Trading pair, e.g. btc_idr")
    parser.add_argument("--live", action="store_true", help="Enable live trading (requires API keys)")
    parser.add_argument("--interval", type=int, default=None, help="Candle interval seconds")
    parser.add_argument("--once", action="store_true", help="Run a single iteration then exit")
    parser.add_argument("--min-confidence", type=float, default=None, help="Minimum confidence to trade")
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
    if args.interval:
        config.interval_seconds = args.interval
    if args.min_confidence is not None:
        config.min_confidence = args.min_confidence
    if args.live:
        config.dry_run = False
        config.require_auth()

    trader = Trader(config)
    while True:
        try:
            snapshot = trader.analyze_market()
            summary = snapshot["decision"]
            logging.info(
                "mode=%s action=%s price=%.2f conf=%.2f reason=%s",
                summary.mode,
                summary.action,
                snapshot["price"],
                summary.confidence,
                summary.reason,
            )
            outcome = trader.maybe_execute(snapshot)
            logging.info("result=%s", outcome)
        except Exception as exc:  # pragma: no cover - runtime guard
            logging.exception("Failed to execute bot loop: %s", exc)

        if args.once:
            break
        time.sleep(config.interval_seconds)


if __name__ == "__main__":
    main()
