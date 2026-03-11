from __future__ import annotations

import logging
import sys
import time

import requests

from bot.config import BotConfig
from bot.trader import Trader


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
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
            "Auto-resume enabled: restored state pair=%s cash=%s base_position=%s",
            trader.restored_state.get("pair", config.pair),
            restored_portfolio.get("cash"),
            restored_portfolio.get("base_position"),
        )
    pair = config.pair
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
            logging.exception("Recoverable error in bot loop (pair=%s)", pair)
            if config.run_once:
                raise
        except KeyboardInterrupt:
            logging.info("Stopping bot")
            break

        if config.run_once:
            break
        time.sleep(config.interval_seconds)


if __name__ == "__main__":
    main()
