import unittest

from bot.config import BotConfig
from bot.grid import build_grid_plan
from bot.strategies import StrategyDecision
from bot.trader import Trader


class GridPlanTests(unittest.TestCase):
    def test_builds_symmetric_grid_levels(self) -> None:
        config = BotConfig(
            api_key=None,
            grid_enabled=True,
            grid_levels_per_side=2,
            grid_spacing_pct=0.01,
        )
        plan = build_grid_plan(100.0, config)
        self.assertEqual(len(plan.buy_orders), 2)
        self.assertEqual(len(plan.sell_orders), 2)
        self.assertAlmostEqual(plan.buy_orders[0].price, 99.0)
        self.assertAlmostEqual(plan.sell_orders[0].price, 101.0)
        # ensure deeper levels spaced correctly
        self.assertAlmostEqual(plan.buy_orders[1].price, 98.0)
        self.assertAlmostEqual(plan.sell_orders[1].price, 102.0)

    def test_maybe_execute_runs_grid_dry_run(self) -> None:
        config = BotConfig(
            api_key=None,
            dry_run=True,
            grid_enabled=True,
            grid_levels_per_side=1,
            grid_spacing_pct=0.005,
        )
        trader = Trader(config)
        plan = build_grid_plan(200.0, config)
        decision = StrategyDecision(
            mode="grid_trading",
            action="grid",
            confidence=1.0,
            reason="grid",
            target_price=200.0,
            amount=sum(o.amount for o in plan.orders),
            stop_loss=None,
            take_profit=None,
        )
        snapshot = {"pair": "btc_idr", "price": 200.0, "decision": decision, "grid_plan": plan}

        outcome = trader.maybe_execute(snapshot)
        self.assertEqual(outcome["status"], "grid_simulated")
        self.assertEqual(len(outcome["orders"]), 2)
        self.assertEqual(trader.tracker.base_position, 0.0)  # no fills on placement


if __name__ == "__main__":
    unittest.main()
