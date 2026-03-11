import os
import unittest
from unittest.mock import patch

import main


class MainErrorHandlingTests(unittest.TestCase):
    def test_main_exits_gracefully_on_recoverable_error_with_run_once(self) -> None:
        class StubTrader:
            def __init__(self, config) -> None:
                self.config = config
                self.restored_state = None

            def scan_and_choose(self):
                raise RuntimeError("no data")

            def maybe_execute(self, snapshot):
                return {}

        with patch("main.Trader", StubTrader):
            with patch.dict(os.environ, {"RUN_ONCE": "true"}, clear=False):
                main.main()  # should exit without raising


if __name__ == "__main__":
    unittest.main()
