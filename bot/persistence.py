from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class StatePersistence:
    """Minimal JSON-based persistence layer to enable auto-resume."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict):
                return None
            return raw
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, state: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(state)
        payload["saved_at"] = time.time()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp_path.replace(self.path)

    def clear(self) -> None:
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                # Safe to ignore inability to delete; caller may retry later.
                pass
