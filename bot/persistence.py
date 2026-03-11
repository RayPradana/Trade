from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional


class StatePersistence:
    """Minimal JSON-based persistence layer to enable auto-resume.

    When *path* is ``None`` all operations are silent no-ops so callers never
    need to guard against a disabled persistence layer.
    """

    def __init__(self, path: Optional[Path]) -> None:
        self.path = path

    def load(self) -> Optional[Dict[str, Any]]:
        if self.path is None or not self.path.exists():
            return None
        try:
            raw = json.loads(self.path.read_text())
            if not isinstance(raw, dict):
                return None
            return raw
        except (json.JSONDecodeError, OSError):
            return None

    def save(self, state: Dict[str, Any]) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(state)
        payload["saved_at"] = time.time()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        tmp_path.replace(self.path)

    def clear(self) -> None:
        if self.path is None:
            return
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                # Safe to ignore inability to delete; caller may retry later.
                pass

    def backup(self, backup_path: Path) -> None:
        """Copy the current state file to *backup_path* as a safety snapshot.

        Called periodically so that a crash never loses more than one backup
        interval's worth of state.  Does nothing if the state file doesn't exist
        or persistence is disabled (``path=None``).
        """
        if self.path is None or not self.path.exists():
            return
        try:
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.path, backup_path)
        except OSError as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning("State backup failed (%s → %s): %s", self.path, backup_path, exc)
