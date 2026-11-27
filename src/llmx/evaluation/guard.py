"""Enforce the rule "the test set is used once per final system".

A lock file is written the first time a tag is evaluated on test; a second attempt raises,
unless explicitly forced (which is then recorded in the lock file for transparency).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class HoldoutAlreadyUsedError(RuntimeError):
    pass


class HoldoutGuard:
    def __init__(self, lock_dir: Path) -> None:
        self.lock_dir = lock_dir

    def _lock_path(self, tag: str) -> Path:
        return self.lock_dir / f"{tag}.lock.json"

    def is_used(self, tag: str) -> bool:
        return self._lock_path(tag).exists()

    def _history(self, tag: str) -> list[dict]:
        path = self._lock_path(tag)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def check(self, tag: str, force: bool = False) -> None:
        """Raise before any expensive work if the test set was already used for ``tag``."""
        history = self._history(tag)
        if history and not force:
            raise HoldoutAlreadyUsedError(
                f"Test set already evaluated for '{tag}' at {history[0]['at']}. "
                "Tune on validation only. Pass force=True to override (it will be logged)."
            )

    def acquire(self, tag: str, force: bool = False) -> None:
        """Record a use of the test set (call once predictions exist, before scoring)."""
        self.check(tag, force)
        path = self._lock_path(tag)
        history = self._history(tag)
        history.append({"at": datetime.now(timezone.utc).isoformat(), "forced": bool(history)})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, indent=2), encoding="utf-8")
