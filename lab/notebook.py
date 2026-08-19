from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Notebook:
    """Append-only JSONL notes plus a short current-beliefs file."""

    def __init__(self, notes_path: Path, beliefs_path: Path) -> None:
        self.notes_path = notes_path
        self.beliefs_path = beliefs_path
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        if not notes_path.exists():
            notes_path.write_text("", encoding="utf-8")
        if not beliefs_path.exists():
            beliefs_path.write_text("# Current beliefs\n\n(none yet)\n", encoding="utf-8")

    def append(self, **entry: Any) -> dict[str, Any]:
        row = {"ts": _now(), **entry}
        with self.notes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return row

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        if n <= 0:
            return []
        lines = self.notes_path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

    def all(self) -> list[dict[str, Any]]:
        return self.tail(10_000)

    def beliefs(self) -> str:
        return self.beliefs_path.read_text(encoding="utf-8")

    def set_beliefs(self, text: str) -> None:
        tmp = self.beliefs_path.with_suffix(".md.tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.beliefs_path)
