"""A small menu of checkpoints a pack may start from.

Seven slots, overwritten in place. The run never lists every job file.
`parent_checkpoint` may be ``bank:<slot>``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

BANK_SLOTS = (
    "subject",
    "latest",
    "best",
    "pretrain",
    "continue",
    "midtrain",
    "posttrain",
)

_ARCH_KEYS = ("hidden", "layers", "heads", "seq_len")


class CheckpointBank:
    def __init__(self, checkpoints_dir: Path) -> None:
        self.dir = Path(checkpoints_dir)
        self.path = self.dir / "bank.json"

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, dict[str, Any]] = {}
        for slot, entry in raw.items():
            if slot in BANK_SLOTS and isinstance(entry, dict) and entry.get("path"):
                out[slot] = dict(entry)
        return out

    def save(self, data: dict[str, dict[str, Any]]) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        kept = {slot: data[slot] for slot in BANK_SLOTS if slot in data}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(kept, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def entries(self) -> list[dict[str, Any]]:
        """Slots whose file is still on disk, in menu order."""
        data = self.load()
        out = []
        for slot in BANK_SLOTS:
            entry = data.get(slot)
            if not entry:
                continue
            path = Path(str(entry.get("path"))).expanduser()
            if not path.is_file():
                continue
            row = dict(entry)
            row["slot"] = slot
            row["id"] = f"bank:{slot}"
            out.append(row)
        return out

    def resolve(self, token: str) -> Path | None:
        """``bank:best`` or ``best`` -> the file, if that slot is filled."""
        slot = str(token).strip()
        if slot.startswith("bank:"):
            slot = slot.split(":", 1)[1]
        if slot not in BANK_SLOTS:
            return None
        entry = self.load().get(slot)
        if not entry:
            return None
        path = Path(str(entry.get("path"))).expanduser()
        return path if path.is_file() else None

    def put(self, slot: str, entry: dict[str, Any]) -> None:
        if slot not in BANK_SLOTS:
            raise ValueError(f"unknown bank slot {slot!r}")
        data = self.load()
        data[slot] = dict(entry)
        data[slot]["slot"] = slot
        self.save(data)

    def ensure_subject(self, raw: str | None) -> None:
        if not raw:
            return
        path = Path(str(raw)).expanduser()
        if not path.is_file():
            return
        data = self.load()
        current = data.get("subject")
        if current and Path(str(current.get("path"))).expanduser().is_file():
            return
        self.put(
            "subject",
            {
                "path": str(path),
                "stage": "subject",
                "job_id": None,
                "confirm_ppl": None,
                **{key: None for key in _ARCH_KEYS},
            },
        )

    def note_train(
        self,
        *,
        path: Path,
        stage: str,
        job_id: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        """A finished job fills ``latest`` and, when the stage has a slot, that slot."""
        cfg = config or {}
        entry = {
            "path": str(path),
            "stage": stage,
            "job_id": job_id,
            "confirm_ppl": None,
            **{key: cfg.get(key) for key in _ARCH_KEYS},
        }
        self.put("latest", entry)
        if stage in BANK_SLOTS and stage != "latest":
            self.put(stage, dict(entry))

    def note_score(self, *, job_id: str, confirm_ppl: float | None) -> None:
        """Point ``best`` at this job when its confirm perplexity is the lowest so far."""
        if confirm_ppl is None:
            return
        try:
            score = float(confirm_ppl)
        except (TypeError, ValueError):
            return
        data = self.load()
        match = None
        for entry in data.values():
            if entry.get("job_id") == job_id:
                match = dict(entry)
                break
        if match is None:
            return
        match["confirm_ppl"] = score
        # Keep the score on the slot this job already occupies.
        for slot, entry in list(data.items()):
            if entry.get("job_id") == job_id:
                entry["confirm_ppl"] = score
                data[slot] = entry
        best = data.get("best")
        if best is None or best.get("confirm_ppl") is None or score < float(best["confirm_ppl"]):
            data["best"] = match
            data["best"]["slot"] = "best"
        self.save(data)


def bank_parent_error(parent: str, stage: str, bank: CheckpointBank) -> str | None:
    """None when ``parent`` is not a bank id, or when that slot can be used."""
    raw = str(parent or "").strip()
    if not raw.startswith("bank:"):
        return None
    slot = raw.split(":", 1)[1].strip()
    if slot not in BANK_SLOTS:
        choices = ", ".join(f"bank:{name}" for name in BANK_SLOTS)
        return f"unknown checkpoint {raw}; choose from {choices}"
    if stage != "pretrain" and bank.resolve(slot) is None:
        filled = ", ".join(row["id"] for row in bank.entries()) or "none yet"
        return f"{raw} is empty; filled slots: {filled}"
    return None
