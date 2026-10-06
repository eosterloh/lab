"""A second harness that heals a run from the outside.

It is not the ensemble repairing itself. It reads that run's trace.jsonl,
asks Jev whether the run is good, and when it is not, writes a heal card.
Call it once after a run, or on a stride while the run is still going.
No model is loaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from lab.jev import classify
from lab.trace_report import load_trace


def _signature(verdict: dict[str, Any]) -> str:
    codes = [str(f.get("code")) for f in verdict.get("findings") or [] if f.get("severity") == "repair"]
    return "|".join([str(verdict.get("action") or ""), *codes])


class Healer:
    """Look, decide with Jev, and write a heal when the run is not good."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.root = self.run_dir / "healer"
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"passes": 0, "heals": 0, "last_signature": None, "last_good": None}
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"passes": 0, "heals": 0, "last_signature": None, "last_good": None}
        return raw if isinstance(raw, dict) else {"passes": 0, "heals": 0, "last_signature": None, "last_good": None}

    def _save(self) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def pass_once(self, *, when: str = "after") -> dict[str, Any]:
        """One look at the current trace. ``when`` is ``during`` or ``after``."""
        rows = load_trace(self.run_dir)
        verdict = classify(rows)
        good = not bool(verdict.get("repair"))
        signature = _signature(verdict)
        self.state["passes"] = int(self.state.get("passes") or 0) + 1
        self.state["last_good"] = good
        self.state["last_seen"] = len(rows)
        record: dict[str, Any] = {
            "when": when,
            "pass": self.state["passes"],
            "good": good,
            "scanned": verdict.get("scanned"),
            "action": verdict.get("action"),
            "phase": verdict.get("phase"),
            "cycle": verdict.get("cycle"),
            "findings": verdict.get("findings") or [],
        }
        if good:
            self.state["last_signature"] = None
            self._append("passes.jsonl", record)
            self._save()
            return record
        if signature == self.state.get("last_signature"):
            record["heal_id"] = self.state.get("last_heal_id")
            record["repeated"] = True
            self._append("passes.jsonl", record)
            self._save()
            return record
        heal_id = self._write_heal(record, signature)
        record["heal_id"] = heal_id
        record["repeated"] = False
        self.state["last_signature"] = signature
        self.state["last_heal_id"] = heal_id
        self.state["heals"] = int(self.state.get("heals") or 0) + 1
        self._append("passes.jsonl", record)
        self._save()
        return record

    def watch(self, *, every: float, passes: int, when: str = "during") -> list[dict[str, Any]]:
        """Look ``passes`` times, sleeping ``every`` seconds between looks."""
        if passes < 1:
            raise ValueError("passes must be >= 1")
        out = []
        for i in range(passes):
            if i:
                time.sleep(max(0.0, every))
            out.append(self.pass_once(when=when))
        return out

    def _write_heal(self, record: dict[str, Any], signature: str) -> str:
        n = int(self.state.get("heals") or 0) + 1
        heal_id = f"heal-{n:04d}"
        codes = [str(f.get("code")) for f in record["findings"] if f.get("severity") == "repair"]
        card = {
            "id": heal_id,
            "when": record["when"],
            "good": False,
            "action": record.get("action"),
            "codes": codes,
            "signature": signature,
            "phase": record.get("phase"),
            "cycle": record.get("cycle"),
            "findings": record["findings"],
            "note": _note(record),
        }
        path = self.root / f"{heal_id}.json"
        path.write_text(json.dumps(card, indent=2, sort_keys=True), encoding="utf-8")
        self._append("heals.jsonl", {"id": heal_id, "action": card["action"], "codes": codes})
        return heal_id

    def _append(self, name: str, row: dict[str, Any]) -> None:
        path = self.root / name
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


def _note(record: dict[str, Any]) -> str:
    action = record.get("action") or "inspect"
    details = "; ".join(
        str(f.get("detail"))
        for f in record.get("findings") or []
        if f.get("severity") == "repair"
    )
    if action == "enter_eval":
        lead = "The job is finished and the run is still in train. The ensemble should enter eval."
    elif action == "cancel_job":
        lead = "Train has been open past its budget. The ensemble should cancel the job and enter eval."
    elif action == "enter_research":
        lead = "Eval has already run and the run is still there. The ensemble should enter research."
    else:
        lead = "Jev wants a repair and did not name a phase change. Read the findings before the next cycle."
    return f"{lead} {details}".strip()
