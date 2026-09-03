from __future__ import annotations

from typing import Any
import json
from pathlib import Path

from lab.state import RunState


def promote(run_dir: Path, min_delta: float = 0.0) -> dict[str, Any]:
    """Promote on confirmation metric. Direction comes from the suite (PPL: lower)."""
    run_dir = Path(run_dir)
    state = RunState.from_dict(json.loads((run_dir / "state.json").read_text(encoding="utf-8")))
    ev = state.last_eval or {}
    if ev.get("confirm_score") is None:
        return {"promoted": False, "reason": "no confirmation eval"}
    if ev.get("suite_known") is False:
        return {"promoted": False, "reason": "unknown eval suite version"}

    higher = bool(ev.get("higher_is_better", False))
    champ_path = run_dir / "champion.json"
    if champ_path.is_file():
        champ = json.loads(champ_path.read_text(encoding="utf-8"))
        if champ.get("suite_id") != ev.get("suite_id") or champ.get("suite_version") != ev.get("suite_version"):
            return {"promoted": False, "reason": "eval suite changed; bump is a domain jump, not a win"}
        confirm = float(ev["confirm_score"])
        best = float(champ["confirm_score"])
        beat = confirm >= best + min_delta if higher else confirm <= best - min_delta
        if not beat:
            return {
                "promoted": False,
                "reason": "confirm_score did not beat champion",
                "confirm_score": confirm,
                "champion": best,
                "higher_is_better": higher,
            }

    record = {
        "pack_hash": state.pack_hash,
        "job_id": state.current_job_id,
        "confirm_score": ev["confirm_score"],
        "confirm_ppl": ev.get("confirm_ppl"),
        "score": ev.get("score"),
        "suite_id": ev.get("suite_id"),
        "suite_version": ev.get("suite_version"),
        "higher_is_better": higher,
        "metrics": state.last_metrics,
        "loss": ev.get("loss"),
        "benchmarks": ev.get("benchmarks"),
    }
    champ_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return {"promoted": True, **record}
