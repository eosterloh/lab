"""Phase hooks. The supervisor fires these; listeners must not break the run.

Live points run on every transition. Open points are the ones worth adding
between phases; they are named here so a listener can subscribe, and so
``python -m lab dry-eval`` can list them. None of the open points change
control flow until one is wired.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json

# status=live means the supervisor fires this point today.
# status=open means the slot is reserved; the idea is the listener to write.
HOOK_POINTS: dict[str, dict[str, str]] = {
    "after_eval": {
        "status": "live",
        "when": "run_eval has stored last_eval",
        "idea": (
            "Compare confirm_ppl to the best sealed episode. A non-dummy "
            "trainer_val_proxy is a warning. Next: drop a trial that regressed "
            "past a margin before research opens."
        ),
    },
    "before_research": {
        "status": "live",
        "when": "enter_research, before the phase flips",
        "idea": (
            "Warn when eval did not run. Next: refuse the transition outright, "
            "and on a mid-trial hop skip research when the queue is already loaded."
        ),
    },
    "before_train": {
        "status": "live",
        "when": "enter_train, after the pack and checklist pass, before the GPU lock",
        "idea": (
            "Record the pack budget. Next: reject a parent_checkpoint that is "
            "not last_checkpoint, and skip a pack hash that already failed."
        ),
    },
    "before_eval": {
        "status": "live",
        "when": "enter_eval, once the job is terminal, before the lock drops",
        "idea": (
            "Jev records train dwell. Next: if train.log loss is flat, mark the "
            "episode uninformative so research does not treat it as a signal."
        ),
    },
    "on_step": {
        "status": "live",
        "when": "each policy turn, before the tool call",
        "idea": (
            "Jev scans on a stride, and every step once a train job is terminal. "
            "Next: at 80% of the research tool cap with no pack, stop browsing."
        ),
    },
    "between_trials": {
        "status": "live",
        "when": "enter_research while candidates are still queued, before the next one loads",
        "idea": (
            "Logged today, no decision yet. Next: if this trial's confirm_ppl is worse "
            "than the best by a margin, clear the rest of the candidate queue."
        ),
    },
}

HookFn = Callable[[dict[str, Any]], dict[str, Any] | None]


class HookBus:
    """Append-only log plus listeners. A listener that raises is recorded and swallowed."""

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._handlers: dict[str, list[HookFn]] = {}

    def on(self, point: str, fn: HookFn) -> None:
        if point not in HOOK_POINTS:
            raise KeyError(f"unknown hook {point!r}")
        self._handlers.setdefault(point, []).append(fn)

    def fire(self, point: str, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for fn in list(self._handlers.get(point, [])):
            try:
                out = fn(ctx) or {}
                if not isinstance(out, dict):
                    out = {"ok": True, "value": out}
            except Exception as exc:
                out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            results.append(out)
        self._write({"point": point, "results": results, **_brief(ctx)})
        return results

    def _write(self, row: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            return


def _brief(ctx: dict[str, Any]) -> dict[str, Any]:
    keep = ("cycle", "phase", "trial", "step", "to", "eval_ran", "pack_hash", "candidates_left")
    return {k: ctx[k] for k in keep if k in ctx}


def install_default_hooks(sup: Any) -> None:
    """Boundary scans. ``on_step`` stays open for callers; the loop scans itself."""

    def _scan(ctx: dict[str, Any]) -> dict[str, Any]:
        verdict = sup.scan_jev(reason=str(ctx.get("point") or "hook"))
        return {
            "ok": True,
            "repair": bool(verdict.get("repair")),
            "action": verdict.get("action"),
        }

    for point in ("after_eval", "before_research", "before_train", "before_eval", "between_trials"):
        sup.hooks.on(point, _scan)
