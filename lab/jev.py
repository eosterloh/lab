"""Jev: a trace classifier that decides whether the harness needs a repair.

Rules over ``trace.jsonl``. No model is loaded, so a scan never touches the GPU.
The supervisor calls this on a stride and on every phase boundary, and writes
the verdict to ``jev.jsonl``.
"""

from __future__ import annotations

from typing import Any

FINISHED = frozenset({"succeeded", "failed", "cancelled"})

# Tool calls in the current phase before Jev calls it stuck. Train is special:
# a running job may poll ``job_status`` for a long time; that is not a repair.
# Lingering *after the job is terminal* is.
EVAL_STEP_CAP = 8
REPEAT_CAP = 5
ERROR_WINDOW = 8
ERROR_RATE = 0.5


def tool_name(row: dict[str, Any]) -> str:
    tool = row.get("tool")
    if tool:
        return str(tool)
    return str(row.get("name") or "").removeprefix("tool.")


def tool_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("kind") == "tool" or str(row.get("name") or "").startswith("tool."):
            out.append(row)
    return out


def _status(out: Any) -> str | None:
    if not isinstance(out, dict):
        return None
    job = out.get("job")
    if isinstance(job, dict) and job.get("status"):
        return str(job["status"])
    status = out.get("status")
    if status in FINISHED or status == "running":
        return str(status)
    return None


def job_status_in(rows: list[dict[str, Any]], cycle: Any) -> str | None:
    """Last job status recorded on tool spans for this cycle."""
    status = None
    for row in rows:
        if cycle is not None and row.get("cycle") != cycle:
            continue
        found = _status(row.get("outputs"))
        if found:
            status = found
    return status


def _runs(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in tools:
        phase = str(row.get("phase") or "")
        cycle = row.get("cycle")
        if not runs or runs[-1]["phase"] != phase or runs[-1]["cycle"] != cycle:
            runs.append({"phase": phase, "cycle": cycle, "rows": [row]})
        else:
            runs[-1]["rows"].append(row)
    return runs


def _dwell_s(
    rows: list[dict[str, Any]],
    *,
    phase_entered_at: float | None,
    now: float | None,
) -> float | None:
    stamps = [float(r["ts"]) for r in rows if isinstance(r.get("ts"), (int, float))]
    span = (max(stamps) - min(stamps)) if len(stamps) >= 2 else 0.0
    clock = 0.0
    if phase_entered_at is not None and now is not None:
        clock = max(0.0, float(now) - float(phase_entered_at))
    if phase_entered_at is None and len(stamps) < 2:
        return None
    return max(span, clock)


def _finding(
    code: str,
    severity: str,
    phase: str,
    cycle: Any,
    detail: str,
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "phase": phase,
        "cycle": cycle,
        "detail": detail,
    }


def classify(
    rows: list[dict[str, Any]],
    *,
    eval_cap: int = EVAL_STEP_CAP,
    research_cap: int = 40,
    train_tail_steps: int = 2,
    train_budget_s: float | None = None,
    phase_entered_at: float | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Judge the open phase. Older cycles are context, not a fresh repair.

    ``train_tail_steps`` is how many train-phase tool calls are allowed after
    the job is already finished. The next call should be ``enter_eval``.
    ``train_budget_s`` is the pack's wall clock; a *running* job past that
    is ``train_over_budget``. A running job under budget is not a repair,
    however many times it polled ``job_status``.
    """
    tools = tool_rows(rows)
    empty: dict[str, Any] = {
        "repair": False,
        "action": None,
        "phase": None,
        "cycle": None,
        "scanned": 0,
        "train_steps": 0,
        "findings": [],
    }
    if not tools:
        return empty

    runs = _runs(tools)
    last = runs[-1]
    phase = str(last["phase"])
    cycle = last["cycle"]
    n = len(last["rows"])
    findings: list[dict[str, Any]] = []
    status = job_status_in(tools, cycle)
    train_steps = n if phase == "train" else 0

    if phase == "train" and status in FINISHED and n >= train_tail_steps:
        findings.append(
            _finding(
                "train_linger",
                "repair",
                phase,
                cycle,
                f"job is {status} and train still has {n} tool calls (cap {train_tail_steps})",
            )
        )
    elif phase == "train" and status == "running" and train_budget_s:
        dwell = _dwell_s(last["rows"], phase_entered_at=phase_entered_at, now=now)
        if dwell is not None and dwell > float(train_budget_s):
            findings.append(
                _finding(
                    "train_over_budget",
                    "repair",
                    phase,
                    cycle,
                    f"train has been open {dwell:.0f}s, budget {float(train_budget_s):.0f}s",
                )
            )

    if phase == "eval" and n > eval_cap:
        ran = any(tool_name(r) == "run_eval" for r in last["rows"])
        findings.append(
            _finding(
                "stuck_phase",
                "repair",
                phase,
                cycle,
                f"eval has {n} tool calls (cap {eval_cap})",
            )
        )
        if not ran:
            findings[-1]["detail"] += "; run_eval has not happened"
    elif phase == "research" and research_cap and n > research_cap:
        findings.append(
            _finding(
                "stuck_phase",
                "repair",
                phase,
                cycle,
                f"research has {n} tool calls (cap {research_cap})",
            )
        )

    window = tools[-ERROR_WINDOW:]
    if len(window) >= 4:
        bad = sum(1 for r in window if r.get("ok") is False)
        if bad / len(window) >= ERROR_RATE:
            findings.append(
                _finding(
                    "error_burst",
                    "repair",
                    phase,
                    cycle,
                    f"{bad}/{len(window)} recent tool calls failed",
                )
            )

    tail = [tool_name(r) for r in tools[-REPEAT_CAP:]]
    if len(tail) >= REPEAT_CAP and len(set(tail)) == 1 and tail[0] not in {
        "job_status",
        "run_eval",
        "enter_eval",
        "enter_train",
        "enter_research",
    }:
        findings.append(
            _finding(
                "repeat_call",
                "repair",
                phase,
                cycle,
                f"{tail[0]} repeated {len(tail)} times",
            )
        )

    codes = {f["code"] for f in findings if f["severity"] == "repair"}
    action = None
    if "train_linger" in codes:
        action = "enter_eval"
    elif "train_over_budget" in codes:
        action = "cancel_job"
    elif "stuck_phase" in codes and phase == "eval" and any(
        tool_name(r) == "run_eval" for r in last["rows"]
    ):
        action = "enter_research"

    return {
        "repair": bool(codes),
        "action": action,
        "phase": phase,
        "cycle": cycle,
        "scanned": len(tools),
        "train_steps": train_steps,
        "job_status": status,
        "findings": findings,
    }
