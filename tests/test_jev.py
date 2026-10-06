"""Jev classifies traces. No model, no GPU."""

from __future__ import annotations

import json

from lab.jobs import Job
from lab.jev import classify
from lab.policy import DummyPolicy


def _tool(name: str, *, phase: str, cycle: int = 1, ok: bool = True, outputs=None, n: int = 1) -> list[dict]:
    return [
        {
            "name": f"tool.{name}",
            "kind": "tool",
            "phase": phase,
            "cycle": cycle,
            "tool": name,
            "ok": ok,
            "ts": 1_000.0 + i,
            "outputs": outputs,
        }
        for i in range(n)
    ]


def test_healthy_cycle_is_not_a_repair() -> None:
    rows = (
        _tool("run_eval", phase="eval")
        + _tool("enter_research", phase="eval")
        + _tool("write_hypothesis", phase="research")
        + _tool("write_pack", phase="research")
        + _tool("enter_train", phase="research", outputs={"job": {"status": "succeeded"}})
        + _tool("enter_eval", phase="train", outputs={"ok": True})
    )
    verdict = classify(rows, train_budget_s=3600, phase_entered_at=1000, now=1001)
    assert verdict["repair"] is False
    assert verdict["action"] is None
    assert verdict["train_steps"] == 1


def test_finished_job_that_stays_in_train_asks_for_eval() -> None:
    rows = _tool("enter_train", phase="research", outputs={"job": {"status": "succeeded"}})
    rows += _tool("read_notebook", phase="train", n=2)
    verdict = classify(rows, train_tail_steps=2)
    assert verdict["repair"] is True
    assert verdict["action"] == "enter_eval"
    assert verdict["findings"][0]["code"] == "train_linger"


def test_running_job_under_budget_is_not_a_repair() -> None:
    rows = _tool("job_status", phase="train", n=6, outputs={"job": {"status": "running"}})
    verdict = classify(
        rows,
        train_tail_steps=2,
        train_budget_s=3600,
        phase_entered_at=1000,
        now=1010,
    )
    assert verdict["repair"] is False
    assert verdict["train_steps"] == 6


def test_running_job_past_budget_asks_to_cancel() -> None:
    rows = _tool("job_status", phase="train", n=2, outputs={"job": {"status": "running"}})
    verdict = classify(
        rows,
        train_budget_s=30,
        phase_entered_at=0,
        now=100,
    )
    assert verdict["action"] == "cancel_job"
    assert any(f["code"] == "train_over_budget" for f in verdict["findings"])


def test_error_burst_and_repeat_call() -> None:
    rows = _tool("web_fetch", phase="research", n=4, ok=False)
    burst = classify(rows)
    assert any(f["code"] == "error_burst" for f in burst["findings"])

    repeated = classify(_tool("read_notebook", phase="eval", n=5))
    assert any(f["code"] == "repeat_call" for f in repeated["findings"])


def test_stuck_eval_after_run_eval_suggests_research() -> None:
    rows = _tool("run_eval", phase="eval") + _tool("read_notebook", phase="eval", n=9)
    verdict = classify(rows, eval_cap=8)
    assert verdict["action"] == "enter_research"
    assert verdict["findings"][0]["code"] == "stuck_phase"


def test_override_enters_eval_but_does_not_cancel_a_lab_job(sup) -> None:
    sup.state.phase = "train"
    sup.state.current_job_id = "job-0009"
    job = Job(id="job-0009", pack_hash="abc", backend="lab", status="running")
    sup.jobs.store.save(job)
    sup._jev_verdict = {"repair": True, "action": "cancel_job", "phase": "train"}
    sup.cfg.jev_apply = True
    assert sup._jev_override() is None

    job.status = "succeeded"
    job.backend = "dummy"
    sup.jobs.store.save(job)
    sup._jev_verdict = {"repair": True, "action": "enter_eval", "phase": "train"}
    sup.cfg.jev_apply = False
    assert sup._jev_override() == ("enter_eval", {})


def test_skipped_eval_is_a_warning_not_a_repair(sup) -> None:
    assert sup.call("enter_research")["ok"]
    rows = [json.loads(line) for line in (sup.cfg.run_dir / "jev.jsonl").read_text().splitlines()]
    warns = [f for row in rows for f in row["findings"] if f["code"] == "skipped_eval"]
    assert warns
    assert all(f["severity"] == "warn" for f in warns)
    assert all(row["repair"] is False for row in rows)
    assert not any(n.get("kind") == "jev" for n in sup.notebook.tail(50))


def test_dummy_policy_still_finishes_one_cycle(sup) -> None:
    result = sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=40)
    assert result["halted"]
    assert result["completed_cycles"] >= 1
    assert result["halt_reason"] == "max_cycles reached"
