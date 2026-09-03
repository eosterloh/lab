"""End-to-end dummy trainer: Eval → Research → Train → Eval.

This is the harness smoke path. Dummy jobs do not load weights; they still
must pass the live-hypothesis checklist, seal an episode, and halt cleanly.
"""

from __future__ import annotations

from pathlib import Path
import json

from lab.cli import main
from lab.policy import DummyPolicy, InterleavePolicy, ScriptedPolicy
from lab.promote import promote


def test_dummy_policy_completes_one_full_cycle_and_seals_memory(sup) -> None:
    """One cycle: eval, hypothesis, pack, dummy train, post-eval episode, halt."""
    result = sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)

    assert result["halted"] is True
    assert result["halt_reason"] == "max_cycles reached"
    assert result["completed_cycles"] == 1
    assert result["phase"] == "eval"
    assert result["eval_ran"] is True
    assert result["last_metrics"]["backend"] == "dummy"
    assert result["last_eval"]["confirm_ppl"] is not None

    kinds = [e["kind"] for e in sup.notebook.all()]
    assert kinds.count("eval") >= 2  # baseline + post-train
    assert "pack" in kinds
    assert "job" in kinds
    assert "episode" in kinds

    hyp = result["hypothesis"]
    assert hyp["status"] == "testing"
    assert hyp["open_train"] == []
    assert "claim_written" not in hyp["open_train"]
    done = {c["id"]: c["done"] for c in hyp["checklist"]}
    assert done["claim_written"] is True
    assert done["pack_ready"] is True
    assert done["post_eval"] is True
    assert done["episode_sealed"] is True
    # Dummy trainer has no checkpoint, so promotion box stays open on purpose.
    assert done["holdout_not_proxy"] is False

    episodes = result["episodes"]
    assert len(episodes) == 1
    assert episodes[0]["job_status"] == "succeeded"
    assert "confirm_ppl" in episodes[0]["title"]
    assert episodes[0]["id"] in sup.notebook.beliefs()


def test_dummy_policy_can_run_two_cycles_and_index_both_episodes(sup) -> None:
    """Second cycle must not clobber the first episode card."""
    result = sup.run_policy(DummyPolicy(), max_cycles=2, max_steps=80)
    assert result["completed_cycles"] == 2
    rows = sup.episodes.summaries()
    assert len(rows) == 2
    assert rows[0]["id"] != rows[1]["id"]


def test_scripted_control_samples_config_then_still_trains_dummy(sup) -> None:
    """Scripted policy is the dumb baseline: random lr/steps, same dummy trainer."""
    result = sup.run_policy(ScriptedPolicy(seed=3), max_cycles=1, max_steps=50)
    assert result["completed_cycles"] == 1
    pack = sup.packs.load(result["pack_hash"])
    assert pack.config["lr"] in {5e-4, 1e-3, 2e-3, 3e-3}


def test_interleave_runs_dummy_then_scripted_on_even_cycles(sup) -> None:
    """Cycle 1 dummy, cycle 2 scripted — both must seal episodes."""
    policy = InterleavePolicy(DummyPolicy(), ScriptedPolicy(seed=1), every=2)
    result = sup.run_policy(policy, max_cycles=2, max_steps=80)
    assert result["completed_cycles"] == 2
    notes = [e for e in sup.notebook.all() if e.get("kind") == "pack"]
    assert any("scripted" in (e.get("hypothesis") or "") for e in notes)
    assert len(sup.episodes.summaries()) == 2


def test_promote_uses_confirmation_metric_and_rejects_tiny_deltas(sup) -> None:
    """First seal can become champion; a huge min_delta must not re-promote."""
    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    first = promote(sup.cfg.run_dir)
    assert first["promoted"] is True
    second = promote(sup.cfg.run_dir, min_delta=10.0)
    assert second["promoted"] is False


def test_cli_dummy_two_cycles_writes_run_dir(tmp_path: Path, monkeypatch) -> None:
    """python -m lab run --policy dummy --cycles 2 must exit 0 and leave artifacts."""
    monkeypatch.setenv("LAB_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("LAB_ALLOW_NETWORK", "0")
    run_dir = tmp_path / "cli-dummy"
    code = main(
        [
            "run",
            "--policy",
            "dummy",
            "--cycles",
            "2",
            "--run-dir",
            str(run_dir),
            "--max-steps",
            "80",
        ]
    )
    assert code == 0
    assert (run_dir / "state.json").is_file()
    assert (run_dir / "hypothesis.json").is_file()
    assert (run_dir / "episodes" / "index.jsonl").stat().st_size > 0
    assert (run_dir / "notebook.jsonl").stat().st_size > 0
    tools = (run_dir / "tools.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert tools
    names = [json.loads(line)["tool"] for line in tools]
    assert "run_eval" in names
    assert "enter_research" in names
    assert "write_hypothesis" in names
