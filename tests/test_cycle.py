from __future__ import annotations

from lab.policy import DummyPolicy, InterleavePolicy, ScriptedPolicy
from lab.promote import promote


def test_dummy_cycle_completes(sup) -> None:
    result = sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    assert result["halted"] is True
    assert result["completed_cycles"] == 1
    assert result["phase"] == "eval"
    assert result["eval_ran"] is True
    assert result["last_eval"]["score"] is not None
    assert result["last_metrics"]["backend"] == "dummy"
    kinds = [e["kind"] for e in sup.notebook.all()]
    assert "pack" in kinds
    assert "job" in kinds
    assert "eval" in kinds


def test_scripted_cycle_varies_config(sup) -> None:
    result = sup.run_policy(ScriptedPolicy(seed=3), max_cycles=1)
    assert result["completed_cycles"] == 1
    pack = sup.packs.load(result["pack_hash"])
    assert pack.config["lr"] in {5e-4, 1e-3, 2e-3, 3e-3}


def test_interleave_runs_two_cycles(sup) -> None:
    policy = InterleavePolicy(DummyPolicy(), ScriptedPolicy(seed=1), every=2)
    result = sup.run_policy(policy, max_cycles=2, max_steps=80)
    assert result["completed_cycles"] == 2
    notes = [e for e in sup.notebook.all() if e.get("kind") == "pack"]
    assert len(notes) >= 2
    assert any("scripted" in (e.get("hypothesis") or "") for e in notes)


def test_promote_uses_confirm_not_tuning(sup) -> None:
    sup.run_policy(DummyPolicy(), max_cycles=1)
    first = promote(sup.cfg.run_dir)
    assert first["promoted"] is True
    second = promote(sup.cfg.run_dir, min_delta=10.0)
    assert second["promoted"] is False
