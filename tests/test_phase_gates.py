"""Phase tool gates, sandbox jail, pack validation, research caps."""

from __future__ import annotations

from lab.pack import ArtifactPack, Budgets
from lab.policy import dummy_pack
from tests.helpers import arm_for_train


def test_eval_phase_rejects_web_fetch_and_exec(sup) -> None:
    """Eval is read-only: no browsing, no shell."""
    fetch = sup.call("web_fetch", {"url": "https://example.com"})
    assert fetch["ok"] is False
    assert "not allowed" in fetch["error"]
    exe = sup.call("exec", {"argv": ["python3", "-c", "print(1)"]})
    assert exe["ok"] is False


def test_eval_phase_rejects_write_hypothesis(sup) -> None:
    """Hypothesis writes belong in research; eval looping write_hypothesis was a Nano hang."""
    out = sup.call(
        "write_hypothesis",
        {"claim": "too early", "why": "eval loop", "falsify": "never trains"},
    )
    assert out["ok"] is False
    assert "not allowed" in out["error"]


def test_research_phase_allows_search_fetch_and_sandboxed_exec(sup, transport) -> None:
    assert sup.call("enter_research")["ok"]
    search = sup.call("web_search", {"query": "tinystories"})
    assert search["ok"] is True
    assert transport.urls
    assert search["hits"]
    wrote = sup.call("write_file", {"path": "hello.py", "content": "print('ok')\n"})
    assert wrote["ok"]
    exe = sup.call("exec", {"argv": ["python3", "hello.py"]})
    assert exe["ok"] is True
    assert exe["returncode"] == 0
    assert "ok" in exe["stdout"]


def test_sandbox_write_cannot_escape_into_frozen_eval(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call("write_file", {"path": "../frozen_eval/core_v1.json", "content": "nope"})
    assert out["ok"] is False


def test_pack_rejects_train_budget_above_hard_ceiling(sup) -> None:
    assert sup.call("enter_research")["ok"]
    pack = dummy_pack(sup.observe())
    pack["budgets"] = {"max_hours": 9.0, "max_steps": 10}
    out = sup.call("write_pack", {"pack": pack})
    assert out["ok"] is False


def test_research_tool_call_cap_is_enforced(sup) -> None:
    sup.cfg.research_max_tool_calls = 2
    assert sup.call("enter_research")["ok"]
    assert sup.call("list_files")["ok"]
    assert sup.call("list_files")["ok"]
    out = sup.call("list_files")
    assert out["ok"] is False
    assert "tool-call cap" in out["error"]


def test_artifact_pack_canonical_digest_is_stable() -> None:
    pack = ArtifactPack.from_dict(
        {
            "hypothesis": "try more steps",
            "trainer": "dummy",
            "config": {"lr": 0.002, "steps": 12},
            "data_manifest": {"sources": []},
            "eval_suite_id": "core",
            "eval_suite_version": 1,
            "parent_checkpoint": "subjects/tinytrain-8m",
            "budgets": {"max_hours": 0.01, "max_steps": 12},
        }
    )
    assert pack.digest() == pack.digest()
    assert pack.budgets.max_hours == 0.01
    Budgets(max_hours=3.5).validate()


def test_armed_dummy_train_job_succeeds(sup) -> None:
    arm_for_train(sup)
    started = sup.call("enter_train")
    assert started["ok"] is True
    assert started["job"]["backend"] == "dummy"
    assert started["job"]["metrics"]["val_ppl"] > 1
