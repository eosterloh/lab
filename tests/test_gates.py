from __future__ import annotations

from lab.pack import ArtifactPack, Budgets
from lab.policy import dummy_pack


def test_eval_cannot_fetch(sup) -> None:
    out = sup.call("web_fetch", {"url": "https://example.com"})
    assert out["ok"] is False
    assert "not allowed" in out["error"]


def test_eval_cannot_exec(sup) -> None:
    out = sup.call("exec", {"argv": ["python3", "-c", "print(1)"]})
    assert out["ok"] is False


def test_enter_train_requires_pack(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call("enter_train")
    assert out["ok"] is False
    assert "write_pack" in out["error"]


def test_research_can_search_and_exec(sup, transport) -> None:
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


def test_sandbox_blocks_escape(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call("write_file", {"path": "../frozen_eval/story_v0.json", "content": "nope"})
    assert out["ok"] is False


def test_pack_rejects_over_budget(sup) -> None:
    assert sup.call("enter_research")["ok"]
    pack = dummy_pack(sup.observe())
    pack["budgets"] = {"max_hours": 9.0, "max_steps": 10}
    out = sup.call("write_pack", {"pack": pack})
    assert out["ok"] is False


def test_research_tool_cap(sup) -> None:
    sup.cfg.research_max_tool_calls = 2
    assert sup.call("enter_research")["ok"]
    assert sup.call("list_files")["ok"]
    assert sup.call("list_files")["ok"]
    out = sup.call("list_files")
    assert out["ok"] is False
    assert "tool-call cap" in out["error"]


def test_artifact_pack_roundtrip() -> None:
    pack = ArtifactPack.from_dict(
        {
            "hypothesis": "try more steps",
            "trainer": "dummy",
            "config": {"lr": 0.002, "steps": 12},
            "data_manifest": {"sources": []},
            "eval_suite_id": "story",
            "eval_suite_version": 0,
            "parent_checkpoint": "subjects/tinytrain-8m",
            "budgets": {"max_hours": 0.01, "max_steps": 12},
        }
    )
    assert pack.digest() == pack.digest()
    assert pack.budgets.max_hours == 0.01
    Budgets(max_hours=3.5).validate()
