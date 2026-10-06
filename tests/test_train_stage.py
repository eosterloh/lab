"""Train stage: pretrain, continue, midtrain, posttrain. No GPU."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from lab.pack import ArtifactPack, parent_plan
from lab.policy import dummy_pack, lab_pack
from lab.train.loop import step_ceiling


def _pack(**extra: object) -> dict:
    payload = lab_pack({"cycle": 1, "subject_checkpoint": "subjects/x"})
    payload.update(extra)
    return payload


def test_omitted_stage_is_continue_and_a_long_pretrain_is_legal() -> None:
    continued = ArtifactPack.from_dict(_pack(budgets={"max_hours": 9.0, "max_steps": 10}))
    assert continued.stage == "continue"
    pretrained = ArtifactPack.from_dict(
        _pack(stage="pretrain", budgets={"max_hours": 84.0, "max_steps": 1000})
    )
    assert pretrained.stage == "pretrain"
    assert pretrained.canonical()["stage"] == "pretrain"


def test_stage_caps_reject_a_posttrain_that_asks_for_a_pretrain_budget() -> None:
    with pytest.raises(ValueError, match="posttrain"):
        ArtifactPack.from_dict(_pack(stage="posttrain", budgets={"max_hours": 10, "max_steps": 10}))
    with pytest.raises(ValueError, match="midtrain"):
        ArtifactPack.from_dict(_pack(stage="midtrain", budgets={"max_hours": 30, "max_steps": 10}))
    with pytest.raises(ValueError, match="cannot exceed"):
        ArtifactPack.from_dict(_pack(stage="pretrain", budgets={"max_hours": 100, "max_steps": 10}))
    with pytest.raises(ValueError, match="stage must be"):
        ArtifactPack.from_dict(_pack(stage="lora"))


def test_parent_plan_matches_the_four_stages() -> None:
    assert parent_plan("pretrain", parent_exists=True) == {
        "load_weights": False,
        "load_optim": False,
        "required": False,
    }
    assert parent_plan("continue", parent_exists=False)["required"] is False
    assert parent_plan("continue", parent_exists=True)["load_optim"] is True
    assert parent_plan("midtrain", parent_exists=True) == {
        "load_weights": True,
        "load_optim": False,
        "required": True,
    }
    assert parent_plan("posttrain", parent_exists=False)["required"] is True


def test_step_ceiling_honors_steps_on_a_long_clock() -> None:
    assert step_ceiling({"steps": 12}, {"max_hours": 84, "max_steps": 1_000_000}) == 12


def test_dummy_pack_still_loads_without_a_stage() -> None:
    pack = ArtifactPack.from_dict(dummy_pack({"cycle": 1}))
    assert pack.stage == "continue"


def _tiny(stage: str, steps: int = 1) -> dict:
    return {
        "hypothesis": stage,
        "trainer": "lab",
        "stage": stage,
        "config": {
            "lr": 0.01,
            "steps": steps,
            "hidden": 16,
            "layers": 1,
            "heads": 1,
            "seq_len": 8,
            "batch": 2,
        },
        "data_manifest": {"sources": ["builtin:tiny"]},
        "eval_suite_id": "core",
        "eval_suite_version": 2,
        "parent_checkpoint": "subjects/x",
        "budgets": {"max_hours": 0.05, "max_steps": steps},
    }


def test_stages_load_or_ignore_a_parent_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch")
    monkeypatch.setenv("LAB_TRAIN_DEVICE", "cpu")
    from lab.train.loop import train

    first = tmp_path / "first"
    first.mkdir()
    (first / "data.txt").write_text("abcd " * 80, encoding="utf-8")
    (first / "pack.json").write_text(json.dumps(_tiny("pretrain")), encoding="utf-8")
    made = train(first)
    assert made["stage"] == "pretrain"
    assert made["resumed"] is False

    def run(stage: str, dest: Path) -> dict:
        dest.mkdir()
        (dest / "data.txt").write_text("abcd " * 80, encoding="utf-8")
        (dest / "parent.pt").write_bytes((first / "checkpoint.pt").read_bytes())
        (dest / "pack.json").write_text(json.dumps(_tiny(stage)), encoding="utf-8")
        return train(dest)

    again = run("pretrain", tmp_path / "pre2")
    assert again["resumed"] is False
    assert again["optim_resumed"] is False

    kept = run("continue", tmp_path / "cont")
    assert kept["resumed"] is True
    assert kept["optim_resumed"] is True

    shifted = run("midtrain", tmp_path / "mid")
    assert shifted["resumed"] is True
    assert shifted["optim_resumed"] is False

    tuned = run("posttrain", tmp_path / "post")
    assert tuned["resumed"] is True
    assert tuned["optim_resumed"] is False

    bare = tmp_path / "noparent"
    bare.mkdir()
    (bare / "data.txt").write_text("abcd " * 80, encoding="utf-8")
    (bare / "pack.json").write_text(json.dumps(_tiny("posttrain")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent.pt"):
        train(bare)
