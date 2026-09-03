"""In-repo lab trainer: pack validation, sandbox env, and a real job if torch is present."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.pack import ArtifactPack
from lab.policy import LabPolicy, lab_pack
from lab.train import sandbox_env
from tests.helpers import arm_for_train


def test_lab_pack_validates_without_torch() -> None:
    pack = ArtifactPack.from_dict(lab_pack({"cycle": 1, "subject_checkpoint": "subjects/x"}))
    assert pack.trainer == "lab"
    assert pack.config["hidden"] == 32
    assert pack.config["heads"] == 1
    from lab.data_cache import DEFAULT_MIX

    assert pack.data_manifest["sources"] == list(DEFAULT_MIX)


def test_lab_pack_rejects_hidden_not_divisible_by_heads() -> None:
    payload = lab_pack({"cycle": 1}, extra_config={"hidden": 32, "heads": 3})
    with pytest.raises(ValueError, match="divisible"):
        ArtifactPack.from_dict(payload)


def test_train_sandbox_env_is_stripped_and_homes_in_job_dir(tmp_path: Path) -> None:
    job_dir = tmp_path / "jobs" / "job-0001"
    job_dir.mkdir(parents=True)
    env = sandbox_env(job_dir)
    assert env["HOME"] == str(job_dir)
    assert env["LAB_JOB_DIR"] == str(job_dir)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["LAB_TRAIN_DEVICE"] == "cpu"
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "OPENAI_API_KEY" not in env
    assert (job_dir / "tmp").is_dir()


def test_lab_policy_one_cycle_stores_artifacts_under_run_dir(sup) -> None:
    pytest.importorskip("torch")
    result = sup.run_policy(LabPolicy(), max_cycles=1, max_steps=50)
    assert result["halted"] is True
    assert result["completed_cycles"] == 1
    assert result["last_metrics"]["backend"] == "lab"
    assert result["last_eval"]["confirm_source"] == "frozen_holdout"
    assert result["last_eval"]["confirm_ppl"] is not None
    assert result["last_eval"]["loss"]["holdout"]["confirm"]["n"] > 0
    done = {c["id"]: c["done"] for c in result["hypothesis"]["checklist"]}
    assert done["holdout_not_proxy"] is True

    job_id = result["job"]["id"]
    job_dir = sup.cfg.jobs_dir / job_id
    for name in ("pack.json", "train.log", "metrics.json", "checkpoint.pt", "job.json", "data.txt"):
        assert (job_dir / name).is_file(), name
    assert (sup.cfg.checkpoints_dir / f"{job_id}.pt").is_file()
    assert (sup.cfg.run_dir / "state.json").is_file()
    assert (sup.cfg.run_dir / "hypothesis.json").is_file()
    assert (sup.cfg.run_dir / "notebook.jsonl").is_file()
    assert "step=" in (job_dir / "train.log").read_text(encoding="utf-8")


def test_armed_lab_train_job_succeeds(sup) -> None:
    pytest.importorskip("torch")
    arm_for_train(sup, pack=lab_pack(sup.observe()))
    started = sup.call("enter_train")
    assert started["ok"] is True
    assert started["job"]["backend"] == "lab"
    assert started["job"]["status"] == "succeeded"
    assert started["job"]["metrics"]["val_ppl"] > 1
    assert started["job"]["metrics"]["resumed"] is False


def test_lab_second_cycle_resumes_parent_and_scores_holdout(sup) -> None:
    pytest.importorskip("torch")
    result = sup.run_policy(LabPolicy(), max_cycles=2, max_steps=80)
    assert result["completed_cycles"] == 2
    job = result["job"]
    assert job["metrics"]["resumed"] is True
    assert (sup.cfg.jobs_dir / job["id"] / "parent.pt").is_file()
    assert result["last_eval"]["confirm_source"] == "frozen_holdout"
    assert (sup.cfg.checkpoints_dir / "latest.pt").is_file()
