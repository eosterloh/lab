"""Extra tooler tools: registration, phase gating, and behaviour through sup.call."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

from lab import actions
from lab.policy import DummyPolicy, dummy_pack
from lab.tools_extra import EXTRA_REGISTRY
from lab.types import INSPECT_TOOLS, PHASE_TOOLS, RESEARCH_EXTRA_TOOLS, Phase
from tests.helpers import arm_for_train

RESEARCH_ONLY = {"run_python", "write_and_run", "data_stats"}


def _write_two_packs(sup) -> tuple[str, str]:
    obs = sup.observe()
    a = sup.call("write_pack", {"pack": dummy_pack(obs, extra_config={"lr": 1e-3, "steps": 8})})
    b = sup.call("write_pack", {"pack": dummy_pack(obs, extra_config={"lr": 3e-3, "steps": 16})})
    assert a["ok"] and b["ok"]
    return a["pack_hash"], b["pack_hash"]


# --------------------------------------------------------------------------- registration


def test_every_extra_tool_is_registered_and_phase_gated() -> None:
    all_phase_tools = set().union(*PHASE_TOOLS.values())
    for name in EXTRA_REGISTRY:
        assert name in actions.REGISTRY, name
        assert name in all_phase_tools, f"{name} is not allowed in any phase"
    extras_in_phases = (INSPECT_TOOLS | RESEARCH_EXTRA_TOOLS) & all_phase_tools
    assert extras_in_phases == set(EXTRA_REGISTRY)
    for name in extras_in_phases:
        assert name in actions.REGISTRY, f"{name} in PHASE_TOOLS but not in REGISTRY"


def test_research_only_tools_are_rejected_in_eval(sup) -> None:
    for name in RESEARCH_ONLY:
        out = sup.call(name, {"path": "x.py", "content": "", "source": "x"})
        assert out["ok"] is False
        assert "not allowed" in out["error"]
    for name in RESEARCH_ONLY:
        assert name in PHASE_TOOLS[Phase.RESEARCH]
        assert name not in PHASE_TOOLS[Phase.EVAL]
        assert name not in PHASE_TOOLS[Phase.TRAIN]


def test_read_only_extras_allowed_in_eval(sup) -> None:
    assert sup.call("sandbox_usage")["ok"]
    assert sup.call("list_packs")["ok"]
    assert sup.call("list_jobs")["ok"]
    assert sup.call("read_trace")["ok"]
    assert sup.call("read_episode_metrics")["ok"]


# --------------------------------------------------------------------------- code execution


def test_write_and_run_executes_python(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call(
        "write_and_run",
        {"path": "hello.py", "content": "import sys\nprint('hi', sys.argv[1])\n", "args": ["there"]},
    )
    assert out["ok"] is True, out
    assert out["path"] == "hello.py"
    assert out["returncode"] == 0
    assert out["stdout"].strip() == "hi there"
    again = sup.call("run_python", {"path": "hello.py", "args": ["again"]})
    assert again["ok"] and again["stdout"].strip() == "hi again"


def test_run_python_validates_args(sup) -> None:
    assert sup.call("enter_research")["ok"]
    assert "path" in sup.call("run_python", {})["error"]
    assert "args" in sup.call("run_python", {"path": "x.py", "args": "nope"})["error"]
    assert "timeout_s" in sup.call("run_python", {"path": "x.py", "timeout_s": "slow"})["error"]
    missing = sup.call("run_python", {"path": "missing.py"})
    assert missing["ok"] is False and "missing.py" in missing["error"]
    bad = sup.call("write_and_run", {"path": "x.py", "content": 5})
    assert bad["ok"] is False and "content" in bad["error"]


def test_write_and_run_reports_timeout(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call(
        "write_and_run",
        {"path": "slow.py", "content": "import time; time.sleep(5)\n", "timeout_s": 0.5},
    )
    assert out["ok"] is True
    assert out["timed_out"] is True


def test_grep_files_finds_pattern(sup) -> None:
    assert sup.call("enter_research")["ok"]
    sup.sandbox.write_file("a/one.txt", "alpha\nneedle here\nomega\n")
    sup.sandbox.write_file("b/two.py", "x = 'needle'\n")
    sup.sandbox.write_file("c/bin.dat", "junk\x00needle\n")
    out = sup.call("grep_files", {"pattern": r"need\w+"})
    assert out["ok"] is True
    paths = {(h["path"], h["line"]) for h in out["hits"]}
    assert ("a/one.txt", 2) in paths
    assert ("b/two.py", 1) in paths
    assert not any(h["path"].endswith("bin.dat") for h in out["hits"])
    assert out["skipped"] == 1
    scoped = sup.call("grep_files", {"pattern": "needle", "path": "b", "max_hits": 1})
    assert scoped["ok"] and scoped["n_hits"] == 1 and scoped["hits"][0]["path"] == "b/two.py"
    bad = sup.call("grep_files", {"pattern": "("})
    assert bad["ok"] is False and "regex" in bad["error"]
    assert "pattern" in sup.call("grep_files", {})["error"]


# --------------------------------------------------------------------------- packs


def test_list_read_diff_packs(sup) -> None:
    assert sup.call("enter_research")["ok"]
    a, b = _write_two_packs(sup)
    listed = sup.call("list_packs", {"n": 10})
    assert listed["ok"] is True
    hashes = [p["pack_hash"] for p in listed["packs"]]
    assert set(hashes) == {a, b}
    assert listed["current"] == b
    assert all(p["trainer"] == "dummy" and p["hypothesis"] for p in listed["packs"])

    read = sup.call("read_pack", {"pack_hash": a})
    assert read["ok"] is True
    assert read["pack"]["config"]["lr"] == 1e-3

    diff = sup.call("diff_packs", {"a": a, "b": b})
    assert diff["ok"] is True
    assert diff["config_diff"]["lr"] == [1e-3, 3e-3]
    assert diff["config_diff"]["steps"] == [8, 16]
    assert diff["data_diff"] == {}
    assert "budgets" in diff["other_diff"]
    assert diff["same"] is False

    same = sup.call("diff_packs", {"a": a, "b": a})
    assert same["ok"] and same["same"] is True

    assert sup.call("read_pack", {"pack_hash": "deadbeefcafe"})["ok"] is False
    assert "pack_hash" in sup.call("read_pack", {})["error"]
    assert "hex" in sup.call("read_pack", {"pack_hash": "../etc"})["error"]


# --------------------------------------------------------------------------- jobs / logs / traces


def test_list_jobs_and_read_train_log_after_dummy_train(sup) -> None:
    arm_for_train(sup)
    started = sup.call("enter_train")
    assert started["ok"] is True
    job_id = started["job"]["id"]

    jobs = sup.call("list_jobs", {"n": 5})
    assert jobs["ok"] is True
    assert jobs["jobs"][0]["id"] == job_id
    assert jobs["jobs"][0]["status"] == "succeeded"
    assert jobs["jobs"][0]["backend"] == "dummy"
    assert jobs["jobs"][0]["val_ppl"] > 1
    assert jobs["jobs"][0]["started_at"] and jobs["jobs"][0]["ended_at"]
    assert jobs["current"] == job_id

    log = sup.call("read_train_log")
    assert log["ok"] is True
    assert log["job"]["id"] == job_id
    assert log["job"]["status"] == "succeeded"
    assert log["log_exists"] is True
    assert any("val_ppl" in line for line in log["lines"])
    short = sup.call("read_train_log", {"job_id": job_id, "tail": 2})
    assert short["ok"] and len(short["lines"]) == 2 and short["total_lines"] > 2

    assert sup.call("read_train_log", {"job_id": "job-9999"})["ok"] is False
    assert "tail" in sup.call("read_train_log", {"tail": "many"})["error"]


def test_read_train_log_without_job_errs(sup) -> None:
    out = sup.call("read_train_log")
    assert out["ok"] is False
    assert "job_id" in out["error"]


def _run_one_cycle(sup) -> None:
    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    # run_policy halts on max_cycles; lift it so inspection tools can be called.
    sup.state.halted = False
    sup.state.halt_reason = None


def test_read_trace_after_policy_run(sup) -> None:
    _run_one_cycle(sup)
    out = sup.call("read_trace", {"kind": "tool", "n": 10})
    assert out["ok"] is True
    assert out["rows"]
    assert all(r["kind"] == "tool" for r in out["rows"])
    assert all(r["name"].startswith("tool.") for r in out["rows"])
    for r in out["rows"]:
        for key in ("inputs", "outputs"):
            if r.get(key) is not None:
                assert len(r[key]) <= 300
    by_cycle = sup.call("read_trace", {"cycle": 1, "n": 500})
    assert by_cycle["ok"] and all(r.get("cycle") == 1 for r in by_cycle["rows"])
    assert "kind" in sup.call("read_trace", {"kind": "bogus"})["error"]


def test_read_trace_without_file(sup) -> None:
    out = sup.call("read_trace")
    assert out["ok"] is True
    assert out["rows"] == []


# --------------------------------------------------------------------------- models / checkpoints


def test_list_models_reads_config_json(sup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = tmp_path / "models"
    fake = models / "tiny-llm"
    fake.mkdir(parents=True)
    (fake / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}),
        encoding="utf-8",
    )
    (fake / "model.safetensors").write_bytes(b"\x00")
    (models / "not-a-model").mkdir()
    monkeypatch.setenv("LAB_MODELS_DIR", str(models))
    out = sup.call("list_models")
    assert out["ok"] is True
    assert len(out["models"]) == 1
    m = out["models"][0]
    assert m["name"] == "tiny-llm"
    assert m["model_type"] == "qwen3"
    assert m["architectures"] == ["Qwen3ForCausalLM"]
    assert m["has_safetensors"] is True
    assert m["path"] == str(fake)


def test_list_models_missing_dir(sup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_MODELS_DIR", str(tmp_path / "nope"))
    out = sup.call("list_models")
    assert out["ok"] is True and out["models"] == []


def test_inspect_model_errs_when_infer_missing(sup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = tmp_path / "models" / "m"
    fake.mkdir(parents=True)
    (fake / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sup.cfg, "infer_root", tmp_path / "no-infer")
    for mod in [m for m in sys.modules if m == "engine" or m.startswith("engine.")]:
        monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setitem(sys.modules, "engine", None)
    out = sup.call("inspect_model", {"model_dir": str(fake)})
    assert out["ok"] is False
    assert "infer" in out["error"]
    assert "model_dir" in sup.call("inspect_model", {})["error"]
    assert sup.call("inspect_model", {"model_dir": str(tmp_path / "missing")})["ok"] is False


def test_read_checkpoint_meta_refuses_paths_outside_run_dir(sup, tmp_path: Path) -> None:
    outside = tmp_path / "outside.pt"
    outside.write_bytes(b"not a checkpoint")
    out = sup.call("read_checkpoint_meta", {"path": str(outside)})
    assert out["ok"] is False
    assert "run_dir" in out["error"]
    rel = sup.call("read_checkpoint_meta", {"path": "../outside.pt"})
    assert rel["ok"] is False
    missing = sup.call("read_checkpoint_meta", {"path": "checkpoints/nope.pt"})
    assert missing["ok"] is False and "not a file" in missing["error"]
    assert "path" in sup.call("read_checkpoint_meta", {})["error"]


def test_read_checkpoint_meta_reads_lab_checkpoint(sup) -> None:
    torch = pytest.importorskip("torch")
    ckpt = sup.cfg.checkpoints_dir / "job-0001.pt"
    torch.save(
        {"model": {"w": torch.zeros(3, 4), "b": torch.zeros(4)}, "config": {"hidden": 4}},
        ckpt,
    )
    out = sup.call("read_checkpoint_meta", {"path": "checkpoints/job-0001.pt"})
    assert out["ok"] is True
    assert out["n_params"] == 16
    assert out["keys"] == ["w", "b"]
    assert out["config"] == {"hidden": 4}


# --------------------------------------------------------------------------- skills


def test_read_skill_errs_when_module_absent(sup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "lab.ensemble.skills", None)
    out = sup.call("read_skill", {"name": "anything"})
    assert out["ok"] is False
    assert "skills module not available" in out["error"]
    listed = sup.call("list_skills")
    assert listed["ok"] is False
    assert "name" in sup.call("read_skill", {})["error"]


def test_read_skill_happy_path_when_module_exists(sup) -> None:
    try:
        from lab.ensemble.skills import list_skills, load_skill  # noqa: F401
    except Exception:
        pytest.skip("lab.ensemble.skills not present yet")
    listed = sup.call("list_skills")
    assert listed["ok"] is True
    assert isinstance(listed["skills"], list)
    if listed["skills"]:
        name = listed["skills"][0]
        out = sup.call("read_skill", {"name": name})
        assert out["ok"] is True
        assert out["name"] == name
        assert isinstance(out["content"], str)
    bad = sup.call("read_skill", {"name": "definitely-not-a-skill-xyz"})
    assert bad["ok"] is False
    assert "skills" in bad


# --------------------------------------------------------------------------- data / episodes / sandbox


def test_data_stats(sup) -> None:
    assert sup.call("enter_research")["ok"]
    cached = sup.call("data_stats", {"source": "hf:roneneldan/TinyStories:train:10000"})
    assert cached["ok"] is True, cached
    assert cached["bytes"] > 0 and cached["lines"] > 0 and cached["unique_bytes"] > 0
    assert len(cached["sample"]) <= 300
    miss = sup.call("data_stats", {"source": "hf:wikimedia/wikipedia:train:5"})
    assert miss["ok"] is False
    assert "not cached" in miss["error"]
    assert sup.call("data_stats", {"source": "hf:evil/notallowed"})["ok"] is False
    assert "source" in sup.call("data_stats", {})["error"]


def test_read_episode_metrics_after_full_cycle(sup) -> None:
    empty = sup.call("read_episode_metrics")
    assert empty["ok"] and empty["episodes"] == [] and empty["best"] is None
    _run_one_cycle(sup)
    out = sup.call("read_episode_metrics", {"n": 5})
    assert out["ok"] is True
    assert len(out["episodes"]) == 1
    ep = out["episodes"][0]
    assert set(ep) == {"id", "cycle", "confirm_ppl", "job_status"}
    assert ep["job_status"] == "succeeded"
    assert out["best"]["id"] == ep["id"]
    assert "n" in sup.call("read_episode_metrics", {"n": "lots"})["error"]


def test_sandbox_usage_shape(sup) -> None:
    before = sup.call("sandbox_usage")
    assert before["ok"] is True
    assert set(before) >= {"ok", "files", "bytes", "cap_bytes", "free_bytes"}
    assert before["cap_bytes"] == sup.cfg.sandbox_max_bytes
    sup.sandbox.write_file("blob.txt", "x" * 1000)
    after = sup.call("sandbox_usage")
    assert after["files"] == before["files"] + 1
    assert after["bytes"] == before["bytes"] + 1000


def test_n_arguments_are_clamped(sup) -> None:
    assert sup.call("enter_research")["ok"]
    for name in ("list_packs", "list_jobs", "read_episode_metrics"):
        out = sup.call(name, {"n": 10_000})
        assert out["ok"] is True, (name, out)
    out = sup.call("read_trace", {"n": 10_000})
    assert out["ok"] is True
    out = sup.call("grep_files", {"pattern": "x", "max_hits": 10_000})
    assert out["ok"] is True
