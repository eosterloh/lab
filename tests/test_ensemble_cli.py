"""``python -m lab run --policy ensemble`` end to end with scripted roles and with no models."""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from lab.cli import main
from lab.data_cache import seed_default_mix
from lab.policy import dummy_pack

ENV_VARS = ("LAB_THINKER_MODEL", "LAB_TOOLER_MODEL", "LAB_CODER_MODEL", "LAB_ENSEMBLES", "LAB_ENSEMBLE_ROUNDS")


@pytest.fixture(autouse=True)
def _offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LAB_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("LAB_ALLOW_NETWORK", "0")
    cache = tmp_path / "hf_cache"
    seed_default_mix(cache)
    monkeypatch.setenv("LAB_DATA_CACHE", str(cache))


def _thought(**kw) -> dict:
    base = {"thought": "", "hypotheses": [], "need": None, "pack": None, "done": False}
    base.update(kw)
    return base


def _scripted_roles(path: Path) -> Path:
    obs = {"cycle": 1, "subject_checkpoint": "subjects/tinytrain-8m"}
    thinker = []
    for steps, claim in ((12, "twelve steps"), (40, "forty steps")):
        pack = dummy_pack(obs, extra_config={"steps": steps})
        pack["hypothesis"] = f"ensemble pack {claim}"
        thinker.append(_thought(pack=pack, hypotheses=[{"claim": claim, "why": "w", "falsify": "f"}]))
    path.write_text(json.dumps({"thinker": thinker, "tooler": [], "coder": []}), encoding="utf-8")
    return path


def test_cli_ensemble_with_scripted_roles_runs_two_trials(tmp_path: Path, capsys) -> None:
    roles_path = _scripted_roles(tmp_path / "roles-script.json")
    run_dir = tmp_path / "ens"
    code = main(
        [
            "run",
            "--policy",
            "ensemble",
            "--cycles",
            "1",
            "--ensembles",
            "2",
            "--scripted-roles",
            str(roles_path),
            "--run-dir",
            str(run_dir),
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "roles: thinker=scripted" in out
    assert "fallback pack authored by harness" not in out

    episodes = [json.loads(l) for l in (run_dir / "episodes" / "index.jsonl").read_text().splitlines()]
    assert len(episodes) == 2
    assert len({e["hypothesis_id"] for e in episodes}) == 2
    assert [e["trial"] for e in episodes] == [1, 2]

    roles = json.loads((run_dir / "roles.json").read_text())
    assert roles["roles"]["thinker"]["backend"] == "scripted"
    hyps = list((run_dir / "hypotheses").glob("hyp-*.json"))
    assert len(hyps) >= 2
    sources = {json.loads(p.read_text())["source"] for p in hyps}
    assert sources == {"ensemble-1", "ensemble-2"}

    names = {json.loads(l)["name"] for l in (run_dir / "trace.jsonl").read_text().splitlines()}
    assert "ensemble 1" in names and "ensemble 2" in names
    assert "ensemble 1 round 1" in names
    assert "ensemble 1 thinker" in names
    summary = json.loads((run_dir / "ensembles" / "cycle-0001.json").read_text())
    assert len(summary["candidates"]) == 2 and summary["fallback"] is False

    assert main(["roles", "--run-dir", str(run_dir)]) == 0
    table = capsys.readouterr().out
    assert "| thinker | scripted |" in table


def test_cli_ensemble_without_models_falls_back_and_still_completes(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "ens-null"
    code = main(["run", "--policy", "ensemble", "--cycles", "1", "--run-dir", str(run_dir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "WARNING: unconfigured roles" in out
    assert "LAB_THINKER_MODEL" in out
    assert "fallback pack authored by harness, not the models" in out

    episodes = [json.loads(l) for l in (run_dir / "episodes" / "index.jsonl").read_text().splitlines()]
    assert len(episodes) == 1
    hyp = json.loads((run_dir / "hypotheses" / f"{episodes[0]['hypothesis_id']}.json").read_text())
    assert hyp["source"] == "harness-fallback"
    roles = json.loads((run_dir / "roles.json").read_text())
    assert all(r["backend"] == "null" for r in roles["roles"].values())
    summary = json.loads((run_dir / "ensembles" / "cycle-0001.json").read_text())
    assert summary["fallback"] is True
    assert all("LAB_THINKER_MODEL" in (r["error"] or "") for r in summary["results"])


def test_cli_roles_without_registry_reports_missing(tmp_path: Path, capsys) -> None:
    assert main(["roles", "--run-dir", str(tmp_path / "nowhere")]) == 1
    assert "no roles.json" in capsys.readouterr().out
