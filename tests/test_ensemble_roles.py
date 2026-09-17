"""Tests for the ensemble role backends, Roles bundle, and RoleRegistry."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import threading
import time

import pytest

from lab.config import LabConfig
from lab.ensemble.registry import RoleRegistry
from lab.ensemble.roles import (
    ROLES,
    InferBackend,
    NullBackend,
    RoleNotConfigured,
    RoleSpec,
    Roles,
    ScriptedBackend,
    SENTINEL_REPLY,
)

ENV_VARS = (
    "LAB_THINKER_MODEL",
    "LAB_TOOLER_MODEL",
    "LAB_CODER_MODEL",
    "LAB_MODELS_DIR",
    "LAB_ENSEMBLES",
    "LAB_ENSEMBLE_ROUNDS",
    "LAB_INFER_ROOT",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ----------------------------------------------------------------- backends


def test_null_backend_names_role_and_env_var() -> None:
    b = NullBackend("thinker")
    with pytest.raises(RoleNotConfigured) as ei:
        b.generate("hi")
    msg = str(ei.value)
    assert "'thinker'" in msg
    assert "LAB_THINKER_MODEL" in msg
    assert b.loaded is False


def test_scripted_backend_fifo_sentinel_and_prompts() -> None:
    b = ScriptedBackend(["a", "b"])
    assert b.generate("p1") == "a"
    assert b.generate("p2") == "b"
    assert b.generate("p3") == SENTINEL_REPLY
    assert json.loads(SENTINEL_REPLY) == {"done": True}
    assert b.prompts == ["p1", "p2", "p3"]
    assert b.remaining == 0


def test_scripted_backend_callable() -> None:
    b = ScriptedBackend(lambda p: p.upper())
    assert b.generate("abc") == "ABC"
    assert b.prompts == ["abc"]


# -------------------------------------------------------------------- Roles


def test_roles_from_env_all_null(tmp_path: Path) -> None:
    cfg = LabConfig(run_dir=tmp_path)
    roles = Roles.from_env(cfg)
    for role in ROLES:
        assert isinstance(roles.backend(role), NullBackend)
    desc = roles.describe()
    assert set(desc) == set(ROLES)
    for role in ROLES:
        assert desc[role]["backend"] == "null"
        assert desc[role]["loaded"] is False
        assert desc[role]["model_dir"] is None
    with pytest.raises(RoleNotConfigured):
        roles.generate("coder", "write code")


def test_roles_from_env_thinker_infer_lazy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "models" / "thinker-1b"
    model.mkdir(parents=True)
    monkeypatch.setenv("LAB_THINKER_MODEL", str(model))
    cfg = LabConfig.from_env(tmp_path / "run")
    roles = Roles.from_env(cfg)

    thinker = roles.backend("thinker")
    assert isinstance(thinker, InferBackend)
    assert thinker.loaded is False
    spec = roles.spec("thinker")
    assert spec.backend == "infer"
    assert spec.model_dir == str(model)
    assert roles.describe()["thinker"]["loaded"] is False
    # other roles untouched
    assert isinstance(roles.backend("tooler"), NullBackend)
    assert isinstance(roles.backend("coder"), NullBackend)


def test_roles_generate_uses_spec_max_new_tokens_and_kwargs_override() -> None:
    class Capture(ScriptedBackend):
        def __init__(self) -> None:
            super().__init__([])
            self.calls: list[dict[str, Any]] = []

        def generate(self, prompt: str, *, max_new_tokens: int = 512, **kwargs: Any) -> str:
            self.calls.append({"max_new_tokens": max_new_tokens, **kwargs})
            return super().generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    cap = Capture()
    specs = {
        "thinker": RoleSpec(role="thinker", backend="scripted", max_new_tokens=77),
        "tooler": RoleSpec(role="tooler", backend="scripted"),
        "coder": RoleSpec(role="coder", backend="scripted"),
    }
    backends = {"thinker": cap, "tooler": ScriptedBackend([]), "coder": ScriptedBackend([])}
    roles = Roles(backends, specs)

    roles.generate("thinker", "x")
    roles.generate("thinker", "y", max_new_tokens=9, temperature=0.1)
    assert cap.calls[0] == {"max_new_tokens": 77}
    assert cap.calls[1] == {"max_new_tokens": 9, "temperature": 0.1}
    with pytest.raises(KeyError):
        roles.generate("critic", "z")


def test_roles_scripted_and_from_specs() -> None:
    roles = Roles.scripted(["t1"], lambda p: "tool:" + p, ["c1"])
    assert roles.generate("thinker", "a") == "t1"
    assert roles.generate("tooler", "b") == "tool:b"
    assert roles.generate("coder", "c") == "c1"
    assert roles.generate("thinker", "d") == SENTINEL_REPLY

    specs = {
        "thinker": RoleSpec(role="thinker", backend="infer", model_dir="/m/think"),
        "tooler": RoleSpec(role="tooler", backend="null"),
        "coder": RoleSpec(role="coder", backend="scripted"),
    }
    built = Roles.from_specs(specs, infer_root="/tmp/infer")
    assert isinstance(built.backend("thinker"), InferBackend)
    assert built.backend("thinker").loaded is False
    assert isinstance(built.backend("tooler"), NullBackend)
    assert isinstance(built.backend("coder"), ScriptedBackend)
    assert built.describe()["thinker"]["model_dir"] == "/m/think"

    with pytest.raises(ValueError):
        Roles.from_specs({"thinker": RoleSpec(role="thinker", backend="null")})


def test_infer_backend_lazy_load_and_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    state = {"active": 0, "max_active": 0, "loads": 0}
    guard = threading.Lock()
    seen_kwargs: list[dict[str, Any]] = []

    class FakeEngine:
        def generate(self, prompt: str, max_new_tokens: int, **kw: Any) -> str:
            with guard:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            seen_kwargs.append({"max_new_tokens": max_new_tokens, **kw})
            time.sleep(0.05)
            with guard:
                state["active"] -= 1
            return "ok:" + prompt

    def fake_load(model_dir: Any, infer_root: Any, device: Any = None) -> FakeEngine:
        state["loads"] += 1
        return FakeEngine()

    import lab.llm_policy as llm_policy

    monkeypatch.setattr(llm_policy, "load_infer_engine", fake_load)

    b = InferBackend("/models/x", "/tmp/infer")
    assert b.loaded is False
    assert state["loads"] == 0

    results: list[str] = []
    threads = [
        threading.Thread(target=lambda i=i: results.append(b.generate(f"p{i}", max_new_tokens=8)))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert b.loaded is True
    assert state["loads"] == 1
    assert state["max_active"] == 1
    assert sorted(results) == [f"ok:p{i}" for i in range(4)]
    assert len(seen_kwargs) == 4
    for kw in seen_kwargs:
        assert kw["enable_thinking"] is False
        assert kw["max_new_tokens"] == 8

    b.generate("think", enable_thinking=True)
    assert seen_kwargs[-1]["enable_thinking"] is True


# ----------------------------------------------------------------- registry


def test_registry_init_writes_null_specs(tmp_path: Path) -> None:
    path = tmp_path / "run" / "roles.json"
    reg = RoleRegistry(path)
    assert path.exists()
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert set(raw["roles"]) == set(ROLES)
    assert raw["candidates"] == [] and raw["history"] == []
    for role in ROLES:
        assert reg.current(role) == RoleSpec(role=role, backend="null")
    assert not (tmp_path / "run" / "roles.json.tmp").exists()


def test_registry_propose_accept_reject_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "roles.json"
    reg = RoleRegistry(path)

    cid = reg.propose("tooler", "/models/tooler-v2", note="better JSON")
    assert cid == "cand-0001"
    assert reg.candidates("proposed")[0]["id"] == cid
    assert reg.current("tooler").version == 1

    spec = reg.accept(cid)
    assert spec.backend == "infer"
    assert spec.model_dir == "/models/tooler-v2"
    assert spec.version == 2
    assert reg.current("tooler") == spec
    assert reg.candidates("accepted")[0]["id"] == cid
    hist = reg.to_dict()["history"]
    assert len(hist) == 1
    assert hist[0]["role"] == "tooler"
    assert hist[0]["from_version"] == 1 and hist[0]["to_version"] == 2
    with pytest.raises(ValueError):
        reg.accept(cid)

    cid2 = reg.propose("coder", "/models/coder-bad")
    assert cid2 == "cand-0002"
    reg.reject(cid2, note="hallucinates imports")
    assert reg.candidates("rejected")[0]["id"] == cid2
    assert "hallucinates" in reg.candidates("rejected")[0]["note"]
    assert reg.current("coder").version == 1
    assert len(reg.candidates()) == 2

    with pytest.raises(ValueError):
        reg.propose("critic", "/models/x")
    with pytest.raises(KeyError):
        reg.accept("cand-9999")

    # reload from disk
    again = RoleRegistry(path)
    assert again.specs() == reg.specs()
    assert again.to_dict() == reg.to_dict()
    assert again.current("tooler").version == 2

    md = reg.markdown()
    for role in ROLES:
        assert role in md
    assert "/models/tooler-v2" in md
    assert "| 2 |" in md


def test_registry_initial_specs_and_roles_from_specs(tmp_path: Path) -> None:
    initial = {
        "thinker": RoleSpec(role="thinker", backend="infer", model_dir="/m/t", version=3),
    }
    reg = RoleRegistry(tmp_path / "roles.json", initial=initial)
    assert reg.current("thinker").version == 3
    assert reg.current("tooler").backend == "null"
    roles = Roles.from_specs(reg.specs(), infer_root="/tmp/infer")
    assert isinstance(roles.backend("thinker"), InferBackend)
    assert roles.describe()["thinker"]["version"] == 3


# ------------------------------------------------------------------- config


def test_config_defaults(tmp_path: Path) -> None:
    cfg = LabConfig.from_env(tmp_path)
    assert cfg.thinker_model is None
    assert cfg.tooler_model is None
    assert cfg.coder_model is None
    assert cfg.models_dir == Path.home() / "models"
    assert cfg.ensembles_per_cycle == 3
    assert cfg.ensemble_max_rounds == 12
    assert cfg.roles_path == cfg.run_dir / "roles.json"


def test_config_reads_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_THINKER_MODEL", "~/models/think")
    monkeypatch.setenv("LAB_TOOLER_MODEL", str(tmp_path / "tool"))
    monkeypatch.setenv("LAB_CODER_MODEL", str(tmp_path / "code"))
    monkeypatch.setenv("LAB_MODELS_DIR", "~/zoo")
    monkeypatch.setenv("LAB_ENSEMBLES", "5")
    monkeypatch.setenv("LAB_ENSEMBLE_ROUNDS", "20")
    cfg = LabConfig.from_env(tmp_path)
    assert cfg.thinker_model == Path.home() / "models" / "think"
    assert cfg.tooler_model == tmp_path / "tool"
    assert cfg.coder_model == tmp_path / "code"
    assert cfg.models_dir == Path.home() / "zoo"
    assert cfg.ensembles_per_cycle == 5
    assert cfg.ensemble_max_rounds == 20
