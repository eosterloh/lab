"""Sandbox exec timeouts, run_python, and the LAB_EXEC_MAX_TIMEOUT_S cap."""

from __future__ import annotations

import sys

import pytest

from lab.sandbox import Sandbox

SLEEP = [sys.executable, "-c", "import time; time.sleep(5)"]


def test_exec_timeout_returns_timed_out_dict(sup) -> None:
    out = sup.sandbox.exec(SLEEP, timeout_s=0.5)
    assert out["timed_out"] is True
    assert out["returncode"] == -1
    assert "timeout after" in out["stderr"]
    assert isinstance(out["stdout"], str)


def test_exec_default_timeout_is_cfg_value(sup) -> None:
    sup.sandbox.cfg.exec_timeout_s = 0.3
    out = sup.sandbox.exec(SLEEP)
    assert out["timed_out"] is True


def test_exec_timeout_is_capped_by_env(sup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_EXEC_MAX_TIMEOUT_S", "0.4")
    out = sup.sandbox.exec(SLEEP, timeout_s=60)
    assert out["timed_out"] is True
    assert "0.4" in out["stderr"]


def test_exec_still_rejects_forbidden_commands(sup) -> None:
    with pytest.raises(ValueError):
        sup.sandbox.exec(["rm", "-rf", "."])


def test_exec_passes_device_env_through(sup, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAB_TRAIN_DEVICE", "cpu")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SECRET_TOKEN", "leak")
    out = sup.sandbox.exec(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('LAB_TRAIN_DEVICE'), 'SECRET_TOKEN' in os.environ)",
        ]
    )
    assert out["returncode"] == 0
    assert out["stdout"].strip() == "cpu False"


def test_run_python_runs_script_and_can_import_lab(sup) -> None:
    sup.sandbox.write_file(
        "scripts/probe.py",
        "import sys, lab\nfrom lab.types import Phase\nprint(Phase.RESEARCH.value, sys.argv[1:])\n",
    )
    out = sup.sandbox.run_python("scripts/probe.py", ["a", "b"])
    assert out["returncode"] == 0, out["stderr"]
    assert out["stdout"].strip() == "research ['a', 'b']"
    assert out["path"] == "scripts/probe.py"
    assert not (sup.sandbox.root / "scripts" / "__pycache__").exists()


def test_run_python_missing_file_and_escape(sup) -> None:
    with pytest.raises(FileNotFoundError):
        sup.sandbox.run_python("nope.py")
    with pytest.raises(ValueError):
        sup.sandbox.run_python("../state.json")


def test_run_python_honours_timeout(sup) -> None:
    sup.sandbox.write_file("slow.py", "import time; time.sleep(5)\n")
    out = sup.sandbox.run_python("slow.py", timeout_s=0.5)
    assert out["timed_out"] is True
    assert out["path"] == "slow.py"


def test_sandbox_is_the_supervisor_sandbox(sup) -> None:
    assert isinstance(sup.sandbox, Sandbox)
