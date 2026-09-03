"""Isolated subprocess for the lab trainer.

cwd = job dir under runs/<id>/jobs/<job>/. Stripped env, wall-clock timeout.
All artifacts stay in that job dir (pack, log, metrics, checkpoint).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import subprocess

from lab.train.loop import BUILTIN_CORPUS

__all__ = ["BUILTIN_CORPUS", "repo_root", "run_train_job", "sandbox_env"]


def repo_root() -> Path:
    # lab/train/__init__.py → parents[2] is the import root (repo or site-packages)
    return Path(__file__).resolve().parents[2]


def sandbox_env(job_dir: Path) -> dict[str, str]:
    path = os.environ.get("PATH", "/usr/bin:/bin")
    root = str(repo_root())
    existing = os.environ.get("PYTHONPATH", "")
    pythonpath = root if not existing else f"{root}{os.pathsep}{existing}"
    tmp = job_dir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": path,
        "HOME": str(job_dir),
        "PYTHONPATH": pythonpath,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "LAB_JOB_DIR": str(job_dir),
        "TMPDIR": str(tmp),
        "TMP": str(tmp),
        "XDG_CACHE_HOME": str(job_dir / ".cache"),
        "LAB_TRAIN_DEVICE": os.environ.get("LAB_TRAIN_DEVICE", "cpu"),
        "CUDA_VISIBLE_DEVICES": "",
    }
    if env["LAB_TRAIN_DEVICE"].strip().lower() in {"cuda", "gpu"}:
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "CUDA_HOME",
            "CUDA_PATH",
            "LD_LIBRARY_PATH",
            "DYLD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES",
        ):
            val = os.environ.get(key)
            if val:
                env[key] = val
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        env["VIRTUAL_ENV"] = venv
    return env


def run_train_job(job_dir: Path, timeout_s: float) -> subprocess.CompletedProcess[str]:
    job_dir.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "lab.train", "--job-dir", str(job_dir)]
    log = job_dir / "train.log"
    with log.open("w", encoding="utf-8") as fh:
        return subprocess.run(
            argv,
            cwd=job_dir,
            env=sandbox_env(job_dir),
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
