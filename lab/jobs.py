from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import math
import shutil
import subprocess
import sys
import time

from lab.config import LabConfig
from lab.data_cache import materialize_sources
from lab.pack import ArtifactPack
from lab.train import BUILTIN_CORPUS, run_train_job


def resolve_lab_parent(pack: ArtifactPack, cfg: LabConfig) -> Path | None:
    for raw in (pack.parent_checkpoint, pack.config.get("checkpoint"), pack.config.get("parent")):
        if not raw:
            continue
        path = Path(str(raw)).expanduser()
        if path.is_file():
            return path
    ckpts = sorted(p for p in cfg.checkpoints_dir.glob("job-*.pt") if p.is_file())
    return ckpts[-1] if ckpts else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    pack_hash: str
    backend: str
    status: str = "queued"
    metrics: dict[str, Any] = field(default_factory=dict)
    log_path: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    error: str | None = None
    pid: int | None = None
    command: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, job_id: str) -> Path:
        return self.root / job_id

    def save(self, job: Job) -> None:
        d = self._dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        (d / "job.json").write_text(json.dumps(job.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def load(self, job_id: str) -> Job:
        path = self._dir(job_id) / "job.json"
        return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def next_id(self) -> str:
        n = len(list(self.root.glob("job-*"))) + 1
        return f"job-{n:04d}"


def dummy_metrics(pack: ArtifactPack) -> dict[str, Any]:
    cfg = pack.config
    lr = float(cfg.get("lr", 2e-3))
    steps = int(cfg.get("steps", pack.budgets.max_steps or 10))
    val_loss = 1.25 - min(steps, 200) * 0.001
    if lr >= 1e-3:
        val_loss -= 0.02
    val_loss = max(0.4, round(val_loss, 4))
    train_loss = round(val_loss + 0.04, 4)
    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_ppl": round(math.exp(val_loss), 4),
        "steps": steps,
        "lr": lr,
        "tokens_seen": steps * 1024,
        "backend": "dummy",
    }


class JobManager:
    def __init__(self, cfg: LabConfig) -> None:
        self.cfg = cfg
        self.store = JobStore(cfg.jobs_dir)
        self._procs: dict[str, subprocess.Popen[str]] = {}

    def submit(self, pack: ArtifactPack, pack_hash: str) -> Job:
        job = Job(
            id=self.store.next_id(),
            pack_hash=pack_hash,
            backend=pack.trainer,
            status="running",
            started_at=_now(),
        )
        job_dir = self.store._dir(job.id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "pack.json").write_text(
            json.dumps(pack.canonical(), indent=2, sort_keys=True), encoding="utf-8"
        )
        log_path = job_dir / "train.log"
        job.log_path = str(log_path)

        if pack.trainer == "dummy":
            if self.cfg.dummy_job_seconds:
                time.sleep(self.cfg.dummy_job_seconds)
            job.metrics = dummy_metrics(pack)
            log_path.write_text(json.dumps(job.metrics, indent=2) + "\n", encoding="utf-8")
            (job_dir / "metrics.json").write_text(
                json.dumps(job.metrics, indent=2, sort_keys=True), encoding="utf-8"
            )
            job.status = "succeeded"
            job.ended_at = _now()
            self.store.save(job)
            return job

        if pack.trainer == "lab":
            return self._submit_lab(job, pack, job_dir, log_path)

        if pack.trainer == "tinytrain":
            root = self.cfg.tinytrain_root
            if root is None or not root.is_dir():
                job.status = "failed"
                job.error = "LAB_TINYTRAIN_ROOT is not set or missing"
                job.ended_at = _now()
                self.store.save(job)
                return job
            argv = list(pack.config["command"])
            job.command = argv
            log_f = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                argv,
                cwd=root,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            job.pid = proc.pid
            self._procs[job.id] = proc
            self.store.save(job)
            return job

        job.status = "failed"
        job.error = f"unknown trainer {pack.trainer}"
        job.ended_at = _now()
        self.store.save(job)
        return job

    def _submit_lab(self, job: Job, pack: ArtifactPack, job_dir: Path, log_path: Path) -> Job:
        sources = list((pack.data_manifest or {}).get("sources") or ["builtin:tiny"])
        try:
            materialize_sources(
                sources,
                self.cfg.data_cache_dir,
                job_dir,
                allow_network=self.cfg.allow_network,
                frozen_eval_dir=self.cfg.frozen_eval_dir,
            )
        except Exception as e:
            job.status = "failed"
            job.error = f"materialize data: {e}"
            job.ended_at = _now()
            log_path.write_text(job.error + "\n", encoding="utf-8")
            self.store.save(job)
            return job
        if any(s in {"builtin:tiny", "dummy://tinystories"} for s in sources):
            shutil.copyfile(BUILTIN_CORPUS, job_dir / "corpus.txt")
        parent = resolve_lab_parent(pack, self.cfg)
        if parent is not None:
            shutil.copy2(parent, job_dir / "parent.pt")
        timeout_s = max(30.0, float(pack.budgets.max_hours) * 3600.0)
        job.command = [sys.executable, "-m", "lab.train", "--job-dir", str(job_dir)]
        try:
            proc = run_train_job(job_dir, timeout_s=timeout_s)
        except subprocess.TimeoutExpired:
            job.status = "failed"
            job.error = f"timeout after {timeout_s:.0f}s"
            job.ended_at = _now()
            self.store.save(job)
            return job
        job.ended_at = _now()
        metrics_path = job_dir / "metrics.json"
        if proc.returncode == 0 and metrics_path.is_file():
            job.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            job.status = "succeeded"
            self._publish_checkpoint(job, job_dir)
        else:
            job.status = "failed"
            tail = ""
            if log_path.is_file():
                tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
            job.error = f"exit {proc.returncode}" + (f": {tail.strip()}" if tail.strip() else "")
        self.store.save(job)
        return job

    def _publish_checkpoint(self, job: Job, job_dir: Path) -> None:
        src = job_dir / "checkpoint.pt"
        if not src.is_file():
            return
        dest_dir = self.cfg.checkpoints_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{job.id}.pt"
        shutil.copy2(src, dest)
        shutil.copy2(src, dest_dir / "latest.pt")
        if job.metrics is not None:
            job.metrics["checkpoint"] = str(src)
            job.metrics["run_checkpoint"] = str(dest)
            (job_dir / "metrics.json").write_text(
                json.dumps(job.metrics, indent=2, sort_keys=True), encoding="utf-8"
            )

    def poll(self, job: Job) -> Job:
        if job.status in {"succeeded", "failed", "cancelled"}:
            return job
        proc = self._procs.get(job.id)
        if proc is None and job.pid:
            job.status = "failed"
            job.error = "lost child process"
            job.ended_at = _now()
            self.store.save(job)
            return job
        if proc is None:
            return job
        code = proc.poll()
        if code is None:
            return job
        job.ended_at = _now()
        metrics_path = self.store._dir(job.id) / "metrics.json"
        if code == 0:
            job.status = "succeeded"
            if metrics_path.is_file():
                job.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            job.status = "failed"
            job.error = f"exit {code}"
        self._procs.pop(job.id, None)
        self.store.save(job)
        return job

    def cancel(self, job: Job) -> Job:
        proc = self._procs.get(job.id)
        if proc and proc.poll() is None:
            proc.terminate()
        job.status = "cancelled"
        job.ended_at = _now()
        self.store.save(job)
        return job
