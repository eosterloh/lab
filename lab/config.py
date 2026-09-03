from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from lab.data_cache import default_cache_dir


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


@dataclass
class LabConfig:
    run_dir: Path
    research_max_tool_calls: int = 50
    research_max_seconds: float = 600.0
    sandbox_max_bytes: int = 50 * 1024 * 1024
    train_max_hours: float = 3.5
    dummy_job_seconds: float = 0.0
    fetch_max_bytes: int = 2 * 1024 * 1024
    exec_timeout_s: float = 30.0
    search_timeout_s: float = 15.0
    fetch_timeout_s: float = 15.0
    subject_checkpoint: str = "subjects/tinytrain-8m"
    tinytrain_root: Path | None = None
    infer_root: Path | None = None
    allow_network: bool = True
    gpu_lock_path: Path | None = None
    data_cache_dir: Path | None = None

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir).expanduser().resolve()
        if self.tinytrain_root is not None:
            self.tinytrain_root = Path(self.tinytrain_root).expanduser().resolve()
        if self.infer_root is not None:
            self.infer_root = Path(self.infer_root).expanduser().resolve()
        if self.gpu_lock_path is None:
            self.gpu_lock_path = self.run_dir / "gpu.lock"
        else:
            self.gpu_lock_path = Path(self.gpu_lock_path).expanduser().resolve()
        if self.data_cache_dir is None:
            self.data_cache_dir = default_cache_dir()
        else:
            self.data_cache_dir = Path(self.data_cache_dir).expanduser().resolve()

    @classmethod
    def from_env(cls, run_dir: Path) -> LabConfig:
        return cls(
            run_dir=run_dir,
            tinytrain_root=_env_path("LAB_TINYTRAIN_ROOT"),
            infer_root=_env_path("LAB_INFER_ROOT"),
            gpu_lock_path=_env_path("LAB_GPU_LOCK") or Path("/tmp/lab-gpu.lock"),
            subject_checkpoint=os.environ.get(
                "LAB_SUBJECT_CHECKPOINT", "subjects/tinytrain-8m"
            ),
            allow_network=os.environ.get("LAB_ALLOW_NETWORK", "1") != "0",
            data_cache_dir=_env_path("LAB_DATA_CACHE") or default_cache_dir(),
        )

    @property
    def sandbox_dir(self) -> Path:
        return self.run_dir / "sandbox"

    @property
    def packs_dir(self) -> Path:
        return self.run_dir / "packs"

    @property
    def jobs_dir(self) -> Path:
        return self.run_dir / "jobs"

    @property
    def frozen_eval_dir(self) -> Path:
        return self.run_dir / "frozen_eval"

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    @property
    def notebook_path(self) -> Path:
        return self.run_dir / "notebook.jsonl"

    @property
    def beliefs_path(self) -> Path:
        return self.run_dir / "beliefs.md"

    @property
    def hypothesis_path(self) -> Path:
        return self.run_dir / "hypothesis.json"

    @property
    def episodes_dir(self) -> Path:
        return self.run_dir / "episodes"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

