from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

# 3.5 days. The old 3.5 hour figure was this number with the unit left as hours.
MAX_TRAIN_HOURS = 84.0
# Suggested clock for a short continue of the byte-level subject. Not a cap.
DEFAULT_CONTINUE_HOURS = 20 / 60
# Kept so older imports still resolve. No longer forced onto every lab pack.
FIXED_TRAIN_HOURS = DEFAULT_CONTINUE_HOURS
TRAINERS = frozenset({"dummy", "tinytrain", "lab"})
STAGES = ("pretrain", "continue", "midtrain", "posttrain")
# pretrain and continue may run the full 3.5 days. Mid and post are shorter on
# purpose: a data-mix shift or a role fine-tune should not occupy the GPU for a pretrain.
STAGE_MAX_HOURS: dict[str, float] = {
    "pretrain": 84.0,
    "continue": 84.0,
    "midtrain": 24.0,
    "posttrain": 8.0,
}


def normalize_stage(value: Any) -> str:
    stage = str(value or "continue").strip().lower()
    if stage not in STAGE_MAX_HOURS:
        raise ValueError(f"stage must be one of {list(STAGES)}")
    return stage


def parent_plan(stage: str, *, parent_exists: bool) -> dict[str, bool]:
    """How a lab job treats parent.pt.

    pretrain ignores it. continue loads weights and Adam when the file is there,
    and starts from scratch when it is not (the first run of a subject).
    midtrain and posttrain load weights, reset Adam, and refuse to start
    without a checkpoint.
    """
    stage = normalize_stage(stage)
    if stage == "pretrain":
        return {"load_weights": False, "load_optim": False, "required": False}
    if stage == "continue":
        return {
            "load_weights": parent_exists,
            "load_optim": parent_exists,
            "required": False,
        }
    return {"load_weights": True, "load_optim": False, "required": True}


@dataclass
class Budgets:
    max_hours: float = 0.01
    max_tokens: int | None = None
    max_steps: int | None = 10

    def validate(self) -> None:
        if self.max_hours <= 0:
            raise ValueError("budgets.max_hours must be > 0")
        if self.max_hours > MAX_TRAIN_HOURS:
            raise ValueError(f"budgets.max_hours cannot exceed {MAX_TRAIN_HOURS}")


@dataclass
class ArtifactPack:
    hypothesis: str
    trainer: str
    config: dict[str, Any]
    data_manifest: dict[str, Any]
    eval_suite_id: str
    eval_suite_version: int
    parent_checkpoint: str
    budgets: Budgets = field(default_factory=Budgets)
    stage: str = "continue"

    def validate(self) -> None:
        if not self.hypothesis.strip():
            raise ValueError("hypothesis is required")
        if self.trainer not in TRAINERS:
            raise ValueError(f"trainer must be one of {sorted(TRAINERS)}")
        if not isinstance(self.config, dict) or not self.config:
            raise ValueError("config must be a non-empty object")
        if not isinstance(self.data_manifest, dict):
            raise ValueError("data_manifest must be an object")
        if not self.eval_suite_id.strip():
            raise ValueError("eval_suite_id is required")
        if not isinstance(self.eval_suite_version, int) or self.eval_suite_version < 0:
            raise ValueError("eval_suite_version must be a non-negative int")
        if not self.parent_checkpoint.strip():
            raise ValueError("parent_checkpoint is required")
        self.budgets.validate()
        self.stage = normalize_stage(self.stage)
        cap = STAGE_MAX_HOURS[self.stage]
        if self.budgets.max_hours > cap:
            raise ValueError(f"{self.stage} max_hours cannot exceed {cap}")
        if self.trainer == "tinytrain" and not self.config.get("command"):
            raise ValueError("tinytrain packs require config.command (argv list)")
        if self.trainer == "lab":
            hidden = int(self.config.get("hidden", 32))
            heads = int(self.config.get("heads", 1))
            if heads < 1 or hidden % heads != 0:
                raise ValueError("lab config: hidden must be divisible by heads")

    def canonical(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.strip(),
            "trainer": self.trainer,
            "config": self.config,
            "data_manifest": self.data_manifest,
            "eval_suite_id": self.eval_suite_id,
            "eval_suite_version": self.eval_suite_version,
            "parent_checkpoint": self.parent_checkpoint,
            "budgets": asdict(self.budgets),
            "stage": self.stage,
        }

    def digest(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return sha256(blob.encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactPack:
        raw = dict(data)
        budgets = raw.pop("budgets", {}) or {}
        extra = set(raw) - {
            "hypothesis",
            "trainer",
            "config",
            "data_manifest",
            "eval_suite_id",
            "eval_suite_version",
            "parent_checkpoint",
            "stage",
        }
        if extra:
            raise ValueError(f"unknown pack fields: {sorted(extra)}")
        pack = cls(
            hypothesis=str(raw.get("hypothesis", "")),
            trainer=str(raw.get("trainer", "")),
            config=raw.get("config") or {},
            data_manifest=raw.get("data_manifest") or {},
            eval_suite_id=str(raw.get("eval_suite_id", "")),
            eval_suite_version=int(raw.get("eval_suite_version", -1)),
            parent_checkpoint=str(raw.get("parent_checkpoint", "")),
            stage=str(raw.get("stage") or "continue"),
            budgets=Budgets(
                max_hours=float(budgets.get("max_hours", 0.01)),
                max_tokens=budgets.get("max_tokens"),
                max_steps=budgets.get("max_steps", 10),
            ),
        )
        pack.validate()
        return pack


class PackStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, pack: ArtifactPack) -> str:
        digest = pack.digest()
        dest = self.root / digest
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "pack.json"
        path.write_text(json.dumps(pack.canonical(), indent=2, sort_keys=True), encoding="utf-8")
        return digest

    def load(self, digest: str) -> ArtifactPack:
        path = self.root / digest / "pack.json"
        if not path.is_file():
            raise FileNotFoundError(digest)
        return ArtifactPack.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def exists(self, digest: str) -> bool:
        return (self.root / digest / "pack.json").is_file()
