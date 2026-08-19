from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import shutil
import sys

from lab.config import LabConfig
from lab.pack import ArtifactPack


SUITE_FILE = Path(__file__).parent / "data" / "frozen_eval" / "story_v0.json"


def install_frozen_eval(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "story_v0.json"
    if not target.exists():
        shutil.copyfile(SUITE_FILE, target)
        target.chmod(0o444)
    return target


def load_suite(frozen_dir: Path, suite_id: str, version: int) -> dict[str, Any]:
    path = frozen_dir / f"{suite_id}_v{version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown eval suite {suite_id} v{version}")
    return json.loads(path.read_text(encoding="utf-8"))


def dummy_eval(pack: ArtifactPack, metrics: dict[str, Any] | None) -> dict[str, Any]:
    suite_ok = pack.eval_suite_id == "story" and pack.eval_suite_version == 0
    val_loss = float((metrics or {}).get("val_loss", 9.0))
    score = round(1.0 / max(val_loss, 1e-6), 4)
    confirm_salt = hashlib.sha256(pack.digest().encode()).hexdigest()
    tweak = (int(confirm_salt[:4], 16) % 21 - 10) / 1000.0
    confirm = round(max(0.01, score + tweak), 4)
    return {
        "suite_id": pack.eval_suite_id,
        "suite_version": pack.eval_suite_version,
        "suite_known": suite_ok,
        "score": score,
        "confirm_score": confirm,
        "metric": "dummy_score",
        "higher_is_better": True,
        "backend": "dummy",
        "val_loss": val_loss,
    }


def infer_samples(cfg: LabConfig, prompts: list[str], model_dir: Path) -> list[str] | None:
    if cfg.infer_root is None or not model_dir.is_dir():
        return None
    root = str(cfg.infer_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from engine.agent_api import load_engine  # type: ignore
    except Exception:
        return None
    try:
        eng = load_engine(model_dir, device="cpu")
        return [eng.generate(p, max_new_tokens=24) for p in prompts]
    except Exception:
        return None


def run_eval(
    cfg: LabConfig,
    pack: ArtifactPack | None,
    metrics: dict[str, Any] | None,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    if pack is None:
        return {
            "suite_id": "story",
            "suite_version": 0,
            "score": None,
            "confirm_score": None,
            "backend": "none",
            "note": "no pack yet — baseline subject only",
        }
    result = dummy_eval(pack, metrics)
    suite = load_suite(cfg.frozen_eval_dir, pack.eval_suite_id, pack.eval_suite_version)
    result["prompts"] = suite.get("prompts", [])
    if model_dir is not None:
        samples = infer_samples(cfg, result["prompts"], model_dir)
        if samples is not None:
            result["samples"] = samples
            result["backend"] = "infer+dummy"
    return result
