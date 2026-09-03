from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import math
import shutil

from lab.config import LabConfig
from lab.pack import ArtifactPack
from lab.scorer import Scorer, load_scorer, mean_nll, ppl

SUITE_DIR = Path(__file__).parent / "data" / "frozen_eval"
DEFAULT_SUITE_ID = "core"
DEFAULT_SUITE_VERSION = 1


def install_frozen_eval(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for src in SUITE_DIR.rglob("*"):
        if not src.is_file():
            continue
        target = dest / src.relative_to(SUITE_DIR)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
        target.chmod(0o444)


def load_suite(frozen_dir: Path, suite_id: str, version: int) -> dict[str, Any]:
    path = frozen_dir / f"{suite_id}_v{version}.json"
    if not path.is_file():
        raise FileNotFoundError(f"unknown eval suite {suite_id} v{version}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _round(x: float | None, nd: int = 4) -> float | None:
    if x is None:
        return None
    return round(float(x), nd)


def _split_nll(scorer: Scorer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    parts = [scorer.sequence_nll(str(r["text"])) for r in rows if r.get("text")]
    loss = mean_nll(parts)
    tokens = sum(n for _, n in parts)
    return {
        "loss": _round(loss),
        "ppl": _round(ppl(loss)),
        "n": len(parts),
        "tokens": tokens,
    }


def _mc_acc(scorer: Scorer, rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    n = 0
    for row in rows:
        endings = list(row.get("endings") or [])
        if not endings:
            continue
        ctx = str(row.get("context") or "")
        scores: list[float] = []
        for end in endings:
            s, k = scorer.continuation_nll(ctx, " " + str(end).lstrip())
            scores.append(s / k if k else float("inf"))
        pred = min(range(len(scores)), key=lambda i: scores[i])
        if pred == int(row["label"]):
            correct += 1
        n += 1
    acc = (correct / n) if n else None
    return {"acc": _round(acc), "n": n, "correct": correct}


def _trainer_loss(metrics: dict[str, Any] | None) -> dict[str, Any]:
    m = metrics or {}
    val_loss = m.get("val_loss")
    train_loss = m.get("train_loss")
    val_ppl = m.get("val_ppl")
    if val_loss is not None and val_ppl is None:
        val_ppl = math.exp(float(val_loss))
    return {
        "train_loss": _round(float(train_loss) if train_loss is not None else None),
        "val_loss": _round(float(val_loss) if val_loss is not None else None),
        "val_ppl": _round(float(val_ppl) if val_ppl is not None else None),
    }


def _checkpoint(cfg: LabConfig, pack: ArtifactPack, metrics: dict[str, Any] | None) -> Path | None:
    m = metrics or {}
    for cand in (
        m.get("checkpoint"),
        m.get("run_checkpoint"),
        pack.config.get("checkpoint"),
        pack.parent_checkpoint,
        cfg.subject_checkpoint,
    ):
        if not cand:
            continue
        path = Path(str(cand)).expanduser()
        if path.is_file() or path.is_dir():
            return path
    return None


def run_eval(
    cfg: LabConfig,
    pack: ArtifactPack | None,
    metrics: dict[str, Any] | None,
    scorer: Scorer | None = None,
) -> dict[str, Any]:
    trainer = _trainer_loss(metrics)
    if pack is None:
        return {
            "suite_id": DEFAULT_SUITE_ID,
            "suite_version": DEFAULT_SUITE_VERSION,
            "suite_known": True,
            "primary": "confirm_ppl",
            "higher_is_better": False,
            "score": None,
            "confirm_score": None,
            "confirm_ppl": None,
            "backend": "none",
            "loss": trainer,
            "benchmarks": {},
            "note": "no pack yet — trainer metrics only",
        }

    try:
        suite = load_suite(cfg.frozen_eval_dir, pack.eval_suite_id, pack.eval_suite_version)
        suite_known = True
    except FileNotFoundError:
        return {
            "suite_id": pack.eval_suite_id,
            "suite_version": pack.eval_suite_version,
            "suite_known": False,
            "primary": "confirm_ppl",
            "higher_is_better": False,
            "score": None,
            "confirm_score": None,
            "confirm_ppl": None,
            "backend": "unknown_suite",
            "loss": trainer,
            "benchmarks": {},
        }

    ckpt = _checkpoint(cfg, pack, metrics)
    if scorer is None:
        scorer = load_scorer(ckpt, cfg.infer_root)

    holdout: dict[str, Any] = {}
    benches: dict[str, Any] = {}
    backend = "trainer"

    if scorer is not None:
        backend = "model"
        frozen = cfg.frozen_eval_dir
        for split, filename in (suite.get("lm") or {}).items():
            holdout[split] = _split_nll(scorer, load_jsonl(frozen / filename))
        for name, spec in (suite.get("mc") or {}).items():
            rec = _mc_acc(scorer, load_jsonl(frozen / spec["file"]))
            rec["chance"] = spec.get("chance")
            benches[name] = rec

    confirm_ppl = (holdout.get("confirm") or {}).get("ppl")
    confirm_source = "frozen_holdout"
    if confirm_ppl is None and trainer["val_ppl"] is not None:
        # Dummy / no-checkpoint path: proxy from trainer val, not a frozen holdout.
        confirm_ppl = trainer["val_ppl"]
        confirm_source = "trainer_val_proxy"
        backend = "trainer_proxy"

    return {
        "suite_id": suite.get("id", pack.eval_suite_id),
        "suite_version": suite.get("version", pack.eval_suite_version),
        "suite_known": suite_known,
        "primary": suite.get("primary", "confirm_ppl"),
        "higher_is_better": bool(suite.get("higher_is_better", False)),
        "score": confirm_ppl,
        "confirm_score": confirm_ppl,
        "confirm_ppl": confirm_ppl,
        "confirm_source": confirm_source,
        "backend": backend,
        "loss": {**trainer, "holdout": holdout},
        "benchmarks": benches,
        "checkpoint": str(ckpt) if ckpt else None,
    }
