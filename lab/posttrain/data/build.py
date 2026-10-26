"""Assemble SFT, DPO, and held-out JSONL for one role."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import random

from lab.posttrain.data.convert import convert_row
from lab.posttrain.data.examples import Example
from lab.posttrain.data.sources import SYNTHETIC, Source, sources_for
from lab.posttrain.data.synthetic import generate
from lab.posttrain.data.validate import validate_example

FORMAT = {
    "sft": (
        "TRL messages. The user content is the harness prompt from thinker_prompt, "
        "tooler_prompt, or coder_prompt. The assistant content is the raw completion "
        "that role's parser accepts. Extra keys role, source, and kind are metadata."
    ),
    "dpo": (
        "TRL prompt, chosen, and rejected strings. prompt is the harness prompt "
        "for the chosen turn (the last user message; the transcript inside it "
        "already has earlier turns). chosen is that turn's assistant completion. "
        "rejected is a wrong call that fails the same parser and arg checks."
    ),
    "eval": (
        "Same shape as the SFT file. Held out by sha256 of the user turns, so no "
        "prompt hash is in both train and eval. DPO rows for a held-out prompt are omitted."
    ),
    "inference": (
        "The ensemble calls backend.generate with the raw prompt and no chat template. "
        "Train on these strings the same way, or the model sees a different prefix than it will at inference."
    ),
}

_HOLDOUT_MOD = 10


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    data = text.encode("utf-8")
    return {"sha256": sha256(data).hexdigest(), "rows": len(rows), "bytes": len(data)}


def _load_rows(source: Source) -> tuple[list[dict[str, Any]] | None, str | None]:
    try:
        from datasets import load_dataset
    except ImportError:
        return None, "datasets package not installed; pip install -e '.[data]'"
    try:
        if source.config:
            ds = load_dataset(source.repo, source.config, split=source.split)
        else:
            ds = load_dataset(source.repo, split=source.split)
    except Exception as exc:  # network, gating, auth, schema
        return None, f"{type(exc).__name__}: {exc}"
    return [dict(row) for row in ds], None


def _split(examples: list[Example]) -> tuple[list[Example], list[Example]]:
    train: list[Example] = []
    held: list[Example] = []
    for example in examples:
        bucket = int(example.prompt_hash()[:8], 16) % _HOLDOUT_MOD
        (held if bucket == 0 else train).append(example)
    if len(examples) >= 2 and not held:
        pick = min(examples, key=lambda e: e.prompt_hash())
        train = [e for e in train if e.prompt_hash() != pick.prompt_hash()]
        held = [pick]
    if len(examples) >= 2 and not train and len(held) > 1:
        train.append(held.pop())
    return train, held


def _source_row(source: Source, *, status: str, detail: str = "", rows_in: int = 0, rows_kept: int = 0) -> dict[str, Any]:
    row = source.to_dict()
    row["status"] = status
    row["detail"] = detail
    row["rows_in"] = rows_in
    row["rows_kept"] = rows_kept
    return row


def build_role(
    role: str,
    out: str | Path,
    *,
    max_n: int | None = None,
    seed: int = 0,
    offline: bool | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    """Write ``{role}.sft.jsonl``, ``{role}.dpo.jsonl``, ``{role}.eval.jsonl``, and merge ``manifest.json``."""
    if role not in {"thinker", "tooler", "coder"}:
        raise ValueError(f"unknown role {role!r}")
    if max_n is not None and max_n < 0:
        raise ValueError("--max must be >= 0")
    if offline is None:
        offline = os.environ.get("LAB_POSTTRAIN_OFFLINE", "0") == "1"
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dropped: Counter[str] = Counter()
    generated = generate(role)
    source_rows = [
        _source_row(SYNTHETIC, status="generated", rows_in=len(generated), rows_kept=0)
    ]
    examples: list[Example] = list(generated)

    for source in sources_for(role):
        if not source.usable:
            source_rows.append(_source_row(source, status="skipped", detail="license"))
            continue
        if offline:
            source_rows.append(_source_row(source, status="skipped", detail="offline"))
            continue
        rows, err = _load_rows(source)
        if err or rows is None:
            source_rows.append(_source_row(source, status="skipped", detail=err or "empty"))
            continue
        before = dropped.copy()
        converted: list[Example] = []
        for row in rows:
            converted.extend(
                convert_row(source.fmt, row, source=source.id, prefix="ens-1", dropped=dropped)
            )
        convert_dropped = dropped - before
        examples.extend(converted)
        source_rows.append(
            _source_row(
                source,
                status="loaded",
                detail="",
                rows_in=len(rows),
                rows_kept=0,
            )
        )
        source_rows[-1]["convert_dropped"] = dict(convert_dropped)

    seen: set[str] = set()
    kept: list[Example] = []
    for example in examples:
        digest = example.prompt_hash()
        if digest in seen:
            dropped["duplicate_prompt"] += 1
            continue
        reason = validate_example(example, execute=execute and example.role == "coder")
        if reason:
            dropped[reason] += 1
            continue
        seen.add(digest)
        kept.append(example)

    rng = random.Random(seed)
    rng.shuffle(kept)
    if max_n is not None:
        kept = kept[:max_n]
    train, held = _split(kept)

    by_source = Counter(ex.source for ex in train + held)
    for row in source_rows:
        row["rows_kept"] = int(by_source.get(row["id"], 0))

    sft_rows = [ex.sft_row() for ex in train]
    eval_rows = [ex.sft_row() for ex in held]
    dpo_rows = [row for ex in train if (row := ex.dpo_row()) is not None]

    files = {
        f"{role}.sft.jsonl": _write_jsonl(out_dir / f"{role}.sft.jsonl", sft_rows),
        f"{role}.dpo.jsonl": _write_jsonl(out_dir / f"{role}.dpo.jsonl", dpo_rows),
        f"{role}.eval.jsonl": _write_jsonl(out_dir / f"{role}.eval.jsonl", eval_rows),
    }
    role_report = {
        "sft": len(sft_rows),
        "eval": len(eval_rows),
        "dpo": len(dpo_rows),
        "dropped": dict(dropped),
        "dropped_total": int(sum(dropped.values())),
        "sources": source_rows,
    }
    manifest = _merge_manifest(out_dir, role, role_report, files)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"role": role, "out": str(out_dir), **role_report, "manifest": str(manifest_path)}


def _merge_manifest(
    out_dir: Path,
    role: str,
    role_report: dict[str, Any],
    files: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = out_dir / "manifest.json"
    if path.is_file():
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    else:
        manifest = {}
    if not isinstance(manifest, dict):
        manifest = {}
    manifest["version"] = 1
    manifest["format"] = FORMAT
    roles = manifest.get("roles")
    if not isinstance(roles, dict):
        roles = {}
    roles[role] = role_report
    manifest["roles"] = roles
    file_map = manifest.get("files")
    if not isinstance(file_map, dict):
        file_map = {}
    file_map.update(files)
    # Drop records for files this role used to write if the name changed.
    manifest["files"] = {name: rec for name, rec in file_map.items() if (out_dir / name).is_file()}
    counts = {}
    for name, rec in roles.items():
        if isinstance(rec, dict):
            counts[name] = {"sft": rec.get("sft", 0), "eval": rec.get("eval", 0), "dpo": rec.get("dpo", 0)}
    manifest["counts"] = counts
    return manifest


def format_report(report: dict[str, Any]) -> str:
    lines = [
        f"role: {report['role']}",
        f"sft: {report['sft']}",
        f"eval: {report['eval']}",
        f"dpo: {report['dpo']}",
        f"dropped: {report['dropped_total']}",
    ]
    for reason, count in sorted((report.get("dropped") or {}).items()):
        lines.append(f"  {reason}: {count}")
    lines.append("sources:")
    for source in report.get("sources") or []:
        detail = f" ({source['detail']})" if source.get("detail") else ""
        lines.append(
            f"  {source['id']}: {source['status']}{detail} "
            f"license={source['license']} in={source.get('rows_in', 0)} kept={source.get('rows_kept', 0)}"
        )
    lines.append(f"manifest: {report['manifest']}")
    return "\n".join(lines)
