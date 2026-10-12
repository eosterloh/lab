"""JSON Schemas for the experimenter's replies.

infer's ``Engine.generate(json_schema=...)`` only lets the model write tokens
that keep the reply valid for the schema. With these schemas every turn parses
into a tool that exists in the current phase, with arguments of the right shape,
and a write_pack can only name the parent checkpoint and the train sources the
harness would accept. The model still picks the tool and every value.

Schemas use the subset infer enforces: type, properties, required, items,
min/maxItems, min/maxLength, enum, const, anyOf. Object keys come out in the
order listed here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import os
import re

from lab.data_cache import ALLOWLIST, DEFAULT_MIX, PKG_FROZEN_EVAL

_INT = {"type": "integer"}
_NUM = {"type": "number"}
_NAME = {"type": "string", "minLength": 1, "maxLength": 120}
_LINE = {"type": "string", "minLength": 1, "maxLength": 300}
_TEXT = {"type": "string", "minLength": 1, "maxLength": 1200}


def _args(required: dict[str, Any] | None = None, optional: dict[str, Any] | None = None) -> dict[str, Any]:
    props = {**(required or {}), **(optional or {})}
    return {"type": "object", "properties": props, "required": list(required or {})}


# Argument shapes as lab/actions.py reads them. write_pack is built per turn.
TOOL_ARGS: dict[str, dict[str, Any]] = {
    "read_notebook": _args(optional={"n": _INT}),
    "write_note": _args({"kind": _NAME, "text": _TEXT}),
    "write_beliefs": _args({"text": _TEXT}),
    "halt": _args(optional={"reason": _LINE}),
    "run_eval": _args(),
    "read_metrics": _args(),
    "read_samples": _args(),
    "enter_research": _args(),
    "list_episodes": _args(optional={"n": _INT, "query": _LINE}),
    "read_episode": _args({"id": _NAME}),
    "read_hypothesis": _args(),
    "list_files": _args(optional={"path": _NAME}),
    "read_file": _args({"path": _NAME}),
    "write_file": _args({"path": _NAME, "content": {"type": "string", "maxLength": 4000}}),
    "exec": _args({"argv": {"type": "array", "items": _LINE, "minItems": 1, "maxItems": 16}}),
    "web_fetch": _args({"url": _LINE}, {"path": _NAME}),
    "web_search": _args({"query": _LINE}),
    "write_hypothesis": _args({"claim": _LINE, "why": _LINE, "falsify": _LINE}),
    "prefetch_data": _args({"source": {"enum": sorted(ALLOWLIST)}}, {"split": _NAME, "n": _INT}),
    "enter_train": _args(),
    "job_status": _args(),
    "cancel_job": _args(),
    "enter_eval": _args(),
}

LAB_CONFIG_KEYS = ("lr", "steps", "hidden", "layers", "heads", "seq_len", "batch")


def structured_enabled() -> bool:
    """``LAB_STRUCTURED=0`` turns schema-constrained replies off."""
    return os.environ.get("LAB_STRUCTURED", "1") != "0"


def parent_checkpoint(obs: dict[str, Any]) -> str:
    return str(obs.get("last_checkpoint") or obs.get("subject_checkpoint") or "subjects/tinytrain-8m")


def train_sources(obs: dict[str, Any]) -> list[str]:
    """The default mix, plus any allowlisted slice prefetch_data has already cached."""
    sources = list(DEFAULT_MIX)
    files = (obs.get("data_cache") or {}).get("files") or []
    for entry in files:
        name = Path(str(entry.get("path", ""))).name
        for key, spec in ALLOWLIST.items():
            # data_cache.cache_file names slices <repo>__<split>__<n>__<digest>.txt
            m = re.fullmatch(re.escape(spec.repo.replace("/", "_")) + r"__(.+)__(\d+)__[0-9a-f]+\.txt", name)
            if m and f"{key}:{m.group(1)}:{m.group(2)}" not in sources:
                sources.append(f"{key}:{m.group(1)}:{m.group(2)}")
    return sources


def eval_suites(frozen_dir: Path = PKG_FROZEN_EVAL) -> list[tuple[str, int]]:
    found = (re.fullmatch(r"(.+)_v(\d+)\.json", p.name) for p in frozen_dir.glob("*_v*.json"))
    return sorted((m.group(1), int(m.group(2))) for m in found if m) or [("core", 1)]


def pack_schema(obs: dict[str, Any] | None = None) -> dict[str, Any]:
    """A lab-trainer ArtifactPack for this turn's observation."""
    obs = obs or {}
    sources = train_sources(obs)
    suites = eval_suites()
    return {
        "type": "object",
        "properties": {
            "hypothesis": _LINE,
            "trainer": {"const": "lab"},
            "config": {
                "type": "object",
                "properties": {key: _NUM if key == "lr" else _INT for key in LAB_CONFIG_KEYS},
                "required": list(LAB_CONFIG_KEYS),
            },
            "data_manifest": {
                "type": "object",
                "properties": {
                    "sources": {"type": "array", "items": {"enum": sources}, "minItems": 1, "maxItems": len(sources)}
                },
                "required": ["sources"],
            },
            "eval_suite_id": {"enum": sorted({s for s, _ in suites})},
            "eval_suite_version": {"enum": sorted({v for _, v in suites})},
            "parent_checkpoint": {"const": parent_checkpoint(obs)},
            "budgets": _args({"max_hours": _NUM, "max_steps": _INT}),
        },
        "required": [
            "hypothesis",
            "trainer",
            "config",
            "data_manifest",
            "eval_suite_id",
            "eval_suite_version",
            "parent_checkpoint",
            "budgets",
        ],
    }


def tool_args_schema(tool: str, obs: dict[str, Any] | None = None) -> dict[str, Any]:
    if tool == "write_pack":
        return _args({"pack": pack_schema(obs)})
    return TOOL_ARGS.get(tool, {"type": "object"})


def tool_call_schema(tools: Iterable[str], obs: dict[str, Any] | None = None) -> dict[str, Any]:
    """``{"tool": ..., "args": {...}}`` for exactly one of ``tools``."""
    branches = [
        {
            "type": "object",
            "properties": {"tool": {"const": tool}, "args": tool_args_schema(tool, obs)},
            "required": ["tool", "args"],
        }
        for tool in sorted(tools)
    ]
    if not branches:
        raise ValueError("tool_call_schema needs at least one tool")
    return branches[0] if len(branches) == 1 else {"anyOf": branches}
