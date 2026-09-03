from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = (
        text.replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<|endoftext|>", "")
    )
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    blob = fenced.group(1) if fenced else cleaned
    start = blob.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(blob[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(blob[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return data if isinstance(data, dict) else None
    return None


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    data = extract_json_object(text)
    if not data:
        return _salvage_tool_call(text)
    name = data.get("tool") or data.get("name")
    if not isinstance(name, str) or not name.strip():
        return _salvage_tool_call(text)
    args = data.get("args") or data.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    name = name.strip()
    if name == "write_pack" and "pack" not in args:
        pack_keys = {
            "hypothesis",
            "trainer",
            "config",
            "data_manifest",
            "eval_suite_id",
            "eval_suite_version",
            "parent_checkpoint",
            "budgets",
        }
        if pack_keys & set(data):
            args = {"pack": {k: data[k] for k in pack_keys if k in data}}
    if name == "write_hypothesis":
        args = _flatten_hypothesis_args(args, data)
    return name, args


def _flatten_hypothesis_args(args: dict[str, Any], data: dict[str, Any] | None = None) -> dict[str, Any]:
    out = dict(args)
    nested = out.get("hypothesis")
    if isinstance(nested, dict):
        for key in ("claim", "why", "falsify", "status"):
            if key in nested and key not in out:
                out[key] = nested[key]
    if data and isinstance(data.get("hypothesis"), dict):
        nested = data["hypothesis"]
        for key in ("claim", "why", "falsify", "status"):
            if key in nested and not out.get(key):
                out[key] = nested[key]
    claim = out.get("claim")
    if isinstance(claim, dict):
        out["claim"] = str(claim.get("claim") or claim.get("text") or "").strip()
    return out


def _salvage_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    cleaned = text.replace("<|im_end|>", "")
    name_m = re.search(r'"tool"\s*:\s*"(\w+)"', cleaned)
    if not name_m:
        return None
    name = name_m.group(1)
    if name != "write_hypothesis":
        return None
    claim_m = re.search(r'"claim"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned)
    if not claim_m:
        return None
    claim = json.loads(f'"{claim_m.group(1)}"') if "\\" in claim_m.group(1) else claim_m.group(1)
    return name, {
        "claim": claim,
        "why": "salvaged truncated json",
        "falsify": "job fails or confirm_ppl does not drop",
    }
