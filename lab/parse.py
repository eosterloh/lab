from __future__ import annotations

import json
import re
from typing import Any


_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)

_PARSE_HINT = (
    'Expected one JSON object {"tool": "<name>", "args": {...}}. '
    "A ```json fence or a <tool_call>...</tool_call> tag is fine, and a single trailing comma is fine."
)


def _clean_tokens(text: str) -> str:
    return (
        text.replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<|endoftext|>", "")
    )


def _strip_trailing_commas(text: str) -> str:
    """Drop a single comma that sits immediately before ``}`` or ``]``.

    Commas inside strings are left alone. A doubled comma (``,,``) is not
    repaired: after one pass the JSON is still invalid, so a genuinely broken
    object stays broken.
    """
    out: list[str] = []
    in_str = False
    esc = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _loads_dict(blob: str) -> dict[str, Any] | None:
    blob = blob.strip()
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        repaired = _strip_trailing_commas(blob)
        if repaired == blob:
            return None
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _object_slice(text: str, start: int) -> str | None:
    """Balanced ``{...}`` slice starting at ``start``, strings respected."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
            if depth < 0:
                return None
    return None


def _iter_dicts(text: str) -> list[dict[str, Any]]:
    """Every JSON object in ``text``, in order. Invalid slices are skipped."""
    out: list[dict[str, Any]] = []
    idx = 0
    while True:
        start = text.find("{", idx)
        if start < 0:
            break
        blob = _object_slice(text, start)
        if blob is None:
            idx = start + 1
            continue
        data = _loads_dict(blob)
        if data is not None:
            out.append(data)
        idx = start + len(blob)
    return out


def _candidate_blobs(text: str) -> list[str]:
    """Places a small model actually puts the object, most specific first."""
    cleaned = _clean_tokens(text)
    blobs: list[str] = []
    blobs.extend(_TOOL_CALL.findall(cleaned))
    blobs.extend(_FENCE.findall(cleaned))
    blobs.append(cleaned)
    return blobs


def _tool_name(data: dict[str, Any]) -> str | None:
    name = data.get("tool")
    if name is None:
        name = data.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    """First JSON object in ``text``, or None.

    Tolerates chat special tokens, ```json fences, leading prose, a single
    trailing comma, and a ``<tool_call>`` wrapper. Does not invent an object
    from non-JSON text.
    """
    if not isinstance(text, str):
        return None
    for blob in _candidate_blobs(text):
        found = _iter_dicts(blob)
        if found:
            return found[0]
    return None


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Tolerant "one JSON object" parser for role replies.

    Same recovery rules as ``extract_json_object``. Returns None when no JSON
    object can be recovered (arrays, prose, doubled commas, empty tool tags).
    """
    return extract_json_object(text)


def _normalize_args(name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
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
    return args


def _coerce_args(raw: Any) -> dict[str, Any] | str:
    """Return an args dict, or a short reason the value cannot be one."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        parsed = _loads_dict(raw.strip())
        if parsed is None:
            # A fenced or tagged string still counts.
            parsed = extract_json_object(raw)
        if not isinstance(parsed, dict):
            return (
                "'args' is a string and not a JSON object. "
                'Pass an object, for example {"path": "..."}.'
            )
        return parsed
    if not isinstance(raw, dict):
        return f"'args' must be an object, got {type(raw).__name__}."
    return raw


def explain_tool_parse(text: str) -> tuple[str, dict[str, Any]] | str:
    """Parse one tool call, or return a reason the model can act on.

    A tuple means the call is well-formed. A string means nothing was called
    and names what was missing (no ``tool`` field, ``args`` not an object,
    unrecoverable text). Genuinely invalid JSON is not repaired into a call.
    """
    if not isinstance(text, str) or not text.strip():
        return f"empty tooler reply. {_PARSE_HINT} No tool was called."
    cleaned = _clean_tokens(text)
    data: dict[str, Any] | None = None
    for blob in _candidate_blobs(cleaned):
        for obj in _iter_dicts(blob):
            if _tool_name(obj):
                data = obj
                break
        if data is not None:
            break
    if data is None:
        salvaged = _salvage_tool_call(cleaned)
        if salvaged is not None:
            return salvaged
        if extract_json_object(cleaned) is not None:
            return (
                "JSON object has no string field 'tool' (alias 'name'). "
                f"{_PARSE_HINT} No tool was called."
            )
        return f"could not parse a tool call. {_PARSE_HINT} No tool was called."
    name = _tool_name(data)
    assert name is not None
    if "args" in data:
        raw_args: Any = data.get("args")
    elif "arguments" in data:
        raw_args = data.get("arguments")
    else:
        raw_args = {}
    coerced = _coerce_args(raw_args)
    if isinstance(coerced, str):
        return f"tool {name!r}: {coerced} {_PARSE_HINT}"
    return name, _normalize_args(name, coerced, data)


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    got = explain_tool_parse(text)
    if isinstance(got, str):
        return None
    return got


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
