"""Adapters over the ensemble prompts, parsers, and tool schema.

Post-train code imports the harness from here so a parser or schema move
inside ``lab.ensemble`` is picked up without editing call sites. Argument
checks follow the same rules the tool handlers apply before they touch the
sandbox: required keys, types from ``TOOL_SCHEMAS``, relative paths, the
trace ``kind`` enum, allowlisted data sources, and forbidden exec commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib
import re

from lab.ensemble.ensemble import FORBIDDEN_TOOLS
from lab.ensemble.prompts import (
    VARIANT_HINTS,
    coder_prompt,
    coerce_config_numbers,
    knob_for_hint,
    pack_template,
    pack_tests_claim,
    placeholder_fields,
    placeholder_in_text,
    thinker_prompt,
    tool_catalog,
    tooler_prompt,
    ungrounded_ppl,
)
from lab.ensemble.protocol import CodeAction, Thought, Turn
from lab.ensemble.skills import list_skills, skills_for_role
from lab.types import PHASE_TOOLS, Phase

# Mirrors lab.sandbox. Sandbox.resolve rejects these before any read or write.
_FORBIDDEN_PREFIXES = ("/", "~")
_FORBIDDEN_COMMANDS = frozenset({"rm", "sudo", "chmod", "chown", "mkfs"})
_TRACE_KINDS = frozenset({"tool", "llm", "chain"})
_INT = re.compile(r"^-?\d+$")
_NETWORK = re.compile(r"\b(subprocess|socket|urllib|requests|http\.client|ftplib|webbrowser)\b")

# Tools whose path argument is not a sandbox path (ensemble._NON_SANDBOX_PATH_TOOLS
# plus model dirs, which live under LAB_MODELS_DIR).
_NON_SANDBOX_PATH_TOOLS = frozenset({"read_checkpoint_meta", "inspect_model"})

_SKILLS = {
    "thinker": skills_for_role("thinker", max_chars=4000),
    "tooler": skills_for_role("tooler", max_chars=3000),
    "coder": skills_for_role("coder", max_chars=6000),
}


def tool_schemas() -> dict[str, dict[str, Any]]:
    """Schema exported by the ensemble package, else the prompt catalog."""
    protocol = importlib.import_module("lab.ensemble.protocol")
    for name in ("TOOL_SCHEMA", "TOOL_SCHEMAS", "tool_schemas"):
        value = getattr(protocol, name, None)
        if isinstance(value, dict) and value:
            return value
        if callable(value):
            got = value()
            if isinstance(got, dict) and got:
                return got
    prompts = importlib.import_module("lab.ensemble.prompts")
    return dict(prompts.TOOL_SCHEMAS)


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """The tooler parser. Prefer one re-exported on the protocol module."""
    protocol = importlib.import_module("lab.ensemble.protocol")
    fn = getattr(protocol, "parse_tool_call", None)
    if not callable(fn):
        fn = importlib.import_module("lab.parse").parse_tool_call
    parsed = fn(text)
    if parsed is None:
        return None
    name, args = parsed
    if not isinstance(args, dict):
        args = {}
    return str(name), dict(args)


def allowed_tooler_tools() -> set[str]:
    """Research tools the tooler is allowed to call (phase gate minus forbidden)."""
    return set(PHASE_TOOLS[Phase.RESEARCH]) - set(FORBIDDEN_TOOLS)


def catalog_text() -> str:
    return tool_catalog(Phase.RESEARCH, exclude=FORBIDDEN_TOOLS)


def skill_names() -> list[str]:
    return list(list_skills())


def make_obs(
    *,
    cycle: int = 2,
    tool_calls: int = 1,
    confirm_ppl: float = 12.4,
) -> dict[str, Any]:
    """A research-phase observation shaped like ``Supervisor.observe``."""
    return {
        "cycle": cycle,
        "phase": "research",
        "trial": 0,
        "trials_planned": 1,
        "halted": False,
        "last_eval": {"confirm_ppl": confirm_ppl, "confirm_source": "frozen"},
        "last_metrics": {"val_ppl": confirm_ppl},
        "subject_checkpoint": "subjects/tinytrain-8m",
        "last_checkpoint": "checkpoints/job-0001.pt",
        "episodes": [
            {
                "id": "ep-0001",
                "confirm_ppl": 18.5,
                "title": "baseline short run",
                "job_status": "succeeded",
                "config": {"lr": 0.001, "steps": 64},
            },
            {
                "id": "ep-0002",
                "confirm_ppl": confirm_ppl,
                "title": "longer train",
                "job_status": "succeeded",
                "config": {"lr": 0.001, "steps": 128},
            },
        ],
        "hypothesis": {},
        "hypotheses": [],
        "data_cache": {"files": [{"path": "cache/tinystories.txt", "name": "tinystories.txt"}]},
        "research": {
            "tool_calls": tool_calls,
            "max_tool_calls": 10,
            "seconds_left": 480,
        },
    }


def render_tooler(request: str, transcript: list[Turn], *, obs: dict[str, Any], prefix: str) -> str:
    return tooler_prompt(
        request,
        obs,
        transcript,
        catalog=catalog_text(),
        sandbox_prefix=prefix,
        skills=_SKILLS["tooler"],
    )


def render_thinker(transcript: list[Turn], *, obs: dict[str, Any], variant_hint: str) -> str:
    return thinker_prompt(
        obs,
        transcript,
        variant_hint=variant_hint,
        skills=_SKILLS["thinker"],
    )


def render_coder(
    request: str,
    transcript: list[Turn],
    *,
    obs: dict[str, Any],
    prefix: str,
    default_path: str,
) -> str:
    return coder_prompt(
        request,
        obs,
        transcript,
        sandbox_prefix=prefix,
        skills=_SKILLS["coder"],
        default_path=default_path,
    )


def schema_fields(tool: str) -> list[tuple[str, str, bool]]:
    """``(name, type, required)`` for one tool. Unknown tool -> empty."""
    spec = tool_schemas().get(tool) or {}
    out: list[tuple[str, str, bool]] = []
    for raw, typ in (spec.get("args") or {}).items():
        required = not str(raw).endswith("?")
        name = str(raw)[:-1] if not required else str(raw)
        out.append((name, str(typ), required))
    return out


def _is_int(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(_INT.match(value.strip()))


def _is_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value.strip())
        except ValueError:
            return False
        return True
    return False


def _type_ok(value: Any, typ: str) -> bool:
    if typ == "str":
        return isinstance(value, str)
    if typ == "int":
        return _is_int(value)
    if typ == "number":
        return _is_number(value)
    if typ == "list[str]":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    if typ == "object":
        return isinstance(value, dict)
    if typ.startswith("list["):
        return isinstance(value, list) and all(isinstance(x, dict) for x in value)
    return True


def sandbox_path_error(path: str) -> str | None:
    """None when ``path`` would pass ``Sandbox.resolve``."""
    if not isinstance(path, str) or not path.strip():
        return "bad_path"
    if path.startswith(_FORBIDDEN_PREFIXES) or ".." in Path(path).parts:
        return "bad_path"
    return None


def _source_error(tool: str, args: dict[str, Any]) -> str | None:
    from lab.data_cache import compose_hf_source

    source = args.get("source")
    if not isinstance(source, str) or not source.strip():
        return "bad_source"
    if tool == "data_stats" and source.strip() in {"builtin:tiny", "dummy://tinystories"}:
        return None
    try:
        compose_hf_source(
            {
                "source": source,
                **({"split": args["split"]} if args.get("split") else {}),
                **({"n": int(args["n"])} if args.get("n") is not None else {}),
            }
        )
    except (TypeError, ValueError):
        return "bad_source"
    return None


def check_tool_args(tool: str, args: dict[str, Any]) -> str | None:
    """Return a drop reason, or None when the call would pass arg checks.

    Missing files, empty caches, and other runtime misses are not arg errors:
    the handler accepts the call and returns ``ok: false`` from the world.
    """
    if not isinstance(args, dict):
        return "bad_type"
    fields = schema_fields(tool)
    if not fields and tool not in tool_schemas():
        return "unknown_tool"
    for name, typ, required in fields:
        if name not in args or args[name] is None:
            if required:
                return "missing_arg"
            continue
        if not _type_ok(args[name], typ):
            return "bad_type"
        if typ == "str" and required and not str(args[name]).strip():
            return "missing_arg"
        if typ == "number" and _is_number(args[name]):
            try:
                if float(args[name]) <= 0:
                    return "bad_type"
            except (TypeError, ValueError):
                return "bad_type"
    # Keys the schema does not list are ignored by the handlers.

    if tool in {"read_file", "write_file", "run_python", "write_and_run", "grep_files", "list_files"}:
        if "path" in args and args["path"] is not None:
            if sandbox_path_error(str(args["path"])):
                return "bad_path"
    if tool == "web_fetch" and args.get("path") is not None:
        if sandbox_path_error(str(args["path"])):
            return "bad_path"
    if tool == "read_checkpoint_meta":
        path = str(args.get("path") or "")
        if sandbox_path_error(path):
            return "bad_path"
    if tool == "web_fetch":
        url = str(args.get("url") or "")
        from lab.http import assert_http_url

        try:
            assert_http_url(url)
        except ValueError:
            return "bad_url"
    if tool == "web_search" and not str(args.get("query") or "").strip():
        return "missing_arg"
    if tool == "read_trace" and args.get("kind") is not None:
        if args["kind"] not in _TRACE_KINDS:
            return "bad_kind"
    if tool == "exec":
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            return "bad_type"
        if argv[0] in _FORBIDDEN_COMMANDS:
            return "forbidden_command"
    if tool in {"prefetch_data", "data_stats"}:
        err = _source_error(tool, args)
        if err:
            return err
    if tool == "read_skill":
        name = str(args.get("name") or "")
        if name not in set(skill_names()):
            return "unknown_skill"
    if tool in {"read_pack", "diff_packs"}:
        keys = ("pack_hash",) if tool == "read_pack" else ("a", "b")
        for key in keys:
            digest = str(args.get(key) or "")
            if not re.fullmatch(r"[0-9a-fA-F]{8,64}", digest.strip()):
                return "bad_type"
    if tool == "write_and_run":
        content = args.get("content")
        if not isinstance(content, str) or not content.strip():
            return "missing_arg"
        if len(content.splitlines()) > 200:
            return "coder_too_long"
    return None


def validate_tool_completion(raw: str) -> str | None:
    parsed = parse_tool_call(raw)
    if parsed is None:
        return "parse_failed"
    name, args = parsed
    if name in FORBIDDEN_TOOLS:
        return "forbidden_tool"
    if name not in allowed_tooler_tools():
        return "unknown_tool"
    return check_tool_args(name, args)


def validate_thinker_completion(
    raw: str,
    *,
    obs: dict[str, Any] | None,
    knob: str | None,
    claim: str | None,
) -> str | None:
    thought = Thought.parse(raw)
    if thought is None:
        return "parse_failed"
    if thought.hypotheses:
        for hyp in thought.hypotheses:
            if placeholder_in_text(hyp.get("claim") or ""):
                return "placeholder"
            if not hyp.get("why") or not hyp.get("falsify"):
                return "incomplete_hypothesis"
    if thought.need is not None:
        if thought.need.get("kind") not in {"tool", "code"}:
            return "bad_need"
        if not str(thought.need.get("request") or "").strip():
            return "empty_request"
    if thought.pack is not None:
        obs = obs or {}
        stale = placeholder_fields(thought.pack)
        if stale:
            return "placeholder"
        pack, err = coerce_config_numbers(thought.pack)
        if err:
            return "pack_invalid"
        if placeholder_in_text(str(pack.get("hypothesis") or "")):
            return "placeholder"
        from lab.pack import ArtifactPack

        try:
            ArtifactPack.from_dict({k: v for k, v in pack.items() if k != "hypothesis_id"})
        except (TypeError, ValueError, KeyError):
            return "pack_invalid"
        use_claim = claim
        if not use_claim and thought.hypotheses:
            use_claim = thought.hypotheses[-1]["claim"]
        if use_claim:
            if pack_tests_claim(pack, use_claim, knob):
                return "pack_claim_mismatch"
            if ungrounded_ppl(use_claim, obs):
                return "ungrounded_ppl"
        if float((pack.get("budgets") or {}).get("max_hours") or 0) > 0.5:
            return "pack_invalid"
    if thought.pack is None and thought.need is None and not thought.done and not thought.hypotheses:
        return "idle"
    return None


def coder_static_error(content: str, path: str) -> str | None:
    if sandbox_path_error(path):
        return "bad_path"
    if not path.endswith(".py"):
        return "bad_path"
    if not isinstance(content, str) or not content.strip():
        return "empty_content"
    if len(content.splitlines()) > 200:
        return "coder_too_long"
    if "print" not in content or "json" not in content:
        return "coder_no_json"
    if _NETWORK.search(content):
        return "coder_network"
    return None


def execute_coder(content: str, args: list[str], *, timeout_s: float = 5.0) -> str | None:
    """Run a script the way the sandbox would, and require a final JSON line."""
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="lab-posttrain-") as tmp:
        path = Path(tmp) / "script.py"
        path.write_text(content, encoding="utf-8")
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmp,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = subprocess.run(
                [sys.executable, str(path), *args],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "coder_exec_failed"
        if proc.returncode != 0:
            return "coder_exec_failed"
        last = ""
        for line in (proc.stdout or "").splitlines():
            if line.strip():
                last = line.strip()
        try:
            parsed = json.loads(last)
        except json.JSONDecodeError:
            return "coder_exec_failed"
        if not isinstance(parsed, dict):
            return "coder_exec_failed"
    return None


def validate_coder_completion(raw: str, *, default_path: str, execute: bool = True) -> str | None:
    action = CodeAction.parse(raw, default_path)
    if action is None:
        return "parse_failed"
    # A bare ```python fence has no path; CodeAction.parse fills default_path.
    path = action.path
    if not path.endswith(".py"):
        path = default_path
    err = coder_static_error(action.content, path)
    if err:
        return err
    if execute and action.run:
        return execute_coder(action.content, list(action.args))
    return None


def refusal_text(kind: str, tool: str = "write_pack") -> str:
    """Wording the ensemble actually writes into the transcript."""
    if kind == "forbidden":
        return (
            f"tool {tool!r} is not available to the ensemble "
            "(harness/runner job); pick a read or sandbox tool"
        )
    if kind == "repeated":
        return "repeated call; use the earlier result"
    if kind == "unparseable_tool":
        return "tooler: unparseable; no tool was called"
    if kind == "unparseable_code":
        return "coder: unparseable; expected a JSON action or a ```python fence"
    if kind == "pack":
        return "pack rejected: config.lr is 'low', which is not a number. Fix it and emit the pack again."
    raise KeyError(kind)


__all__ = [
    "VARIANT_HINTS",
    "allowed_tooler_tools",
    "catalog_text",
    "check_tool_args",
    "execute_coder",
    "knob_for_hint",
    "make_obs",
    "pack_template",
    "parse_tool_call",
    "refusal_text",
    "render_coder",
    "render_thinker",
    "render_tooler",
    "schema_fields",
    "skill_names",
    "tool_schemas",
    "validate_coder_completion",
    "validate_thinker_completion",
    "validate_tool_completion",
]
