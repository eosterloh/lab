"""Map public function-calling and code rows onto harness completions.

A row is kept only when its function name (or, for code, its solution) can be
turned into one call the role parser and arg checks accept. Everything else
is counted and dropped. This module does not download anything.
"""

from __future__ import annotations

from collections import Counter
from typing import Any
import json
import re
import shlex

from lab.posttrain.data.examples import Example
from lab.posttrain.data.harness import (
    allowed_tooler_tools,
    make_obs,
    render_coder,
    render_tooler,
    schema_fields,
)

# Public names that mean the same harness tool. Anything else is unmapped.
_TOOL_ALIASES = {
    "read_file": "read_file",
    "readfile": "read_file",
    "cat_file": "read_file",
    "cat": "read_file",
    "list_files": "list_files",
    "list_dir": "list_files",
    "listdir": "list_files",
    "ls": "list_files",
    "grep": "grep_files",
    "grep_files": "grep_files",
    "search_files": "grep_files",
    "regex_search": "grep_files",
    "web_search": "web_search",
    "websearch": "web_search",
    "search_web": "web_search",
    "duckduckgo": "web_search",
    "duckduckgo_search": "web_search",
    "web_fetch": "web_fetch",
    "fetch_url": "web_fetch",
    "http_get": "web_fetch",
    "fetch": "web_fetch",
    "write_file": "write_file",
    "exec": "exec",
    "shell": "exec",
    "run_command": "exec",
    "run_python": "run_python",
    "write_and_run": "write_and_run",
    "prefetch_data": "prefetch_data",
    "data_stats": "data_stats",
    "read_notebook": "read_notebook",
    "write_note": "write_note",
    "write_beliefs": "write_beliefs",
    "list_episodes": "list_episodes",
    "read_episode": "read_episode",
    "read_skill": "read_skill",
    "list_skills": "list_skills",
    "list_jobs": "list_jobs",
    "read_train_log": "read_train_log",
    "sandbox_usage": "sandbox_usage",
    "list_models": "list_models",
    "inspect_model": "inspect_model",
    "read_hypothesis": "read_hypothesis",
    "list_hypotheses": "list_hypotheses",
    "list_packs": "list_packs",
    "read_pack": "read_pack",
    "diff_packs": "diff_packs",
    "read_trace": "read_trace",
    "read_checkpoint_meta": "read_checkpoint_meta",
    "read_episode_metrics": "read_episode_metrics",
}

# Public argument names -> harness argument, applied only when the harness
# field is in the schema and not already set.
_ARG_FROM = {
    "path": ("filepath", "file_path", "filename", "file", "dir", "directory", "folder", "dest"),
    "pattern": ("regex", "regexp", "pattern"),
    "query": ("q", "search", "search_query", "query"),
    "url": ("uri", "link", "href", "url"),
    "argv": ("command", "cmd", "argv"),
    "text": ("note", "text", "message"),
    "content": ("code", "script", "source", "content"),
    "n": ("limit", "count", "num", "n"),
    "model_dir": ("model", "model_path", "model_dir"),
    "pack_hash": ("hash", "pack_hash"),
    "source": ("dataset", "hf_source", "source"),
    "name": ("skill", "skill_name", "name"),
    "id": ("episode_id", "episode", "id"),
    "job_id": ("job", "job_id"),
}

_TAG_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FN_CALL = re.compile(r"<functioncall>\s*(\{.*)", re.DOTALL | re.IGNORECASE)
_ASSISTANT_JSON = re.compile(r"ASSISTANT:\s*(\{.*)", re.IGNORECASE | re.DOTALL)


def _loads_obj(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        return dict(data) if isinstance(data, dict) else None
    return None


def _loads_list(raw: Any) -> list[Any] | None:
    if isinstance(raw, list):
        return list(raw)
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return list(data) if isinstance(data, list) else None
    return None


def _balanced_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
                return data if isinstance(data, dict) else None
    return None


def extract_calls(text: str) -> list[dict[str, Any]]:
    """Function calls in Hermes ``<tool_call>`` or Glaive ``<functioncall>`` text."""
    if not isinstance(text, str) or not text.strip():
        return []
    found: list[dict[str, Any]] = []
    for match in _TAG_CALL.finditer(text):
        obj = _balanced_object(match.group(1))
        if obj and (obj.get("name") or obj.get("tool")):
            found.append(obj)
    if found:
        return found
    for cre in (_FN_CALL, _ASSISTANT_JSON):
        match = cre.search(text)
        if not match:
            continue
        obj = _balanced_object(match.group(1))
        if obj and (obj.get("name") or obj.get("tool")):
            return [obj]
    return []


def _norm_tool(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip("_")
    mapped = _TOOL_ALIASES.get(key)
    if mapped is None or mapped not in allowed_tooler_tools():
        return None
    return mapped


def _rename_args(tool: str, raw: dict[str, Any]) -> dict[str, Any]:
    fields = {name: typ for name, typ, _req in schema_fields(tool)}
    out: dict[str, Any] = {}
    used: set[str] = set()
    for canon, aliases in _ARG_FROM.items():
        if canon not in fields:
            continue
        for alias in (canon, *aliases):
            if alias in raw and alias not in used and raw[alias] is not None:
                out[canon] = raw[alias]
                used.add(alias)
                break
    for name in fields:
        if name in raw and name not in out and raw[name] is not None:
            out[name] = raw[name]
    if tool == "exec" and isinstance(out.get("argv"), str):
        try:
            out["argv"] = shlex.split(out["argv"])
        except ValueError:
            pass
    if tool == "exec" and isinstance(out.get("argv"), list):
        out["argv"] = [str(x) for x in out["argv"]]
    for name, typ in fields.items():
        if name not in out:
            continue
        value = out[name]
        if typ == "int" and isinstance(value, float) and value == int(value):
            out[name] = int(value)
        if typ == "int" and isinstance(value, str) and value.strip().isdigit():
            out[name] = int(value.strip())
    return out


def _prefix_path(tool: str, args: dict[str, Any], prefix: str) -> dict[str, Any] | None:
    if "path" not in args or not isinstance(args["path"], str):
        return args
    if tool in {"read_checkpoint_meta", "inspect_model"}:
        return args
    path = args["path"].strip()
    if path.startswith("/") or path.startswith("~") or ".." in path.split("/"):
        return None
    while path.startswith("./"):
        path = path[2:]
    if path and path != prefix and not path.startswith(prefix + "/"):
        path = f"{prefix}/{path}"
    out = dict(args)
    out["path"] = path
    return out


def _call_example(query: str, tool: str, args: dict[str, Any], *, source: str, prefix: str) -> Example:
    obs = make_obs()
    raw = json.dumps({"tool": tool, "args": args}, ensure_ascii=False)
    prompt = render_tooler(query.strip() or f"Use {tool}.", [], obs=obs, prefix=prefix)
    return Example(
        role="tooler",
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": raw}],
        source=source,
        kind="single",
        obs=obs,
    )


def map_call(
    query: str,
    call: dict[str, Any],
    *,
    source: str,
    prefix: str = "ens-1",
    dropped: Counter | None = None,
) -> Example | None:
    """Map one public function call onto a tooler example, or count a drop."""
    dropped = dropped if dropped is not None else Counter()
    name = str(call.get("name") or call.get("tool") or "").strip()
    tool = _norm_tool(name)
    if tool is None:
        if name in {"write_pack", "write_hypothesis", "halt", "enter_train", "queue_candidates"}:
            dropped["forbidden_tool"] += 1
        else:
            dropped["unmapped_tool"] += 1
        return None
    raw_args = call.get("arguments", call.get("args", {}))
    parsed = _loads_obj(raw_args)
    if parsed is None:
        if raw_args in ({}, None, "", []):
            parsed = {}
        else:
            dropped["bad_type"] += 1
            return None
    args = _rename_args(tool, parsed)
    prefixed = _prefix_path(tool, args, prefix)
    if prefixed is None:
        dropped["bad_path"] += 1
        return None
    return _call_example(query, tool, prefixed, source=source, prefix=prefix)


def _one_call(calls: list[dict[str, Any]], dropped: Counter) -> dict[str, Any] | None:
    if not calls:
        dropped["no_function_call"] += 1
        return None
    if len(calls) != 1:
        dropped["multiple_calls"] += 1
        return None
    return calls[0]


def convert_xlam_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> Example | None:
    answers = _loads_list(row.get("answers") if "answers" in row else row.get("answer"))
    if answers is None:
        dropped["no_function_call"] += 1
        return None
    calls = [c for c in answers if isinstance(c, dict)]
    call = _one_call(calls, dropped)
    if call is None:
        return None
    query = str(row.get("query") or row.get("question") or "")
    return map_call(query, call, source=source, prefix=prefix, dropped=dropped)


def convert_glaive_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> Example | None:
    chat = str(row.get("chat") or row.get("text") or "")
    calls = extract_calls(chat)
    call = _one_call(calls, dropped)
    if call is None:
        return None
    query = ""
    for match in re.finditer(r"USER:\s*(.*?)(?:<\|endoftext\|>|ASSISTANT:|$)", chat, re.DOTALL):
        query = match.group(1).strip()
    return map_call(query, call, source=source, prefix=prefix, dropped=dropped)


def convert_hermes_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> list[Example]:
    """One example per assistant turn that contains exactly one mapped call."""
    conv = row.get("conversations") or row.get("conversation") or []
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError:
            dropped["no_function_call"] += 1
            return []
    if not isinstance(conv, list):
        dropped["no_function_call"] += 1
        return []
    pending = ""
    out: list[Example] = []
    saw_call = False
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        who = str(turn.get("from") or turn.get("role") or "").lower()
        value = str(turn.get("value") or turn.get("content") or "")
        if who in {"human", "user"}:
            pending = value.strip()
            continue
        if who not in {"gpt", "assistant", "model"}:
            continue
        calls = extract_calls(value)
        if not calls:
            continue
        saw_call = True
        if len(calls) != 1:
            dropped["multiple_calls"] += 1
            continue
        example = map_call(pending, calls[0], source=source, prefix=prefix, dropped=dropped)
        if example is not None:
            out.append(example)
    if not saw_call:
        dropped["no_function_call"] += 1
    return out


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    lines = text.splitlines() or [""]
    return "\n".join(pad + line if line.strip() else line for line in lines)


def render_mbpp_script(row: dict[str, Any]) -> str:
    code = str(row.get("code") or "").rstrip()
    setup = str(row.get("test_setup_code") or "").rstrip()
    tests = row.get("test_list") or []
    if isinstance(tests, str):
        tests = [tests]
    body = "\n".join(str(t) for t in tests if str(t).strip())
    task_id = row.get("task_id", row.get("id", ""))
    checker = "\n".join(
        [
            "import json",
            "",
            "def _lab_main():",
            "    try:",
            _indent(body or "pass", 8),
            f"        return {{'ok': True, 'task_id': {task_id!r}}}",
            "    except Exception as exc:",
            "        return {'ok': False, 'error': type(exc).__name__ + ': ' + str(exc)}",
            "",
            "if __name__ == '__main__':",
            "    _result = _lab_main()",
            "    print(json.dumps(_result))",
            "    if not _result.get('ok'):",
            "        raise SystemExit(1)",
            "",
        ]
    )
    parts = [p for p in (setup, code, checker) if p]
    return "\n\n".join(parts)


def convert_mbpp_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> Example | None:
    text = str(row.get("text") or row.get("prompt") or "").strip()
    code = str(row.get("code") or "").strip()
    if not text or not code:
        dropped["empty_content"] += 1
        return None
    task_id = row.get("task_id", row.get("id", "0"))
    path = f"{prefix}/mbpp_{task_id}.py"
    content = render_mbpp_script(row)
    if len(content.splitlines()) > 200:
        dropped["coder_too_long"] += 1
        return None
    raw = json.dumps({"path": path, "content": content, "run": True, "args": []}, ensure_ascii=False)
    obs = make_obs()
    request = (
        "Write one Python script that solves this task and prints a final JSON line "
        f"with ok and task_id. Task: {text}"
    )
    prompt = render_coder(request, [], obs=obs, prefix=prefix, default_path=path)
    return Example(
        role="coder",
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": raw}],
        source=source,
        kind="single",
        default_path=path,
        obs=obs,
    )


def convert_codealpaca_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> Example | None:
    """Shape adapter only. The source is not usable; ``build`` does not call this."""
    instruction = str(row.get("instruction") or "").strip()
    output = str(row.get("output") or row.get("response") or "").strip()
    extra = str(row.get("input") or "").strip()
    if not instruction or not output:
        dropped["empty_content"] += 1
        return None
    request = instruction if not extra else f"{instruction}\n{extra}"
    if "json" not in output or "print" not in output:
        output = output.rstrip() + "\nimport json\nprint(json.dumps({'ok': True}))\n"
    path = f"{prefix}/codealpaca.py"
    raw = json.dumps({"path": path, "content": output, "run": True, "args": []})
    obs = make_obs()
    prompt = render_coder(request, [], obs=obs, prefix=prefix, default_path=path)
    return Example(
        role="coder",
        messages=[{"role": "user", "content": prompt}, {"role": "assistant", "content": raw}],
        source=source,
        kind="single",
        default_path=path,
        obs=obs,
    )


def convert_row(fmt: str, row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> list[Example]:
    if fmt == "xlam":
        example = convert_xlam_row(row, source=source, prefix=prefix, dropped=dropped)
        return [example] if example else []
    if fmt == "glaive":
        example = convert_glaive_row(row, source=source, prefix=prefix, dropped=dropped)
        return [example] if example else []
    if fmt == "hermes":
        return convert_hermes_row(row, source=source, prefix=prefix, dropped=dropped)
    if fmt == "mbpp":
        example = convert_mbpp_row(row, source=source, prefix=prefix, dropped=dropped)
        return [example] if example else []
    if fmt == "codealpaca":
        example = convert_codealpaca_row(row, source=source, prefix=prefix, dropped=dropped)
        return [example] if example else []
    dropped["unmapped_tool"] += 1
    return []
