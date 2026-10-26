"""Map public function-calling and code rows onto harness completions.

Function-calling rows (Glaive, Hermes, xLAM) keep their own tool names.
Each one is rendered with ``tooler_prompt``, with that example's tool list
in place of the harness catalog, and the assistant reply is
``{"tool", "args"}``. Those rows are tagged ``source=foreign_tools`` and
carry ``origin`` = the dataset id. Several calls in one example become
sequential single-call turns. A row with no call is dropped.

Code rows still have to be a script the coder parser and the sandbox accept.
This module does not download anything.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any
import json
import re

from lab.ensemble.protocol import Turn
from lab.posttrain.data.examples import Example
from lab.posttrain.data.harness import make_obs, render_coder, render_tooler

_TAG_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FN_MARK = re.compile(r"<functioncall>\s*", re.IGNORECASE)
_JSON_TYPES = {
    "string": "str",
    "str": "str",
    "integer": "int",
    "int": "int",
    "number": "number",
    "float": "number",
    "boolean": "bool",
    "bool": "bool",
    "array": "list",
    "list": "list",
    "object": "object",
    "dict": "object",
}
_INT_RE = re.compile(r"^-?\d+$")
_DESC_MAX = 180


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


def _repair_single_quoted_arguments(text: str) -> str:
    """Glaive writes ``"arguments": '{...}'`` with single quotes around JSON.

    That is not JSON. Lift the inner object out so ``json.loads`` can read the call.
    """
    mark = re.search(r'"arguments"\s*:\s*\'', text)
    if mark is None:
        return text
    rest = text[mark.end() :]
    if not rest.startswith("{"):
        return text
    _obj, end = _scan_object(rest, 0)
    if end <= 0 or not rest[end:].startswith("'"):
        return text
    inner = rest[:end]
    return text[: mark.start()] + '"arguments": ' + inner + rest[end + 1 :]


def _object_after(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    if start < 0:
        return None
    obj, _end = _scan_object(text, start)
    if obj is not None:
        return obj
    repaired = _repair_single_quoted_arguments(text[start:])
    if repaired == text[start:]:
        return None
    obj, _end = _scan_object(repaired, 0)
    return obj


@dataclass
class ForeignTool:
    name: str
    description: str
    params: list[tuple[str, str, bool]]  # name, type, required


def _scan_object(text: str, start: int) -> tuple[dict[str, Any] | None, int]:
    """JSON object at ``start``, or None and the index just after this attempt."""
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
                try:
                    data = json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None, i + 1
                return (data, i + 1) if isinstance(data, dict) else (None, i + 1)
    return None, start + 1


def _iter_objects(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        obj, end = _scan_object(text, i)
        if obj is not None:
            out.append(obj)
        i = end if end > i else i + 1
    return out


def extract_calls(text: str) -> list[dict[str, Any]]:
    """Every function call in Hermes ``<tool_call>`` or Glaive ``<functioncall>`` text."""
    if not isinstance(text, str) or not text.strip():
        return []
    found: list[dict[str, Any]] = []
    for match in _TAG_CALL.finditer(text):
        obj = _object_after(match.group(1))
        if obj and (obj.get("name") or obj.get("tool")):
            found.append(obj)
    if found:
        return found
    for match in _FN_MARK.finditer(text):
        obj = _object_after(text[match.end() :])
        if obj and (obj.get("name") or obj.get("tool")):
            found.append(obj)
    if found:
        return found
    for obj in _iter_objects(text):
        if obj.get("name") or obj.get("tool"):
            found.append(obj)
    return found


def _json_type(raw: Any) -> str:
    if isinstance(raw, list):
        for item in raw:
            if item != "null":
                return _json_type(item)
        return "any"
    if raw is None or not isinstance(raw, str):
        return "any"
    return _JSON_TYPES.get(raw.strip().lower(), "any")


def _params_from(raw: Any) -> list[tuple[str, str, bool]]:
    if isinstance(raw, str):
        raw = _loads_obj(raw) or {}
    if not isinstance(raw, dict):
        return []
    props = raw.get("properties")
    if isinstance(props, dict):
        req = raw.get("required")
        if isinstance(req, bool):
            required = {str(name) for name in props} if req else set()
        elif isinstance(req, list):
            required = {str(name) for name in req}
        else:
            required = set()
        out: list[tuple[str, str, bool]] = []
        for key, spec in props.items():
            spec = spec if isinstance(spec, dict) else {}
            out.append((str(key), _json_type(spec.get("type")), str(key) in required))
        return out
    out = []
    for key, spec in raw.items():
        if key in {"type", "required", "description", "$schema", "additionalProperties"}:
            continue
        if isinstance(spec, dict) and ("type" in spec or "properties" in spec or "description" in spec):
            required = spec.get("required", True)
            out.append((str(key), _json_type(spec.get("type")), bool(required) if isinstance(required, bool) else True))
        elif isinstance(spec, str):
            out.append((str(key), _json_type(spec), True))
    return out


def _tool_from_obj(obj: dict[str, Any]) -> ForeignTool | None:
    body = obj
    if obj.get("type") == "function" and isinstance(obj.get("function"), dict):
        body = obj["function"]
    name = body.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if "parameters" not in body and "params" not in body and "description" not in body and "properties" not in body:
        # A bare {"name": "..."} still counts when that is all the row declared.
        if set(body) - {"name", "type"} and "arguments" in body:
            return None
    desc = str(body.get("description") or "").strip().replace("\n", " ")
    if len(desc) > _DESC_MAX:
        desc = desc[: _DESC_MAX - 3] + "..."
    params = _params_from(body.get("parameters") if "parameters" in body else body.get("params"))
    return ForeignTool(name.strip(), desc, params)


def parse_tool_defs(raw: Any) -> list[ForeignTool]:
    """Tool definitions from a Hermes/xLAM list or a Glaive system string."""
    if isinstance(raw, list):
        blobs: list[Any] = list(raw)
    else:
        blobs = _loads_list(raw) or []
    tools: list[ForeignTool] = []
    seen: set[str] = set()
    if blobs:
        for item in blobs:
            obj = item if isinstance(item, dict) else _loads_obj(item)
            if not isinstance(obj, dict):
                continue
            try:
                tool = _tool_from_obj(obj)
            except (TypeError, ValueError):
                continue
            if tool is None or tool.name in seen:
                continue
            seen.add(tool.name)
            tools.append(tool)
        return tools
    if isinstance(raw, str):
        for obj in _iter_objects(raw):
            try:
                tool = _tool_from_obj(obj)
            except (TypeError, ValueError):
                continue
            if tool is None or tool.name in seen:
                continue
            seen.add(tool.name)
            tools.append(tool)
    return tools


def _infer_type(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "str"


def _infer_tools(calls: list[tuple[str, dict[str, Any]]]) -> list[ForeignTool]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for name, args in calls:
        if name not in grouped:
            order.append(name)
            grouped[name] = []
        grouped[name].append(args)
    tools: list[ForeignTool] = []
    for name in order:
        argsets = grouped[name]
        keys: list[str] = []
        for args in argsets:
            for key in args:
                if key not in keys:
                    keys.append(key)
        params: list[tuple[str, str, bool]] = []
        for key in keys:
            present = [args[key] for args in argsets if key in args and args[key] is not None]
            required = len(present) == len(argsets)
            params.append((key, _infer_type(present[0]) if present else "str", required))
        tools.append(ForeignTool(name, "", params))
    return tools


def _merge_tools(declared: list[ForeignTool], calls: list[tuple[str, dict[str, Any]]]) -> list[ForeignTool]:
    by = {tool.name: tool for tool in declared}
    order = [tool.name for tool in declared]
    for tool in _infer_tools(calls):
        if tool.name not in by:
            by[tool.name] = tool
            order.append(tool.name)
    return [by[name] for name in order]


def _call_parts(call: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    name = str(call.get("name") or call.get("tool") or "").strip()
    if not name:
        return None
    raw = call.get("arguments", call.get("args"))
    if raw in (None, "", [], {}):
        return name, {}
    parsed = _loads_obj(raw)
    if isinstance(parsed, str):
        parsed = _loads_obj(parsed)
    if parsed is None:
        return None
    return name, parsed


def _coerce_value(value: Any, typ: str) -> Any:
    if typ == "any":
        return value
    if typ == "str":
        return value if isinstance(value, str) else None
    if typ == "int":
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value == int(value):
            return int(value)
        if isinstance(value, str) and _INT_RE.match(value.strip()):
            return int(value.strip())
        return None
    if typ == "number":
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None
    if typ == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            return value.strip().lower() == "true"
        return None
    if typ == "list":
        return value if isinstance(value, list) else None
    if typ == "object":
        return value if isinstance(value, dict) else None
    return value


def _coerce_args(args: dict[str, Any], tool: ForeignTool) -> dict[str, Any] | str:
    out: dict[str, Any] = {}
    for name, typ, required in tool.params:
        if name not in args or args[name] is None:
            if required:
                return "missing_arg"
            continue
        value = _coerce_value(args[name], typ)
        if value is None:
            return "bad_type"
        if required and typ == "str" and isinstance(value, str) and not value.strip():
            return "missing_arg"
        out[name] = value
    for key, value in args.items():
        if key not in out and value is not None:
            out[key] = value
    return out


def _tool_meta(tools: list[ForeignTool]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "params": [{"name": name, "type": typ, "required": required} for name, typ, required in tool.params],
        }
        for tool in tools
    ]


def _catalog(tools: list[ForeignTool]) -> str:
    lines: list[str] = []
    for tool in tools:
        args = ", ".join(
            f"{name}{'?' if not required else ''}: {typ}" for name, typ, required in tool.params
        )
        line = f"{tool.name} {{{args}}}"
        if tool.description:
            line += f" — {tool.description}"
        lines.append(line)
    return "\n".join(lines)


def _foreign_example(
    steps: list[tuple[str, str, dict[str, Any]]],
    tools: list[ForeignTool],
    *,
    origin: str,
    prefix: str,
) -> Example:
    """One tooler conversation. Each step is one ``{"tool", "args"}`` turn."""
    obs = make_obs()
    catalog = _catalog(tools)
    messages: list[dict[str, str]] = []
    transcript: list[Turn] = []
    for i, (query, name, args) in enumerate(steps, start=1):
        request = query.strip() or f"Call {name}."
        if i > 1:
            request = (
                f"{request}\nThe previous call already ran. "
                f"Fulfil the rest of the request with the next call, {name}, and nothing else."
            )
        raw = json.dumps({"tool": name, "args": args}, ensure_ascii=False)
        prompt = render_tooler(request, transcript, obs=obs, prefix=prefix, catalog=catalog)
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": raw})
        transcript.append(
            Turn(
                role="tooler",
                prompt_chars=len(prompt),
                raw=raw,
                parsed_ok=True,
                result={"ok": True, "tool": name},
                ms=1.0,
                round=i,
                tool=name,
            )
        )
    return Example(
        role="tooler",
        messages=messages,
        source="foreign_tools",
        kind="episode" if len(steps) > 1 else "single",
        obs=obs,
        meta={"origin": origin, "tools": _tool_meta(tools)},
    )


def _steps_from_calls(
    groups: list[tuple[str, list[dict[str, Any]]]],
    *,
    declared: list[ForeignTool],
    origin: str,
    prefix: str,
    dropped: Counter,
) -> Example | None:
    """``groups`` is (user text, calls in the assistant span that follows it)."""
    flat: list[tuple[str, str, dict[str, Any]]] = []
    parsed_calls: list[tuple[str, dict[str, Any]]] = []
    for query, calls in groups:
        for call in calls:
            parts = _call_parts(call)
            if parts is None:
                dropped["bad_type"] += 1
                return None
            parsed_calls.append(parts)
            flat.append((query, parts[0], parts[1]))
    if not flat:
        dropped["no_function_call"] += 1
        return None
    tools = _merge_tools(declared, [(name, args) for _q, name, args in flat])
    by_name = {tool.name: tool for tool in tools}
    coerced: list[tuple[str, str, dict[str, Any]]] = []
    for query, name, args in flat:
        tool = by_name.get(name)
        if tool is None:
            dropped["unknown_tool"] += 1
            return None
        checked = _coerce_args(args, tool)
        if isinstance(checked, str):
            dropped[checked] += 1
            return None
        coerced.append((query, name, checked))
    return _foreign_example(coerced, tools, origin=origin, prefix=prefix)


def _glaive_groups(chat: str) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    parts = re.split(r"(?:<\|endoftext\|>\s*)?USER:\s*", chat, flags=re.IGNORECASE)
    for part in parts:
        if not part.strip():
            continue
        match = re.search(r"ASSISTANT:\s*(.*)$", part, flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            continue
        user = re.sub(r"<\|endoftext\|>", "", part[: match.start()]).strip()
        calls = extract_calls(match.group(1))
        if calls:
            groups.append((user, calls))
    return groups


def convert_xlam_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> Example | None:
    answers = _loads_list(row.get("answers") if "answers" in row else row.get("answer"))
    if answers is None:
        dropped["no_function_call"] += 1
        return None
    calls = [c for c in answers if isinstance(c, dict)]
    query = str(row.get("query") or row.get("question") or "")
    return _steps_from_calls(
        [(query, calls)],
        declared=parse_tool_defs(row.get("tools")),
        origin=source,
        prefix=prefix,
        dropped=dropped,
    )


def convert_glaive_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> Example | None:
    chat = str(row.get("chat") or row.get("text") or "")
    groups = _glaive_groups(chat)
    if not groups:
        # Some rows put the only call in ``chat`` without a USER/ASSISTANT split.
        calls = extract_calls(chat)
        if calls:
            groups = [(str(row.get("query") or ""), calls)]
    return _steps_from_calls(
        groups,
        declared=parse_tool_defs(row.get("system") or ""),
        origin=source,
        prefix=prefix,
        dropped=dropped,
    )


def convert_hermes_row(row: dict[str, Any], *, source: str, prefix: str, dropped: Counter) -> list[Example]:
    """One example per conversation. Several calls become sequential turns."""
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
    groups: list[tuple[str, list[dict[str, Any]]]] = []
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
        if calls:
            groups.append((pending, calls))
    example = _steps_from_calls(
        groups,
        declared=parse_tool_defs(row.get("tools")),
        origin=source,
        prefix=prefix,
        dropped=dropped,
    )
    return [example] if example else []


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
