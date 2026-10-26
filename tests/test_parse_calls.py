"""Tool-call parsing: small-model wrappers, and refusals that name the mistake."""

from __future__ import annotations

from lab.ensemble.prompts import TOOL_SCHEMAS, check_tool_args, tool_json_schemas
from lab.ensemble.protocol import CodeAction
from lab.parse import explain_tool_parse, parse_json_object, parse_tool_call


def test_parse_accepts_fence_prose_trailing_comma_and_tool_call_tag() -> None:
    fenced = 'Sure.\n```json\n{"tool": "read_file", "args": {"path": "a.txt",}}\n```'
    assert parse_tool_call(fenced) == ("read_file", {"path": "a.txt"})

    prose = 'I will list the sandbox.\n{"tool": "list_files", "args": {"path": "."}}'
    assert parse_tool_call(prose) == ("list_files", {"path": "."})

    tagged = (
        'note {"skip": true}\n'
        '<tool_call>\n{"name": "read_file", "arguments": {"path": "notes.txt"}}\n</tool_call>'
    )
    assert parse_tool_call(tagged) == ("read_file", {"path": "notes.txt"})

    string_args = '<tool_call>{"tool": "read_file", "arguments": "{\\"path\\": \\"b.txt\\"}"}</tool_call>'
    assert parse_tool_call(string_args) == ("read_file", {"path": "b.txt"})


def test_parse_json_object_accepts_a_trailing_comma_and_a_tool_call_tag() -> None:
    assert parse_json_object('```json\n{"a": 1,}\n```') == {"a": 1}
    assert parse_json_object('<tool_call>{"done": true,}</tool_call>') == {"done": True}
    assert parse_json_object('{"path": "a,}b"}') == {"path": "a,}b"}


def test_parse_rejects_genuinely_broken_calls() -> None:
    assert parse_tool_call('{"tool": "read_file", "args": {"path": "a",,}}') is None
    assert parse_tool_call("<tool_call>not json</tool_call>") is None
    assert parse_tool_call("no braces here") is None
    assert parse_tool_call("[1, 2, 3]") is None
    assert parse_tool_call('{"tool": ""}') is None
    assert parse_tool_call('{"tool": 12, "args": {}}') is None
    reason = explain_tool_parse('{"thought": "hi"}')
    assert isinstance(reason, str)
    assert "tool" in reason
    args_reason = explain_tool_parse('{"tool": "read_file", "args": ["nope"]}')
    assert isinstance(args_reason, str)
    assert "args" in args_reason
    assert "object" in args_reason


def test_check_tool_args_names_the_field_and_coerces_numeric_strings() -> None:
    missing = check_tool_args("read_file", {})
    assert missing is not None
    assert "path" in missing
    assert "Allowed args" in missing
    assert "str" in missing

    bad = check_tool_args("read_file", {"path": 3})
    assert bad is not None
    assert "path" in bad and "int" in bad

    args = {"n": "5"}
    assert check_tool_args("list_episodes", args) is None
    assert args["n"] == 5

    assert check_tool_args("list_files", {}) is None
    assert check_tool_args("not_a_tool", {"x": 1}) is None


def test_tool_json_schemas_cover_the_catalog() -> None:
    specs = tool_json_schemas()
    assert set(specs) == set(TOOL_SCHEMAS)
    read = specs["read_file"]["parameters"]
    assert read["type"] == "object"
    assert read["required"] == ["path"]
    assert read["properties"]["path"] == {"type": "string"}
    listed = specs["list_files"]["parameters"]
    assert listed["required"] == []
    assert "path" in listed["properties"]
    assert specs["read_file"]["description"]


def test_coder_refusal_names_the_missing_field() -> None:
    raw = '{"path": "x.py", "run": true}'
    assert CodeAction.parse(raw, "d.py") is None
    msg = CodeAction.refusal(raw)
    assert "content" in msg
    assert "Allowed fields" in msg
    assert CodeAction.parse("```python\nprint(1)\n```", "d.py") is not None
