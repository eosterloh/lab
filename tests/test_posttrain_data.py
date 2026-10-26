"""Converters, synthetic traces, and the build CLI for lab.posttrain.data."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
import json

import pytest

from lab.ensemble.protocol import CodeAction, Thought
from lab.parse import parse_tool_call
from lab.posttrain.data.build import build_role
from lab.posttrain.data.convert import convert_row
from lab.posttrain.data.examples import Example
from lab.posttrain.data.harness import refusal_text, validate_coder_completion, validate_tool_completion
from lab.posttrain.data.sources import SOURCES, source_by_fmt
from lab.posttrain.data.synthetic import generate
from lab.posttrain.data.validate import validate_completion, validate_example

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "posttrain"


def _rows(name: str) -> list[dict]:
    path = FIXTURES / name
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _convert(fmt: str, filename: str) -> tuple[list[Example], Counter]:
    source = source_by_fmt(fmt)
    assert source is not None
    dropped: Counter = Counter()
    out: list[Example] = []
    for row in _rows(filename):
        out.extend(convert_row(fmt, row, source=source.id, prefix="ens-1", dropped=dropped))
    return out, dropped


def _hashes(path: Path) -> set[str]:
    found = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        users = [m["content"] for m in row["messages"] if m["role"] == "user"]
        found.add(sha256("\n<|user|>\n".join(users).encode("utf-8")).hexdigest())
    return found


def test_sources_record_licenses_and_skip_codealpaca() -> None:
    by_id = {s.id: s for s in SOURCES}
    assert by_id["Salesforce/xlam-function-calling-60k"].license == "cc-by-4.0"
    assert by_id["Salesforce/xlam-function-calling-60k"].usable is True
    assert by_id["glaiveai/glaive-function-calling-v2"].license == "apache-2.0"
    assert by_id["NousResearch/hermes-function-calling-v1"].license == "apache-2.0"
    assert by_id["NousResearch/hermes-function-calling-v1"].config == "func_calling_singleturn"
    assert by_id["google-research-datasets/mbpp"].license == "cc-by-4.0"
    assert by_id["google-research-datasets/mbpp"].split == "train"
    skipped = by_id["sahil2801/CodeAlpaca-20k"]
    assert skipped.usable is False
    assert skipped.license == "cc-by-nc-4.0"
    assert "OpenAI" in skipped.notes


def test_xlam_fixture_maps_harness_tools_and_drops_the_rest() -> None:
    examples, dropped = _convert("xlam", "xlam.jsonl")
    tools = []
    for ex in examples:
        parsed = parse_tool_call(ex.messages[-1]["content"])
        assert parsed is not None
        tools.append(parsed[0])
        assert validate_example(ex) is None
        assert "You are the TOOLER" in ex.messages[0]["content"]
    assert tools == ["read_file", "web_search"]
    read_args = parse_tool_call(examples[0].messages[-1]["content"])[1]
    assert read_args["path"] == "ens-1/notes.txt"
    assert dropped["unmapped_tool"] == 1
    assert dropped["multiple_calls"] == 1
    assert dropped["forbidden_tool"] == 1
    assert dropped["bad_path"] == 1


def test_glaive_and_hermes_fixtures() -> None:
    glaive, gdrop = _convert("glaive", "glaive.jsonl")
    assert len(glaive) == 2
    assert gdrop["no_function_call"] == 1
    assert gdrop["unmapped_tool"] == 1
    names = [parse_tool_call(ex.messages[-1]["content"])[0] for ex in glaive]
    assert names == ["read_file", "web_search"]
    for ex in glaive:
        assert validate_tool_completion(ex.messages[-1]["content"]) is None

    hermes, hdrop = _convert("hermes", "hermes.jsonl")
    assert [parse_tool_call(ex.messages[-1]["content"])[0] for ex in hermes] == ["web_search", "read_skill"]
    assert hdrop["multiple_calls"] == 1
    assert hdrop["no_function_call"] == 1
    skill = parse_tool_call(hermes[-1].messages[-1]["content"])
    assert skill == ("read_skill", {"name": "lab_harness"})
    for ex in hermes:
        assert validate_example(ex) is None


def test_mbpp_fixture_executes_and_drops_a_failing_check() -> None:
    examples, dropped = _convert("mbpp", "mbpp.jsonl")
    assert dropped["empty_content"] == 1
    assert len(examples) == 2
    reasons = [validate_example(ex, execute=True) for ex in examples]
    assert reasons[0] is None
    assert reasons[1] == "coder_exec_failed"
    action = CodeAction.parse(examples[0].messages[-1]["content"], examples[0].default_path)
    assert action is not None
    assert action.path == "ens-1/mbpp_9001.py"
    assert "assert add(1, 2) == 3" in action.content
    assert "You are the CODER" in examples[0].messages[0]["content"]


def test_synthetic_positives_cover_recovery_and_fail_their_rejections() -> None:
    for role, banner in (
        ("tooler", "You are the TOOLER"),
        ("thinker", "You are the THINKER"),
        ("coder", "You are the CODER"),
    ):
        rows = generate(role)
        kinds = {ex.kind for ex in rows}
        assert {"single", "recovery"} <= kinds
        assert any(banner in m["content"] for ex in rows for m in ex.messages if m["role"] == "user")
        for ex in rows:
            assert validate_example(ex, execute=(role == "coder")) is None
            if ex.rejected:
                assert validate_completion(ex, ex.rejected, execute=False) is not None
    tooler = generate("tooler")
    episode = next(ex for ex in tooler if ex.kind == "episode")
    assert len([m for m in episode.messages if m["role"] == "assistant"]) >= 2
    blob = "\n".join(m["content"] for m in episode.messages if m["role"] == "user")
    assert refusal_text("forbidden", "write_pack") in blob
    assert refusal_text("repeated") in blob
    recovery = next(ex for ex in tooler if ex.kind == "recovery")
    assert refusal_text("forbidden", "write_pack") in recovery.messages[0]["content"]
    thinker = next(ex for ex in generate("thinker") if ex.kind == "recovery")
    assert "pack rejected:" in thinker.messages[0]["content"]
    thought = Thought.parse(thinker.messages[-1]["content"])
    assert thought is not None and thought.pack is not None
    coder = next(ex for ex in generate("coder") if ex.kind == "recovery")
    assert "unparseable" in coder.messages[0]["content"]


def test_arg_checks_reject_the_harness_failures() -> None:
    assert validate_tool_completion('{"tool": "list_skills", "args": {}}') is None
    assert validate_tool_completion('{"tool": "read_file", "args": {"path": "ens-1/notes.txt"}}') is None
    assert validate_tool_completion('{"tool": "halt", "args": {"reason": "x"}}') == "forbidden_tool"
    assert validate_tool_completion('{"tool": "read_file", "args": {}}') == "missing_arg"
    assert validate_tool_completion('{"tool": "read_file", "args": {"path": "/etc/passwd"}}') == "bad_path"
    assert validate_tool_completion('{"tool": "exec", "args": {"argv": ["rm", "x"]}}') == "forbidden_command"
    assert validate_tool_completion('{"tool": "read_trace", "args": {"kind": "nope"}}') == "bad_kind"
    assert validate_tool_completion('{"tool": "prefetch_data", "args": {"source": "hf:nope/nope"}}') == "bad_source"
    assert validate_tool_completion("nope") == "parse_failed"
    # The name/arguments alias is the parser's, and a canonical tool still has to type-check.
    assert validate_tool_completion('{"name": "read_file", "arguments": {"path": 1}}') == "bad_type"


def test_build_each_role_offline_holds_out_by_prompt_hash(tmp_path: Path) -> None:
    out = tmp_path / "data"
    reports = []
    for role in ("thinker", "tooler", "coder"):
        reports.append(build_role(role, out, offline=True, seed=0))
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["counts"]) == {"thinker", "tooler", "coder"}
    assert manifest["format"]["sft"].startswith("TRL messages")
    for report in reports:
        role = report["role"]
        assert report["sft"] > 0 and report["eval"] > 0 and report["dpo"] > 0
        assert report["dropped_total"] == 0
        sft = out / f"{role}.sft.jsonl"
        ev = out / f"{role}.eval.jsonl"
        dpo = out / f"{role}.dpo.jsonl"
        assert _hashes(sft).isdisjoint(_hashes(ev))
        file_rec = manifest["files"][f"{role}.sft.jsonl"]
        assert file_rec["sha256"] == sha256(sft.read_bytes()).hexdigest()
        assert file_rec["rows"] == report["sft"]
        # Every positive assistant turn still parses; every rejection does not.
        for line in sft.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for message in row["messages"]:
                if message["role"] != "assistant":
                    continue
                if role == "tooler":
                    assert parse_tool_call(message["content"]) is not None
                elif role == "thinker":
                    assert Thought.parse(message["content"]) is not None
                else:
                    assert CodeAction.parse(message["content"], "ens-1/round1.py") is not None
                    assert validate_coder_completion(message["content"], default_path="ens-1/round1.py") is None
        for line in dpo.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert {"prompt", "chosen", "rejected"} <= set(row)
            probe = Example(role=role, messages=[{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["chosen"]}], source="synthetic", kind="single")
            # Chosen was already validated at build. Rejection must fail on its own.
            if role == "tooler":
                assert validate_tool_completion(row["rejected"]) is not None
            else:
                assert validate_completion(probe, row["rejected"], execute=False) is not None
        skipped = [s for s in report["sources"] if s["status"] == "skipped"]
        if role == "coder":
            reasons = {s["id"]: s["detail"] for s in skipped}
            assert reasons["sahil2801/CodeAlpaca-20k"] == "license"
            assert reasons["google-research-datasets/mbpp"] == "offline"
        if role == "tooler":
            assert any(s["license"] == "cc-by-4.0" and s["detail"] == "offline" for s in skipped)


def test_format_report_lists_drop_reasons() -> None:
    from lab.posttrain.data.build import format_report

    text = format_report(
        {
            "role": "tooler",
            "sft": 1,
            "eval": 1,
            "dpo": 0,
            "dropped_total": 3,
            "dropped": {"unmapped_tool": 2, "multiple_calls": 1},
            "sources": [],
            "manifest": "manifest.json",
        }
    )
    assert "dropped: 3" in text
    assert "multiple_calls: 1" in text
    assert "unmapped_tool: 2" in text


def test_cli_build_max(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from lab.posttrain.data.cli import main

    out = tmp_path / "cli"
    code = main(["build", "--role", "tooler", "--out", str(out), "--offline", "--max", "6", "--seed", "1"])
    assert code == 0
    text = capsys.readouterr().out
    assert "role: tooler" in text
    assert "dropped: 0" in text
    sft = [json.loads(line) for line in (out / "tooler.sft.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    ev = [json.loads(line) for line in (out / "tooler.eval.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(sft) + len(ev) == 6
    assert len(ev) >= 1 and len(sft) >= 1
