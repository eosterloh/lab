"""Tracing must be free when off, structured when on, and never fatal.

The local JSONL sink is the one that has to work everywhere: Spark runs with
no LangSmith key still need a readable trace.
"""

from __future__ import annotations

from pathlib import Path
import json

from lab.policy import DummyPolicy
from lab.trace_report import markdown, summarize
from lab.tracing import LLM, TOOL, Tracer, langsmith_enabled


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_langsmith_stays_off_without_the_env_var(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    assert langsmith_enabled() is False
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert langsmith_enabled() is False


def test_spans_record_nesting_and_timing(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, use_langsmith=False)
    with tracer.span("outer", metadata={"cycle": 1}) as outer:
        with tracer.span("inner", kind=TOOL, inputs={"a": 1}) as inner:
            inner.set(outputs={"ok": True})
        outer.set(outputs={"done": True})

    rows = _rows(path)
    assert [r["name"] for r in rows] == ["inner", "outer"]
    inner_row, outer_row = rows
    assert inner_row["parent"] == outer_row["id"]
    assert inner_row["kind"] == TOOL
    assert inner_row["outputs"] == {"ok": True}
    assert outer_row["cycle"] == 1
    assert all(isinstance(r["ms"], float) for r in rows)


def test_a_raising_span_is_recorded_then_reraised(tmp_path: Path) -> None:
    tracer = Tracer(tmp_path / "trace.jsonl", use_langsmith=False)
    try:
        with tracer.span("boom"):
            raise ValueError("kaboom")
    except ValueError:
        pass
    row = _rows(tmp_path / "trace.jsonl")[0]
    assert row["ok"] is False
    assert "kaboom" in row["error"]


def test_an_unwritable_sink_does_not_break_the_run(tmp_path: Path) -> None:
    """A trace sink is never worth failing a training run over."""
    tracer = Tracer(tmp_path / "nope" / "trace.jsonl", use_langsmith=False)
    tracer.path = tmp_path  # a directory: writing must fail internally
    with tracer.span("still-fine") as span:
        span.set(outputs={"ok": True})


def test_cycle_spans_group_the_tree(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, use_langsmith=False)
    with tracer.span("lab.run"):
        tracer.enter_cycle(1)
        with tracer.span("step 1"):
            pass
        tracer.enter_cycle(2)
        with tracer.span("step 2"):
            pass
        tracer.close_cycle()

    rows = {r["name"]: r for r in _rows(path)}
    assert rows["step 1"]["parent"] == rows["cycle 1"]["id"]
    assert rows["step 2"]["parent"] == rows["cycle 2"]["id"]
    assert rows["cycle 1"]["parent"] == rows["lab.run"]["id"]


def test_run_policy_traces_tools_and_report_reads_it(sup) -> None:
    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    trace_path = sup.cfg.run_dir / "trace.jsonl"
    assert trace_path.is_file()

    rows = _rows(trace_path)
    tools = [r for r in rows if r["kind"] == TOOL]
    assert {"tool.run_eval", "tool.enter_train"} <= {r["name"] for r in tools}
    assert all(r.get("phase") for r in tools)

    summary = summarize(sup.cfg.run_dir)
    assert summary["tool_calls"] == len(tools)
    assert summary["episodes"]
    report = markdown(summary)
    assert "Trace report" in report
    assert "## Cycles" in report


def test_policy_turns_are_traced_with_prompt_and_parsed_tool(tmp_path: Path, sup) -> None:
    """Policy turns are the spans worth reading, so they carry the most detail."""
    from lab.llm_policy import InferPolicy
    from tests.test_llm_policy import SeqEngine

    path = tmp_path / "trace.jsonl"
    tracer = Tracer(path, use_langsmith=False)
    policy = InferPolicy(
        SeqEngine(['{"tool": "run_eval", "args": {}}']),
        max_new_tokens=64,
        tracer=tracer,
    )
    assert policy.act(sup.observe())[0] == "run_eval"

    row = _rows(path)[0]
    assert row["name"] == "policy.act"
    assert row["kind"] == LLM
    assert row["parsed_tool"] == "run_eval"
    assert row["phase"] == "eval"
    assert row["prompt_chars"] > 0


def test_report_on_a_run_with_no_trace_is_graceful(tmp_path: Path) -> None:
    summary = summarize(tmp_path)
    assert summary["spans"] == 0
    assert "No trace.jsonl" in markdown(summary)
