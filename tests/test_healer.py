"""The healer harness reads a trace and asks Jev if the run is good."""

from __future__ import annotations

import json
from pathlib import Path

from lab.healer import Healer


def _tool(name: str, *, phase: str, n: int = 1, outputs: dict | None = None) -> list[dict]:
    return [
        {
            "name": f"tool.{name}",
            "kind": "tool",
            "phase": phase,
            "cycle": 1,
            "tool": name,
            "ok": True,
            "outputs": outputs,
        }
        for _ in range(n)
    ]


def _write_trace(run: Path, rows: list[dict]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    (run / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_a_healthy_trace_is_good_and_writes_no_heal(tmp_path: Path) -> None:
    run = tmp_path / "run"
    rows = (
        _tool("run_eval", phase="eval")
        + _tool("enter_research", phase="eval")
        + _tool("enter_train", phase="research", outputs={"job": {"status": "succeeded"}})
        + _tool("enter_eval", phase="train")
    )
    _write_trace(run, rows)
    result = Healer(run).pass_once(when="after")
    assert result["good"] is True
    assert "heal_id" not in result
    assert not list((run / "healer").glob("heal-*.json"))
    assert (run / "healer" / "passes.jsonl").is_file()


def test_a_lingering_train_writes_one_heal_and_does_not_repeat_it(tmp_path: Path) -> None:
    run = tmp_path / "run"
    rows = _tool("enter_train", phase="research", outputs={"job": {"status": "succeeded"}})
    rows += _tool("read_notebook", phase="train", n=3)
    _write_trace(run, rows)
    healer = Healer(run)
    first = healer.pass_once(when="during")
    second = healer.pass_once(when="during")
    assert first["good"] is False
    assert first["action"] == "enter_eval"
    assert first["heal_id"] == "heal-0001"
    card = json.loads((run / "healer" / "heal-0001.json").read_text(encoding="utf-8"))
    assert card["good"] is False
    assert "enter eval" in card["note"]
    assert second["repeated"] is True
    assert second["heal_id"] == "heal-0001"
    assert len(list((run / "healer").glob("heal-*.json"))) == 1


def test_watch_looks_more_than_once(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _write_trace(run, _tool("run_eval", phase="eval"))
    results = Healer(run).watch(every=0, passes=2, when="during")
    assert len(results) == 2
    assert all(row["when"] == "during" for row in results)
