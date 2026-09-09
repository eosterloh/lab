"""Digest a run's trace into something a human (or a chat) can read.

Reads trace.jsonl written by lab.tracing plus the episode cards, and answers
the questions that actually come up when a run misbehaves: what did the policy
do each cycle, what got rejected, where did the time go, and did the model or
the harness author the pack.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import json


def load_trace(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "trace.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _episode_scores(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted((Path(run_dir) / "episodes").glob("ep-*.json")):
        try:
            ep = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        metrics = ep.get("metrics") or {}
        out.append(
            {
                "id": ep.get("id"),
                "status": ep.get("job_status"),
                "lr": metrics.get("lr"),
                "steps": metrics.get("steps"),
                "confirm_ppl": (ep.get("eval") or {}).get("confirm_ppl"),
            }
        )
    return out


def summarize(run_dir: Path) -> dict[str, Any]:
    rows = load_trace(run_dir)
    tools = [r for r in rows if r.get("kind") == "tool"]
    llm = [r for r in rows if r.get("kind") == "llm"]
    runs = [r for r in rows if r.get("name") == "lab.run"]

    by_cycle: dict[Any, dict[str, Any]] = defaultdict(
        lambda: {"tools": [], "rejected": [], "llm_ms": 0.0, "tool_ms": 0.0, "nudges": 0}
    )
    for r in tools:
        c = by_cycle[r.get("cycle")]
        name = str(r.get("name", "")).removeprefix("tool.")
        c["tools"].append(name)
        c["tool_ms"] += float(r.get("ms") or 0)
        if not r.get("ok"):
            c["rejected"].append({"tool": name, "error": r.get("error")})
    for r in llm:
        c = by_cycle[r.get("cycle")]
        c["llm_ms"] += float(r.get("ms") or 0)
        if r.get("name") == "policy.nudge":
            c["nudges"] += 1

    rejected = Counter(
        f"{str(r.get('name','')).removeprefix('tool.')}: {r.get('error')}"
        for r in tools
        if not r.get("ok")
    )
    slowest = sorted(llm, key=lambda r: float(r.get("ms") or 0), reverse=True)[:5]

    return {
        "run_dir": str(run_dir),
        "spans": len(rows),
        "tool_calls": len(tools),
        "llm_calls": len(llm),
        "nudges": sum(1 for r in llm if r.get("name") == "policy.nudge"),
        "rejected_calls": sum(1 for r in tools if not r.get("ok")),
        "llm_seconds": round(sum(float(r.get("ms") or 0) for r in llm) / 1000, 1),
        "tool_seconds": round(sum(float(r.get("ms") or 0) for r in tools) / 1000, 1),
        "runs": [
            {"outputs": r.get("outputs"), "seconds": round(float(r.get("ms") or 0) / 1000, 1)}
            for r in runs
        ],
        "cycles": {
            str(k): {
                "tools": v["tools"],
                "rejected": v["rejected"][:5],
                "nudges": v["nudges"],
                "llm_seconds": round(v["llm_ms"] / 1000, 1),
                "tool_seconds": round(v["tool_ms"] / 1000, 1),
            }
            for k, v in sorted(by_cycle.items(), key=lambda kv: (kv[0] is None, kv[0]))
        },
        "top_rejections": rejected.most_common(5),
        "slowest_llm_calls": [
            {
                "name": r.get("name"),
                "cycle": r.get("cycle"),
                "seconds": round(float(r.get("ms") or 0) / 1000, 1),
                "parsed_tool": r.get("parsed_tool"),
            }
            for r in slowest
        ],
        "episodes": _episode_scores(run_dir),
    }


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        f"# Trace report: {summary['run_dir']}",
        "",
        f"- tool calls: {summary['tool_calls']} ({summary['rejected_calls']} rejected)",
        f"- llm calls: {summary['llm_calls']} ({summary['nudges']} nudges)",
        f"- time: {summary['llm_seconds']}s policy, {summary['tool_seconds']}s tools",
        "",
    ]
    if not summary["spans"]:
        lines.append("No trace.jsonl found. Run the harness after enabling tracing.")
        return "\n".join(lines)

    lines += ["## Cycles", "", "| cycle | tools | rejected | nudges | policy s |", "|---|---|---|---|---|"]
    for cycle, c in summary["cycles"].items():
        seq = " → ".join(c["tools"]) or "-"
        if len(seq) > 90:
            seq = seq[:87] + "..."
        lines.append(
            f"| {cycle} | {seq} | {len(c['rejected'])} | {c['nudges']} | {c['llm_seconds']} |"
        )

    if summary["top_rejections"]:
        lines += ["", "## Rejections", ""]
        for reason, count in summary["top_rejections"]:
            lines.append(f"- {count}x {reason}")

    if summary["episodes"]:
        lines += ["", "## Episodes", "", "| lr | steps | confirm_ppl | status |", "|---|---|---|---|"]
        for ep in summary["episodes"]:
            lines.append(
                f"| {ep['lr']} | {ep['steps']} | {ep['confirm_ppl']} | {ep['status']} |"
            )

    if summary["slowest_llm_calls"]:
        lines += ["", "## Slowest policy turns", ""]
        for call in summary["slowest_llm_calls"]:
            lines.append(
                f"- {call['seconds']}s {call['name']} (cycle {call['cycle']})"
                f" -> {call['parsed_tool']}"
            )
    lines.append("")
    return "\n".join(lines)
