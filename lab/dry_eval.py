"""Dry-run evals of the harness itself.

Dummy trainer, scripted ensemble roles, a private GPU lock file. Nothing here
loads a model or submits a ``lab`` / ``tinytrain`` job.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import json
import tempfile

from lab.config import LabConfig
from lab.ensemble.policy import EnsemblePolicy
from lab.ensemble.roles import Roles
from lab.hooks import HOOK_POINTS
from lab.jev import tool_name, tool_rows
from lab.policy import DummyPolicy, dummy_pack
from lab.supervisor import Supervisor
from lab.trace_report import load_trace


Case = Callable[[Path], dict[str, Any]]


def make_supervisor(root: Path, **overrides: Any) -> Supervisor:
    """Isolated run dir and its own lock file. Never ``LabConfig.from_env``."""
    cfg = LabConfig(
        run_dir=root / "run",
        allow_network=False,
        gpu_lock_path=root / "gpu.lock",
        data_cache_dir=root / "cache",
        research_max_tool_calls=int(overrides.pop("research_max_tool_calls", 20)),
        research_max_seconds=30,
        dummy_job_seconds=0.0,
        jev_every=int(overrides.pop("jev_every", 1)),
        train_tail_steps=int(overrides.pop("train_tail_steps", 2)),
        train_dwell_seconds=float(overrides.pop("train_dwell_seconds", 0.0)),
    )
    if overrides:
        raise TypeError(f"unknown dry-eval config {sorted(overrides)}")
    return Supervisor(cfg)


def _thought(**kw: Any) -> str:
    base: dict[str, Any] = {
        "thought": "",
        "hypotheses": [],
        "need": None,
        "pack": None,
        "done": False,
    }
    base.update(kw)
    return json.dumps(base)


def _phase_runs(rows: list[dict[str, Any]]) -> list[str]:
    runs: list[str] = []
    for row in tool_rows(rows):
        phase = str(row.get("phase") or "")
        if not runs or runs[-1] != phase:
            runs.append(phase)
    return runs


def _train_steps(rows: list[dict[str, Any]]) -> int:
    """Tool calls in the last train run."""
    count = 0
    seen = False
    for row in tool_rows(rows):
        if row.get("phase") == "train":
            seen = True
            count += 1
        elif seen:
            count = 0
            seen = False
    return count if seen else 0


def _case_phase_cadence(root: Path) -> dict[str, Any]:
    sup = make_supervisor(root)
    result = sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=40)
    rows = load_trace(sup.cfg.run_dir)
    runs = _phase_runs(rows)
    train_steps = _train_steps(rows)
    ok = (
        runs == ["eval", "research", "train", "eval"]
        and train_steps <= 2
        and result["completed_cycles"] >= 1
        and result["halted"]
    )
    return {
        "name": "phase_cadence",
        "ok": ok,
        "detail": {"phases": runs, "train_steps": train_steps, "halted": result["halted"]},
    }


class _Linger:
    """Stays in train after the job has finished, reading the notebook."""

    def __init__(self) -> None:
        self.inner = DummyPolicy()
        self.train_reads = 0

    def act(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if obs.get("phase") == "train":
            job = obs.get("job") or {}
            if job.get("status") in {"succeeded", "failed", "cancelled"} and self.train_reads < 6:
                self.train_reads += 1
                return "read_notebook", {}
        return self.inner.act(obs)


def _case_train_linger(root: Path) -> dict[str, Any]:
    sup = make_supervisor(root, train_tail_steps=2, jev_every=1)
    policy = _Linger()
    result = sup.run_policy(policy, max_cycles=1, max_steps=40)
    train_steps = _train_steps(load_trace(sup.cfg.run_dir))
    verdicts = _jsonl(sup.cfg.run_dir / "jev.jsonl")
    repaired = [v for v in verdicts if v.get("repair") and any(
        f.get("code") == "train_linger" for f in v.get("findings") or []
    )]
    # Two wasted reads, then the harness enters eval itself. Not six.
    ok = (
        bool(repaired)
        and policy.train_reads <= 2
        and train_steps <= sup.cfg.train_tail_steps + 1
        and result["completed_cycles"] >= 1
    )
    return {
        "name": "train_linger",
        "ok": ok,
        "detail": {
            "train_reads": policy.train_reads,
            "train_steps": train_steps,
            "repairs": len(repaired),
            "completed_cycles": result["completed_cycles"],
        },
    }


def _case_roles(root: Path) -> dict[str, Any]:
    """Thinker, tooler, and coder each take a turn. Dummy pack, no model."""
    sup = make_supervisor(root, research_max_tool_calls=40)
    seen = {"tool": False, "code": False}

    def thinker(prompt: str) -> str:
        del prompt
        if not seen["tool"]:
            seen["tool"] = True
            return _thought(need={"kind": "tool", "request": "list episodes"})
        if not seen["code"]:
            seen["code"] = True
            return _thought(need={"kind": "code", "request": "print a line"})
        pack = dummy_pack(sup.observe(), extra_config={"steps": 17})
        pack["hypothesis"] = "dry-eval steps=17"
        return _thought(
            pack=pack,
            hypotheses=[{"claim": "dry-eval steps=17 lowers confirm_ppl", "why": "w", "falsify": "f"}],
        )

    roles = Roles.scripted(
        thinker=thinker,
        tooler=lambda prompt: '{"tool": "list_episodes", "args": {"n": 1}}',
        coder=lambda prompt: json.dumps(
            {"path": "dry.py", "content": "print('dry-eval')\n", "run": True}
        ),
    )
    policy = EnsemblePolicy(roles, sup, n=1, max_rounds=6, tracer=sup.tracer)
    result = sup.run_policy(policy, max_cycles=1, max_steps=40)
    text = (sup.cfg.run_dir / "trace.jsonl").read_text(encoding="utf-8") if (
        sup.cfg.run_dir / "trace.jsonl"
    ).is_file() else ""
    parts = {
        "thinker": "ensemble 1 thinker" in text,
        "tooler": "ensemble 1 tooler" in text,
        "coder": "ensemble 1 coder" in text,
    }
    rows = load_trace(sup.cfg.run_dir)
    names = {tool_name(r) for r in tool_rows(rows)}
    ok = (
        all(parts.values())
        and "write_and_run" in names
        and _train_steps(rows) <= 2
        and result["completed_cycles"] >= 1
        and result.get("halt_reason") != "ensemble commit produced no trainable pack in cycle 1"
    )
    return {
        "name": "ensemble_roles",
        "ok": ok,
        "detail": {
            "parts": parts,
            "train_steps": _train_steps(rows),
            "halt_reason": result.get("halt_reason"),
            "completed_cycles": result["completed_cycles"],
        },
    }


def _case_hooks(root: Path) -> dict[str, Any]:
    sup = make_supervisor(root)
    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=40)
    points = [row.get("point") for row in _jsonl(sup.cfg.run_dir / "hooks.jsonl")]
    needed = ["after_eval", "before_research", "before_train", "before_eval", "on_step"]
    missing = [p for p in needed if p not in points]
    # Eval, then research, then train, then the post-train eval.
    order = [p for p in points if p in {"before_research", "before_train", "before_eval", "after_eval"}]
    ok = not missing and order[:4] == ["after_eval", "before_research", "before_train", "before_eval"]
    return {
        "name": "hooks",
        "ok": ok,
        "detail": {"order": order[:6], "missing": missing},
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


CASES: tuple[Case, ...] = (
    _case_phase_cadence,
    _case_train_linger,
    _case_roles,
    _case_hooks,
)


def run_dry_evals(root: Path | None = None) -> dict[str, Any]:
    """Run every dry case. ``root`` is a temp dir when omitted."""
    if root is None:
        with tempfile.TemporaryDirectory(prefix="lab-dry-") as tmp:
            return _run(Path(tmp))
    return _run(root)


def _run(root: Path) -> dict[str, Any]:
    cases = []
    for fn in CASES:
        case_root = root / fn.__name__
        case_root.mkdir(parents=True, exist_ok=True)
        try:
            cases.append(fn(case_root))
        except Exception as exc:
            cases.append(
                {
                    "name": fn.__name__,
                    "ok": False,
                    "detail": {"error": f"{type(exc).__name__}: {exc}"},
                }
            )
    return {
        "ok": all(c["ok"] for c in cases),
        "cases": cases,
        "hooks": HOOK_POINTS,
        "gpu": False,
    }

