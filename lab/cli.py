from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from lab.config import LabConfig
from lab.llm_policy import DEFAULT_POLICY_MODEL, InferPolicy, load_infer_engine
from lab.policy import DummyPolicy, InterleavePolicy, LabPolicy, ScriptedPolicy
from lab.promote import promote
from lab.supervisor import Supervisor


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _cfg(run_dir: Path) -> LabConfig:
    return LabConfig.from_env(run_dir)


def cmd_init(run_dir: Path) -> int:
    sup = Supervisor(_cfg(run_dir))
    sup.save()
    print(run_dir)
    return 0


def cmd_run(
    run_dir: Path,
    policy_name: str,
    cycles: int,
    seed: int,
    model: Path | None,
    max_steps: int | None,
    max_new_tokens: int,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    sup = Supervisor(_cfg(run_dir))
    if max_steps is None:
        max_steps = max(40, cycles * 16)
    if policy_name == "dummy":
        policy: object = DummyPolicy()
    elif policy_name == "lab":
        policy = LabPolicy()
    elif policy_name == "scripted":
        policy = ScriptedPolicy(seed=seed)
    elif policy_name == "interleave":
        policy = InterleavePolicy(DummyPolicy(), ScriptedPolicy(seed=seed), every=2)
    elif policy_name in {"nano", "qwen"}:
        cfg = sup.cfg
        model_dir = model or Path(
            os.environ.get("LAB_POLICY_MODEL", str(DEFAULT_POLICY_MODEL))
        )
        infer_root = cfg.infer_root or Path.home() / "Projects" / "infer"
        sup.cfg.research_max_seconds = max(sup.cfg.research_max_seconds, 1800.0)
        print(
            f"loading {policy_name} policy model={model_dir} infer_root={infer_root}",
            flush=True,
        )
        engine = load_infer_engine(model_dir, infer_root)
        policy = InferPolicy(
            engine,
            max_new_tokens=max_new_tokens,
            log_path=run_dir / "policy.jsonl",
        )
    else:
        raise SystemExit(f"unknown policy {policy_name}")
    result = sup.run_policy(policy, max_cycles=cycles, max_steps=max_steps)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("halted") else 1


def cmd_status(run_dir: Path) -> int:
    sup = Supervisor(_cfg(run_dir))
    print(json.dumps(sup.observe(), indent=2, sort_keys=True, default=str))
    return 0


def cmd_promote(run_dir: Path, min_delta: float) -> int:
    result = promote(run_dir, min_delta=min_delta)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("promoted") else 1


def cmd_seal(run_dir: Path) -> int:
    """Write an episode card from the latest finished job+eval (idempotent)."""
    sup = Supervisor(_cfg(run_dir))
    if not sup.state.current_job_id or not sup.state.pack_hash or not sup.state.last_eval:
        print(json.dumps({"sealed": False, "reason": "no completed job+eval"}))
        return 1
    pack = sup.packs.load(sup.state.pack_hash)
    job = sup.jobs.store.load(sup.state.current_job_id)
    ep = sup.episodes.record(
        cycle=max(sup.state.completed_cycles, 1),
        pack=pack,
        pack_hash=sup.state.pack_hash,
        job=job,
        ev=sup.state.last_eval,
    )
    sup.notebook.set_beliefs(sup.episodes.beliefs_markdown())
    print(json.dumps({"sealed": True, "id": ep["id"], "title": ep["title"]}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lab", description="Eval → Research → Train harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_run_dir(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--run-dir",
            type=Path,
            default=None,
            help="run directory (default: ./runs/<timestamp>)",
        )

    p_init = sub.add_parser("init", help="create a run directory")
    add_run_dir(p_init)

    p_run = sub.add_parser("run", help="run a policy for N cycles")
    add_run_dir(p_run)
    p_run.add_argument(
        "--policy",
        choices=["dummy", "lab", "scripted", "interleave", "nano", "qwen"],
        default="dummy",
    )
    p_run.add_argument("--cycles", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument(
        "--model",
        type=Path,
        default=None,
        help="LLM experimenter checkpoint (policy=qwen/nano; default ~/models/Qwen3.8-27B)",
    )
    p_run.add_argument("--max-steps", type=int, default=None, help="policy tool-call cap (default: 16 per cycle)")
    p_run.add_argument("--max-new-tokens", type=int, default=384)

    p_status = sub.add_parser("status", help="print run state")
    add_run_dir(p_status)

    p_promote = sub.add_parser("promote", help="promote last confirmation eval if it wins")
    add_run_dir(p_promote)
    p_promote.add_argument("--min-delta", type=float, default=0.0)

    p_seal = sub.add_parser("seal", help="seal latest job+eval into an episode card")
    add_run_dir(p_seal)

    args = p.parse_args(argv)
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = Path("runs") / _timestamp()
        if args.cmd == "status" or args.cmd == "promote" or args.cmd == "seal":
            p.error("--run-dir is required")

    if args.cmd == "init":
        return cmd_init(run_dir)
    if args.cmd == "run":
        return cmd_run(
            run_dir,
            args.policy,
            args.cycles,
            args.seed,
            args.model,
            args.max_steps,
            args.max_new_tokens,
        )
    if args.cmd == "status":
        return cmd_status(run_dir)
    if args.cmd == "promote":
        return cmd_promote(run_dir, args.min_delta)
    if args.cmd == "seal":
        return cmd_seal(run_dir)
    raise SystemExit("unknown command")
