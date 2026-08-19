from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from lab.config import LabConfig
from lab.policy import DummyPolicy, InterleavePolicy, ScriptedPolicy
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


def cmd_run(run_dir: Path, policy_name: str, cycles: int, seed: int) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    sup = Supervisor(_cfg(run_dir))
    if policy_name == "dummy":
        policy: object = DummyPolicy()
    elif policy_name == "scripted":
        policy = ScriptedPolicy(seed=seed)
    elif policy_name == "interleave":
        policy = InterleavePolicy(DummyPolicy(), ScriptedPolicy(seed=seed), every=2)
    else:
        raise SystemExit(f"unknown policy {policy_name}")
    result = sup.run_policy(policy, max_cycles=cycles)
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
    p_run.add_argument("--policy", choices=["dummy", "scripted", "interleave"], default="dummy")
    p_run.add_argument("--cycles", type=int, default=1)
    p_run.add_argument("--seed", type=int, default=0)

    p_status = sub.add_parser("status", help="print run state")
    add_run_dir(p_status)

    p_promote = sub.add_parser("promote", help="promote last confirmation eval if it wins")
    add_run_dir(p_promote)
    p_promote.add_argument("--min-delta", type=float, default=0.0)

    args = p.parse_args(argv)
    run_dir = args.run_dir
    if run_dir is None:
        run_dir = Path("runs") / _timestamp()
        if args.cmd == "status" or args.cmd == "promote":
            p.error("--run-dir is required")

    if args.cmd == "init":
        return cmd_init(run_dir)
    if args.cmd == "run":
        return cmd_run(run_dir, args.policy, args.cycles, args.seed)
    if args.cmd == "status":
        return cmd_status(run_dir)
    if args.cmd == "promote":
        return cmd_promote(run_dir, args.min_delta)
    raise SystemExit("unknown command")
