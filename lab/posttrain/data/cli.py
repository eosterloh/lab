"""CLI for the post-train data builder."""

from __future__ import annotations

from pathlib import Path
import argparse

from lab.posttrain.data.build import build_role, format_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lab.posttrain.data")
    sub = parser.add_subparsers(dest="cmd", required=True)
    build = sub.add_parser("build", help="write SFT, DPO, and held-out JSONL for one role")
    build.add_argument("--role", required=True, choices=("thinker", "tooler", "coder"))
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--max", type=int, default=None, help="cap kept SFT rows before the held-out split")
    build.add_argument("--seed", type=int, default=0)
    build.add_argument(
        "--offline",
        action="store_true",
        help="skip Hugging Face downloads and build from the synthetic generator only",
    )
    args = parser.parse_args(argv)
    if args.cmd == "build":
        report = build_role(
            args.role,
            args.out,
            max_n=args.max,
            seed=args.seed,
            offline=True if args.offline else None,
        )
        print(format_report(report))
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
