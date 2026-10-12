"""A/B the experimenter with and without schema-constrained replies.

Drives a real Supervisor through one cycle to get the observations the policy
actually sees, then asks the model for each turn twice, plain and under
lab.schemas, using InferPolicy's own prompt. Every reply is scored with the
harness's own parser and pack validator.

    LAB_INFER_ROOT=~/Projects/infer python scripts/structured_ab.py \
        --model ~/models/Qwen2.5-0.5B-Instruct --out runs/structured_ab.jsonl
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab.config import LabConfig  # noqa: E402
from lab.data_cache import seed_default_mix  # noqa: E402
from lab.llm_policy import InferPolicy, load_infer_engine  # noqa: E402
from lab.pack import ArtifactPack  # noqa: E402
from lab.parse import parse_tool_call  # noqa: E402
from lab.policy import dummy_pack  # noqa: E402
from lab.schemas import parent_checkpoint, train_sources  # noqa: E402
from lab.supervisor import Supervisor  # noqa: E402
from lab.types import PHASE_TOOLS, Phase  # noqa: E402

PACK_NUDGE = "Call write_pack now with the config you want to test."


def scenarios(root: Path) -> list[tuple[str, dict[str, Any], dict[str, Any] | None, set[str], str | None]]:
    """(name, observation, last tool result, tools that follow the script, nudged tool)."""
    cache = root / "cache"
    seed_default_mix(cache)
    sup = Supervisor(LabConfig(run_dir=root / "run", data_cache_dir=cache, allow_network=False))
    out = [("eval/start", sup.observe(), None, {"run_eval"}, None)]
    last = sup.call("run_eval")
    out.append(("eval/after_run", sup.observe(), last, {"enter_research"}, None))
    last = sup.call("enter_research")
    out.append(("research/no_claim", sup.observe(), last, {"write_hypothesis", "list_episodes"}, None))
    last = sup.call(
        "write_hypothesis",
        {"claim": "a higher lr helps a 1-layer TinyGPT", "why": "short runs", "falsify": "confirm_ppl rises"},
    )
    no_pack = sup.observe()
    reads = {"write_pack", "list_episodes", "read_episode", "prefetch_data"}
    out.append(("research/no_pack", no_pack, last, reads, None))
    out.append(("research/nudge_pack", no_pack, last, {"write_pack"}, "write_pack"))
    last = sup.call("write_pack", {"pack": dummy_pack(no_pack)})
    out.append(("research/pack_ready", sup.observe(), last, {"enter_train"}, None))
    sup.call("enter_train")
    last = sup.call("job_status")
    out.append(("train/finished", sup.observe(), last, {"enter_eval", "job_status"}, None))
    return out


def score(raw: str, obs: dict[str, Any], on_script: set[str]) -> dict[str, Any]:
    parsed = parse_tool_call(raw)
    row: dict[str, Any] = {"parsed": parsed is not None, "tool": parsed[0] if parsed else None}
    try:
        json.loads(raw.strip())
        row["bare_json"] = True
    except ValueError:
        row["bare_json"] = False
    row["legal"] = bool(parsed) and parsed[0] in PHASE_TOOLS[Phase(obs["phase"])]
    row["on_script"] = bool(parsed) and parsed[0] in on_script
    if parsed and parsed[0] == "write_pack":
        pack = parsed[1].get("pack") or parsed[1]
        try:
            ArtifactPack.from_dict(pack)
            row["pack_valid"] = True
        except Exception as e:  # noqa: BLE001 - any rejection counts
            row["pack_valid"] = False
            row["pack_error"] = str(e)[:120]
        row["pack_parent_ok"] = pack.get("parent_checkpoint") == parent_checkpoint(obs)
        sources = (pack.get("data_manifest") or {}).get("sources") or []
        row["pack_sources_ok"] = bool(sources) and set(sources) <= set(train_sources(obs))
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, action="append", required=True)
    p.add_argument("--infer-root", default=os.environ.get("LAB_INFER_ROOT"))
    p.add_argument("--device", default=None)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if not args.infer_root:
        p.error("set LAB_INFER_ROOT or pass --infer-root")

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        cases = scenarios(Path(tmp))
        for model_dir in args.model:
            engine = load_infer_engine(model_dir, args.infer_root, device=args.device)
            tokenizer = engine.tokenizer
            for structured in (False, True):
                policy = InferPolicy(engine, max_new_tokens=args.max_new_tokens, structured=structured)
                for name, obs, last, on_script, want in cases:
                    policy._last = last
                    prompt = policy._prompt(obs)
                    tools = PHASE_TOOLS[Phase(obs["phase"])]
                    if want:
                        prompt += f"IMPORTANT: {PACK_NUDGE}\n"
                        tools = [want]
                    start = time.perf_counter()
                    raw = policy._generate(prompt, "ab", obs, tools)
                    seconds = time.perf_counter() - start
                    row = {
                        "model": model_dir.name,
                        "structured": structured,
                        "case": name,
                        "seconds": round(seconds, 2),
                        "tokens": len(tokenizer.encode(raw, add_special_tokens=False)),
                        **score(raw, obs, on_script),
                        "raw": raw,
                    }
                    rows.append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "raw"}), flush=True)
            del engine

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    print("\n| model | mode | parsed | bare JSON | legal tool | on script | valid pack | s/turn |")
    print("|---|---|---|---|---|---|---|---|")
    for model in dict.fromkeys(r["model"] for r in rows):
        for structured in (False, True):
            group = [r for r in rows if r["model"] == model and r["structured"] is structured]
            packs = [r for r in group if r["tool"] == "write_pack"]
            valid = sum(r["pack_valid"] and r["pack_parent_ok"] and r["pack_sources_ok"] for r in packs)

            def frac(key: str) -> str:
                return f"{sum(bool(r[key]) for r in group)}/{len(group)}"

            print(
                f"| {model} | {'schema' if structured else 'plain'} | {frac('parsed')} | {frac('bare_json')} "
                f"| {frac('legal')} | {frac('on_script')} | {valid}/{len(packs)} "
                f"| {sum(r['seconds'] for r in group) / len(group):.1f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
