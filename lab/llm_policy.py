from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys

from lab.data_cache import ALLOWLIST, DEFAULT_MIX
from lab.parse import parse_tool_call
from lab.policy import lab_pack
from lab.types import PHASE_TOOLS, Phase


_ALLOW = ", ".join(sorted(ALLOWLIST))
_MIX = ", ".join(f'"{s}"' for s in DEFAULT_MIX)
_PACK_SOURCES = "[" + _MIX + "]"

SYSTEM = f"""You are the experimenter in a local ML harness.
You may only call tools. Reply with ONE JSON object, nothing else:
{{"tool": "<name>", "args": {{ ... }}}}

Rules:
- One tool per turn. Use the current phase's allowed tools only.
- Complete one cycle: eval → research → train → eval.
- eval: if eval_ran is false, call run_eval. Then enter_research. Do not write_hypothesis or write_pack in eval. Do not halt.
- research: write_hypothesis if claim is empty. Optionally prefetch_data once per allowlisted source. If pack_hash is null, write_pack with trainer "lab" and the default mix sources. If pack_hash is set, call enter_train. Skip web_search/web_fetch/exec. Never write_pack twice in one cycle.
- write_hypothesis args are ONLY {{"claim": "...", "why": "...", "falsify": "..."}}. Never nest a hypothesis object or copy the checklist.
- prefetch_data args are {{"source": "hf:org/name", "split": "train", "n": 10000}}. Allowlist: {_ALLOW}. Do not fetch frozen_eval.
- Keep hidden, layers, heads, seq_len identical to the parent so weights resume. You may change lr and steps.
- parent_checkpoint must be observation.last_checkpoint when it is set.
- Lab pack: {{"tool": "write_pack", "args": {{"pack": {{"hypothesis": "continue TinyGPT overtrain on mixed HF text", "trainer": "lab", "config": {{"lr": 0.003, "steps": 32, "hidden": 32, "layers": 1, "heads": 1, "seq_len": 32, "batch": 4}}, "data_manifest": {{"sources": {_PACK_SOURCES}}}, "eval_suite_id": "core", "eval_suite_version": 1, "parent_checkpoint": "<last_checkpoint or subjects/tinytrain-8m>", "budgets": {{"max_hours": 0.1, "max_steps": 32}}}}}}}}
- Do not set data_manifest.sources to only builtin:tiny. Use the default mix exactly unless prefetch_data already cached extra allowlisted sources. Do not add Wikipedia or FineWeb-Edu on a cache miss.
- train: if job.status is succeeded, failed, or cancelled, call enter_eval. Else job_status.
- After returning to eval, run_eval. The harness will halt after max_cycles.
- Prior cycles live as episode cards. Use list_episodes / read_episode; do not dump the whole notebook.
"""


def _clip(obj: Any, n: int = 4000) -> str:
    text = json.dumps(obj, default=str)
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


class InferPolicy:
    """Greedy JSON tool-calling policy driven by an infer Engine."""

    def __init__(
        self,
        engine: Any,
        *,
        max_new_tokens: int = 384,
        parse_retries: int = 2,
        log_path: Path | None = None,
    ) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.parse_retries = parse_retries
        self.log_path = log_path
        self._last: dict[str, Any] | None = None
        self._parse_fails = 0

    def observe_result(self, result: dict[str, Any]) -> None:
        self._last = result

    def act(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if obs.get("halted"):
            return "halt", {"reason": "already halted"}
        prompt = self._prompt(obs)
        raw = self.engine.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            enable_thinking=False,
        )
        self._log(prompt, raw)
        parsed = parse_tool_call(raw)
        if parsed is None:
            self._parse_fails += 1
            if self._parse_fails > self.parse_retries:
                self._parse_fails = 0
                name, args = self._fallback(obs)
                print(f"[policy] parse_fallback {name}", flush=True)
                return name, args
            return "write_note", {
                "kind": "parse_error",
                "text": "last output was not JSON; next turn emit only {\"tool\":...,\"args\":...}",
            }
        self._parse_fails = 0
        name, args = parsed
        phase = Phase(obs["phase"])
        if name not in PHASE_TOOLS[phase]:
            fb_name, fb_args = self._fallback(obs)
            print(f"[policy] illegal {name} in {phase.value} -> {fb_name}", flush=True)
            return fb_name, fb_args
        if name == "write_hypothesis":
            hyp = obs.get("hypothesis") or {}
            if str(hyp.get("claim") or "").strip():
                fb_name, fb_args = self._fallback(obs)
                print(f"[policy] skip write_hypothesis (claim set) -> {fb_name}", flush=True)
                return fb_name, fb_args
        if name == "write_pack":
            if obs.get("pack_hash"):
                print("[policy] skip write_pack (pack_hash set) -> enter_train", flush=True)
                return "enter_train", {}
        return name, args

    def _fallback(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        phase = obs.get("phase")
        if phase == "eval":
            if not obs.get("eval_ran"):
                return "run_eval", {}
            return "enter_research", {}
        if phase == "research":
            hyp = obs.get("hypothesis") or {}
            if not str(hyp.get("claim") or "").strip():
                return "write_hypothesis", {
                    "claim": "continue TinyGPT overtrain",
                    "why": "policy fallback after a bad tool call",
                    "falsify": "lab job fails or confirm_ppl does not drop",
                }
            if not obs.get("pack_hash"):
                return "write_pack", {"pack": lab_pack(obs)}
            return "enter_train", {}
        if phase == "train":
            job = obs.get("job") or {}
            if job.get("status") in {"succeeded", "failed", "cancelled"}:
                return "enter_eval", {}
            return "job_status", {}
        return "write_note", {"kind": "fallback", "text": f"no fallback for phase {phase}"}

    def _prompt(self, obs: dict[str, Any]) -> str:
        phase = Phase(obs["phase"])
        allowed = sorted(PHASE_TOOLS[phase])
        return (
            f"{SYSTEM}\n"
            f"phase={phase.value} cycle={obs.get('cycle')} eval_ran={obs.get('eval_ran')} "
            f"completed={obs.get('completed_cycles')}\n"
            f"pack_hash={obs.get('pack_hash')}\n"
            f"job_status={(obs.get('job') or {}).get('status')}\n"
            f"allowed_tools={allowed}\n"
            f"observation={_clip(obs)}\n"
            f"last_tool_result={_clip(self._last)}\n"
        )

    def _log(self, prompt: str, raw: str) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": prompt, "completion": raw}) + "\n")


DEFAULT_POLICY_MODEL = Path.home() / "models" / "Qwen3.8-27B"


def load_infer_engine(model_dir: str | Path, infer_root: str | Path, device: str | None = None) -> Any:
    root = str(Path(infer_root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from engine.agent_api import load_engine  # type: ignore

    return load_engine(model_dir, device=device)
