from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import sys

from lab.data_cache import ALLOWLIST, DEFAULT_MIX
from lab.parse import parse_tool_call
from lab.policy import lab_pack
from lab.tracing import LLM, Tracer
from lab.types import MEMORY_TOOLS, PHASE_TOOLS, Phase

# Reads are cheap per call but a read/read loop can eat a whole step budget.
READ_TOOLS = MEMORY_TOOLS | {"read_notebook", "list_files", "read_file"}


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
- Keep hidden, layers, heads, seq_len identical to the parent so weights resume. lr and steps are yours to choose.
- You are running a search. Before write_pack, call list_episodes once (and read_episode on the best one) and pick lr and steps that differ from the last episode. Repeating the previous config learns nothing.
- Research has a tool-call budget. Never repeat a call that just failed, and never call list_episodes twice in a row: read once, then write_pack, then enter_train.
- The example below shows the schema, not the values to use. Vary lr in 1e-4..1e-2 and steps in 8..512.
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
        max_hyp_writes_per_cycle: int = 2,
        max_error_streak: int = 3,
        max_reads_per_cycle: int = 4,
        tracer: Tracer | None = None,
    ) -> None:
        self.engine = engine
        self.max_new_tokens = max_new_tokens
        self.parse_retries = parse_retries
        self.max_hyp_writes_per_cycle = max_hyp_writes_per_cycle
        self.max_error_streak = max_error_streak
        self.max_reads_per_cycle = max_reads_per_cycle
        self.tracer = tracer or Tracer(None)
        self.log_path = log_path
        self._last: dict[str, Any] | None = None
        self._parse_fails = 0
        self._hyp_writes: dict[int, int] = {}
        self._reads: dict[int, int] = {}
        self._err_streak = 0

    def observe_result(self, result: dict[str, Any]) -> None:
        self._last = result
        self._err_streak = 0 if result.get("ok") else self._err_streak + 1

    def act(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if obs.get("halted"):
            return "halt", {"reason": "already halted"}
        if self._err_streak >= self.max_error_streak:
            streak = self._err_streak
            self._err_streak = 0
            want, want_args = self._fallback(obs)
            print(f"[policy] {streak} rejected calls -> {want}", flush=True)
            if want in {"write_pack", "write_hypothesis"}:
                return self._nudge_forward(
                    obs, f"Your last {streak} tool calls were rejected."
                )
            return want, want_args
        prompt = self._prompt(obs)
        raw = self._generate(prompt, "policy.act", obs)
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
            cycle = int(obs.get("cycle") or 1)
            self._hyp_writes[cycle] = self._hyp_writes.get(cycle, 0) + 1
            if self._hyp_writes[cycle] > self.max_hyp_writes_per_cycle:
                print(f"[policy] write_hypothesis loop in cycle {cycle}", flush=True)
                return self._nudge(
                    obs,
                    "You already wrote this cycle's hypothesis. Call write_pack now "
                    "with the config you want to test. Do not call write_hypothesis "
                    "or list_episodes again.",
                    want="write_pack",
                )
        if name == "write_pack":
            if obs.get("pack_hash"):
                print("[policy] skip write_pack (pack_hash set) -> enter_train", flush=True)
                return "enter_train", {}
        if name in READ_TOOLS and phase is Phase.RESEARCH:
            cycle = int(obs.get("cycle") or 1)
            self._reads[cycle] = self._reads.get(cycle, 0) + 1
            if self._reads[cycle] > self.max_reads_per_cycle:
                print(f"[policy] read budget spent in cycle {cycle}", flush=True)
                return self._nudge_forward(
                    obs, "You have read enough history for this cycle."
                )
        return name, args

    def _nudge_forward(
        self, obs: dict[str, Any], why: str
    ) -> tuple[str, dict[str, Any]]:
        """Push research toward train, asking for whichever gate is still open."""
        hyp = obs.get("hypothesis") or {}
        if not str(hyp.get("claim") or "").strip():
            return self._nudge(
                obs,
                f"{why} Call write_hypothesis now with this cycle's claim.",
                want="write_hypothesis",
            )
        return self._nudge(
            obs,
            f"{why} Call write_pack now with the config you want to test.",
            want="write_pack",
        )

    def _generate(
        self, prompt: str, label: str, obs: dict[str, Any], **meta: Any
    ) -> str:
        meta = {"cycle": obs.get("cycle"), "phase": obs.get("phase"), **meta}
        with self.tracer.span(label, kind=LLM, inputs=prompt, metadata=meta) as span:
            raw = self.engine.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                enable_thinking=False,
            )
            parsed = parse_tool_call(raw)
            span.set(
                outputs=raw,
                parsed_tool=parsed[0] if parsed else None,
                prompt_chars=len(prompt),
                completion_chars=len(raw),
            )
        self._log(prompt, raw)
        return raw

    def _nudge(
        self, obs: dict[str, Any], instruction: str, *, want: str
    ) -> tuple[str, dict[str, Any]]:
        """Re-ask for one specific tool so the model keeps authoring the pack.

        Substituting a harness pack here would silently take over experiment
        design, so the fallback is only a last resort.
        """
        prompt = f"{self._prompt(obs)}IMPORTANT: {instruction}\n"
        raw = self._generate(prompt, "policy.nudge", obs, want=want)
        parsed = parse_tool_call(raw)
        if parsed is not None and parsed[0] == want:
            print(f"[policy] nudge produced {want}", flush=True)
            return parsed
        fb_name, fb_args = self._fallback(obs)
        print(f"[policy] nudge failed, fallback {fb_name}", flush=True)
        return fb_name, fb_args

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
                print("[policy] fallback pack authored by harness, not the model", flush=True)
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
