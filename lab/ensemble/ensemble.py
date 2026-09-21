"""One ensemble: a thinker -> (tooler | coder) -> thinker loop that ends in a pack.

The ensemble never writes to the hypothesis board or the pack store itself.
It collects hypotheses and (at most) one validated pack in an ``EnsembleResult``;
``EnsembleRunner.commit`` opens them on the supervisor serially. Every
``sup.call`` goes through the shared ``call_lock`` so N ensembles can run in
parallel threads against one supervisor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import json
import threading
import time

from lab.ensemble.prompts import (
    coder_prompt,
    coerce_config_numbers,
    knob_for_hint,
    pack_fill_prompt,
    pack_nudge,
    pack_tests_claim,
    pack_template,
    placeholder_fields,
    placeholder_in_text,
    ungrounded_ppl,
    thinker_prompt,
    tool_catalog,
    tooler_prompt,
)
from lab.ensemble.protocol import CodeAction, Thought, Turn
from lab.ensemble.roles import Roles
from lab.ensemble.skills import skills_for_role
from lab.pack import ArtifactPack
from lab.parse import parse_json_object, parse_tool_call
from lab.tracing import CHAIN, LLM, TOOL, Tracer
from lab.types import PHASE_TOOLS, Phase

# Tools the tooler may not call: committing packs/hypotheses and moving phases
# is the runner's / harness's job.
FORBIDDEN_TOOLS = frozenset(
    {
        "write_pack",
        "enter_train",
        "queue_candidates",
        "write_hypothesis",
        "halt",
        "enter_research",
        "enter_eval",
    }
)

# Tools whose ``path`` argument is not a sandbox path (so it must not be prefixed).
_NON_SANDBOX_PATH_TOOLS = frozenset({"read_checkpoint_meta"})

MAX_HYPOTHESES = 3
MAX_NUDGES = 3
RETRY_SUFFIX = "\nReply with only the JSON object."


@dataclass
class EnsembleConfig:
    index: int  # 1-based
    max_rounds: int = 12
    max_tool_calls: int = 10
    max_new_tokens: dict[str, int] = field(
        default_factory=lambda: {"thinker": 768, "tooler": 384, "coder": 1536}
    )
    variant_hint: str = ""

    @property
    def sandbox_prefix(self) -> str:
        return f"ens-{self.index}"


@dataclass
class EnsembleResult:
    index: int
    hypotheses: list[dict]
    pack: dict | None
    rounds: int
    tool_calls: int
    code_runs: int
    transcript: list[Turn]
    error: str | None = None
    nudges: int = 0
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "hypotheses": list(self.hypotheses),
            "pack": self.pack,
            "rounds": self.rounds,
            "tool_calls": self.tool_calls,
            "code_runs": self.code_runs,
            "nudges": self.nudges,
            "done": self.done,
            "error": self.error,
            "transcript": [t.to_dict() for t in self.transcript],
        }

    def compact(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "hypotheses": [h.get("claim") for h in self.hypotheses],
            "pack": bool(self.pack),
            "rounds": self.rounds,
            "tool_calls": self.tool_calls,
            "code_runs": self.code_runs,
            "nudges": self.nudges,
            "done": self.done,
            "error": self.error,
        }


class _GenerateFailed(Exception):
    def __init__(self, role: str, exc: BaseException) -> None:
        self.role = role
        self.exc = exc
        super().__init__(f"{role}: {exc}")


class Ensemble:
    def __init__(
        self,
        roles: Roles,
        sup: Any,
        cfg: EnsembleConfig,
        *,
        call_lock: threading.Lock,
        tracer: Tracer | None = None,
    ) -> None:
        self.roles = roles
        self.sup = sup
        self.cfg = cfg
        self.lock = call_lock
        self.tracer = tracer or Tracer(None)
        self._skills = {
            "thinker": skills_for_role("thinker", max_chars=4000),
            "tooler": skills_for_role("tooler", max_chars=3000),
            "coder": skills_for_role("coder", max_chars=6000),
        }
        self._knob = knob_for_hint(cfg.variant_hint)
        self._catalog = tool_catalog(Phase.RESEARCH, exclude=FORBIDDEN_TOOLS)
        self._allowed = PHASE_TOOLS[Phase.RESEARCH] - FORBIDDEN_TOOLS

    # ------------------------------------------------------------------ run

    def run(self, obs: dict[str, Any]) -> EnsembleResult:
        i = self.cfg.index
        res = EnsembleResult(
            index=i, hypotheses=[], pack=None, rounds=0, tool_calls=0, code_runs=0, transcript=[]
        )
        last_call: tuple[str, str] | None = None
        self._template_nudged = False
        try:
            for r in range(1, self.cfg.max_rounds + 1):
                res.rounds = r
                meta = {"cycle": obs.get("cycle"), "ensemble": i}
                with self.tracer.span(f"ensemble {i} round {r}", kind=CHAIN, metadata=meta) as span:
                    thought = self._think(obs, res, r)
                    if thought is None:
                        res.error = "thinker: unparseable"
                        span.set(outputs={"error": res.error})
                        break
                    absorb_err = self._absorb_hypotheses(thought, res)

                    if thought.pack is not None:
                        claim = res.hypotheses[-1]["claim"] if res.hypotheses else None
                        checked, pack_err = self._validate_pack(thought.pack, claim=claim, obs=obs)
                        if pack_err is None and not self._template_nudged and self._is_template(thought.pack, obs):
                            # Small models sometimes hand the nudge template back
                            # verbatim; ask once for the edit the claim promised.
                            self._template_nudged = True
                            res.nudges += 1
                            self._synthetic(
                                res,
                                r,
                                "your pack is the template unchanged. Apply the change your claim names "
                                "(e.g. set config.lr or config.steps to the value in your hypothesis) and emit it again.",
                            )
                            span.set(outputs={"pack_error": "template unchanged"})
                            continue
                        if pack_err is None:
                            res.pack = _strip_pack(checked)
                            res.done = True
                            span.set(outputs={"pack": True})
                            break
                        res.nudges += 1
                        self._synthetic(res, r, f"pack rejected: {pack_err}. Fix it and emit the pack again.")
                        span.set(outputs={"pack_error": pack_err})
                        continue

                    need = thought.need
                    if need and need.get("kind") == "tool":
                        if res.tool_calls >= self.cfg.max_tool_calls:
                            res.nudges += 1
                            self._synthetic(res, r, "tool budget spent; decide now: emit a pack or set done.")
                            span.set(outputs={"tool_budget": "spent"})
                            continue
                        last_call = self._tool_round(obs, res, r, need, last_call)
                        span.set(outputs={"tool": res.transcript[-1].tool, "ok": (res.transcript[-1].result or {}).get("ok")})
                        continue

                    if need and need.get("kind") == "code":
                        self._code_round(obs, res, r, need)
                        span.set(outputs={"code": res.transcript[-1].tool})
                        continue

                    if thought.done:
                        res.done = True
                        span.set(outputs={"done": True})
                        break

                    res.nudges += 1
                    if absorb_err:
                        self._synthetic(res, r, absorb_err)
                        span.set(outputs={"nudge": res.nudges, "why": "placeholder claim"})
                        if res.nudges >= MAX_NUDGES:
                            res.error = "thinker: no progress after 3 nudges"
                            break
                        continue
                    if res.hypotheses and self._fill_pack(obs, res, r):
                        span.set(outputs={"pack": True, "via": "fill"})
                        break
                    if not res.hypotheses:
                        self._synthetic(res, r, pack_nudge(obs, res.hypotheses, knob=self._knob))
                    span.set(outputs={"nudge": res.nudges})
                    if res.nudges >= MAX_NUDGES:
                        res.error = "thinker: no progress after 3 nudges"
                        break
        except _GenerateFailed as e:
            res.error = str(e)
        except Exception as e:  # never take the whole cycle down
            res.error = f"{type(e).__name__}: {e}"
        return res

    # ---------------------------------------------------------------- roles

    def _generate(self, role: str, prompt: str, obs: dict[str, Any], round_: int) -> tuple[str, float]:
        meta = {"cycle": obs.get("cycle"), "ensemble": self.cfg.index, "role": role, "round": round_}
        started = time.time()
        with self.tracer.span(f"ensemble {self.cfg.index} {role}", kind=LLM, inputs=prompt, metadata=meta) as span:
            try:
                raw = self.roles.generate(role, prompt, max_new_tokens=self.cfg.max_new_tokens.get(role, 512))
            except Exception as e:
                span.set(outputs={"error": str(e)}, error=str(e), prompt_chars=len(prompt))
                raise _GenerateFailed(role, e) from e
            if not isinstance(raw, str):
                raw = str(raw)
            span.set(
                outputs=raw,
                prompt_chars=len(prompt),
                completion_chars=len(raw),
                parsed_tool=_parsed_label(role, raw),
            )
        return raw, round((time.time() - started) * 1000, 1)

    def _call(self, tool: str, args: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
        """``sup.call`` under the shared lock, as a tool span nested in this round."""
        meta = {"cycle": obs.get("cycle"), "phase": "research", "tool": tool, "ensemble": self.cfg.index}
        with self.tracer.span(f"tool.{tool}", kind=TOOL, inputs=args, metadata=meta) as span:
            with self.lock:
                result = self.sup.call(tool, args)
            span.set(outputs=result, error=None if result.get("ok") else str(result.get("error")))
        return result

    def _think(self, obs: dict[str, Any], res: EnsembleResult, r: int) -> Thought | None:
        prompt = thinker_prompt(
            obs, res.transcript, variant_hint=self.cfg.variant_hint, skills=self._skills["thinker"]
        )
        raw, ms = self._generate("thinker", prompt, obs, r)
        thought = Thought.parse(raw)
        if thought is None:
            res.transcript.append(
                Turn(role="thinker", prompt_chars=len(prompt), raw=raw, parsed_ok=False, result=None, ms=ms, round=r)
            )
            raw, ms = self._generate("thinker", prompt + RETRY_SUFFIX, obs, r)
            thought = Thought.parse(raw)
            if thought is None:
                res.transcript.append(
                    Turn(role="thinker", prompt_chars=len(prompt), raw=raw, parsed_ok=False, result=None, ms=ms, round=r)
                )
                return None
        summary = {
            "need": thought.need,
            "pack": thought.pack is not None,
            "hypotheses": len(thought.hypotheses),
            "done": thought.done,
            "thought": thought.thought[:300],
        }
        res.transcript.append(
            Turn(role="thinker", prompt_chars=len(prompt), raw=raw, parsed_ok=True, result=summary, ms=ms, round=r)
        )
        return thought

    def _tool_round(
        self,
        obs: dict[str, Any],
        res: EnsembleResult,
        r: int,
        need: dict[str, Any],
        last_call: tuple[str, str] | None,
    ) -> tuple[str, str] | None:
        prompt = tooler_prompt(
            need.get("request") or "",
            obs,
            res.transcript,
            catalog=self._catalog,
            sandbox_prefix=self.cfg.sandbox_prefix,
            skills=self._skills["tooler"],
        )
        raw, ms = self._generate("tooler", prompt, obs, r)
        parsed = parse_tool_call(raw)
        if parsed is None:
            res.nudges += 1
            res.transcript.append(
                Turn(
                    role="tooler",
                    prompt_chars=len(prompt),
                    raw=raw,
                    parsed_ok=False,
                    result={"ok": False, "error": "tooler: unparseable; no tool was called"},
                    ms=ms,
                    round=r,
                )
            )
            return last_call
        tool, args = parsed
        if tool in FORBIDDEN_TOOLS or tool not in self._allowed:
            res.nudges += 1
            res.transcript.append(
                Turn(
                    role="tooler",
                    prompt_chars=len(prompt),
                    raw=raw,
                    parsed_ok=True,
                    result={
                        "ok": False,
                        "error": f"tool {tool!r} is not available to the ensemble (harness/runner job); pick a read or sandbox tool",
                    },
                    ms=ms,
                    round=r,
                    tool=tool,
                )
            )
            return last_call
        args = self._prefix_paths(tool, dict(args))
        key = (tool, json.dumps(args, sort_keys=True, default=str))
        if key == last_call:
            res.nudges += 1
            res.transcript.append(
                Turn(
                    role="tooler",
                    prompt_chars=len(prompt),
                    raw=raw,
                    parsed_ok=True,
                    result={"ok": False, "error": "repeated call; use the earlier result"},
                    ms=ms,
                    round=r,
                    tool=tool,
                )
            )
            return last_call
        result = self._call(tool, args, obs)
        res.tool_calls += 1
        res.transcript.append(
            Turn(role="tooler", prompt_chars=len(prompt), raw=raw, parsed_ok=True, result=result, ms=ms, round=r, tool=tool)
        )
        return key

    def _code_round(self, obs: dict[str, Any], res: EnsembleResult, r: int, need: dict[str, Any]) -> None:
        prefix = self.cfg.sandbox_prefix
        default_path = f"{prefix}/round{r}.py"
        if need.get("path"):
            default_path = self._under_prefix(str(need["path"]))
        prompt = coder_prompt(
            need.get("request") or "",
            obs,
            res.transcript,
            sandbox_prefix=prefix,
            skills=self._skills["coder"],
            default_path=default_path,
        )
        raw, ms = self._generate("coder", prompt, obs, r)
        action = CodeAction.parse(raw, default_path)
        if action is None:
            res.nudges += 1
            res.transcript.append(
                Turn(
                    role="coder",
                    prompt_chars=len(prompt),
                    raw=raw,
                    parsed_ok=False,
                    result={"ok": False, "error": "coder: unparseable; expected a JSON action or a ```python fence"},
                    ms=ms,
                    round=r,
                )
            )
            return
        path = self._under_prefix(action.path)
        if not path.endswith(".py"):
            path += ".py"
        if action.run:
            tool = "write_and_run"
            args: dict[str, Any] = {"path": path, "content": action.content, "args": list(action.args)}
            if action.timeout_s is not None:
                args["timeout_s"] = action.timeout_s
        else:
            tool = "write_file"
            args = {"path": path, "content": action.content}
        result = self._call(tool, args, obs)
        res.code_runs += 1
        res.transcript.append(
            Turn(role="coder", prompt_chars=len(prompt), raw=raw, parsed_ok=True, result=result, ms=ms, round=r, tool=tool)
        )

    def _fill_pack(self, obs: dict[str, Any], res: EnsembleResult, r: int) -> bool:
        """Second stage: ask for the pack alone. True when we got a valid one."""
        claim = res.hypotheses[-1]["claim"]
        prompt = pack_fill_prompt(obs, claim, knob=self._knob, skills=self._skills["thinker"])
        raw, ms = self._generate("thinker", prompt, obs, r)
        pack = parse_json_object(raw)
        if pack is None:
            err: str | None = "reply was not a JSON object"
        else:
            pack, err = self._validate_pack(pack, claim=claim, obs=obs)
        if err is None and not self._template_nudged and self._is_template(pack, obs):
            self._template_nudged = True
            err = "that is the template unchanged; apply the change your claim names"
        res.transcript.append(
            Turn(
                role="thinker",
                prompt_chars=len(prompt),
                raw=raw,
                parsed_ok=pack is not None,
                result=None if err is None else {"ok": False, "error": err},
                ms=ms,
                round=r,
                tool="pack_fill",
            )
        )
        if err is not None:
            self._synthetic(res, r, f"pack rejected: {err}")
            return False
        res.pack = _strip_pack(pack)
        res.done = True
        return True

    # -------------------------------------------------------------- helpers

    def _absorb_hypotheses(self, thought: Thought, res: EnsembleResult) -> str | None:
        """Take the usable claims; say why if a stated one was thrown away.

        A claim becomes the episode title and the thing the run is judged
        against, so one with an unfilled slot in it is worse than none.
        """
        seen = {h["claim"].strip().lower() for h in res.hypotheses}
        refused: str | None = None
        added = 0
        for h in thought.hypotheses:
            claim = h["claim"]
            slot = placeholder_in_text(claim)
            if slot:
                refused = slot
                continue
            key = claim.strip().lower()
            if key in seen or len(res.hypotheses) >= MAX_HYPOTHESES:
                continue
            seen.add(key)
            res.hypotheses.append(dict(h))
            added += 1
        return refused if added == 0 else None

    def _is_template(self, pack: dict[str, Any], obs: dict[str, Any]) -> bool:
        """True when the pack's experimental content equals the nudge template."""
        tpl = pack_template(obs, knob=self._knob)
        return pack.get("config") == tpl["config"] and pack.get("data_manifest") == tpl["data_manifest"]

    def _validate_pack(
        self,
        pack: dict[str, Any],
        *,
        claim: str | None = None,
        obs: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Check a pack, returning it normalized plus the first problem found.

        Beyond the schema this asks the two questions the schema cannot: are
        the numbers actually numbers, and does the experiment test the claim
        it was written for.
        """
        # A placeholder left in place means the model copied the example
        # instead of answering it.
        stale = placeholder_fields(pack)
        if stale:
            return pack, (
                f"{', '.join(stale)} still holds a placeholder. Replace it with the value your claim names."
            )
        pack, err = coerce_config_numbers(pack)
        if err:
            return pack, err
        try:
            ArtifactPack.from_dict(_strip_pack(pack))
        except (ValueError, TypeError, KeyError) as e:
            return pack, str(e)
        # Report every remaining problem at once. Observed on Spark: told only
        # its first mistake, the thinker fixed that one, tripped the next, and
        # spent its whole nudge budget alternating between two.
        problems = []
        problems.append(placeholder_in_text(str(pack.get("hypothesis") or "")))
        if claim:
            problems.append(pack_tests_claim(pack, claim, self._knob))
            if obs is not None:
                problems.append(ungrounded_ppl(claim, obs))
        found = [p for p in problems if p]
        return pack, (" Also: ".join(found) if found else None)

    def _under_prefix(self, path: str) -> str:
        prefix = self.cfg.sandbox_prefix
        p = str(path).strip()
        while p.startswith("./"):
            p = p[2:]
        if p == prefix or p.startswith(prefix + "/"):
            return p
        return f"{prefix}/{p}" if p else prefix

    def _prefix_paths(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool in _NON_SANDBOX_PATH_TOOLS:
            return args
        for key in ("path", "dest"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                args[key] = self._under_prefix(val)
        return args

    @staticmethod
    def _synthetic(res: EnsembleResult, r: int, message: str) -> None:
        res.transcript.append(
            Turn(
                role="harness",
                prompt_chars=0,
                raw="",
                parsed_ok=True,
                result={"ok": False, "error": message},
                ms=0.0,
                round=r,
            )
        )


def _parsed_label(role: str, raw: str) -> str | None:
    """What the trace should show as the parsed outcome of one role turn."""
    if role == "tooler":
        parsed = parse_tool_call(raw)
        return parsed[0] if parsed else None
    if role == "coder":
        return "code" if CodeAction.parse(raw, "x.py") else None
    thought = Thought.parse(raw)
    if thought is None:
        return None
    if thought.pack is not None:
        return "pack"
    if thought.need:
        return f"need:{thought.need.get('kind')}"
    return "done" if thought.done else "idle"


def _strip_pack(pack: dict[str, Any]) -> dict[str, Any]:
    """Drop link-only keys the pack schema rejects (``hypothesis_id``)."""
    out = dict(pack)
    out.pop("hypothesis_id", None)
    return out


__all__ = ["Ensemble", "EnsembleConfig", "EnsembleResult", "FORBIDDEN_TOOLS"]
