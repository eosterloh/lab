from __future__ import annotations

from typing import Any
from pathlib import Path
import json
import time

from lab import actions
from lab.config import LabConfig
from lab.episodes import EpisodeStore
from lab.evals import install_frozen_eval, run_eval
from lab.gpu import GpuLock
from lab.http import Transport
from lab.hypothesis import HypothesisBoard, HypothesisStore, LiveHypothesis
from lab.jobs import Job, JobManager
from lab.notebook import Notebook
from lab.pack import ArtifactPack, PackStore
from lab.sandbox import Sandbox
from lab.scorer import Scorer
from lab.state import RunState
from lab.tracing import CHAIN, TOOL, Tracer
from lab.types import PHASE_TOOLS, Phase, ToolResult, err, ok
from lab.data_cache import list_cache

FINISHED = {"succeeded", "failed", "cancelled"}

# Tools whose phase gate lives here rather than in PHASE_TOOLS (that table is
# shared with other workstreams; see dispatch()).
RESEARCH_ONLY_EXTRA = frozenset({"queue_candidates"})


class Supervisor:
    def __init__(
        self,
        cfg: LabConfig,
        transport: Transport | None = None,
        scorer: Scorer | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.cfg = cfg
        self.scorer = scorer
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        install_frozen_eval(cfg.frozen_eval_dir)
        cfg.sandbox_dir.mkdir(parents=True, exist_ok=True)
        cfg.packs_dir.mkdir(parents=True, exist_ok=True)
        cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        cfg.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.notebook = Notebook(cfg.notebook_path, cfg.beliefs_path)
        self.episodes = EpisodeStore(cfg.episodes_dir)
        self.packs = PackStore(cfg.packs_dir)
        self.jobs = JobManager(cfg)
        self.sandbox = Sandbox(cfg, transport=transport)
        self.gpu = GpuLock(cfg.gpu_lock_path)
        self.tracer = tracer or Tracer(cfg.run_dir / "trace.jsonl")
        self.state = self._load_state()
        self.board = HypothesisBoard(cfg.run_dir / "hypotheses")
        # Single-hypothesis view (``sup.hypothesis.current`` etc.) over the board.
        self.hypothesis = HypothesisStore(
            cfg.hypothesis_path,
            self.board,
            current_id=lambda: self.state.hypothesis_id,
            cycle=lambda: self.state.cycle,
        )
        if not cfg.notebook_path.exists() or cfg.notebook_path.stat().st_size == 0:
            self.notebook.append(
                cycle=1,
                phase=Phase.EVAL.value,
                kind="init",
                text="run started",
                parent_checkpoint=cfg.subject_checkpoint,
            )

    def _load_state(self) -> RunState:
        if self.cfg.state_path.is_file():
            return RunState.from_dict(json.loads(self.cfg.state_path.read_text(encoding="utf-8")))
        return RunState()

    def save(self) -> None:
        tmp = self.cfg.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.cfg.state_path)
        self.hypothesis.save()

    @property
    def current_hypothesis(self) -> LiveHypothesis:
        """The hypothesis under test: ``state.hypothesis_id`` if set, else the
        board's active one, else a blank placeholder (no id)."""
        return self.hypothesis.current

    def _episode_cycle(self) -> int:
        # Mid-trial the cycle counter has not advanced yet, so the trial that
        # just trained belongs to state.cycle; on the last trial transition_eval
        # already bumped cycle, so it belongs to completed_cycles.
        return self.state.cycle if self.state.candidate_queue else self.state.completed_cycles

    def observe(self) -> dict[str, Any]:
        job = None
        if self.state.current_job_id:
            job = self.jobs.store.load(self.state.current_job_id).to_dict()
        research_left = None
        if self.state.research_started_at is not None:
            elapsed = time.time() - self.state.research_started_at
            research_left = max(0.0, self.cfg.research_max_seconds - elapsed)
        return {
            "cycle": self.state.cycle,
            "phase": self.state.phase,
            "halted": self.state.halted,
            "halt_reason": self.state.halt_reason,
            "pack_hash": self.state.pack_hash,
            "job": job,
            "eval_ran": self.state.eval_ran_this_phase,
            "completed_cycles": self.state.completed_cycles,
            "last_eval": self.state.last_eval,
            "last_metrics": self.state.last_metrics,
            "subject_checkpoint": self.cfg.subject_checkpoint,
            "last_checkpoint": self._last_checkpoint(),
            "episodes": self.episodes.summaries(n=8),
            "hypothesis": self.current_hypothesis.summary(),
            "hypotheses": self.board.summary(cycle=self.state.cycle),
            "trial": self.state.trial,
            "trials_planned": self.state.trials_planned,
            "candidate_queue": list(self.state.candidate_queue),
            "trial_results": list(self.state.trial_results),
            "data_cache": list_cache(self.cfg.data_cache_dir),
            "research": {
                "tool_calls": self.state.research_tool_calls,
                "max_tool_calls": self.cfg.research_max_tool_calls,
                "seconds_left": research_left,
            },
        }

    def _last_checkpoint(self) -> str | None:
        ckpts = sorted(p for p in self.cfg.checkpoints_dir.glob("job-*.pt") if p.is_file())
        if ckpts:
            return str(ckpts[-1])
        metrics = self.state.last_metrics or {}
        for key in ("run_checkpoint", "checkpoint"):
            raw = metrics.get(key)
            if raw and Path(str(raw)).is_file():
                return str(raw)
        return None

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.dispatch(name, args or {})
        self._log_tool(name, result)
        self.save()
        return result.to_dict()

    def _log_tool(self, name: str, result: ToolResult) -> None:
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cycle": self.state.cycle,
            "trial": self.state.trial,
            "phase": self.state.phase,
            "tool": name,
            "ok": result.ok,
            "error": result.error,
        }
        path = self.cfg.run_dir / "tools.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        flag = "ok" if result.ok else f"err={result.error}"
        where = f"c{self.state.cycle}"
        if self.state.trials_planned > 1:
            where += f".t{self.state.trial}"
        print(f"[tool] {self.state.phase} {where} {name} {flag}", flush=True)

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        if self.state.halted and name not in {
            "read_notebook",
            "list_episodes",
            "read_episode",
            "read_hypothesis",
            "list_hypotheses",
        }:
            return err("run is halted", reason=self.state.halt_reason)
        phase = self.state.phase_enum()
        allowed = PHASE_TOOLS[phase]
        if phase is Phase.RESEARCH:
            # queue_candidates is research-only; gated here instead of in
            # PHASE_TOOLS so this workstream does not edit the shared table.
            allowed = allowed | RESEARCH_ONLY_EXTRA
        if name not in allowed:
            return err(f"tool {name!r} is not allowed in phase {phase.value}")
        if name not in actions.REGISTRY:
            return err(f"unknown tool {name!r}")
        if phase is Phase.RESEARCH and name not in {"enter_train", "halt"}:
            cap = self._research_cap()
            if not cap.ok:
                return cap
            self.state.research_tool_calls += 1
        try:
            return actions.REGISTRY[name](self, args)
        except Exception as e:
            return err(str(e))

    def _research_cap(self) -> ToolResult:
        if self.state.research_tool_calls >= self.cfg.research_max_tool_calls:
            return err("research tool-call cap reached")
        if self.state.research_started_at is None:
            return ok()
        if time.time() - self.state.research_started_at > self.cfg.research_max_seconds:
            return err("research wall-clock cap reached")
        return ok()

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason
        if self.gpu.held:
            self.gpu.release()
            self.state.gpu_held = False
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="halt",
            text=reason,
        )
        self.save()

    def open_hypothesis(
        self,
        *,
        claim: str,
        why: str = "",
        falsify: str = "",
        source: str = "policy",
        status: str | None = None,
    ) -> LiveHypothesis:
        """Open a new hypothesis for this cycle and make it the current one."""
        hyp = self.board.open(
            claim=claim, why=why, falsify=falsify, cycle=self.state.cycle, source=source
        )
        if status is not None:
            hyp = self.board.update(hyp.id, status=status)
        self.state.hypothesis_id = hyp.id
        # Single-hypothesis compatibility: a pack written earlier in this
        # research phase (before any claim) belongs to the claim that follows.
        if (
            self.state.pack_hash
            and self.state.phase_enum() is Phase.RESEARCH
            and not self.state.mid_trials()
            and self.packs.exists(self.state.pack_hash)
        ):
            self.board.link_pack(hyp.id, self.state.pack_hash)
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="hypothesis",
            hypothesis=hyp.claim,
            text=f"{hyp.id} ({hyp.source}): {hyp.claim}",
        )
        return hyp

    def save_pack(self, pack: ArtifactPack, hypothesis_id: str | None = None) -> str:
        if hypothesis_id and not self.board.has(hypothesis_id):
            raise KeyError(f"unknown hypothesis {hypothesis_id!r}")
        digest = self.packs.save(pack)
        self.state.pack_hash = digest
        hyp_id = hypothesis_id or self.current_hypothesis.id or None
        if hyp_id:
            self.board.link_pack(hyp_id, digest)
            self.state.hypothesis_id = hyp_id
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="pack",
            pack_hash=digest,
            hypothesis=pack.hypothesis,
            hypothesis_id=hyp_id,
            text=f"pack {digest[:12]}",
        )
        return digest

    def queue_candidates(self, candidates: list[dict[str, Any]]) -> ToolResult:
        """Plan this cycle's trials: the first candidate is loaded now, the rest
        are queued and loaded one per mid-trial enter_research."""
        if self.state.phase_enum() is not Phase.RESEARCH:
            return err("queue_candidates is only allowed in research")
        if not isinstance(candidates, list):
            return err("candidates must be a list of {hypothesis_id, pack_hash}")
        loaded: list[tuple[str, str]] = []
        seen: set[str] = set()
        for raw in candidates:
            if not isinstance(raw, dict):
                return err("each candidate must be an object {hypothesis_id, pack_hash}")
            hyp_id = str(raw.get("hypothesis_id") or "")
            pack_hash = str(raw.get("pack_hash") or "")
            if not hyp_id or not pack_hash:
                return err("each candidate needs hypothesis_id and pack_hash")
            if not self.board.has(hyp_id):
                return err(f"unknown hypothesis {hyp_id!r}")
            if not self.packs.exists(pack_hash):
                return err(f"unknown pack {pack_hash!r}")
            if self.packs.load(pack_hash).budgets.max_hours > self.cfg.train_max_hours:
                return err(f"pack {pack_hash[:12]} budget exceeds harness train_max_hours")
            hyp = self.board.get(hyp_id)
            if not hyp.is_live():
                return err(f"hypothesis {hyp_id} is {hyp.status}")
            if not hyp.claim.strip():
                return err(f"hypothesis {hyp_id} has no claim")
            if hyp_id in seen:
                return err(f"hypothesis {hyp_id} listed twice")
            seen.add(hyp_id)
            loaded.append((hyp_id, pack_hash))
        if not loaded:
            # Escape hatch: drop any remaining candidates; the cycle ends after
            # the trial currently loaded.
            self.state.candidate_queue = []
            self.state.trials_planned = max(1, self.state.trial)
            return ok(trial=self.state.trial, trials_planned=self.state.trials_planned, candidate_queue=[])
        for hyp_id, pack_hash in loaded:
            self.board.link_pack(hyp_id, pack_hash)
        first_hyp, first_pack = loaded[0]
        self.state.candidate_queue = [
            {"hypothesis_id": h, "pack_hash": p} for h, p in loaded[1:]
        ]
        # Re-queueing mid-cycle keeps the trials already sealed in the count.
        self.state.trials_planned = (self.state.trial - 1) + len(loaded)
        self.state.hypothesis_id = first_hyp
        self.state.pack_hash = first_pack
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="candidates",
            pack_hash=first_pack,
            hypothesis_id=first_hyp,
            candidates=[{"hypothesis_id": h, "pack_hash": p} for h, p in loaded],
            text=f"queued {len(loaded)} candidate(s); trial {self.state.trial}/{self.state.trials_planned} is {first_hyp}",
        )
        return ok(
            trial=self.state.trial,
            trials_planned=self.state.trials_planned,
            current={"hypothesis_id": first_hyp, "pack_hash": first_pack},
            candidate_queue=list(self.state.candidate_queue),
        )

    def _best_prior_confirm(self) -> float | None:
        numbered = [
            float(r["confirm_ppl"])
            for r in self.episodes.summaries(n=10_000)
            if r.get("confirm_ppl") is not None
        ]
        return min(numbered) if numbered else None

    def run_eval(self) -> dict[str, Any]:
        pack = self.packs.load(self.state.pack_hash) if self.state.pack_hash else None
        result = run_eval(self.cfg, pack, self.state.last_metrics, scorer=self.scorer)
        self.state.last_eval = result
        self.state.eval_ran_this_phase = True
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="eval",
            pack_hash=self.state.pack_hash,
            metrics=result,
            text=(
                f"eval confirm_ppl={result.get('confirm_ppl')} "
                f"val_ppl={(result.get('loss') or {}).get('val_ppl')} "
                f"backend={result.get('backend')}"
            ),
        )
        job = self.jobs.store.load(self.state.current_job_id) if self.state.current_job_id else None
        if job is not None and job.status in FINISHED and pack is not None:
            self._seal_trial(pack, job, result)
        return result

    def _seal_trial(self, pack: ArtifactPack, job: Job, result: dict[str, Any]) -> None:
        """Seal the episode for the finished job and link it to its hypothesis."""
        already = self.episodes.exists_for_job(job.id)
        hyp_id = self.state.hypothesis_id if self.board.has(self.state.hypothesis_id) else None
        best_before = self._best_prior_confirm()
        ep = self.episodes.record(
            cycle=self._episode_cycle(),
            pack=pack,
            pack_hash=self.state.pack_hash or "",
            job=job,
            ev=result,
            hypothesis_id=hyp_id,
            trial=self.state.trial,
        )
        if already:
            return
        self.notebook.set_beliefs(self.episodes.beliefs_markdown())
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="episode",
            pack_hash=self.state.pack_hash,
            hypothesis_id=hyp_id,
            trial=self.state.trial,
            text=f"sealed {ep['id']}: {ep['title']}",
        )
        confirm = result.get("confirm_ppl")
        supporting = (
            job.status == "succeeded"
            and confirm is not None
            and (best_before is None or float(confirm) < best_before)
        )
        if hyp_id:
            self.board.link_episode(hyp_id, ep["id"], supporting=supporting)
            if result.get("confirm_source") and result.get("confirm_source") != "trainer_val_proxy":
                self.board.mark(hyp_id, "holdout_not_proxy", True)
        self.state.trial_results.append(
            {
                "trial": self.state.trial,
                "hypothesis_id": hyp_id,
                "pack_hash": self.state.pack_hash,
                "job_id": job.id,
                "job_status": job.status,
                "episode_id": ep["id"],
                "confirm_ppl": confirm,
                "supporting": supporting,
            }
        )
        if not self.state.candidate_queue:
            # Last trial of the cycle: summarise all of them for the notebook.
            # (state.trial_results itself stays visible until the next cycle's
            # enter_research resets it.)
            self._write_cycle_summary()

    def _write_cycle_summary(self) -> None:
        results = list(self.state.trial_results)
        scored = [r for r in results if r.get("confirm_ppl") is not None]
        best = min(scored, key=lambda r: float(r["confirm_ppl"])) if scored else None
        self.notebook.append(
            cycle=self._episode_cycle(),
            phase=self.state.phase,
            kind="cycle_summary",
            trials_planned=self.state.trials_planned,
            trials=results,
            best_hypothesis_id=best["hypothesis_id"] if best else None,
            best_confirm_ppl=best["confirm_ppl"] if best else None,
            text=(
                f"cycle {self._episode_cycle()}: {len(results)} trial(s), "
                f"best confirm_ppl={best['confirm_ppl'] if best else None} "
                f"({best['hypothesis_id'] if best else 'n/a'})"
            ),
        )

    def transition_research(self) -> ToolResult:
        # Mid-trial research is distinguished by a non-empty candidate_queue:
        # queue_candidates loads the first candidate and queues the rest, and
        # each enter_research after a sealed trial pops the next one. The board
        # is not rolled and the cycle's hypotheses stay live. An empty queue at
        # enter_research means a new cycle: retire earlier hypotheses and reset
        # the trial counters.
        self.state.phase = Phase.RESEARCH.value
        self.state.research_tool_calls = 0
        self.state.research_started_at = time.time()
        self.state.eval_ran_this_phase = False
        if self.state.candidate_queue:
            nxt = self.state.candidate_queue.pop(0)
            self.state.trial += 1
            self.state.hypothesis_id = str(nxt["hypothesis_id"])
            self.state.pack_hash = str(nxt["pack_hash"])
            text = (
                f"entered research (trial {self.state.trial}/{self.state.trials_planned}: "
                f"{self.state.hypothesis_id})"
            )
        else:
            self.state.trial = 1
            self.state.trials_planned = 1
            self.state.trial_results = []
            self.state.pack_hash = None
            self.state.hypothesis_id = None
            killed = self.board.roll_cycle(self.state.cycle)
            text = "entered research"
            if killed:
                text += f"; killed untested {', '.join(killed)}"
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="phase",
            pack_hash=self.state.pack_hash,
            hypothesis_id=self.state.hypothesis_id,
            text=text,
        )
        return ok(
            phase=self.state.phase,
            cycle=self.state.cycle,
            trial=self.state.trial,
            trials_planned=self.state.trials_planned,
            hypothesis_id=self.state.hypothesis_id,
        )

    def transition_train(self) -> ToolResult:
        if not self.state.pack_hash or not self.packs.exists(self.state.pack_hash):
            return err("write_pack before enter_train")
        open_train = self.current_hypothesis.open_for("train")
        if open_train:
            return err(
                "hypothesis checklist blocks train",
                open=[c.id for c in open_train],
            )
        pack = self.packs.load(self.state.pack_hash)
        if pack.budgets.max_hours > self.cfg.train_max_hours:
            return err("pack budget exceeds harness train_max_hours")
        if not self.gpu.acquire(blocking=False):
            return err("gpu lock held")
        self.state.gpu_held = True
        job = self.jobs.submit(pack, self.state.pack_hash)
        self.state.current_job_id = job.id
        self.state.last_metrics = job.metrics or None
        self.state.phase = Phase.TRAIN.value
        self.state.research_started_at = None
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="job",
            pack_hash=self.state.pack_hash,
            hypothesis_id=self.state.hypothesis_id,
            trial=self.state.trial,
            metrics=job.metrics,
            text=f"job {job.id} {job.status}",
        )
        self.save()
        return ok(phase=self.state.phase, job=job.to_dict(), trial=self.state.trial)

    def poll_job(self) -> Job | None:
        if not self.state.current_job_id:
            return None
        job = self.jobs.poll(self.jobs.store.load(self.state.current_job_id))
        if job.metrics:
            self.state.last_metrics = job.metrics
        return job

    def cancel_job(self) -> Job | None:
        if not self.state.current_job_id:
            return None
        job = self.jobs.cancel(self.jobs.store.load(self.state.current_job_id))
        return job

    def transition_eval(self) -> ToolResult:
        job = self.poll_job()
        if job and job.status == "running":
            return err("job still running")
        if self.gpu.held:
            self.gpu.release()
            self.state.gpu_held = False
        text = "entered eval"
        if job and job.status in FINISHED:
            if self.state.candidate_queue:
                # More candidates queued: the cycle stays open. pack_hash and
                # hypothesis_id are left pointing at this trial so run_eval
                # seals its episode; the next enter_research loads the next one.
                text = (
                    f"entered eval (trial {self.state.trial}/{self.state.trials_planned}, "
                    f"{len(self.state.candidate_queue)} queued)"
                )
            else:
                self.state.completed_cycles += 1
                self.state.cycle += 1
        self.state.phase = Phase.EVAL.value
        self.state.eval_ran_this_phase = False
        self.state.research_started_at = None
        self.state.research_tool_calls = 0
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="phase",
            pack_hash=self.state.pack_hash,
            hypothesis_id=self.state.hypothesis_id,
            text=text,
        )
        return ok(
            phase=self.state.phase,
            cycle=self.state.cycle,
            completed_cycles=self.state.completed_cycles,
            trial=self.state.trial,
            trials_planned=self.state.trials_planned,
            candidates_left=len(self.state.candidate_queue),
        )

    def run_policy(
        self,
        policy: Any,
        max_cycles: int = 1,
        max_steps: int = 200,
        max_error_streak: int = 12,
    ) -> dict[str, Any]:
        steps = 0
        err_streak = 0
        run_inputs = {
            "policy": type(policy).__name__,
            "max_cycles": max_cycles,
            "max_steps": max_steps,
        }
        with self.tracer.span("lab.run", kind=CHAIN, inputs=run_inputs) as run_span:
            while steps < max_steps and not self.state.halted:
                obs = self.observe()
                if (
                    self.state.completed_cycles >= max_cycles
                    and self.state.phase_enum() is Phase.EVAL
                    and self.state.eval_ran_this_phase
                ):
                    self.halt("max_cycles reached")
                    break
                self.tracer.enter_cycle(self.state.cycle)
                phase = self.state.phase
                with self.tracer.span(
                    f"step {steps + 1}",
                    kind=CHAIN,
                    metadata={"cycle": self.state.cycle, "phase": phase},
                ) as step_span:
                    name, args = policy.act(obs)
                    with self.tracer.span(
                        f"tool.{name}",
                        kind=TOOL,
                        inputs=args,
                        metadata={"cycle": self.state.cycle, "phase": phase, "tool": name},
                    ) as tool_span:
                        result = self.call(name, args)
                        tool_span.set(
                            outputs=result,
                            error=None if result.get("ok") else str(result.get("error")),
                        )
                    step_span.set(outputs={"tool": name, "ok": result.get("ok")})
                observe = getattr(policy, "observe_result", None)
                if callable(observe):
                    observe(result)
                steps += 1
                # A policy that ignores rejections would otherwise burn the whole
                # step budget in one phase, as a wedged read loop once did.
                err_streak = 0 if result.get("ok") else err_streak + 1
                if err_streak >= max_error_streak:
                    self.halt(f"policy wedged: {err_streak} consecutive tool errors")
                    break
            self.tracer.close_cycle()
            final = {"steps": steps, **self.observe()}
            run_span.set(
                outputs={
                    "steps": steps,
                    "completed_cycles": self.state.completed_cycles,
                    "halted": self.state.halted,
                    "halt_reason": self.state.halt_reason,
                    "confirm_ppl": (self.state.last_eval or {}).get("confirm_ppl"),
                }
            )
        return final
