from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from lab import actions
from lab.config import LabConfig
from lab.evals import install_frozen_eval, run_eval
from lab.gpu import GpuLock
from lab.http import Transport
from lab.jobs import Job, JobManager
from lab.notebook import Notebook
from lab.pack import ArtifactPack, PackStore
from lab.sandbox import Sandbox
from lab.state import RunState
from lab.types import PHASE_TOOLS, Phase, ToolResult, err, ok


class Supervisor:
    def __init__(self, cfg: LabConfig, transport: Transport | None = None) -> None:
        self.cfg = cfg
        cfg.run_dir.mkdir(parents=True, exist_ok=True)
        install_frozen_eval(cfg.frozen_eval_dir)
        cfg.sandbox_dir.mkdir(parents=True, exist_ok=True)
        cfg.packs_dir.mkdir(parents=True, exist_ok=True)
        cfg.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.notebook = Notebook(cfg.notebook_path, cfg.beliefs_path)
        self.packs = PackStore(cfg.packs_dir)
        self.jobs = JobManager(cfg)
        self.sandbox = Sandbox(cfg, transport=transport)
        self.gpu = GpuLock(cfg.gpu_lock_path)
        self.state = self._load_state()
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
            "research": {
                "tool_calls": self.state.research_tool_calls,
                "max_tool_calls": self.cfg.research_max_tool_calls,
                "seconds_left": research_left,
            },
        }

    def call(self, name: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.dispatch(name, args or {})
        self.save()
        return result.to_dict()

    def dispatch(self, name: str, args: dict[str, Any]) -> ToolResult:
        if self.state.halted and name != "read_notebook":
            return err("run is halted", reason=self.state.halt_reason)
        phase = self.state.phase_enum()
        allowed = PHASE_TOOLS[phase]
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

    def save_pack(self, pack: ArtifactPack) -> str:
        digest = self.packs.save(pack)
        self.state.pack_hash = digest
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="pack",
            pack_hash=digest,
            hypothesis=pack.hypothesis,
            text=f"pack {digest[:12]}",
        )
        return digest

    def run_eval(self) -> dict[str, Any]:
        pack = self.packs.load(self.state.pack_hash) if self.state.pack_hash else None
        result = run_eval(self.cfg, pack, self.state.last_metrics)
        self.state.last_eval = result
        self.state.eval_ran_this_phase = True
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="eval",
            pack_hash=self.state.pack_hash,
            metrics=result,
            text=f"eval score={result.get('score')} confirm={result.get('confirm_score')}",
        )
        return result

    def transition_research(self) -> ToolResult:
        self.state.phase = Phase.RESEARCH.value
        self.state.research_tool_calls = 0
        self.state.research_started_at = time.time()
        self.state.eval_ran_this_phase = False
        self.notebook.append(
            cycle=self.state.cycle,
            phase=self.state.phase,
            kind="phase",
            text="entered research",
        )
        return ok(phase=self.state.phase, cycle=self.state.cycle)

    def transition_train(self) -> ToolResult:
        if not self.state.pack_hash or not self.packs.exists(self.state.pack_hash):
            return err("write_pack before enter_train")
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
            metrics=job.metrics,
            text=f"job {job.id} {job.status}",
        )
        self.save()
        return ok(phase=self.state.phase, job=job.to_dict())

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
        if job and job.status in {"succeeded", "failed", "cancelled"}:
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
            text="entered eval",
        )
        return ok(phase=self.state.phase, cycle=self.state.cycle, completed_cycles=self.state.completed_cycles)

    def run_policy(self, policy: Any, max_cycles: int = 1, max_steps: int = 200) -> dict[str, Any]:
        steps = 0
        while steps < max_steps and not self.state.halted:
            obs = self.observe()
            if (
                self.state.completed_cycles >= max_cycles
                and self.state.phase_enum() is Phase.EVAL
                and self.state.eval_ran_this_phase
            ):
                self.halt("max_cycles reached")
                break
            name, args = policy.act(obs)
            self.call(name, args)
            steps += 1
        return {"steps": steps, **self.observe()}
