"""Policy that plugs the ensemble runner into ``Supervisor.run_policy``.

Research-phase walk (against the real supervisor):

    eval:      run_eval                      (eval_ran False)
               enter_research                (eval_ran True)
    research:  trial 1, no pack:  run_research + commit -> write_hypothesis xK,
                                  write_pack xK, queue_candidates (all inside
                                  act, serial) -> return enter_train
               trial > 1 or queue non-empty (mid-trial; enter_research already
                                  loaded the next candidate) -> enter_train
               pack_hash already set (committed this cycle) -> enter_train
    train:     enter_eval when the job finished, else job_status

Each trial then loops eval -> research -> train until the queue is empty and
``transition_eval`` advances the cycle.
"""

from __future__ import annotations

from typing import Any

from lab.ensemble.roles import Roles
from lab.ensemble.runner import EnsembleRunner
from lab.tracing import Tracer

FINISHED = {"succeeded", "failed", "cancelled"}


class EnsemblePolicy:
    def __init__(
        self,
        roles: Roles,
        sup: Any,
        *,
        n: int,
        max_rounds: int,
        tracer: Tracer | None = None,
    ) -> None:
        self.roles = roles
        self.sup = sup
        self.runner = EnsembleRunner(roles, sup, n=n, max_rounds=max_rounds, tracer=tracer)
        self.last_result: dict[str, Any] | None = None
        self.last_commit: dict[str, Any] | None = None
        self.commits: list[dict[str, Any]] = []
        # Cycles whose research we already ran; a rejected enter_train must not
        # re-run the ensembles and re-queue candidates.
        self._researched: set[int] = set()

    def observe_result(self, result: dict[str, Any]) -> None:
        self.last_result = result

    def act(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if obs.get("halted"):
            return "halt", {"reason": "already halted"}
        phase = obs.get("phase")
        if phase == "eval":
            if not obs.get("eval_ran"):
                return "run_eval", {}
            return "enter_research", {}
        if phase == "research":
            return self._research(obs)
        if phase == "train":
            job = obs.get("job") or {}
            if job.get("status") in FINISHED:
                return "enter_eval", {}
            return "job_status", {}
        return "halt", {"reason": f"unknown phase {phase}"}

    def _research(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cycle = int(obs.get("cycle") or 1)
        mid_trial = bool(obs.get("candidate_queue")) or int(obs.get("trial") or 1) > 1
        if mid_trial or obs.get("pack_hash"):
            return "enter_train", {}
        if cycle in self._researched:
            # Research ran and commit could not load a pack; do not wedge on enter_train.
            return "halt", {"reason": f"ensemble commit produced no trainable pack in cycle {cycle}"}
        self._researched.add(cycle)
        results = self.runner.run_research(obs)
        self.last_commit = self.runner.commit(results, obs)
        self.commits.append(self.last_commit)
        return "enter_train", {}


__all__ = ["EnsemblePolicy"]
