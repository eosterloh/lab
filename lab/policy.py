from __future__ import annotations

from typing import Any
import random


def dummy_pack(obs: dict[str, Any], extra_config: dict[str, Any] | None = None) -> dict[str, Any]:
    cycle = int(obs.get("cycle") or 1)
    cfg = {"lr": 2e-3, "steps": 10 + cycle}
    if extra_config:
        cfg.update(extra_config)
    return {
        "hypothesis": f"cycle {cycle}: dummy overtrain of subject {obs.get('subject_checkpoint')}",
        "trainer": "dummy",
        "config": cfg,
        "data_manifest": {"sources": ["dummy://tinystories"], "unique_tokens": 0},
        "eval_suite_id": "story",
        "eval_suite_version": 0,
        "parent_checkpoint": str(obs.get("subject_checkpoint") or "subjects/tinytrain-8m"),
        "budgets": {"max_hours": 0.01, "max_steps": cfg["steps"]},
    }


class DummyPolicy:
    """Walks Eval → Research → Train with a valid dummy pack."""

    def __init__(self) -> None:
        self._pack_cycle: int | None = None

    def act(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if obs.get("halted"):
            return "halt", {"reason": "already halted"}
        phase = obs["phase"]
        if phase == "eval":
            if not obs.get("eval_ran"):
                return "run_eval", {}
            return "enter_research", {}
        if phase == "research":
            if self._pack_cycle != obs["cycle"]:
                self._pack_cycle = obs["cycle"]
                return "write_pack", {"pack": dummy_pack(obs)}
            return "enter_train", {}
        if phase == "train":
            job = obs.get("job") or {}
            if job.get("status") in {"succeeded", "failed", "cancelled"}:
                return "enter_eval", {}
            return "job_status", {}
        return "halt", {"reason": f"unknown phase {phase}"}


SEARCH_SPACE: dict[str, list[Any]] = {
    "lr": [5e-4, 1e-3, 2e-3, 3e-3],
    "steps": [8, 12, 16, 24],
}


class ScriptedPolicy:
    """Random search over a tiny dummy config space. Control baseline."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        self._pack_cycle: int | None = None

    def act(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        dummy = DummyPolicy()
        dummy._pack_cycle = self._pack_cycle
        if obs.get("phase") == "research" and self._pack_cycle != obs["cycle"]:
            self._pack_cycle = obs["cycle"]
            cfg = {k: self.rng.choice(v) for k, v in SEARCH_SPACE.items()}
            pack = dummy_pack(obs, extra_config=cfg)
            pack["hypothesis"] = f"scripted random search {cfg}"
            return "write_pack", {"pack": pack}
        name, args = dummy.act(obs)
        self._pack_cycle = dummy._pack_cycle
        return name, args


class InterleavePolicy:
    """Primary policy, with scripted control on cycles divisible by `every`."""

    def __init__(self, primary: Any, control: Any | None = None, every: int = 2) -> None:
        self.primary = primary
        self.control = control or ScriptedPolicy(seed=1)
        self.every = every
        self._cycle: int | None = None
        self._active: Any = None

    def act(self, obs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        cycle = obs["cycle"]
        if self._cycle != cycle:
            self._cycle = cycle
            self._active = self.control if cycle % self.every == 0 else self.primary
        return self._active.act(obs)
