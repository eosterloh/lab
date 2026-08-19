from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lab.types import Phase


@dataclass
class RunState:
    cycle: int = 1
    phase: str = Phase.EVAL.value
    halted: bool = False
    pack_hash: str | None = None
    current_job_id: str | None = None
    completed_cycles: int = 0
    eval_ran_this_phase: bool = False
    research_tool_calls: int = 0
    research_started_at: float | None = None
    last_eval: dict[str, Any] | None = None
    last_metrics: dict[str, Any] | None = None
    gpu_held: bool = False
    halt_reason: str | None = None

    def phase_enum(self) -> Phase:
        return Phase(self.phase)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunState:
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
