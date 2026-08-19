from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    EVAL = "eval"
    RESEARCH = "research"
    TRAIN = "train"


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"ok": self.ok, **self.data}
        if self.error:
            out["error"] = self.error
        return out


def ok(**data: Any) -> ToolResult:
    return ToolResult(ok=True, data=data)


def err(message: str, **data: Any) -> ToolResult:
    return ToolResult(ok=False, error=message, data=data)


ALWAYS_TOOLS = frozenset(
    {"read_notebook", "write_note", "write_beliefs", "halt"}
)

PHASE_TOOLS: dict[Phase, frozenset[str]] = {
    Phase.EVAL: ALWAYS_TOOLS
    | {
        "run_eval",
        "read_metrics",
        "read_samples",
        "enter_research",
    },
    Phase.RESEARCH: ALWAYS_TOOLS
    | {
        "list_files",
        "read_file",
        "write_file",
        "exec",
        "web_fetch",
        "web_search",
        "write_pack",
        "enter_train",
    },
    Phase.TRAIN: ALWAYS_TOOLS
    | {
        "job_status",
        "cancel_job",
        "enter_eval",
    },
}
