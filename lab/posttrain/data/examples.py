"""In-memory training rows before they are written as JSONL."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any


@dataclass
class Example:
    """One SFT conversation, optionally with a DPO rejection for the last turn.

    ``messages`` alternate user (harness prompt) and assistant (raw completion).
    Every assistant turn of a positive example must pass the role validator.
    ``rejected`` is a completion for the last user turn that must fail it.
    """

    role: str
    messages: list[dict[str, str]]
    source: str
    kind: str
    rejected: str | None = None
    default_path: str = "ens-1/round1.py"
    obs: dict[str, Any] | None = None
    knob: str | None = None
    claim: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def user_blob(self) -> str:
        return "\n<|user|>\n".join(m["content"] for m in self.messages if m["role"] == "user")

    def prompt_hash(self) -> str:
        return sha256(self.user_blob().encode("utf-8")).hexdigest()

    def sft_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "messages": list(self.messages),
            "role": self.role,
            "source": self.source,
            "kind": self.kind,
        }
        origin = self.meta.get("origin")
        if isinstance(origin, str) and origin:
            row["origin"] = origin
        return row

    def dpo_row(self) -> dict[str, str] | None:
        if not self.rejected:
            return None
        users = [m["content"] for m in self.messages if m["role"] == "user"]
        assistants = [m["content"] for m in self.messages if m["role"] == "assistant"]
        if not users or not assistants:
            return None
        # The last user turn is the harness prompt for this completion. Earlier
        # turns are already inside that prompt's transcript.
        return {"prompt": users[-1], "chosen": assistants[-1], "rejected": self.rejected}
