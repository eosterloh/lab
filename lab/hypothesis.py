from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


STATUSES = ("open", "testing", "killed", "closed")

DEFAULT_CHECKLIST: tuple[dict[str, Any], ...] = (
    {
        "id": "claim_written",
        "required_for": "train",
        "done": False,
        "note": "Falsifiable claim is written",
    },
    {
        "id": "pack_ready",
        "required_for": "train",
        "done": False,
        "note": "Hashed artifact pack exists",
    },
    {
        "id": "post_eval",
        "required_for": "close",
        "done": False,
        "note": "Post-train eval has run",
    },
    {
        "id": "episode_sealed",
        "required_for": "close",
        "done": False,
        "note": "Episode card written",
    },
    {
        "id": "holdout_not_proxy",
        "required_for": "promote",
        "done": False,
        "note": "confirm_ppl is frozen holdout, not trainer-val proxy",
    },
)


@dataclass
class CheckItem:
    id: str
    required_for: str
    done: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveHypothesis:
    """Single working theory. Supervisor writes; policy proposes."""

    claim: str = ""
    why: str = ""
    falsify: str = ""
    status: str = "open"
    supporting_episodes: list[str] = field(default_factory=list)
    refuting_episodes: list[str] = field(default_factory=list)
    checklist: list[CheckItem] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if not self.checklist:
            self.checklist = [CheckItem(**x) for x in DEFAULT_CHECKLIST]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "why": self.why,
            "falsify": self.falsify,
            "status": self.status,
            "supporting_episodes": list(self.supporting_episodes),
            "refuting_episodes": list(self.refuting_episodes),
            "checklist": [c.to_dict() for c in self.checklist],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiveHypothesis:
        raw_items = data.get("checklist") or DEFAULT_CHECKLIST
        items = [CheckItem(**{k: x[k] for k in ("id", "required_for", "done", "note") if k in x}) for x in raw_items]
        return cls(
            claim=str(data.get("claim") or ""),
            why=str(data.get("why") or ""),
            falsify=str(data.get("falsify") or ""),
            status=str(data.get("status") or "open"),
            supporting_episodes=list(data.get("supporting_episodes") or []),
            refuting_episodes=list(data.get("refuting_episodes") or []),
            checklist=items,
        )

    def item(self, item_id: str) -> CheckItem:
        for c in self.checklist:
            if c.id == item_id:
                return c
        raise KeyError(item_id)

    def mark(self, item_id: str, done: bool = True) -> None:
        self.item(item_id).done = done

    def open_for(self, gate: str) -> list[CheckItem]:
        return [c for c in self.checklist if c.required_for == gate and not c.done]

    def summary(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "status": self.status,
            "open_train": [c.id for c in self.open_for("train")],
            "open_close": [c.id for c in self.open_for("close")],
            "open_promote": [c.id for c in self.open_for("promote")],
            "checklist": [c.to_dict() for c in self.checklist],
        }

    def markdown(self) -> str:
        lines = [
            "# Live hypothesis",
            "",
            f"- status: `{self.status}`",
            f"- claim: {self.claim or '(none)'}",
            f"- why: {self.why or '(none)'}",
            f"- falsify: {self.falsify or '(none)'}",
            "",
            "## Checklist",
            "",
        ]
        for c in self.checklist:
            box = "[x]" if c.done else "[ ]"
            lines.append(f"- {box} `{c.id}` ({c.required_for}): {c.note}")
        lines.append("")
        return "\n".join(lines)


class HypothesisStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.md_path = path.with_suffix(".md")
        if path.is_file():
            self.current = LiveHypothesis.from_dict(json.loads(path.read_text(encoding="utf-8")))
        else:
            self.current = LiveHypothesis()
            self.save()

    def save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.current.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)
        self.md_path.write_text(self.current.markdown(), encoding="utf-8")

    def update(
        self,
        *,
        claim: str | None = None,
        why: str | None = None,
        falsify: str | None = None,
        status: str | None = None,
    ) -> LiveHypothesis:
        if claim is not None:
            self.current.claim = claim.strip()
        if why is not None:
            self.current.why = why.strip()
        if falsify is not None:
            self.current.falsify = falsify.strip()
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"status must be one of {STATUSES}")
            self.current.status = status
        if self.current.claim:
            self.current.mark("claim_written", True)
            if self.current.status == "open":
                self.current.status = "testing"
        self.save()
        return self.current
