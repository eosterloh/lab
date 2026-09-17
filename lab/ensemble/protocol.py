"""JSON contracts between the three ensemble roles.

The thinker replies with a ``Thought``; the tooler with a ``ToolCall`` (parsed by
``lab.parse.parse_tool_call``); the coder with a ``CodeAction``. Every parser is
tolerant: fences and chat tokens are stripped, missing keys take defaults, and
``None`` means "no usable JSON at all" so the caller can re-ask once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import re

from lab.parse import parse_json_object

NEED_KINDS = ("tool", "code")

_FENCE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.DOTALL)
_ANY_FENCE = re.compile(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", re.DOTALL)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "done"}
    return bool(value)


def _clean_hypotheses(raw: Any) -> list[dict[str, str]]:
    """Keep ``{claim, why, falsify}`` rows with a non-empty claim; drop the rest."""
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, str):
            item = {"claim": item}
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim") or "").strip()
        if not claim:
            continue
        out.append(
            {
                "claim": claim,
                "why": str(item.get("why") or "").strip(),
                "falsify": str(item.get("falsify") or "").strip(),
            }
        )
    return out


def _clean_need(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, str):
        # "need": "tool" / "code" without a request is still a request.
        raw = {"kind": raw}
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or raw.get("type") or "").strip().lower()
    if kind not in NEED_KINDS:
        return None
    need: dict[str, Any] = {
        "kind": kind,
        "request": str(raw.get("request") or raw.get("prompt") or raw.get("ask") or "").strip(),
    }
    if kind == "code" and raw.get("path"):
        need["path"] = str(raw["path"])
    return need


@dataclass
class Thought:
    """Thinker output."""

    thought: str
    hypotheses: list[dict]
    need: dict | None
    pack: dict | None
    done: bool

    @classmethod
    def parse(cls, raw: str) -> Thought | None:
        data = parse_json_object(raw)
        if data is None:
            return None
        pack = data.get("pack")
        return cls(
            thought=str(data.get("thought") or data.get("reasoning") or "").strip(),
            hypotheses=_clean_hypotheses(data.get("hypotheses")),
            need=_clean_need(data.get("need")),
            pack=dict(pack) if isinstance(pack, dict) and pack else None,
            done=_as_bool(data.get("done", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "thought": self.thought,
            "hypotheses": list(self.hypotheses),
            "need": self.need,
            "pack": self.pack,
            "done": self.done,
        }


@dataclass
class ToolCall:
    """Tooler output; parse with ``lab.parse.parse_tool_call``."""

    tool: str
    args: dict


@dataclass
class CodeAction:
    """Coder output: one script, written (and by default run) in the sandbox."""

    path: str
    content: str
    run: bool = True
    args: list[str] = field(default_factory=list)
    timeout_s: float | None = None

    @classmethod
    def parse(cls, raw: str, default_path: str) -> CodeAction | None:
        """Accept ``{"path","content","run","args"}`` JSON or a bare python fence."""
        if not isinstance(raw, str) or not raw.strip():
            return None
        data = parse_json_object(raw)
        if isinstance(data, dict) and isinstance(data.get("content"), str) and data["content"].strip():
            args_raw = data.get("args") or []
            if not isinstance(args_raw, list):
                args_raw = []
            timeout_raw = data.get("timeout_s")
            timeout: float | None = None
            if isinstance(timeout_raw, (int, float)) and not isinstance(timeout_raw, bool) and timeout_raw > 0:
                timeout = float(timeout_raw)
            path = str(data.get("path") or "").strip() or default_path
            return cls(
                path=path,
                content=data["content"],
                run=_as_bool(data.get("run", True)),
                args=[str(x) for x in args_raw],
                timeout_s=timeout,
            )
        fence = _FENCE.search(raw) or _ANY_FENCE.search(raw)
        if fence and fence.group(1).strip():
            return cls(path=default_path, content=fence.group(1), run=True)
        return None


@dataclass
class Turn:
    """One entry in an ensemble transcript (a role reply or a synthetic harness note)."""

    role: str
    prompt_chars: int
    raw: str
    parsed_ok: bool
    result: dict | None
    ms: float
    round: int
    tool: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "round": self.round,
            "tool": self.tool,
            "prompt_chars": self.prompt_chars,
            "raw": self.raw,
            "parsed_ok": self.parsed_ok,
            "result": self.result,
            "ms": self.ms,
        }


__all__ = ["CodeAction", "NEED_KINDS", "Thought", "ToolCall", "Turn"]
