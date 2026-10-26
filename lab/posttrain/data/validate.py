"""Valid completions only: every assistant turn, and every DPO rejection fails."""

from __future__ import annotations

from lab.posttrain.data.examples import Example
from lab.posttrain.data.harness import (
    validate_coder_completion,
    validate_foreign_completion,
    validate_thinker_completion,
    validate_tool_completion,
)


def validate_completion(example: Example, raw: str, *, execute: bool) -> str | None:
    if example.role == "tooler":
        if example.source == "foreign_tools":
            tools = example.meta.get("tools")
            return validate_foreign_completion(raw, tools if isinstance(tools, list) else [])
        return validate_tool_completion(raw)
    if example.role == "thinker":
        return validate_thinker_completion(raw, obs=example.obs, knob=example.knob, claim=example.claim)
    if example.role == "coder":
        return validate_coder_completion(raw, default_path=example.default_path, execute=execute)
    return "unknown_tool"


def validate_example(example: Example, *, execute: bool = True) -> str | None:
    """None when every assistant turn is a positive and any rejection is not."""
    assistants = [m["content"] for m in example.messages if m["role"] == "assistant"]
    users = [m for m in example.messages if m["role"] == "user"]
    if not assistants or len(assistants) != len(users):
        return "parse_failed"
    for raw in assistants:
        reason = validate_completion(example, raw, execute=execute)
        if reason:
            return reason
    if example.rejected is not None:
        reason = validate_completion(example, example.rejected, execute=False)
        if reason is None:
            return "negative_passed"
    return None
