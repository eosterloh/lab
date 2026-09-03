"""JSON tool-call parsing and InferPolicy against a scripted engine (no GPU)."""

from __future__ import annotations

import json

from lab.llm_policy import SYSTEM, InferPolicy
from lab.parse import parse_tool_call
from lab.policy import dummy_pack
from lab.supervisor import Supervisor


class SeqEngine:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)

    def generate(self, prompt: str, max_new_tokens: int = 384, **kwargs) -> str:
        if not self.replies:
            return '{"tool": "halt", "args": {"reason": "out of script"}}'
        return self.replies.pop(0)


def test_nano_system_prompt_asks_for_lab_trainer_not_dummy() -> None:
    assert 'trainer "lab"' in SYSTEM
    assert '"trainer": "dummy"' not in SYSTEM
    assert "prefetch_data" in SYSTEM
    assert "hf:roneneldan/TinyStories" in SYSTEM
    assert "builtin:tiny" in SYSTEM  # mentioned as what not to use alone


def test_parse_tool_call_strips_fences_and_im_end() -> None:
    raw = 'Sure.\n```json\n{"tool": "run_eval", "args": {}}\n```\n<|im_end|>'
    assert parse_tool_call(raw) == ("run_eval", {})


def test_parse_write_hypothesis_flattens_nested_object() -> None:
    raw = json.dumps(
        {
            "tool": "write_hypothesis",
            "args": {
                "hypothesis": {
                    "claim": "continue TinyGPT overtrain",
                    "status": "testing",
                    "open_train": ["pack_ready"],
                }
            },
        }
    )
    name, args = parse_tool_call(raw)
    assert name == "write_hypothesis"
    assert args["claim"] == "continue TinyGPT overtrain"


def test_parse_salvages_truncated_write_hypothesis() -> None:
    raw = (
        '{"tool": "write_hypothesis", "args": {"hypothesis": {"claim": "continue TinyGPT overtrain", '
        '"status": "testing", "open_train": ["pack_ready"], "checklist"'
    )
    parsed = parse_tool_call(raw)
    assert parsed is not None
    name, args = parsed
    assert name == "write_hypothesis"
    assert args["claim"] == "continue TinyGPT overtrain"


def test_infer_policy_falls_back_instead_of_halting_on_garbage(sup: Supervisor) -> None:
    replies = ["not json", "still not json", "???"]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64, parse_retries=2)
    assert policy.act(sup.observe())[0] == "write_note"
    assert policy.act(sup.observe())[0] == "write_note"
    name, args = policy.act(sup.observe())
    assert name == "run_eval"
    assert args == {}


def test_infer_policy_maps_eval_write_hypothesis_to_enter_research(sup: Supervisor) -> None:
    assert sup.call("run_eval")["ok"]
    replies = [
        '{"tool": "write_hypothesis", "args": {"claim": "loop", "why": "eval", "falsify": "no train"}}'
    ]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64)
    name, args = policy.act(sup.observe())
    assert name == "enter_research"
    assert args == {}


def test_infer_policy_skips_repeat_write_hypothesis_when_claim_exists(sup: Supervisor) -> None:
    assert sup.call("enter_research")["ok"]
    assert sup.call(
        "write_hypothesis",
        {"claim": "already set", "why": "fixture", "falsify": "job fails"},
    )["ok"]
    replies = [
        '{"tool": "write_hypothesis", "args": {"claim": "again", "why": "loop", "falsify": "x"}}'
    ]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64)
    name, args = policy.act(sup.observe())
    assert name == "write_pack"
    assert args["pack"]["trainer"] == "lab"


def test_write_hypothesis_accepts_nested_payload(sup: Supervisor) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call(
        "write_hypothesis",
        {"hypothesis": {"claim": "nested claim", "why": "nano echoed state", "falsify": "job fails"}},
    )
    assert out["ok"] is True
    assert out["hypothesis"]["claim"] == "nested claim"
    pack = dummy_pack({"cycle": 1})
    name, args = parse_tool_call(json.dumps({"tool": "write_pack", **pack}))
    assert name == "write_pack"
    assert args["pack"]["trainer"] == "dummy"


def test_infer_policy_scripted_replies_complete_dummy_cycle(sup: Supervisor) -> None:
    """Same tool sequence Nano must emit, including write_hypothesis before train."""
    pack = dummy_pack({"cycle": 1, "subject_checkpoint": "subjects/tinytrain-8m"})
    replies = [
        '{"tool": "run_eval", "args": {}}',
        '{"tool": "enter_research", "args": {}}',
        '{"tool": "write_hypothesis", "args": {"claim": "smoke", "why": "loop", "falsify": "job fails"}}',
        '{"tool": "write_pack", "args": {"pack": ' + json.dumps(pack) + "}}",
        '{"tool": "enter_train", "args": {}}',
        '{"tool": "enter_eval", "args": {}}',
        '{"tool": "run_eval", "args": {}}',
    ]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64)
    result = sup.run_policy(policy, max_cycles=1, max_steps=20)
    assert result["completed_cycles"] == 1
    assert result["halted"] is True
    assert result["last_eval"]["confirm_ppl"] is not None
    assert result["hypothesis"]["open_train"] == []
