"""JSON tool-call parsing and InferPolicy against a scripted engine (no GPU)."""

from __future__ import annotations

import json

from lab.llm_policy import SYSTEM, InferPolicy
from lab.parse import parse_tool_call
from lab.policy import dummy_pack, lab_pack
from lab.supervisor import Supervisor
from tests.helpers import arm_for_train


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


def test_infer_policy_lets_the_model_refine_its_hypothesis(sup: Supervisor) -> None:
    """A second write_hypothesis is the model's business, not a fallback pack."""
    assert sup.call("enter_research")["ok"]
    assert sup.call(
        "write_hypothesis",
        {"claim": "already set", "why": "fixture", "falsify": "job fails"},
    )["ok"]
    replies = [
        '{"tool": "write_hypothesis", "args": {"claim": "sharper", "why": "loop", "falsify": "x"}}'
    ]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64)
    name, args = policy.act(sup.observe())
    assert name == "write_hypothesis"
    assert args["claim"] == "sharper"


def _hyp_reply(claim: str = "again") -> str:
    return json.dumps(
        {"tool": "write_hypothesis", "args": {"claim": claim, "why": "loop", "falsify": "x"}}
    )


def test_hypothesis_loop_nudges_the_model_to_author_the_pack(sup: Supervisor) -> None:
    """The loop breaker must ask the model for a pack, not write one for it."""
    assert sup.call("enter_research")["ok"]
    obs = sup.observe()
    model_pack = lab_pack(obs)
    model_pack["config"]["lr"] = 7e-4
    model_pack["config"]["steps"] = 96
    replies = [
        _hyp_reply(),
        _hyp_reply(),
        _hyp_reply(),
        json.dumps({"tool": "write_pack", "args": {"pack": model_pack}}),
    ]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64, max_hyp_writes_per_cycle=2)
    assert policy.act(obs)[0] == "write_hypothesis"
    assert policy.act(obs)[0] == "write_hypothesis"
    name, args = policy.act(obs)
    assert name == "write_pack"
    assert args["pack"]["config"]["lr"] == 7e-4
    assert args["pack"]["config"]["steps"] == 96


def test_repeated_rejections_force_the_policy_forward(sup: Supervisor) -> None:
    """A capped read loop must advance, not burn the whole step budget."""
    assert sup.call("enter_research")["ok"]
    arm_for_train(sup)
    sup.cfg.research_max_tool_calls = 0
    reply = '{"tool": "list_episodes", "args": {}}'
    policy = InferPolicy(SeqEngine([reply] * 12), max_new_tokens=64, max_error_streak=3)

    names = []
    for _ in range(5):
        name, args = policy.act(sup.observe())
        result = sup.call(name, args)
        policy.observe_result(result)
        names.append(name)
    # Reads get capped, then the policy gives up on reading and trains.
    assert names[:3] == ["list_episodes"] * 3
    assert "enter_train" in names


def test_read_budget_pushes_the_model_to_write_its_own_pack(sup: Supervisor) -> None:
    """A read/read loop must end in the model's pack, not a harness pack."""
    assert sup.call("enter_research")["ok"]
    assert sup.call(
        "write_hypothesis", {"claim": "set", "why": "fixture", "falsify": "x"}
    )["ok"]
    obs = sup.observe()
    model_pack = lab_pack(obs)
    model_pack["config"]["lr"] = 4e-4
    replies = [
        '{"tool": "list_episodes", "args": {}}',
        '{"tool": "read_episode", "args": {"id": "ep-0001"}}',
        '{"tool": "list_episodes", "args": {}}',
        '{"tool": "read_episode", "args": {"id": "ep-0001"}}',
        '{"tool": "list_episodes", "args": {}}',
        json.dumps({"tool": "write_pack", "args": {"pack": model_pack}}),
    ]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64, max_reads_per_cycle=4)
    names = [policy.act(obs)[0] for _ in range(4)]
    assert names == ["list_episodes", "read_episode", "list_episodes", "read_episode"]
    name, args = policy.act(obs)
    assert name == "write_pack"
    assert args["pack"]["config"]["lr"] == 4e-4


def test_read_budget_asks_for_the_hypothesis_first_when_the_claim_is_empty(
    sup: Supervisor,
) -> None:
    """Nudging straight to write_pack would trip the train checklist."""
    assert sup.call("enter_research")["ok"]
    obs = sup.observe()
    replies = ['{"tool": "list_episodes", "args": {}}'] * 5 + [
        '{"tool": "write_hypothesis", "args": {"claim": "c", "why": "w", "falsify": "f"}}'
    ]
    policy = InferPolicy(SeqEngine(replies), max_new_tokens=64, max_reads_per_cycle=4)
    for _ in range(4):
        policy.act(obs)
    name, args = policy.act(obs)
    assert name == "write_hypothesis"
    assert args["claim"] == "c"


def test_wedged_policy_halts_instead_of_burning_the_budget(sup: Supervisor) -> None:
    class StuckPolicy:
        def act(self, obs: dict) -> tuple[str, dict]:
            return "prefetch_data", {"source": "hf:not/allowlisted"}

    assert sup.call("enter_research")["ok"]
    result = sup.run_policy(StuckPolicy(), max_cycles=1, max_steps=200, max_error_streak=5)
    assert result["halted"] is True
    assert "wedged" in result["halt_reason"]
    assert result["steps"] == 5


def test_hypothesis_loop_falls_back_only_when_the_nudge_fails(sup: Supervisor) -> None:
    assert sup.call("enter_research")["ok"]
    assert sup.call(
        "write_hypothesis", {"claim": "set", "why": "fixture", "falsify": "x"}
    )["ok"]
    obs = sup.observe()
    policy = InferPolicy(
        SeqEngine([_hyp_reply()] * 6), max_new_tokens=64, max_hyp_writes_per_cycle=2
    )
    assert policy.act(obs)[0] == "write_hypothesis"
    assert policy.act(obs)[0] == "write_hypothesis"
    name, args = policy.act(obs)
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
