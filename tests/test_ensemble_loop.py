"""One ensemble: thinker -> (tooler | coder) -> thinker, driven by scripted roles."""

from __future__ import annotations

import json
import threading

import pytest

from lab.config import LabConfig
from lab.ensemble.ensemble import FORBIDDEN_TOOLS, Ensemble, EnsembleConfig
from lab.ensemble.prompts import (
    TOOL_SCHEMAS,
    VARIANT_HINTS,
    _schema_line,
    obs_digest,
    thinker_prompt,
    tool_catalog,
)
from lab.ensemble.protocol import CodeAction, Thought
from lab.ensemble.roles import Roles
from lab.parse import parse_json_object
from lab.policy import dummy_pack
from lab.supervisor import Supervisor
from lab.types import PHASE_TOOLS, Phase

ENV_VARS = ("LAB_THINKER_MODEL", "LAB_TOOLER_MODEL", "LAB_CODER_MODEL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _pack(obs: dict, steps: int = 12) -> dict:
    return dummy_pack(obs, extra_config={"steps": steps})


def _thought(**kw) -> str:
    base = {"thought": "", "hypotheses": [], "need": None, "pack": None, "done": False}
    base.update(kw)
    return json.dumps(base)


def _hyp(claim: str = "more steps lower confirm_ppl") -> dict:
    return {"claim": claim, "why": "undertrained", "falsify": "confirm_ppl does not drop"}


def _ens(sup: Supervisor, roles: Roles, **cfg_kw) -> Ensemble:
    cfg_kw.setdefault("variant_hint", "training steps")
    cfg = EnsembleConfig(index=1, **cfg_kw)
    return Ensemble(roles, sup, cfg, call_lock=threading.Lock(), tracer=sup.tracer)


def _research(sup: Supervisor) -> dict:
    assert sup.call("enter_research")["ok"]
    return sup.observe()


# ------------------------------------------------------------------ protocol


def test_tool_schemas_cover_every_research_tool() -> None:
    missing = sorted(PHASE_TOOLS[Phase.RESEARCH] - set(TOOL_SCHEMAS))
    assert missing == []
    assert "queue_candidates" in TOOL_SCHEMAS
    catalog = tool_catalog(Phase.RESEARCH)
    for name in PHASE_TOOLS[Phase.RESEARCH]:
        assert f"{name} {{" in catalog
    assert "write_pack" not in tool_catalog(Phase.RESEARCH, exclude=FORBIDDEN_TOOLS)
    assert _schema_line("unknown_tool") == "unknown_tool {}"


def test_parse_json_object_is_tolerant() -> None:
    assert parse_json_object('Sure!\n```json\n{"a": 1}\n```<|im_end|>') == {"a": 1}
    assert parse_json_object('text {"a": {"b": 2}} trailing') == {"a": {"b": 2}}
    assert parse_json_object("no json here") is None
    assert parse_json_object("[1, 2]") is None


def test_thought_parse_defaults_and_need_shapes() -> None:
    t = Thought.parse('{"thought": "x"}')
    assert t is not None
    assert t.hypotheses == [] and t.need is None and t.pack is None and t.done is False
    t = Thought.parse(json.dumps({"need": {"kind": "code", "request": "count", "path": "a.py"}, "hypotheses": [_hyp(), {"why": "no claim"}]}))
    assert t.need == {"kind": "code", "request": "count", "path": "a.py"}
    assert len(t.hypotheses) == 1
    assert Thought.parse("garbage") is None
    assert Thought.parse('{"done": true}').done is True


def test_code_action_parse_accepts_json_and_python_fence() -> None:
    act = CodeAction.parse(json.dumps({"path": "x.py", "content": "print(1)", "run": False, "args": ["a"]}), "d.py")
    assert act == CodeAction(path="x.py", content="print(1)", run=False, args=["a"], timeout_s=None)
    act = CodeAction.parse("here you go\n```python\nprint(2)\n```\n", "ens-1/round1.py")
    assert act is not None
    assert act.path == "ens-1/round1.py" and act.run is True and "print(2)" in act.content
    assert CodeAction.parse("nothing useful", "d.py") is None


def test_prompts_mention_angle_and_digest_fields(sup: Supervisor) -> None:
    obs = sup.observe()
    prompt = thinker_prompt(obs, [], variant_hint=VARIANT_HINTS[0], skills="SKILLS")
    assert "Your angle: learning rate" in prompt
    assert '"trainer": "lab"' in prompt
    assert "SKILLS" in prompt
    digest = obs_digest(obs)
    assert "cycle=1" in digest and "last_checkpoint" in digest and "research_budget" in digest


# ---------------------------------------------------------------------- loop


def test_thinker_asks_tool_then_emits_pack(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=[
            _thought(need={"kind": "tool", "request": "list episodes"}, hypotheses=[_hyp()]),
            _thought(pack=_pack(obs), hypotheses=[_hyp("dup claim"), _hyp()]),
        ],
        tooler=['{"tool": "list_episodes", "args": {"n": 5}}'],
        coder=[],
    )
    res = _ens(sup, roles).run(obs)
    assert res.error is None, res.error
    assert res.pack is not None and res.pack["trainer"] == "dummy"
    assert res.done is True
    assert res.tool_calls == 1 and res.code_runs == 0
    assert [t.role for t in res.transcript] == ["thinker", "tooler", "thinker"]
    assert res.transcript[1].tool == "list_episodes"
    assert res.transcript[1].result["ok"] is True
    assert [h["claim"] for h in res.hypotheses] == ["more steps lower confirm_ppl", "dup claim"]
    # The board and the pack store are untouched: committing is the runner's job.
    assert sup.observe()["pack_hash"] is None
    assert sup.board.list() == []
    # The second thinker prompt carried the tool result.
    prompts = roles.backend("thinker").prompts
    assert len(prompts) == 2 and "list_episodes" in prompts[1]


def test_tooler_paths_are_jailed_under_the_sandbox_prefix(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=[_thought(need={"kind": "tool", "request": "save notes"}), _thought(done=True)],
        tooler=['{"tool": "write_file", "args": {"path": "notes.txt", "content": "hi"}}'],
        coder=[],
    )
    res = _ens(sup, roles).run(obs)
    assert res.error is None
    assert res.transcript[1].result["ok"] is True
    assert (sup.cfg.sandbox_dir / "ens-1" / "notes.txt").read_text() == "hi"
    assert "ens-1/" in roles.backend("tooler").prompts[0]


def test_thinker_asks_code_and_coder_fence_is_written_and_run(sup: Supervisor) -> None:
    obs = _research(sup)
    script = "import json\nprint(json.dumps({'answer': 42}))\n"
    roles = Roles.scripted(
        thinker=[_thought(need={"kind": "code", "request": "compute the answer"}), _thought(pack=_pack(obs))],
        tooler=[],
        coder=[f"```python\n{script}```"],
    )
    res = _ens(sup, roles).run(obs)
    assert res.error is None
    assert res.code_runs == 1 and res.tool_calls == 0
    turn = res.transcript[1]
    assert turn.role == "coder" and turn.tool == "write_and_run"
    assert turn.result["ok"] is True
    assert json.loads(turn.result["stdout"].strip().splitlines()[-1]) == {"answer": 42}
    assert (sup.cfg.sandbox_dir / "ens-1" / "round1.py").read_text() == script
    assert res.pack is not None


def test_coder_json_with_run_false_only_writes(sup: Supervisor) -> None:
    obs = _research(sup)
    action = {"path": "helper.py", "content": "X = 1\n", "run": False}
    roles = Roles.scripted(
        thinker=[_thought(need={"kind": "code", "request": "write helper"}), _thought(done=True)],
        tooler=[],
        coder=[json.dumps(action)],
    )
    res = _ens(sup, roles).run(obs)
    turn = res.transcript[1]
    assert turn.tool == "write_file" and turn.result["ok"] is True
    assert (sup.cfg.sandbox_dir / "ens-1" / "helper.py").read_text() == "X = 1\n"
    names = [json.loads(l)["tool"] for l in (sup.cfg.run_dir / "tools.jsonl").read_text().splitlines()]
    assert "write_and_run" not in names and "run_python" not in names


def test_forbidden_tool_is_rejected_without_calling_the_supervisor(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=[_thought(need={"kind": "tool", "request": "freeze it"}), _thought(done=True)],
        tooler=[json.dumps({"tool": "write_pack", "args": {"pack": _pack(obs)}})],
        coder=[],
    )
    res = _ens(sup, roles).run(obs)
    turn = res.transcript[1]
    assert turn.role == "tooler" and turn.tool == "write_pack"
    assert turn.result["ok"] is False and "not available" in turn.result["error"]
    assert res.tool_calls == 0 and res.nudges == 1
    assert sup.observe()["pack_hash"] is None
    names = [json.loads(l)["tool"] for l in (sup.cfg.run_dir / "tools.jsonl").read_text().splitlines()]
    assert "write_pack" not in names


def test_invalid_pack_is_fed_back_then_valid_pack_succeeds(sup: Supervisor) -> None:
    obs = _research(sup)
    bad = _pack(obs)
    bad["trainer"] = "nope"
    roles = Roles.scripted(thinker=[_thought(pack=bad), _thought(pack=_pack(obs))], tooler=[], coder=[])
    res = _ens(sup, roles).run(obs)
    assert res.error is None and res.pack is not None
    assert res.rounds == 2 and res.nudges == 1
    synthetic = res.transcript[1]
    assert synthetic.role == "harness" and "trainer" in synthetic.result["error"]
    assert "pack rejected" in roles.backend("thinker").prompts[1]


def test_a_thinker_with_no_hypothesis_still_gets_the_template_nudge(sup: Supervisor) -> None:
    """With a hypothesis the harness runs a focused fill turn; with nothing at
    all there is nothing to fill from, so the nudge carries the template."""
    obs = _research(sup)
    roles = Roles.scripted(thinker=[_thought(), _thought(pack=_pack(obs))], tooler=[], coder=[])
    res = _ens(sup, roles).run(obs)
    assert res.error is None and res.pack is not None and res.nudges == 1

    nudge = res.transcript[1].result["error"]
    assert "emitted no pack" in nudge
    template = parse_json_object(nudge[nudge.index("{") :])
    assert template["parent_checkpoint"] == (obs["last_checkpoint"] or obs["subject_checkpoint"])
    assert nudge in roles.backend("thinker").prompts[1]


def test_the_fill_template_carries_the_runs_real_parent_checkpoint(sup: Supervisor) -> None:
    from lab.ensemble.prompts import pack_fill_prompt

    obs = _research(sup)
    prompt = pack_fill_prompt(obs, "steps 64->128", knob="steps")
    template = parse_json_object(prompt[prompt.index("{") :])
    assert template["parent_checkpoint"] == (obs["last_checkpoint"] or obs["subject_checkpoint"])
    assert template["hypothesis"] == "steps 64->128"


def test_the_angles_knob_is_a_placeholder_the_model_must_fill(sup: Supervisor) -> None:
    """Observed on Spark: Llama-3.2-3B copied the documented example pack while
    claiming a different lr, in both cycles and both ensembles. The example it
    is shown must therefore be unusable until the angle's knob is filled in."""
    from lab.ensemble.prompts import PLACEHOLDER, knob_for_hint, pack_template

    obs = _research(sup)
    assert knob_for_hint("learning rate") == "lr"
    assert knob_for_hint("data mix") is None

    prompt = thinker_prompt(obs, [], variant_hint="learning rate", skills="")
    assert PLACEHOLDER in prompt and '"config.lr" above is a placeholder' in prompt
    # the steps angle masks a different key, so two ensembles cannot converge
    # on one pack by copying the same example
    assert pack_template(obs, knob="steps")["config"] != pack_template(obs, knob="lr")["config"]

    copied = pack_template(obs, "lr 1.5e-3 lowers confirm_ppl", knob="lr")
    filled = json.loads(json.dumps(copied))
    filled["config"]["lr"] = 0.0015
    roles = Roles.scripted(thinker=[_thought(pack=copied), _thought(pack=filled)], tooler=[], coder=[])
    cfg = EnsembleConfig(index=1, variant_hint="learning rate")
    res = Ensemble(roles, sup, cfg, call_lock=threading.Lock(), tracer=sup.tracer).run(obs)

    assert res.error is None and res.pack["config"]["lr"] == 0.0015
    assert "config.lr still holds a placeholder" in res.transcript[1].result["error"]


def test_a_hypothesis_without_a_pack_triggers_a_focused_fill_turn(sup: Supervisor) -> None:
    """Observed on Spark: Llama-3.2-3B re-emitted its thought with pack null
    for three straight rounds because the ask was buried in the transcript.
    The second stage asks for the pack alone and parses a bare object."""
    from lab.ensemble.prompts import pack_template

    obs = _research(sup)
    claim = "steps 64->128, vs ep-0001 (confirm_ppl 198.3); expect lower"
    filled = pack_template(obs, claim, knob="steps")
    filled["config"]["steps"] = 128
    roles = Roles.scripted(
        thinker=[_thought(hypotheses=[_hyp(claim)]), json.dumps(filled)], tooler=[], coder=[]
    )
    res = _ens(sup, roles).run(obs)

    assert res.error is None and res.pack["config"]["steps"] == 128
    fill = res.transcript[-1]
    assert fill.role == "thinker" and fill.tool == "pack_fill"
    prompt = roles.backend("thinker").prompts[1]
    assert prompt.startswith("Fill in one ArtifactPack")
    assert claim in prompt and '"config.steps" to the number' in prompt
    # the fill turn asks for a bare object, not another Thought
    assert '"hypotheses"' not in prompt and "ONLY the pack" in prompt


def test_a_fill_turn_that_keeps_the_placeholder_is_refused(sup: Supervisor) -> None:
    from lab.ensemble.prompts import pack_template

    obs = _research(sup)
    copied = pack_template(obs, "steps 64->128", knob="steps")
    # each round is one Thought call plus, when that thought has a hypothesis
    # and no pack, one fill call
    roles = Roles.scripted(
        thinker=[
            _thought(hypotheses=[_hyp("steps 64->128")]),
            json.dumps(copied),
            _thought(),
            "not json",
        ],
        tooler=[],
        coder=[],
    )
    res = _ens(sup, roles).run(obs)
    assert res.pack is None and sup.observe()["pack_hash"] is None
    errors = [str(t.result.get("error", "")) for t in res.transcript if t.result]
    assert any("config.steps still holds a placeholder" in e for e in errors)
    assert any("not a JSON object" in e for e in errors)


def test_a_placeholder_pack_never_reaches_the_supervisor(sup: Supervisor) -> None:
    """Config values are not type-checked downstream, so a placeholder string
    would otherwise reach the trainer as an lr."""
    from lab.ensemble.prompts import pack_template, placeholder_fields

    obs = _research(sup)
    copied = pack_template(obs, "claim", knob="lr")
    assert placeholder_fields(copied) == ["config.lr"]
    roles = Roles.scripted(thinker=[_thought(pack=copied)] * 4, tooler=[], coder=[])
    cfg = EnsembleConfig(index=1, variant_hint="learning rate")
    res = Ensemble(roles, sup, cfg, call_lock=threading.Lock(), tracer=sup.tracer).run(obs)
    assert res.pack is None
    assert sup.observe()["pack_hash"] is None


def test_template_returned_verbatim_is_sent_back_once(sup: Supervisor) -> None:
    """Observed on Spark: after the template nudge the 1.5B thinker claimed
    'lr 1.5e-3' but returned the template with lr untouched. One more nudge
    asks for the edit; if it insists, the pack is accepted (never loop).

    Angles with a knob are covered earlier by the placeholder check; this
    backstop is what catches the angles that have no single key to mask."""
    from lab.ensemble.prompts import knob_for_hint, pack_template

    hint = "data mix"
    assert knob_for_hint(hint) is None
    obs = _research(sup)
    tpl = pack_template(obs, "lr 1.5e-3 lowers confirm_ppl")
    edited = dict(tpl, config=dict(tpl["config"], lr=0.0015))
    roles = Roles.scripted(
        thinker=[
            _thought(hypotheses=[_hyp("lr 1.5e-3")]),
            json.dumps(tpl),
            _thought(hypotheses=[_hyp("lr 1.5e-3")]),
            json.dumps(edited),
        ],
        tooler=[],
        coder=[],
    )
    res = _ens(sup, roles, variant_hint=hint).run(obs)
    assert res.error is None and res.pack["config"]["lr"] == 0.0015
    assert res.nudges == 2
    assert any(
        "template unchanged" in str((t.result or {}).get("error", "")) for t in res.transcript
    )

    # a stubborn thinker is refused once, then its template pack is taken
    stubborn = Roles.scripted(
        thinker=[
            _thought(hypotheses=[_hyp("lr 1.5e-3")]),
            json.dumps(tpl),
            _thought(hypotheses=[_hyp("lr 1.5e-3")]),
            json.dumps(tpl),
        ],
        tooler=[],
        coder=[],
    )
    res2 = _ens(sup, stubborn, variant_hint=hint).run(obs)
    assert res2.error is None and res2.pack is not None


def test_pack_with_hypothesis_id_key_is_accepted(sup: Supervisor) -> None:
    obs = _research(sup)
    pack = _pack(obs)
    pack["hypothesis_id"] = "hyp-0001"
    roles = Roles.scripted(thinker=[_thought(pack=pack)], tooler=[], coder=[])
    res = _ens(sup, roles).run(obs)
    assert res.pack is not None and "hypothesis_id" not in res.pack


def test_unparseable_thinker_twice_sets_error_without_raising(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(thinker=["not json", "still not json"], tooler=[], coder=[])
    res = _ens(sup, roles).run(obs)
    assert res.error == "thinker: unparseable"
    assert res.pack is None
    assert [t.parsed_ok for t in res.transcript] == [False, False]
    prompts = roles.backend("thinker").prompts
    assert len(prompts) == 2 and prompts[1].endswith("Reply with only the JSON object.")


def test_unparseable_once_then_recovers(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(thinker=["???", _thought(pack=_pack(obs))], tooler=[], coder=[])
    res = _ens(sup, roles).run(obs)
    assert res.error is None and res.pack is not None
    assert [t.parsed_ok for t in res.transcript] == [False, True]


def test_role_not_configured_is_recorded_not_raised(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.from_env(LabConfig(run_dir=sup.cfg.run_dir))
    res = _ens(sup, roles).run(obs)
    assert res.error is not None
    assert "LAB_THINKER_MODEL" in res.error
    assert res.pack is None and res.transcript == []


def test_repeated_identical_tool_call_is_not_executed_twice(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=[
            _thought(need={"kind": "tool", "request": "list"}),
            _thought(need={"kind": "tool", "request": "list again"}),
            _thought(pack=_pack(obs)),
        ],
        tooler=['{"tool": "list_episodes", "args": {}}'] * 2,
        coder=[],
    )
    before = sup.observe()["research"]["tool_calls"]
    res = _ens(sup, roles).run(obs)
    assert res.pack is not None
    assert res.tool_calls == 1 and res.nudges == 1
    assert res.transcript[3].result["error"] == "repeated call; use the earlier result"
    assert sup.observe()["research"]["tool_calls"] == before + 1


def test_tool_budget_stops_further_calls(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=[_thought(need={"kind": "tool", "request": f"read {i}"}) for i in range(4)] + [_thought(pack=_pack(obs))],
        tooler=[json.dumps({"tool": "list_episodes", "args": {"n": i + 1}}) for i in range(4)],
        coder=[],
    )
    res = _ens(sup, roles, max_tool_calls=2).run(obs)
    assert res.tool_calls == 2
    assert any("tool budget spent" in (t.result or {}).get("error", "") for t in res.transcript)
    assert res.pack is not None


def test_unparseable_tooler_counts_as_nudge(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=[_thought(need={"kind": "tool", "request": "x"}), _thought(done=True)],
        tooler=["I would call list_episodes"],
        coder=[],
    )
    res = _ens(sup, roles).run(obs)
    assert res.tool_calls == 0 and res.nudges == 1
    assert res.transcript[1].parsed_ok is False


def test_max_rounds_is_respected(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=lambda prompt: _thought(need={"kind": "tool", "request": "more"}),
        tooler=lambda prompt: json.dumps({"tool": "list_episodes", "args": {"n": len(prompt) % 97}}),
        coder=[],
    )
    res = _ens(sup, roles, max_rounds=3, max_tool_calls=50).run(obs)
    assert res.rounds == 3
    assert res.pack is None and res.error is None
    assert len(roles.backend("thinker").prompts) == 3


def test_idle_thinker_is_nudged_three_times_then_stopped(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(thinker=lambda prompt: _thought(thought="hmm"), tooler=[], coder=[])
    res = _ens(sup, roles).run(obs)
    assert res.nudges == 3 and res.rounds == 3
    assert res.error is not None and "nudges" in res.error


def test_sentinel_reply_ends_the_loop_without_a_pack(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(thinker=[], tooler=[], coder=[])
    res = _ens(sup, roles).run(obs)
    assert res.done is True and res.pack is None and res.error is None
    assert res.rounds == 1
