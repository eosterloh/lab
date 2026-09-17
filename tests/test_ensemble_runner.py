"""N ensembles in parallel, one serial commit, trials queued on the supervisor."""

from __future__ import annotations

import json
import threading
import time

import pytest

from lab.ensemble.policy import EnsemblePolicy
from lab.ensemble.roles import Roles
from lab.ensemble.runner import FALLBACK_SOURCE, EnsembleRunner
from lab.policy import dummy_pack
from lab.supervisor import Supervisor

ENV_VARS = ("LAB_THINKER_MODEL", "LAB_TOOLER_MODEL", "LAB_CODER_MODEL")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _thought(**kw) -> str:
    base = {"thought": "", "hypotheses": [], "need": None, "pack": None, "done": False}
    base.update(kw)
    return json.dumps(base)


def _hyp(claim: str) -> dict:
    return {"claim": claim, "why": "w", "falsify": "f"}


def _research(sup: Supervisor) -> dict:
    assert sup.call("run_eval")["ok"]
    assert sup.call("enter_research")["ok"]
    return sup.observe()


def _angle_thinker(obs: dict, *, steps_by_angle: dict[str, int]):
    """Callable thinker keyed on the variant hint in the prompt."""

    def reply(prompt: str) -> str:
        for angle, steps in steps_by_angle.items():
            if f"Your angle: {angle}" in prompt:
                pack = dummy_pack(obs, extra_config={"steps": steps})
                pack["hypothesis"] = f"{angle}: steps={steps}"
                return _thought(pack=pack, hypotheses=[_hyp(f"{angle} with steps={steps} lowers confirm_ppl")])
        return _thought(done=True)

    return reply


def test_two_ensembles_commit_two_trials(sup: Supervisor) -> None:
    obs = _research(sup)
    roles = Roles.scripted(
        thinker=_angle_thinker(obs, steps_by_angle={"learning rate": 12, "training steps": 40}),
        tooler=[],
        coder=[],
    )
    runner = EnsembleRunner(roles, sup, n=2, max_rounds=4, tracer=sup.tracer)
    results = runner.run_research(obs)
    assert [r.index for r in results] == [1, 2]
    assert all(r.pack for r in results)
    assert results[0].pack["config"]["steps"] == 12
    assert results[1].pack["config"]["steps"] == 40

    out = runner.commit(results, obs)
    assert out["fallback"] is False
    assert len(out["candidates"]) == 2
    hashes = {c["pack_hash"] for c in out["candidates"]}
    assert len(hashes) == 2

    after = sup.observe()
    assert after["trials_planned"] == 2 and after["trial"] == 1
    assert after["pack_hash"] == out["candidates"][0]["pack_hash"]
    assert after["hypothesis"]["id"] == out["candidates"][0]["hypothesis_id"]
    assert after["candidate_queue"] == [out["candidates"][1]]
    sources = [h.source for h in sup.board.list()]
    assert sources == ["ensemble-1", "ensemble-2"]
    for hyp in sup.board.list():
        assert hyp.pack_hash in hashes

    summary_path = sup.cfg.run_dir / "ensembles" / "cycle-0001.json"
    assert summary_path.is_file()
    summary = json.loads(summary_path.read_text())
    assert len(summary["results"]) == 2
    assert summary["candidates"] == out["candidates"]
    assert summary["fallback"] is False
    trace = (sup.cfg.run_dir / "trace.jsonl").read_text()
    assert '"name": "ensemble 1"' in trace and '"name": "ensemble 2"' in trace
    assert '"name": "ensemble 1 thinker"' in trace


def test_identical_packs_are_deduped_to_one_candidate(sup: Supervisor) -> None:
    obs = _research(sup)
    pack = dummy_pack(obs)
    roles = Roles.scripted(
        thinker=lambda prompt: _thought(pack=pack, hypotheses=[_hyp("same idea")]),
        tooler=[],
        coder=[],
    )
    runner = EnsembleRunner(roles, sup, n=2, max_rounds=2)
    results = runner.run_research(obs)
    assert all(r.pack for r in results)
    out = runner.commit(results, obs)
    assert len(out["candidates"]) == 1 and out["fallback"] is False
    assert sup.observe()["trials_planned"] == 1
    # The duplicate never opened a hypothesis that would later be killed as untested.
    assert [h.source for h in sup.board.list()] == ["ensemble-1"]
    assert any("duplicate" in n.get("skipped", "") for n in out["notes"])


def test_no_packs_falls_back_to_one_harness_pack(sup: Supervisor, capsys) -> None:
    obs = _research(sup)
    roles = Roles.scripted(thinker=lambda prompt: _thought(done=True), tooler=[], coder=[])
    runner = EnsembleRunner(roles, sup, n=2, max_rounds=2)
    results = runner.run_research(obs)
    assert not any(r.pack for r in results)
    out = runner.commit(results, obs)
    assert out["fallback"] is True
    assert len(out["candidates"]) == 1
    hyps = sup.board.list()
    assert len(hyps) == 1 and hyps[0].source == FALLBACK_SOURCE
    assert sup.observe()["pack_hash"] == out["candidates"][0]["pack_hash"]
    assert sup.packs.load(out["candidates"][0]["pack_hash"]).trainer == "lab"
    assert "fallback pack authored by harness" in capsys.readouterr().out


def test_pack_without_hypotheses_gets_a_synthesized_one(sup: Supervisor) -> None:
    obs = _research(sup)
    pack = dummy_pack(obs)
    pack["hypothesis"] = "silent ensemble pack"
    roles = Roles.scripted(thinker=[_thought(pack=pack)], tooler=[], coder=[])
    runner = EnsembleRunner(roles, sup, n=1, max_rounds=2)
    out = runner.commit(runner.run_research(obs), obs)
    assert len(out["candidates"]) == 1 and out["fallback"] is False
    hyp = sup.board.list()[0]
    assert hyp.claim == "silent ensemble pack"
    assert hyp.why == "ensemble did not state a hypothesis"
    assert hyp.source == "ensemble-1"


def test_unconfigured_roles_record_errors_and_fallback_keeps_the_cycle_alive(sup: Supervisor) -> None:
    from lab.config import LabConfig

    obs = _research(sup)
    roles = Roles.from_env(LabConfig(run_dir=sup.cfg.run_dir))
    runner = EnsembleRunner(roles, sup, n=2, max_rounds=2)
    results = runner.run_research(obs)
    assert all("LAB_THINKER_MODEL" in (r.error or "") for r in results)
    out = runner.commit(results, obs)
    assert out["fallback"] is True and len(out["candidates"]) == 1
    assert sup.call("enter_train")["ok"]


def test_ensembles_run_in_parallel_and_supervisor_calls_are_serialized(sup: Supervisor, monkeypatch) -> None:
    obs = _research(sup)
    real_call = sup.call
    inside = {"n": 0, "max": 0}
    guard = threading.Lock()

    def guarded_call(name, args=None):
        with guard:
            inside["n"] += 1
            inside["max"] = max(inside["max"], inside["n"])
        try:
            time.sleep(0.02)
            return real_call(name, args)
        finally:
            with guard:
                inside["n"] -= 1

    monkeypatch.setattr(sup, "call", guarded_call)

    def thinker(prompt: str) -> str:
        if "tooler -> list_episodes" in prompt:
            steps = 10 + len(prompt) % 50
            for i, angle in enumerate(("learning rate", "training steps", "data mix")):
                if f"Your angle: {angle}" in prompt:
                    steps = 20 + i
            return _thought(pack=dummy_pack(obs, extra_config={"steps": steps}), hypotheses=[_hyp(f"steps {steps}")])
        return _thought(need={"kind": "tool", "request": "episodes"})

    def tooler(prompt: str) -> str:
        time.sleep(0.1)
        return '{"tool": "list_episodes", "args": {}}'

    roles = Roles.scripted(thinker=thinker, tooler=tooler, coder=[])
    runner = EnsembleRunner(roles, sup, n=3, max_rounds=4)
    started = time.time()
    results = runner.run_research(obs)
    wall = time.time() - started
    assert all(r.pack for r in results), [r.error for r in results]
    assert all(r.tool_calls == 1 for r in results)
    assert wall < 0.25, f"ensembles did not overlap: {wall:.3f}s"
    assert inside["max"] == 1, "sup.call overlapped across threads"
    out = runner.commit(results, obs)
    assert len(out["candidates"]) == 3
    assert sup.observe()["trials_planned"] == 3


# --------------------------------------------------------------------- policy


def test_policy_walks_two_trials_through_the_real_supervisor(sup: Supervisor) -> None:
    packs = {
        "learning rate": 12,
        "training steps": 40,
    }
    seen_cycles: list[int] = []

    def thinker(prompt: str) -> str:
        obs = sup.observe()
        seen_cycles.append(obs["cycle"])
        for angle, steps in packs.items():
            if f"Your angle: {angle}" in prompt:
                pack = dummy_pack(obs, extra_config={"steps": steps})
                return _thought(pack=pack, hypotheses=[_hyp(f"{angle} c{obs['cycle']} steps={steps}")])
        return _thought(done=True)

    roles = Roles.scripted(thinker=thinker, tooler=[], coder=[])
    policy = EnsemblePolicy(roles, sup, n=2, max_rounds=3, tracer=sup.tracer)
    result = sup.run_policy(policy, max_cycles=1, max_steps=60)
    assert result["halted"] is True and result["halt_reason"] == "max_cycles reached"
    assert result["completed_cycles"] == 1
    assert result["trials_planned"] == 2 and result["trial"] == 2
    trials = result["trial_results"]
    assert [t["trial"] for t in trials] == [1, 2]
    assert len({t["hypothesis_id"] for t in trials}) == 2
    episodes = sup.episodes.summaries()
    assert len(episodes) == 2 and [e["trial"] for e in episodes] == [1, 2]
    assert len(policy.commits) == 1
    tools = [json.loads(l)["tool"] for l in (sup.cfg.run_dir / "tools.jsonl").read_text().splitlines()]
    # Exact research walk: research runs once, commit queues, then the trials loop.
    assert tools == [
        "run_eval",
        "enter_research",
        "write_hypothesis",
        "write_pack",
        "write_hypothesis",
        "write_pack",
        "queue_candidates",
        "enter_train",
        "enter_eval",
        "run_eval",
        "enter_research",
        "enter_train",
        "enter_eval",
        "run_eval",
    ]


def test_policy_runs_research_once_per_cycle_over_two_cycles(sup: Supervisor) -> None:
    def thinker(prompt: str) -> str:
        obs = sup.observe()
        steps = 10 + obs["cycle"] * 2 + (1 if "Your angle: learning rate" in prompt else 0)
        return _thought(pack=dummy_pack(obs, extra_config={"steps": steps}), hypotheses=[_hyp(f"steps {steps}")])

    roles = Roles.scripted(thinker=thinker, tooler=[], coder=[])
    policy = EnsemblePolicy(roles, sup, n=2, max_rounds=2)
    result = sup.run_policy(policy, max_cycles=2, max_steps=80)
    assert result["completed_cycles"] == 2
    assert len(policy.commits) == 2
    assert len(sup.episodes.summaries()) == 4
    assert (sup.cfg.run_dir / "ensembles" / "cycle-0002.json").is_file()
    assert {h.status for h in sup.board.list(cycle=1)} == {"closed"}
