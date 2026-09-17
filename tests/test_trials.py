"""Many trials per cycle: N (hypothesis, pack) candidates trained and evaluated in turn."""

from __future__ import annotations

from lab.policy import DummyPolicy, dummy_pack
from tests.helpers import arm_for_train


def _write_candidate(sup, claim: str, steps: int) -> tuple[str, str]:
    hyp = sup.call("write_hypothesis", {"claim": claim, "why": "trials test", "falsify": "job fails"})
    assert hyp["ok"], hyp
    pack = dummy_pack(sup.observe(), extra_config={"steps": steps})
    out = sup.call("write_pack", {"pack": pack, "hypothesis_id": hyp["id"]})
    assert out["ok"], out
    assert out["hypothesis_id"] == hyp["id"]
    return hyp["id"], out["pack_hash"]


def _run_trial(sup) -> dict:
    assert sup.call("enter_train")["ok"]
    assert sup.call("enter_eval")["ok"]
    out = sup.call("run_eval")
    assert out["ok"]
    return out


def test_two_candidates_run_as_two_trials_in_one_cycle(sup) -> None:
    assert sup.call("run_eval")["ok"]
    assert sup.call("enter_research")["ok"]
    h1, p1 = _write_candidate(sup, "candidate one: fewer steps", steps=12)
    h2, p2 = _write_candidate(sup, "candidate two: more steps", steps=40)
    assert h1 != h2 and p1 != p2

    queued = sup.call(
        "queue_candidates",
        {"candidates": [{"hypothesis_id": h1, "pack_hash": p1}, {"hypothesis_id": h2, "pack_hash": p2}]},
    )
    assert queued["ok"], queued
    assert queued["trials_planned"] == 2
    obs = sup.observe()
    assert obs["trials_planned"] == 2
    assert obs["trial"] == 1
    assert obs["pack_hash"] == p1
    assert obs["hypothesis"]["id"] == h1
    assert obs["candidate_queue"] == [{"hypothesis_id": h2, "pack_hash": p2}]
    assert len(obs["hypotheses"]) == 2

    # Trial 1: cycle must not advance.
    _run_trial(sup)
    obs = sup.observe()
    assert obs["cycle"] == 1
    assert obs["completed_cycles"] == 0
    assert obs["trial"] == 1
    assert len(obs["trial_results"]) == 1
    assert obs["trial_results"][0]["hypothesis_id"] == h1
    assert obs["hypothesis"]["id"] == h1
    assert obs["hypothesis"]["episode_id"]

    # Mid-trial research loads candidate 2 without rolling the board.
    entered = sup.call("enter_research")
    assert entered["ok"]
    assert entered["trial"] == 2
    obs = sup.observe()
    assert obs["cycle"] == 1
    assert obs["pack_hash"] == p2
    assert obs["hypothesis"]["id"] == h2
    assert obs["hypothesis"]["open_train"] == []
    assert obs["candidate_queue"] == []
    assert len(obs["hypotheses"]) == 2

    # Trial 2: last trial closes the cycle.
    _run_trial(sup)
    obs = sup.observe()
    assert obs["cycle"] == 2
    assert obs["completed_cycles"] == 1
    assert obs["trial"] == 2 and obs["trials_planned"] == 2
    results = obs["trial_results"]
    assert [r["trial"] for r in results] == [1, 2]
    assert [r["hypothesis_id"] for r in results] == [h1, h2]
    assert all(r["confirm_ppl"] is not None for r in results)

    episodes = sup.episodes.summaries()
    assert len(episodes) == 2
    assert {e["hypothesis_id"] for e in episodes} == {h1, h2}
    assert [e["cycle"] for e in episodes] == [1, 1]
    assert [e["trial"] for e in episodes] == [1, 2]
    card = sup.call("read_episode", {"id": episodes[1]["id"]})["card"]
    assert f"- hypothesis: `{h2}`" in card

    summaries = [e for e in sup.notebook.all() if e.get("kind") == "cycle_summary"]
    assert len(summaries) == 1
    assert [t["hypothesis_id"] for t in summaries[0]["trials"]] == [h1, h2]
    assert summaries[0]["cycle"] == 1

    board_rows = sup.board.summary(cycle=1)
    assert [r["id"] for r in board_rows] == [h1, h2]
    assert board_rows[0]["episode_id"] == episodes[0]["id"]
    assert board_rows[1]["episode_id"] == episodes[1]["id"]
    assert len(sup.observe()["hypotheses"]) == 0  # observe() lists the *current* cycle (2)
    assert len(sup.call("list_hypotheses", {"cycle": 1})["hypotheses"]) == 2

    # The next cycle starts clean: counters reset, cycle-1 hypotheses retired.
    assert sup.call("enter_research")["ok"]
    obs = sup.observe()
    assert obs["trial"] == 1 and obs["trials_planned"] == 1
    assert obs["trial_results"] == [] and obs["candidate_queue"] == []
    assert obs["pack_hash"] is None
    assert obs["hypothesis"]["claim"] == ""
    assert {h.status for h in sup.board.list(cycle=1)} == {"closed"}
    assert len(sup.hypothesis.archive) == 2


def test_queue_candidates_validates_inputs_and_phase(sup) -> None:
    denied = sup.call("queue_candidates", {"candidates": []})
    assert denied["ok"] is False
    assert "not allowed" in denied["error"]

    assert sup.call("enter_research")["ok"]
    assert sup.call("queue_candidates", {})["ok"] is False
    h1, p1 = _write_candidate(sup, "real", steps=11)
    bad_hyp = sup.call("queue_candidates", {"candidates": [{"hypothesis_id": "hyp-9999", "pack_hash": p1}]})
    assert bad_hyp["ok"] is False and "hyp-9999" in bad_hyp["error"]
    bad_pack = sup.call("queue_candidates", {"candidates": [{"hypothesis_id": h1, "pack_hash": "nope"}]})
    assert bad_pack["ok"] is False and "nope" in bad_pack["error"]
    dup = sup.call(
        "queue_candidates",
        {"candidates": [{"hypothesis_id": h1, "pack_hash": p1}, {"hypothesis_id": h1, "pack_hash": p1}]},
    )
    assert dup["ok"] is False and "twice" in dup["error"]
    # A failed queue leaves the single-candidate state intact.
    assert sup.observe()["trials_planned"] == 1


def test_queue_candidates_links_pack_so_train_gate_opens(sup) -> None:
    assert sup.call("enter_research")["ok"]
    hyp = sup.call("write_hypothesis", {"claim": "unlinked", "why": "w", "falsify": "f"})
    other = sup.call("write_hypothesis", {"claim": "gets the pack", "why": "w", "falsify": "f"})
    pack = sup.call("write_pack", {"pack": dummy_pack(sup.observe()), "hypothesis_id": other["id"]})
    assert sup.board.get(hyp["id"]).pack_hash is None
    out = sup.call("queue_candidates", {"candidates": [{"hypothesis_id": hyp["id"], "pack_hash": pack["pack_hash"]}]})
    assert out["ok"], out
    assert out["trials_planned"] == 1
    assert sup.board.get(hyp["id"]).pack_hash == pack["pack_hash"]
    assert sup.observe()["hypothesis"]["id"] == hyp["id"]
    assert sup.call("enter_train")["ok"]


def test_single_candidate_is_the_degenerate_case(sup) -> None:
    """DummyPolicy never queues; one hypothesis + one pack behaves exactly as before."""
    result = sup.run_policy(DummyPolicy(), max_cycles=2, max_steps=80)
    assert result["completed_cycles"] == 2
    assert result["trial"] == 1 and result["trials_planned"] == 1
    assert len(sup.episodes.summaries()) == 2
    hyps = sup.board.list()
    assert [h.cycle for h in hyps] == [1, 2]
    assert [h.episode_id for h in hyps] == [e["id"] for e in sup.episodes.summaries()]
    assert [e["hypothesis_id"] for e in sup.episodes.summaries()] == [h.id for h in hyps]
    summaries = [e for e in sup.notebook.all() if e.get("kind") == "cycle_summary"]
    assert [s["cycle"] for s in summaries] == [1, 2]
    assert all(len(s["trials"]) == 1 for s in summaries)


def test_extra_hypotheses_without_a_trial_are_killed_at_the_next_cycle(sup) -> None:
    arm_for_train(sup)
    tested = sup.state.hypothesis_id
    spare = sup.call("write_hypothesis", {"claim": "never tested", "why": "w", "falsify": "f"})
    # The spare inherits the already-written pack (single-hypothesis compat) and
    # becomes current; re-select the armed one by writing its pack again.
    assert sup.observe()["hypothesis"]["id"] == spare["id"]
    assert sup.call("write_pack", {"pack": dummy_pack(sup.observe()), "hypothesis_id": tested})["ok"]
    _run_trial(sup)
    assert sup.call("enter_research")["ok"]
    assert sup.board.get(spare["id"]).status == "killed"
    assert sup.board.get(spare["id"]).note == "untested"
    assert sup.board.get(tested).status == "closed"
    assert sup.board.get(tested).episode_id
    listed = sup.call("list_hypotheses", {"status": "killed"})
    assert listed["ok"] and [r["id"] for r in listed["hypotheses"]] == [spare["id"]]


def test_write_hypothesis_with_id_updates_instead_of_opening(sup) -> None:
    assert sup.call("enter_research")["ok"]
    first = sup.call("write_hypothesis", {"claim": "v1", "why": "w", "falsify": "f"})
    edited = sup.call("write_hypothesis", {"id": first["id"], "claim": "v2"})
    assert edited["ok"] and edited["id"] == first["id"]
    assert edited["hypothesis"]["claim"] == "v2"
    assert len(sup.board.list()) == 1
    got = sup.call("read_hypothesis", {"id": first["id"]})
    assert got["ok"] and got["hypothesis"]["claim"] == "v2"
    missing = sup.call("read_hypothesis", {"id": "hyp-0099"})
    assert missing["ok"] is False


def test_state_round_trips_trial_fields(sup) -> None:
    from lab.state import RunState

    assert sup.call("enter_research")["ok"]
    h1, p1 = _write_candidate(sup, "a", steps=11)
    h2, p2 = _write_candidate(sup, "b", steps=12)
    assert sup.call(
        "queue_candidates",
        {"candidates": [{"hypothesis_id": h1, "pack_hash": p1}, {"hypothesis_id": h2, "pack_hash": p2}]},
    )["ok"]
    import json

    saved = RunState.from_dict(json.loads(sup.cfg.state_path.read_text(encoding="utf-8")))
    assert saved.trials_planned == 2
    assert saved.hypothesis_id == h1
    assert saved.candidate_queue == [{"hypothesis_id": h2, "pack_hash": p2}]
    assert RunState.from_dict({"cycle": 3}).trial == 1
