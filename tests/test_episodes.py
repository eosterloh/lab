"""Episode cards: one skill-like record per finished train+eval."""

from __future__ import annotations

from lab.policy import DummyPolicy
from tests.helpers import arm_for_train


def test_finished_dummy_cycle_is_readable_as_episode_card(sup) -> None:
    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    rows = sup.episodes.summaries()
    assert len(rows) == 1
    ep = sup.episodes.load(rows[0]["id"])
    assert ep["job_status"] == "succeeded"
    assert ep["trainer"] == "dummy"
    assert ep["eval"]["confirm_ppl"] is not None
    listed = sup.call("list_episodes", {"n": 5})
    assert listed["ok"] is True
    assert listed["episodes"][0]["id"] == ep["id"]
    got = sup.call("read_episode", {"id": ep["id"]})
    assert got["ok"] is True
    assert "# " in (got.get("card") or "")
    assert ep["id"] in sup.notebook.beliefs()
    # The card is linked back to the hypothesis it tested.
    assert ep["hypothesis_id"] == sup.state.hypothesis_id == "hyp-0001"
    assert ep["trial"] == 1
    assert rows[0]["hypothesis_id"] == "hyp-0001"
    assert "- hypothesis: `hyp-0001` (trial 1)" in got["card"]
    assert sup.board.get("hyp-0001").episode_id == ep["id"]


def test_record_without_hypothesis_id_omits_the_card_line(sup) -> None:
    arm_for_train(sup)
    assert sup.call("enter_train")["ok"]
    job = sup.jobs.store.load(sup.state.current_job_id)
    pack = sup.packs.load(sup.state.pack_hash)
    ep = sup.episodes.record(cycle=1, pack=pack, pack_hash=sup.state.pack_hash, job=job, ev={"confirm_ppl": 3.0})
    assert ep["hypothesis_id"] is None and ep["trial"] is None
    card = (sup.cfg.episodes_dir / f"{ep['id']}.md").read_text(encoding="utf-8")
    assert "- hypothesis:" not in card


def test_resealing_same_job_does_not_duplicate_episode(sup) -> None:
    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    first = sup.episodes.summaries()[0]["id"]
    pack = sup.packs.load(sup.state.pack_hash)
    job = sup.jobs.store.load(sup.state.current_job_id)
    again = sup.episodes.record(
        cycle=1,
        pack=pack,
        pack_hash=sup.state.pack_hash,
        job=job,
        ev=sup.state.last_eval,
    )
    assert again["id"] == first
    assert len(sup.episodes.summaries()) == 1
