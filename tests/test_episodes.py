"""Episode cards: one skill-like record per finished train+eval."""

from __future__ import annotations

from lab.policy import DummyPolicy


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
