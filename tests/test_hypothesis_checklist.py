"""Live hypothesis is the working theory + checklist.

Train is blocked until claim_written and pack_ready. Post-eval seals close
boxes. holdout_not_proxy stays open on dummy (trainer-val proxy).
"""

from __future__ import annotations

from tests.helpers import arm_for_train
from lab.policy import dummy_pack


def test_write_hypothesis_requires_a_claim(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call("write_hypothesis", {"why": "no claim"})
    assert out["ok"] is False
    assert "claim" in out["error"]


def test_write_hypothesis_marks_claim_written_and_status_testing(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call(
        "write_hypothesis",
        {"claim": "more steps help", "why": "overtrain", "falsify": "ppl up"},
    )
    assert out["ok"] is True
    hyp = out["hypothesis"]
    assert hyp["status"] == "testing"
    assert hyp["claim"] == "more steps help"
    done = {c["id"]: c["done"] for c in hyp["checklist"]}
    assert done["claim_written"] is True
    assert done["pack_ready"] is False


def test_enter_train_blocked_until_checklist_and_pack_are_ready(sup) -> None:
    assert sup.call("enter_research")["ok"]
    blocked = sup.call("enter_train")
    assert blocked["ok"] is False
    assert "write_pack" in blocked["error"]

    pack = dummy_pack(sup.observe())
    assert sup.call("write_pack", {"pack": pack})["ok"]
    # Pack exists but claim_written is still open.
    blocked = sup.call("enter_train")
    assert blocked["ok"] is False
    assert "checklist" in blocked["error"]
    assert "claim_written" in blocked["open"]

    arm_for_train(sup)
    started = sup.call("enter_train")
    assert started["ok"] is True
    assert started["job"]["status"] == "succeeded"


def test_new_cycle_reopens_the_claim_so_the_policy_must_rewrite_it(sup) -> None:
    """Each cycle gets its own theory; a stale claim must not unlock train."""
    arm_for_train(sup)
    assert sup.call("enter_train")["ok"]
    assert sup.call("enter_eval")["ok"]
    assert sup.call("run_eval")["ok"]

    assert sup.call("enter_research")["ok"]
    hyp = sup.observe()["hypothesis"]
    assert hyp["claim"] == ""
    assert "claim_written" in hyp["open_train"]

    pack = dummy_pack(sup.observe())
    assert sup.call("write_pack", {"pack": pack})["ok"]
    blocked = sup.call("enter_train")
    assert blocked["ok"] is False
    assert "claim_written" in blocked["open"]


def test_rolled_hypotheses_are_archived(sup) -> None:
    arm_for_train(sup)
    assert sup.call("enter_train")["ok"]
    assert sup.call("enter_eval")["ok"]
    assert sup.call("run_eval")["ok"]
    assert sup.call("enter_research")["ok"]

    archive = sup.hypothesis.archive
    assert len(archive) == 1
    assert archive[0]["claim"] == "dummy overtrain improves confirm_ppl"
    assert sup.hypothesis.archive_path.is_file()


def test_read_hypothesis_allowed_after_halt(sup) -> None:
    from lab.policy import DummyPolicy

    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    out = sup.call("read_hypothesis")
    assert out["ok"] is True
    assert out["hypothesis"]["claim"]
    listed = sup.call("list_hypotheses")
    assert listed["ok"] is True
    assert listed["hypotheses"][0]["id"] == out["id"]


def test_each_write_hypothesis_opens_a_new_board_entry(sup) -> None:
    """Refining a claim opens a second hypothesis; the newest becomes current."""
    assert sup.call("enter_research")["ok"]
    first = sup.call("write_hypothesis", {"claim": "v1", "why": "w", "falsify": "f"})
    second = sup.call("write_hypothesis", {"claim": "v2 sharper", "why": "w", "falsify": "f"})
    assert first["id"] == "hyp-0001" and second["id"] == "hyp-0002"
    obs = sup.observe()
    assert obs["hypothesis"]["id"] == second["id"]
    assert [h["id"] for h in obs["hypotheses"]] == [first["id"], second["id"]]
    assert sup.hypothesis.current.id == second["id"]
    assert (sup.cfg.run_dir / "hypotheses" / "hyp-0002.json").is_file()
    # The single-hypothesis mirror next to state.json tracks the current one.
    assert (sup.cfg.run_dir / "hypothesis.json").is_file()
    assert "v2 sharper" in (sup.cfg.run_dir / "hypothesis.json").read_text(encoding="utf-8")


def test_pack_written_before_the_claim_still_arms_train(sup) -> None:
    """Old single-hypothesis order (pack, then claim) must keep unlocking train."""
    assert sup.call("enter_research")["ok"]
    assert sup.call("write_pack", {"pack": dummy_pack(sup.observe())})["ok"]
    assert sup.observe()["hypothesis"]["id"] == ""
    wrote = sup.call("write_hypothesis", {"claim": "late claim", "why": "w", "falsify": "f"})
    assert wrote["ok"]
    hyp = sup.observe()["hypothesis"]
    assert hyp["id"] == wrote["id"]
    assert hyp["open_train"] == []
    assert sup.call("enter_train")["ok"]
