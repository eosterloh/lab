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
