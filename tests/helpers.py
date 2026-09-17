"""Helpers shared by harness tests. Not collected as tests."""

from __future__ import annotations

from lab.policy import dummy_pack
from lab.supervisor import Supervisor


def arm_for_train(sup: Supervisor, pack: dict | None = None) -> str:
    """Open a hypothesis on the board, write a pack linked to it, return the id.

    Leaves the supervisor in research with the train gate open: one
    hypothesis + one pack, the degenerate single-trial cycle.
    """
    if sup.observe()["phase"] != "research":
        assert sup.call("enter_research")["ok"], "failed to enter research"
    wrote = sup.call(
        "write_hypothesis",
        {
            "claim": "dummy overtrain improves confirm_ppl",
            "why": "test fixture",
            "falsify": "dummy job fails",
        },
    )
    assert wrote["ok"], wrote
    payload = pack or dummy_pack(sup.observe())
    assert sup.call("write_pack", {"pack": payload, "hypothesis_id": wrote["id"]})["ok"]
    return wrote["id"]
