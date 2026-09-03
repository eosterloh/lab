"""Helpers shared by harness tests. Not collected as tests."""

from __future__ import annotations

from lab.policy import dummy_pack
from lab.supervisor import Supervisor


def arm_for_train(sup: Supervisor, pack: dict | None = None) -> None:
    """Fill the live-hypothesis train checklist and write a dummy pack."""
    if sup.observe()["phase"] != "research":
        assert sup.call("enter_research")["ok"], "failed to enter research"
    assert sup.call(
        "write_hypothesis",
        {
            "claim": "dummy overtrain improves confirm_ppl",
            "why": "test fixture",
            "falsify": "dummy job fails",
        },
    )["ok"]
    payload = pack or dummy_pack(sup.observe())
    assert sup.call("write_pack", {"pack": payload})["ok"]
