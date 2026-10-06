"""Checkpoint bank: a short menu, not every job file."""

from __future__ import annotations

from pathlib import Path

from lab.checkpoints import BANK_SLOTS, CheckpointBank
from lab.policy import dummy_pack


def test_bank_keeps_named_slots_and_picks_the_lower_score(tmp_path: Path) -> None:
    bank = CheckpointBank(tmp_path)
    first = tmp_path / "job-0001.pt"
    second = tmp_path / "job-0002.pt"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    bank.note_train(path=first, stage="pretrain", job_id="job-0001", config={"hidden": 32, "layers": 2})
    bank.note_score(job_id="job-0001", confirm_ppl=4.0)
    bank.note_train(path=second, stage="continue", job_id="job-0002", config={"hidden": 32, "layers": 2})
    bank.note_score(job_id="job-0002", confirm_ppl=5.5)

    slots = {row["slot"]: row for row in bank.entries()}
    assert set(slots) == {"latest", "best", "pretrain", "continue"}
    assert slots["latest"]["job_id"] == "job-0002"
    assert slots["pretrain"]["job_id"] == "job-0001"
    assert slots["best"]["job_id"] == "job-0001"
    assert slots["best"]["confirm_ppl"] == 4.0
    assert bank.resolve("bank:best") == first
    assert bank.resolve("nope") is None


def test_a_better_score_replaces_best(tmp_path: Path) -> None:
    bank = CheckpointBank(tmp_path)
    a = tmp_path / "a.pt"
    b = tmp_path / "b.pt"
    a.write_bytes(b"a")
    b.write_bytes(b"b")
    bank.note_train(path=a, stage="continue", job_id="job-0001", config={})
    bank.note_score(job_id="job-0001", confirm_ppl=3.0)
    bank.note_train(path=b, stage="midtrain", job_id="job-0002", config={})
    bank.note_score(job_id="job-0002", confirm_ppl=2.5)
    assert bank.resolve("bank:best") == b
    assert bank.resolve("bank:midtrain") == b
    assert bank.resolve("bank:continue") == a


def test_list_checkpoints_and_write_pack_use_the_bank(sup) -> None:
    listed = sup.call("list_checkpoints")
    assert listed["ok"]
    assert listed["checkpoints"] == []
    assert listed["slots"] == list(BANK_SLOTS)
    assert sup.call("enter_research")["ok"]

    pack = dummy_pack(sup.observe())
    pack["parent_checkpoint"] = "bank:best"
    pack["stage"] = "continue"
    denied = sup.call("write_pack", {"pack": pack})
    assert denied["ok"] is False
    assert "empty" in denied["error"]

    path = sup.cfg.checkpoints_dir / "job-0001.pt"
    path.write_bytes(b"ckpt")
    sup.bank.note_train(path=path, stage="pretrain", job_id="job-0001", config={"hidden": 16, "layers": 1})
    pack["parent_checkpoint"] = "bank:pretrain"
    pack["stage"] = "continue"
    ok = sup.call("write_pack", {"pack": pack})
    assert ok["ok"], ok
    assert "bank:pretrain" in sup.observe()["checkpoint_bank"][0]["id"] or any(
        row["id"] == "bank:pretrain" for row in sup.observe()["checkpoint_bank"]
    )

    pack["stage"] = "pretrain"
    pack["parent_checkpoint"] = "bank:best"
    # pretrain may name an empty slot; the weights are ignored
    allowed = sup.call("write_pack", {"pack": pack})
    assert allowed["ok"], allowed
