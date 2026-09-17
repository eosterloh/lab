"""HypothesisBoard: many live hypotheses per cycle, each linked to a pack and an episode."""

from __future__ import annotations

from pathlib import Path

from lab.hypothesis import HypothesisBoard, LiveHypothesis


def _board(tmp_path: Path) -> HypothesisBoard:
    return HypothesisBoard(tmp_path / "hypotheses")


def test_open_assigns_incrementing_ids_and_marks_claim(tmp_path: Path) -> None:
    board = _board(tmp_path)
    a = board.open(claim="lr up helps", why="w", falsify="f", cycle=1)
    b = board.open(claim="steps up helps", cycle=1, source="ensemble-2")
    c = board.open(claim="third", cycle=2)
    assert [a.id, b.id, c.id] == ["hyp-0001", "hyp-0002", "hyp-0003"]
    assert a.status == "testing"
    assert a.item("claim_written").done is True
    assert a.item("pack_ready").done is False
    assert b.source == "ensemble-2"
    assert a.source == "policy"
    assert (board.root / "hyp-0001.json").is_file()
    assert (board.root / "index.jsonl").is_file()
    assert (board.root / "board.md").is_file()


def test_list_filters_by_cycle_and_status(tmp_path: Path) -> None:
    board = _board(tmp_path)
    board.open(claim="a", cycle=1)
    b = board.open(claim="b", cycle=1)
    board.open(claim="c", cycle=2)
    board.close(b.id, "killed")
    assert [h.id for h in board.list(cycle=1)] == ["hyp-0001", "hyp-0002"]
    assert [h.id for h in board.list(cycle=2)] == ["hyp-0003"]
    assert [h.id for h in board.list(status="killed")] == ["hyp-0002"]
    assert [h.id for h in board.list(cycle=1, status="testing")] == ["hyp-0001"]
    assert len(board.list()) == 3


def test_link_pack_marks_pack_ready_and_link_episode_closes_boxes(tmp_path: Path) -> None:
    board = _board(tmp_path)
    h = board.open(claim="a", cycle=1)
    assert h.open_for("train") == [h.item("pack_ready")]
    board.link_pack(h.id, "abc123")
    assert board.get(h.id).pack_hash == "abc123"
    assert board.get(h.id).open_for("train") == []

    board.link_episode(h.id, "ep-0001-x", supporting=True)
    got = board.get(h.id)
    assert got.episode_id == "ep-0001-x"
    assert got.supporting_episodes == ["ep-0001-x"]
    assert got.refuting_episodes == []
    assert got.open_for("close") == []

    other = board.open(claim="b", cycle=1)
    board.link_episode(other.id, "ep-0002-y", supporting=False)
    assert board.get(other.id).refuting_episodes == ["ep-0002-y"]


def test_roll_cycle_kills_untested_from_earlier_cycles_only(tmp_path: Path) -> None:
    board = _board(tmp_path)
    tested = board.open(claim="tested", cycle=1)
    board.link_pack(tested.id, "p1")
    board.link_episode(tested.id, "ep-0001", supporting=True)
    untested = board.open(claim="untested", cycle=1)
    board.link_pack(untested.id, "p2")
    current = board.open(claim="current", cycle=2)

    killed = board.roll_cycle(2)
    assert killed == [untested.id]
    assert board.get(untested.id).status == "killed"
    assert board.get(untested.id).note == "untested"
    # Tested ones are closed, not killed; the current cycle is untouched.
    assert board.get(tested.id).status == "closed"
    assert board.get(current.id).status == "testing"
    # Rolling again is a no-op.
    assert board.roll_cycle(2) == []


def test_active_prefers_pack_linked_untested_then_current_cycle(tmp_path: Path) -> None:
    board = _board(tmp_path)
    assert board.active() is None
    a = board.open(claim="a", cycle=1)
    assert board.active().id == a.id  # most recent in current cycle
    b = board.open(claim="b", cycle=1)
    assert board.active().id == b.id
    board.link_pack(a.id, "p1")
    assert board.active().id == a.id  # armed beats newer-but-unarmed
    board.link_episode(a.id, "ep-0001", supporting=True)
    assert board.active().id == b.id  # a is tested, b is the newest live one
    board.roll_cycle(2)
    assert board.active() is None  # b killed, nothing in cycle 2
    assert board.active(cycle=1) is None


def test_summary_and_markdown_list_everything(tmp_path: Path) -> None:
    board = _board(tmp_path)
    a = board.open(claim="alpha", cycle=1)
    b = board.open(claim="beta", cycle=1)
    board.link_pack(b.id, "deadbeefcafe0000")
    rows = board.summary(cycle=1)
    assert [r["id"] for r in rows] == [a.id, b.id]
    assert set(rows[0]) == {"id", "cycle", "claim", "status", "pack_hash", "episode_id", "source"}
    assert rows[1]["pack_hash"] == "deadbeefcafe0000"
    md = board.markdown()
    assert "alpha" in md and "beta" in md
    assert "hyp-0001" in md and "hyp-0002" in md
    assert "deadbeefcafe" in md
    assert md == (board.root / "board.md").read_text(encoding="utf-8")


def test_board_round_trips_from_disk(tmp_path: Path) -> None:
    board = _board(tmp_path)
    a = board.open(claim="a", why="because", falsify="unless", cycle=3, source="ens-1")
    board.link_pack(a.id, "p1")
    board.link_episode(a.id, "ep-0007", supporting=False)
    board.update(a.id, status="closed")
    b = board.open(claim="b", cycle=3)

    reloaded = HypothesisBoard(board.root)
    assert [h.id for h in reloaded.list()] == [a.id, b.id]
    got = reloaded.get(a.id)
    assert got.to_dict() == a.to_dict()
    assert got.cycle == 3 and got.source == "ens-1"
    assert got.pack_hash == "p1" and got.episode_id == "ep-0007"
    assert got.refuting_episodes == ["ep-0007"]
    assert got.status == "closed"
    assert reloaded.cycle == 3
    # Next id continues the sequence after reload.
    assert reloaded.open(claim="c", cycle=3).id == "hyp-0003"


def test_live_hypothesis_dict_and_summary_carry_board_fields() -> None:
    h = LiveHypothesis(id="hyp-0042", cycle=4, source="x", claim="c", pack_hash="p", episode_id="e")
    d = h.to_dict()
    assert d["id"] == "hyp-0042" and d["cycle"] == 4 and d["source"] == "x"
    assert d["pack_hash"] == "p" and d["episode_id"] == "e"
    assert LiveHypothesis.from_dict(d).to_dict() == d
    assert h.summary()["id"] == "hyp-0042"
    # Legacy single-hypothesis files without the new keys still load.
    legacy = LiveHypothesis.from_dict({"claim": "old", "status": "testing"})
    assert legacy.id == "" and legacy.cycle == 0 and legacy.pack_hash is None
