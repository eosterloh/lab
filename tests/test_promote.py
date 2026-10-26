"""Promotion uses the best frozen-holdout trial of the latest cycle."""

from __future__ import annotations

from pathlib import Path
import json

from lab.promote import promote


def _eval(ppl: float, source: str = "frozen_holdout", **extra: object) -> dict:
    ev = {
        "confirm_score": ppl,
        "confirm_ppl": ppl,
        "confirm_source": source,
        "suite_id": "core",
        "suite_version": 1,
        "suite_known": True,
        "higher_is_better": False,
        "score": ppl,
    }
    ev.update(extra)
    return ev


def _plant(tmp_path: Path, episodes: list[dict], last_eval: dict) -> Path:
    run = tmp_path / "run"
    (run / "episodes").mkdir(parents=True)
    (run / "state.json").write_text(
        json.dumps(
            {
                "last_eval": last_eval,
                "pack_hash": last_eval.get("pack_hash", "hash-last"),
                "current_job_id": "job-last",
                "last_metrics": {"backend": "dummy"},
            }
        ),
        encoding="utf-8",
    )
    for ep in episodes:
        (run / "episodes" / f"{ep['id']}.json").write_text(json.dumps(ep), encoding="utf-8")
    return run


def _ep(ep_id: str, ppl: float, *, trial: int, source: str = "frozen_holdout") -> dict:
    return {
        "id": ep_id,
        "cycle": 1,
        "trial": trial,
        "pack_hash": f"hash-{ep_id}",
        "job_id": f"job-{trial}",
        "eval": _eval(ppl, source),
        "metrics": {"val_ppl": ppl},
    }


def test_promote_picks_the_best_holdout_not_the_last_trial(tmp_path: Path) -> None:
    last = _eval(4.0)
    run = _plant(
        tmp_path,
        [_ep("ep-best", 1.5, trial=1), _ep("ep-last", 4.0, trial=2)],
        last,
    )
    result = promote(run)
    assert result["promoted"] is True
    assert result["confirm_score"] == 1.5
    assert result["episode_id"] == "ep-best"
    assert result["pack_hash"] == "hash-ep-best"
    assert result["confirm_source"] == "frozen_holdout"


def test_promote_ignores_a_better_looking_proxy(tmp_path: Path) -> None:
    run = _plant(
        tmp_path,
        [
            _ep("ep-real", 2.5, trial=1),
            _ep("ep-proxy", 0.1, trial=2, source="trainer_val_proxy"),
        ],
        _eval(0.1, "trainer_val_proxy"),
    )
    result = promote(run)
    assert result["promoted"] is True
    assert result["episode_id"] == "ep-real"
    assert result["confirm_score"] == 2.5


def test_promote_refuses_when_the_cycle_has_no_holdout(tmp_path: Path) -> None:
    run = _plant(
        tmp_path,
        [_ep("ep-proxy", 0.1, trial=1, source="trainer_val_proxy")],
        _eval(0.1, "trainer_val_proxy"),
    )
    result = promote(run)
    assert result["promoted"] is False
    assert "trainer_val_proxy" in result["reason"]
    assert not (run / "champion.json").exists()


def test_promote_min_delta_and_tie_still_apply_to_the_best_holdout(tmp_path: Path) -> None:
    run = _plant(
        tmp_path,
        [_ep("ep-a", 1.9, trial=1), _ep("ep-b", 3.0, trial=2)],
        _eval(3.0),
    )
    (run / "champion.json").write_text(
        json.dumps(
            {
                "confirm_score": 2.0,
                "suite_id": "core",
                "suite_version": 1,
                "higher_is_better": False,
            }
        ),
        encoding="utf-8",
    )
    tiny = promote(run, min_delta=0.5)
    assert tiny["promoted"] is False
    assert tiny["reason"] == "confirm_score did not beat champion"
    assert tiny["confirm_score"] == 1.9

    # Same best score as the champion is a tie, not a win.
    tied = _plant(
        tmp_path / "tie",
        [_ep("ep-tie", 2.0, trial=1)],
        _eval(2.0),
    )
    (tied / "champion.json").write_text(
        json.dumps(
            {
                "confirm_score": 2.0,
                "suite_id": "core",
                "suite_version": 1,
                "higher_is_better": False,
            }
        ),
        encoding="utf-8",
    )
    again = promote(tied)
    assert again["promoted"] is False
    assert again["reason"] == "confirm_score did not beat champion"
