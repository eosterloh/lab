from __future__ import annotations

from typing import Any
import json
from pathlib import Path

from lab.state import RunState

_HOLDOUT = "frozen_holdout"


def _score_of(ev: dict[str, Any]) -> float | None:
    raw = ev.get("confirm_score")
    if raw is None:
        raw = ev.get("confirm_ppl")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _load_episodes(run_dir: Path) -> list[dict[str, Any]]:
    root = run_dir / "episodes"
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _select_eval(
    state: RunState, episodes: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Best frozen-holdout eval of the latest cycle, or a refusal reason.

    Returns ``(episode or None, eval, reason)``. ``reason`` is set when there
    is nothing that may be promoted.
    """
    if episodes:
        cycles = [int(ep["cycle"]) for ep in episodes if ep.get("cycle") is not None]
        target = max(cycles) if cycles else None
        pool = [
            ep
            for ep in episodes
            if target is None or int(ep.get("cycle") or -1) == target
        ]
    else:
        pool = []

    eligible: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for ep in pool:
        ev = ep.get("eval") if isinstance(ep.get("eval"), dict) else {}
        if ev.get("confirm_source") != _HOLDOUT:
            continue
        if ev.get("suite_known") is False:
            continue
        if _score_of(ev) is None:
            continue
        eligible.append((ep, ev))

    if not eligible:
        last = state.last_eval or {}
        sources = {
            (ep.get("eval") or {}).get("confirm_source")
            for ep in pool
            if isinstance(ep.get("eval"), dict)
        }
        if last.get("confirm_source") == "trainer_val_proxy" or "trainer_val_proxy" in sources:
            return None, None, "confirm_source is trainer_val_proxy, not frozen_holdout"
        if last.get("suite_known") is False:
            return None, None, "unknown eval suite version"
        if not pool and last.get("confirm_source") == _HOLDOUT and _score_of(last) is not None:
            return None, last, None
        if not pool and _score_of(last) is None:
            return None, None, "no confirmation eval"
        return None, None, "no frozen-holdout confirmation to promote"

    higher = any(bool(ev.get("higher_is_better", False)) for _, ev in eligible)

    def key(pair: tuple[dict[str, Any], dict[str, Any]]) -> float:
        return _score_of(pair[1]) or 0.0

    best_ep, best_ev = (max if higher else min)(eligible, key=key)
    if best_ev.get("suite_known") is False:
        return None, None, "unknown eval suite version"
    return best_ep, best_ev, None


def promote(run_dir: Path, min_delta: float = 0.0) -> dict[str, Any]:
    """Promote the best frozen-holdout trial of the latest cycle.

    Direction comes from the suite (PPL: lower). A trainer_val_proxy score
    never becomes champion, and a later worse trial does not hide an earlier
    better one from the same cycle.
    """
    run_dir = Path(run_dir)
    state = RunState.from_dict(json.loads((run_dir / "state.json").read_text(encoding="utf-8")))
    episode, ev, reason = _select_eval(state, _load_episodes(run_dir))
    if reason or ev is None:
        return {"promoted": False, "reason": reason or "no confirmation eval"}
    if ev.get("suite_known") is False:
        return {"promoted": False, "reason": "unknown eval suite version"}

    higher = bool(ev.get("higher_is_better", False))
    confirm = _score_of(ev)
    if confirm is None:
        return {"promoted": False, "reason": "no confirmation eval"}
    champ_path = run_dir / "champion.json"
    if champ_path.is_file():
        champ = json.loads(champ_path.read_text(encoding="utf-8"))
        if champ.get("suite_id") != ev.get("suite_id") or champ.get("suite_version") != ev.get("suite_version"):
            return {"promoted": False, "reason": "eval suite changed; bump is a domain jump, not a win"}
        best = float(champ["confirm_score"])
        # A tie is not a win: an unchanged score must never re-promote.
        if higher:
            beat = confirm > best and confirm >= best + min_delta
        else:
            beat = confirm < best and confirm <= best - min_delta
        if not beat:
            return {
                "promoted": False,
                "reason": "confirm_score did not beat champion",
                "confirm_score": confirm,
                "champion": best,
                "higher_is_better": higher,
                "confirm_source": ev.get("confirm_source"),
                "episode_id": None if episode is None else episode.get("id"),
            }

    record = {
        "pack_hash": (episode or {}).get("pack_hash") or state.pack_hash,
        "job_id": (episode or {}).get("job_id") or state.current_job_id,
        "episode_id": None if episode is None else episode.get("id"),
        "trial": None if episode is None else episode.get("trial"),
        "confirm_score": ev.get("confirm_score") if ev.get("confirm_score") is not None else ev.get("confirm_ppl"),
        "confirm_ppl": ev.get("confirm_ppl"),
        "confirm_source": ev.get("confirm_source"),
        "score": ev.get("score"),
        "suite_id": ev.get("suite_id"),
        "suite_version": ev.get("suite_version"),
        "higher_is_better": higher,
        "metrics": (episode or {}).get("metrics") if episode is not None else state.last_metrics,
        "loss": ev.get("loss"),
        "benchmarks": ev.get("benchmarks"),
    }
    champ_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return {"promoted": True, **record}
