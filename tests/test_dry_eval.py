"""Dry-run harness evals. Dummy trainer only; the suite must not touch a GPU lock outside tmp."""

from __future__ import annotations

from lab.dry_eval import run_dry_evals
from lab.hooks import HOOK_POINTS


def test_dry_eval_suite_passes() -> None:
    report = run_dry_evals()
    failed = [c for c in report["cases"] if not c["ok"]]
    assert not failed, failed
    assert report["gpu"] is False
    assert report["ok"] is True


def test_hook_points_name_the_open_slot() -> None:
    assert HOOK_POINTS["between_trials"]["status"] == "live"
    live = [name for name, spec in HOOK_POINTS.items() if spec["status"] == "live"]
    assert "between_trials" in live
    assert "on_step" in live
