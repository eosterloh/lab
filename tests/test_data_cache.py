"""Allowlisted HF cache: no Hub, frozen eval stays off-limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from lab.data_cache import (
    DEFAULT_MIX,
    PKG_FROZEN_EVAL,
    cache_file,
    compose_hf_source,
    ensure_hf_text,
    materialize_sources,
    parse_hf_source,
    seed_file,
)


def test_unknown_hf_source_is_rejected() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        parse_hf_source("hf:openai/gsm8k:train:100")


def test_frozen_eval_token_is_rejected_as_source() -> None:
    with pytest.raises(ValueError, match="frozen_eval"):
        parse_hf_source("hf:lab/frozen_eval:train:10")


def test_compose_hf_source_applies_split_and_n() -> None:
    src = compose_hf_source(
        {"source": "roneneldan/TinyStories", "split": "train", "n": 10000}
    )
    assert src == "hf:roneneldan/TinyStories:train:10000"


def test_prefetch_data_serves_seeded_cache(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call(
        "prefetch_data",
        {"source": "hf:roneneldan/TinyStories", "split": "train", "n": 10000},
    )
    assert out["ok"] is True
    assert Path(out["path"]).is_file()
    assert out["bytes"] >= 64
    obs = sup.observe()
    assert obs["data_cache"]["bytes"] >= 64
    assert "hf:roneneldan/TinyStories" in obs["data_cache"]["allowlist"]


def test_eval_phase_rejects_prefetch_data(sup) -> None:
    out = sup.call("prefetch_data", {"source": "hf:roneneldan/TinyStories"})
    assert out["ok"] is False
    assert "not allowed" in out["error"]


def test_prefetch_unknown_source_fails_closed(sup) -> None:
    assert sup.call("enter_research")["ok"]
    out = sup.call("prefetch_data", {"source": "hf:openai/gsm8k", "n": 10})
    assert out["ok"] is False
    assert "allowlist" in out["error"]


def test_cache_miss_fails_when_network_disabled(tmp_path: Path) -> None:
    spec, split, n = parse_hf_source(DEFAULT_MIX[0])
    dest = cache_file(tmp_path, spec, split, n)
    assert not dest.exists()
    with pytest.raises(FileNotFoundError, match="network disabled"):
        ensure_hf_text(DEFAULT_MIX[0], tmp_path, allow_network=False)


def test_materialize_skips_uncached_hf_when_offline(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    cache = tmp_path / "cache"
    a = "hf:roneneldan/TinyStories:train:10000"
    b = "hf:HuggingFaceFW/fineweb-edu:train:10000"
    spec_a, split_a, n_a = parse_hf_source(a)
    seed_file(cache_file(cache, spec_a, split_a, n_a), "AAA stories about cats.\n" * 8)
    dest = materialize_sources([a, b], cache, job_dir, allow_network=False)
    text = dest.read_text(encoding="utf-8")
    assert "AAA stories" in text


def test_materialize_concatenates_multiple_sources(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    cache = tmp_path / "cache"
    a = "hf:roneneldan/TinyStories:train:10000"
    b = "hf:wikimedia/wikipedia:train:5000"
    spec_a, split_a, n_a = parse_hf_source(a)
    spec_b, split_b, n_b = parse_hf_source(b)
    seed_file(cache_file(cache, spec_a, split_a, n_a), "AAA stories about cats.\n" * 8)
    seed_file(cache_file(cache, spec_b, split_b, n_b), "BBB encyclopedia wheat.\n" * 8)
    dest = materialize_sources([a, b], cache, job_dir, allow_network=False)
    text = dest.read_text(encoding="utf-8")
    assert "AAA stories" in text
    assert "BBB encyclopedia" in text


def test_materialize_rejects_frozen_eval_path(tmp_path: Path, sup) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    frozen = sup.cfg.frozen_eval_dir
    sneak = frozen / "lm_confirm.jsonl"
    assert sneak.is_file()
    with pytest.raises(ValueError, match="frozen_eval"):
        materialize_sources(
            [str(sneak)],
            tmp_path / "cache",
            job_dir,
            allow_network=False,
            frozen_eval_dir=frozen,
        )


def test_package_frozen_eval_is_not_a_train_source(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    sneak = next(PKG_FROZEN_EVAL.glob("*.jsonl"))
    with pytest.raises(ValueError, match="frozen_eval"):
        materialize_sources([str(sneak)], tmp_path / "cache", job_dir, allow_network=False)
