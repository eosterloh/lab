"""core v1 eval: trainer loss/PPL always; holdout/benches only with a scorer."""

from __future__ import annotations

from lab.evals import run_eval
from lab.pack import ArtifactPack
from lab.policy import DummyPolicy, dummy_pack
from lab.scorer import mean_nll, ppl


class ConstScorer:
    """Deterministic NLL=1 per token so holdout PPL is exp(1)."""

    def sequence_nll(self, text: str) -> tuple[float, int]:
        n = max(len(text.split()), 1)
        return 1.0 * n, n

    def continuation_nll(self, context: str, ending: str) -> tuple[float, int]:
        n = max(len(ending.split()), 1)
        cheap = ending.strip().lower()
        good = cheap.startswith(("drank", "sunlight", "put it in a sealed", "bridge", "coat"))
        return (0.1 * n, n) if good else (2.0 * n, n)


def _pack() -> ArtifactPack:
    return ArtifactPack.from_dict(dummy_pack({"cycle": 1, "subject_checkpoint": "x"}))


def test_dummy_job_eval_is_trainer_val_proxy_not_holdout(sup) -> None:
    """No checkpoint → confirm_ppl copies trainer val_ppl; benches skipped."""
    sup.run_policy(DummyPolicy(), max_cycles=1, max_steps=50)
    ev = sup.state.last_eval
    assert ev["suite_id"] == "core"
    assert ev["primary"] == "confirm_ppl"
    assert ev["higher_is_better"] is False
    assert ev["loss"]["train_loss"] is not None
    assert ev["loss"]["val_loss"] is not None
    assert ev["loss"]["val_ppl"] is not None
    assert ev["confirm_ppl"] == ev["loss"]["val_ppl"]
    assert ev["confirm_source"] == "trainer_val_proxy"
    assert "hellaswag" not in ev["benchmarks"]


def test_injected_scorer_fills_tune_confirm_ood_and_mc_benches(sup) -> None:
    pack = _pack()
    sup.save_pack(pack)
    result = run_eval(sup.cfg, pack, {"val_loss": 1.2, "val_ppl": 3.32}, scorer=ConstScorer())
    assert result["backend"] == "model"
    assert result["confirm_source"] == "frozen_holdout"
    assert result["loss"]["holdout"]["confirm"]["ppl"] == round(ppl(1.0), 4)
    assert result["loss"]["holdout"]["tune"]["n"] == 20
    assert result["loss"]["holdout"]["ood"]["n"] == 15
    assert result["benchmarks"]["hellaswag"]["n"] == 16
    assert result["benchmarks"]["arc_easy"]["n"] == 16
    assert result["benchmarks"]["piqa"]["n"] == 16
    assert result["benchmarks"]["piqa"]["chance"] == 0.5
    assert result["confirm_ppl"] == result["loss"]["holdout"]["confirm"]["ppl"]


def test_mean_nll_is_token_weighted_and_ppl_is_exp() -> None:
    assert mean_nll([(2.0, 2), (4.0, 2)]) == 1.5
    assert round(ppl(0.0), 4) == 1.0
    assert mean_nll([]) is None


def test_unknown_eval_suite_version_is_not_promotable(sup) -> None:
    pack = ArtifactPack.from_dict({**dummy_pack({"cycle": 1}), "eval_suite_id": "nope", "eval_suite_version": 9})
    result = run_eval(sup.cfg, pack, {"val_loss": 1.0, "val_ppl": 2.718})
    assert result["suite_known"] is False
    assert result["confirm_score"] is None
