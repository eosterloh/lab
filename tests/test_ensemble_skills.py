"""Tests for the ensemble skill loader and the markdown skill cards."""

from __future__ import annotations

import re

import pytest

from lab.ensemble.skills import (
    ROLE_SKILLS,
    SKILLS_DIR,
    list_skills,
    load_skill,
    skill_summary,
    skills_for_role,
)

EXPECTED = sorted(
    ["experiment_design", "infer_api", "lab_harness", "pytorch_transformer", "sandbox_rules"]
)

CHECKPOINT_CONFIG_KEYS = ("hidden", "layers", "heads", "seq_len", "vocab")
METRICS_KEYS = ("train_loss", "val_loss", "val_ppl", "steps", "lr", "tokens_seen", "n_params")


def test_list_skills_exact() -> None:
    assert list_skills() == EXPECTED


def test_load_skill_accepts_stem_and_md() -> None:
    a = load_skill("pytorch_transformer")
    b = load_skill("pytorch_transformer.md")
    assert a == b
    assert a.startswith("# pytorch_transformer")


def test_load_skill_unknown_lists_available() -> None:
    with pytest.raises(KeyError) as ei:
        load_skill("does_not_exist")
    msg = str(ei.value)
    assert "does_not_exist" in msg
    for name in EXPECTED:
        assert name in msg


def test_load_skill_rejects_path_escape() -> None:
    with pytest.raises(KeyError):
        load_skill("../__init__")


def test_skills_for_role_coder_order_and_content() -> None:
    text = skills_for_role("coder")
    assert text.startswith("## skill: pytorch_transformer\n")
    headers = re.findall(r"^## skill: (\w+)$", text, flags=re.M)
    assert headers == ROLE_SKILLS["coder"]
    for name in ROLE_SKILLS["coder"]:
        assert load_skill(name) in text


def test_skills_for_role_other_roles() -> None:
    for role in ("tooler", "thinker"):
        headers = re.findall(r"^## skill: (\w+)$", skills_for_role(role), flags=re.M)
        assert headers == ROLE_SKILLS[role]


def test_skills_for_role_unknown_is_empty() -> None:
    assert skills_for_role("janitor") == ""
    assert skills_for_role("") == ""


def test_skills_for_role_truncates_to_budget() -> None:
    text = skills_for_role("coder", max_chars=2000)
    assert len(text) <= 2000
    assert "[truncated]" in text
    # Every coder skill still gets a header inside the budget.
    for name in ROLE_SKILLS["coder"]:
        assert f"## skill: {name}" in text


def test_skills_for_role_no_truncation_when_it_fits() -> None:
    full = skills_for_role("thinker")
    assert skills_for_role("thinker", max_chars=len(full) + 10) == full
    assert "[truncated]" not in full


def test_skill_summary_one_line_each() -> None:
    summary = skill_summary()
    assert sorted(summary) == EXPECTED
    for name, line in summary.items():
        assert line, name
        assert "\n" not in line
        assert not line.startswith("#")
        assert line == name  # title line is "# <name>"


@pytest.mark.parametrize("name", EXPECTED)
def test_skill_file_shape(name: str) -> None:
    text = load_skill(name)
    assert text.startswith("# "), name
    assert len(text) <= 10_000, (name, len(text))
    assert (SKILLS_DIR / f"{name}.md").is_file()


def _first_python_fence(text: str) -> str:
    m = re.search(r"```python\n(.*?)```", text, flags=re.S)
    assert m, "no python fence found"
    return m.group(1)


def test_pytorch_skill_script_compiles() -> None:
    src = _first_python_fence(load_skill("pytorch_transformer"))
    assert "def build_model" in src
    assert "def main" in src
    compile(src, "skill", "exec")


def test_pytorch_skill_mentions_checkpoint_and_metrics_keys() -> None:
    text = load_skill("pytorch_transformer")
    for key in CHECKPOINT_CONFIG_KEYS:
        assert f'"{key}"' in text, key
    for key in METRICS_KEYS:
        assert f'"{key}"' in text, key
    assert "LAB_TRAIN_DEVICE" in text
    assert "norm_first=True" in text
    assert "batch_first=True" in text
    assert "clip_grad_norm_" in text


def test_lab_harness_lists_phase_tools() -> None:
    from lab.types import PHASE_TOOLS

    text = load_skill("lab_harness")
    for tools in PHASE_TOOLS.values():
        for tool in tools:
            assert tool in text, tool


def test_pyproject_ships_skills() -> None:
    from pathlib import Path

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    assert '"ensemble/skills/*.md"' in pyproject.read_text(encoding="utf-8")
