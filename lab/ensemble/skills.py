"""Markdown "skills" that ground the ensemble roles in real harness facts.

Each ``skills/<name>.md`` is a short reference card written for a small model.
``skills_for_role`` concatenates a role's cards in a fixed order so prompts stay
deterministic; ``max_chars`` trims each card proportionally when the budget is tight.
"""

from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

ROLE_SKILLS: dict[str, list[str]] = {
    "coder": ["pytorch_transformer", "lab_harness", "infer_api", "sandbox_rules"],
    "tooler": ["lab_harness", "sandbox_rules", "experiment_design"],
    "thinker": ["experiment_design", "lab_harness"],
}

_TRUNC = "\n[truncated]"
_SEP = "\n\n"


def _stem(name: str) -> str:
    name = str(name).strip()
    return name[:-3] if name.endswith(".md") else name


def list_skills() -> list[str]:
    """Sorted stem names of every ``*.md`` under SKILLS_DIR."""
    if not SKILLS_DIR.is_dir():
        return []
    return sorted(p.stem for p in SKILLS_DIR.glob("*.md") if p.is_file())


def load_skill(name: str) -> str:
    """Return the markdown for ``name`` (``"foo"`` or ``"foo.md"``)."""
    stem = _stem(name)
    path = SKILLS_DIR / f"{stem}.md"
    if not stem or "/" in stem or "\\" in stem or not path.is_file():
        available = ", ".join(list_skills()) or "(none)"
        raise KeyError(f"{name}: unknown skill; available: {available}")
    return path.read_text(encoding="utf-8")


def _header(name: str) -> str:
    return f"## skill: {name}\n"


def skills_for_role(role: str, max_chars: int | None = None) -> str:
    """Concatenate the role's skills in ROLE_SKILLS order. Unknown role -> ``""``."""
    names = ROLE_SKILLS.get(role)
    if not names:
        return ""
    bodies = [load_skill(n) for n in names]
    full = _SEP.join(_header(n) + b for n, b in zip(names, bodies))
    if max_chars is None or len(full) <= max_chars:
        return full
    if max_chars <= 0:
        return ""
    # Proportional truncation: fixed overhead first, then split what is left by body size.
    overhead = sum(len(_header(n)) + len(_TRUNC) for n in names) + len(_SEP) * (len(names) - 1)
    budget = max(max_chars - overhead, 0)
    total = sum(len(b) for b in bodies) or 1
    parts: list[str] = []
    for name, body in zip(names, bodies):
        allot = (budget * len(body)) // total
        parts.append(_header(name) + body[:allot].rstrip() + _TRUNC)
    out = _SEP.join(parts)
    return out[:max_chars]


def skill_summary() -> dict[str, str]:
    """``{name: first non-empty line without leading '#'}`` for every skill."""
    out: dict[str, str] = {}
    for name in list_skills():
        summary = ""
        for line in load_skill(name).splitlines():
            line = line.strip()
            if line:
                summary = line.lstrip("#").strip()
                break
        out[name] = summary
    return out


__all__ = [
    "ROLE_SKILLS",
    "SKILLS_DIR",
    "list_skills",
    "load_skill",
    "skill_summary",
    "skills_for_role",
]
