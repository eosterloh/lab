"""Persistent, versioned record of which model fills each ensemble role.

The roles themselves are experimental candidates: a later cycle may propose a
different model for ``tooler`` and, once it proves out, accept it. Every swap
bumps the role's version and lands in ``history`` so a run can be replayed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import time

from lab.ensemble.roles import ROLES, RoleSpec, null_specs

CANDIDATE_STATUSES = ("proposed", "accepted", "rejected")


class RoleRegistry:
    """JSON-backed registry at ``path``.

    Shape::

        {"roles": {role: RoleSpec.to_dict()},
         "candidates": [{id, role, model_dir, note, status, ts}],
         "history": [{ts, role, from_version, to_version, model_dir, note}]}
    """

    def __init__(self, path: Path, initial: dict[str, RoleSpec] | None = None) -> None:
        self.path = Path(path)
        self._roles: dict[str, RoleSpec] = {}
        self._candidates: list[dict[str, Any]] = []
        self._history: list[dict[str, Any]] = []
        if self.path.exists():
            self._load()
        else:
            specs = initial if initial is not None else null_specs()
            for role in ROLES:
                self._roles[role] = specs.get(role) or RoleSpec(role=role, backend="null")
            self.save()

    # ------------------------------------------------------------------ io

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        roles = raw.get("roles") or {}
        for role in ROLES:
            d = roles.get(role)
            self._roles[role] = (
                RoleSpec.from_dict(d) if d else RoleSpec(role=role, backend="null")
            )
        self._candidates = list(raw.get("candidates") or [])
        self._history = list(raw.get("history") or [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp.replace(self.path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "roles": {role: self._roles[role].to_dict() for role in ROLES},
            "candidates": [dict(c) for c in self._candidates],
            "history": [dict(h) for h in self._history],
        }

    # --------------------------------------------------------------- reads

    def current(self, role: str) -> RoleSpec:
        self._check_role(role)
        return self._roles[role]

    def specs(self) -> dict[str, RoleSpec]:
        return {role: self._roles[role] for role in ROLES}

    def candidates(self, status: str | None = None) -> list[dict[str, Any]]:
        if status is None:
            return [dict(c) for c in self._candidates]
        return [dict(c) for c in self._candidates if c.get("status") == status]

    def markdown(self) -> str:
        lines = [
            "| role | backend | model_dir | version |",
            "|---|---|---|---|",
        ]
        for role in ROLES:
            s = self._roles[role]
            lines.append(f"| {role} | {s.backend} | {s.model_dir or '-'} | {s.version} |")
        return "\n".join(lines) + "\n"

    # -------------------------------------------------------------- writes

    def propose(self, role: str, model_dir: str, note: str = "") -> str:
        self._check_role(role)
        if not model_dir:
            raise ValueError("model_dir must be non-empty")
        cid = f"cand-{len(self._candidates) + 1:04d}"
        self._candidates.append(
            {
                "id": cid,
                "role": role,
                "model_dir": str(model_dir),
                "note": note,
                "status": "proposed",
                "ts": time.time(),
            }
        )
        self.save()
        return cid

    def accept(self, candidate_id: str) -> RoleSpec:
        cand = self._find(candidate_id)
        if cand["status"] != "proposed":
            raise ValueError(f"candidate {candidate_id} is already {cand['status']}")
        role = cand["role"]
        prior = self._roles[role]
        new = RoleSpec(
            role=role,
            backend="infer",
            model_dir=cand["model_dir"],
            max_new_tokens=prior.max_new_tokens,
            version=prior.version + 1,
            note=cand.get("note", ""),
        )
        self._roles[role] = new
        cand["status"] = "accepted"
        self._history.append(
            {
                "ts": time.time(),
                "role": role,
                "from_version": prior.version,
                "to_version": new.version,
                "model_dir": new.model_dir,
                "note": new.note,
            }
        )
        self.save()
        return new

    def reject(self, candidate_id: str, note: str = "") -> None:
        cand = self._find(candidate_id)
        if cand["status"] != "proposed":
            raise ValueError(f"candidate {candidate_id} is already {cand['status']}")
        cand["status"] = "rejected"
        if note:
            cand["note"] = f"{cand['note']} | {note}".strip(" |") if cand.get("note") else note
        self.save()

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _check_role(role: str) -> None:
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")

    def _find(self, candidate_id: str) -> dict[str, Any]:
        for c in self._candidates:
            if c.get("id") == candidate_id:
                return c
        raise KeyError(candidate_id)
