from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import json


STATUSES = ("open", "testing", "killed", "closed")
LIVE_STATUSES = ("open", "testing")

DEFAULT_CHECKLIST: tuple[dict[str, Any], ...] = (
    {
        "id": "claim_written",
        "required_for": "train",
        "done": False,
        "note": "Falsifiable claim is written",
    },
    {
        "id": "pack_ready",
        "required_for": "train",
        "done": False,
        "note": "Hashed artifact pack exists",
    },
    {
        "id": "post_eval",
        "required_for": "close",
        "done": False,
        "note": "Post-train eval has run",
    },
    {
        "id": "episode_sealed",
        "required_for": "close",
        "done": False,
        "note": "Episode card written",
    },
    {
        "id": "holdout_not_proxy",
        "required_for": "promote",
        "done": False,
        "note": "confirm_ppl is frozen holdout, not trainer-val proxy",
    },
)


@dataclass
class CheckItem:
    id: str
    required_for: str
    done: bool = False
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LiveHypothesis:
    """One working theory. Supervisor writes; policy proposes.

    A blank instance (``id == ""``) is the "no hypothesis yet" placeholder the
    supervisor hands out when nothing is selected for the current cycle.
    """

    claim: str = ""
    why: str = ""
    falsify: str = ""
    status: str = "open"
    supporting_episodes: list[str] = field(default_factory=list)
    refuting_episodes: list[str] = field(default_factory=list)
    checklist: list[CheckItem] = field(default_factory=list)
    id: str = ""
    cycle: int = 0
    source: str = ""
    pack_hash: str | None = None
    episode_id: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        if not self.checklist:
            self.checklist = [CheckItem(**x) for x in DEFAULT_CHECKLIST]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cycle": self.cycle,
            "source": self.source,
            "claim": self.claim,
            "why": self.why,
            "falsify": self.falsify,
            "status": self.status,
            "pack_hash": self.pack_hash,
            "episode_id": self.episode_id,
            "note": self.note,
            "supporting_episodes": list(self.supporting_episodes),
            "refuting_episodes": list(self.refuting_episodes),
            "checklist": [c.to_dict() for c in self.checklist],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LiveHypothesis:
        raw_items = data.get("checklist") or DEFAULT_CHECKLIST
        items = [CheckItem(**{k: x[k] for k in ("id", "required_for", "done", "note") if k in x}) for x in raw_items]
        return cls(
            claim=str(data.get("claim") or ""),
            why=str(data.get("why") or ""),
            falsify=str(data.get("falsify") or ""),
            status=str(data.get("status") or "open"),
            supporting_episodes=list(data.get("supporting_episodes") or []),
            refuting_episodes=list(data.get("refuting_episodes") or []),
            checklist=items,
            id=str(data.get("id") or ""),
            cycle=int(data.get("cycle") or 0),
            source=str(data.get("source") or ""),
            pack_hash=data.get("pack_hash") or None,
            episode_id=data.get("episode_id") or None,
            note=str(data.get("note") or ""),
        )

    def item(self, item_id: str) -> CheckItem:
        for c in self.checklist:
            if c.id == item_id:
                return c
        raise KeyError(item_id)

    def mark(self, item_id: str, done: bool = True) -> None:
        self.item(item_id).done = done

    def open_for(self, gate: str) -> list[CheckItem]:
        return [c for c in self.checklist if c.required_for == gate and not c.done]

    def is_live(self) -> bool:
        return self.status in LIVE_STATUSES

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cycle": self.cycle,
            "source": self.source,
            "claim": self.claim,
            "status": self.status,
            "pack_hash": self.pack_hash,
            "episode_id": self.episode_id,
            "open_train": [c.id for c in self.open_for("train")],
            "open_close": [c.id for c in self.open_for("close")],
            "open_promote": [c.id for c in self.open_for("promote")],
            "checklist": [c.to_dict() for c in self.checklist],
        }

    def board_row(self) -> dict[str, Any]:
        """Compact row for the board index and ``list_hypotheses``."""
        return {
            "id": self.id,
            "cycle": self.cycle,
            "claim": self.claim,
            "status": self.status,
            "pack_hash": self.pack_hash,
            "episode_id": self.episode_id,
            "source": self.source,
        }

    def markdown(self) -> str:
        lines = [
            "# Live hypothesis",
            "",
        ]
        if self.id:
            lines.append(f"- id: `{self.id}` (cycle {self.cycle}, source `{self.source or 'policy'}`)")
        lines += [
            f"- status: `{self.status}`",
            f"- claim: {self.claim or '(none)'}",
            f"- why: {self.why or '(none)'}",
            f"- falsify: {self.falsify or '(none)'}",
        ]
        if self.pack_hash:
            lines.append(f"- pack: `{self.pack_hash}`")
        if self.episode_id:
            lines.append(f"- episode: `{self.episode_id}`")
        if self.note:
            lines.append(f"- note: {self.note}")
        lines += ["", "## Checklist", ""]
        for c in self.checklist:
            box = "[x]" if c.done else "[ ]"
            lines.append(f"- {box} `{c.id}` ({c.required_for}): {c.note}")
        lines.append("")
        return "\n".join(lines)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class HypothesisBoard:
    """Many live hypotheses per cycle.

    Directory layout under ``root``: ``index.jsonl`` (one compact row per
    hypothesis, rewritten on every save), ``<id>.json`` (full record) and
    ``board.md`` (human view). All hypotheses are held in memory; ``get``
    returns the live object, so callers mutate and then ``save`` it.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = root / "index.jsonl"
        self.md_path = root / "board.md"
        self._items: dict[str, LiveHypothesis] = {}
        for path in sorted(self.root.glob("hyp-*.json")):
            hyp = LiveHypothesis.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if hyp.id:
                self._items[hyp.id] = hyp
        # The cycle the board believes is current: bumped by open() and roll_cycle().
        self.cycle = max((h.cycle for h in self._items.values()), default=0)
        if not self.index_path.exists():
            self._write_index()

    # -- persistence -------------------------------------------------------

    def _ordered(self) -> list[LiveHypothesis]:
        return [self._items[k] for k in sorted(self._items)]

    def _write_index(self) -> None:
        rows = [json.dumps(h.board_row(), sort_keys=True) for h in self._ordered()]
        _write_atomic(self.index_path, "".join(r + "\n" for r in rows))
        _write_atomic(self.md_path, self.markdown())

    def save(self, hyp: LiveHypothesis) -> None:
        if not hyp.id:
            raise ValueError("hypothesis has no id; open() it on the board first")
        self._items[hyp.id] = hyp
        _write_atomic(
            self.root / f"{hyp.id}.json",
            json.dumps(hyp.to_dict(), indent=2, sort_keys=True),
        )
        self._write_index()

    def _next_id(self) -> str:
        n = 0
        for key in self._items:
            try:
                n = max(n, int(key.split("-", 1)[1]))
            except (IndexError, ValueError):
                continue
        return f"hyp-{n + 1:04d}"

    # -- public API --------------------------------------------------------

    def open(
        self,
        *,
        claim: str,
        why: str = "",
        falsify: str = "",
        cycle: int,
        source: str = "policy",
    ) -> LiveHypothesis:
        hyp = LiveHypothesis(
            id=self._next_id(),
            cycle=int(cycle),
            source=str(source or "policy"),
            claim=str(claim or "").strip(),
            why=str(why or "").strip(),
            falsify=str(falsify or "").strip(),
        )
        if hyp.claim:
            hyp.mark("claim_written", True)
            hyp.status = "testing"
        self.cycle = max(self.cycle, hyp.cycle)
        self.save(hyp)
        return hyp

    def get(self, hyp_id: str) -> LiveHypothesis:
        try:
            return self._items[hyp_id]
        except KeyError:
            raise KeyError(f"unknown hypothesis {hyp_id!r}") from None

    def has(self, hyp_id: str | None) -> bool:
        return bool(hyp_id) and hyp_id in self._items

    def list(
        self, *, cycle: int | None = None, status: str | None = None
    ) -> list[LiveHypothesis]:
        out = self._ordered()
        if cycle is not None:
            out = [h for h in out if h.cycle == int(cycle)]
        if status is not None:
            out = [h for h in out if h.status == status]
        return out

    def update(
        self,
        hyp_id: str,
        *,
        claim: str | None = None,
        why: str | None = None,
        falsify: str | None = None,
        status: str | None = None,
    ) -> LiveHypothesis:
        hyp = self.get(hyp_id)
        if claim is not None:
            hyp.claim = str(claim).strip()
        if why is not None:
            hyp.why = str(why).strip()
        if falsify is not None:
            hyp.falsify = str(falsify).strip()
        if status is not None:
            if status not in STATUSES:
                raise ValueError(f"status must be one of {STATUSES}")
            hyp.status = status
        if hyp.claim:
            hyp.mark("claim_written", True)
            if hyp.status == "open":
                hyp.status = "testing"
        self.save(hyp)
        return hyp

    def mark(self, hyp_id: str, item_id: str, done: bool = True) -> None:
        hyp = self.get(hyp_id)
        hyp.mark(item_id, done)
        self.save(hyp)

    def link_pack(self, hyp_id: str, pack_hash: str) -> None:
        hyp = self.get(hyp_id)
        hyp.pack_hash = str(pack_hash)
        hyp.mark("pack_ready", True)
        self.save(hyp)

    def link_episode(self, hyp_id: str, episode_id: str, *, supporting: bool) -> None:
        hyp = self.get(hyp_id)
        hyp.episode_id = str(episode_id)
        bucket = hyp.supporting_episodes if supporting else hyp.refuting_episodes
        if episode_id not in bucket:
            bucket.append(str(episode_id))
        hyp.mark("post_eval", True)
        hyp.mark("episode_sealed", True)
        self.save(hyp)

    def close(self, hyp_id: str, status: str = "closed") -> None:
        if status not in ("closed", "killed"):
            raise ValueError("close status must be 'closed' or 'killed'")
        hyp = self.get(hyp_id)
        hyp.status = status
        self.save(hyp)

    def roll_cycle(self, new_cycle: int) -> list[str]:
        """Retire everything from cycles before ``new_cycle``.

        Live hypotheses that never got an episode are killed with note
        "untested"; live ones that did get an episode are closed (their verdict
        lives on the episode card). Returns the killed ids.
        """
        killed: list[str] = []
        for hyp in self._ordered():
            if hyp.cycle >= int(new_cycle) or not hyp.is_live():
                continue
            if hyp.episode_id is None:
                hyp.status = "killed"
                hyp.note = "untested"
                killed.append(hyp.id)
            else:
                hyp.status = "closed"
            self.save(hyp)
        self.cycle = max(self.cycle, int(new_cycle))
        return killed

    def active(self, cycle: int | None = None) -> LiveHypothesis | None:
        """Best guess at "the" hypothesis when none is explicitly selected.

        Most recent live hypothesis with a pack but no episode yet; else the
        most recent live one in the current cycle; else None.
        """
        current = self.cycle if cycle is None else int(cycle)
        live = [h for h in self._ordered() if h.is_live()]
        armed = [h for h in live if h.pack_hash and h.episode_id is None]
        if armed:
            return armed[-1]
        this_cycle = [h for h in live if h.cycle == current]
        if this_cycle:
            return this_cycle[-1]
        return None

    def summary(self, cycle: int | None = None) -> list[dict[str, Any]]:
        return [h.board_row() for h in self.list(cycle=cycle)]

    def markdown(self) -> str:
        lines = ["# Hypothesis board", ""]
        items = self._ordered()
        if not items:
            lines += ["(no hypotheses yet)", ""]
            return "\n".join(lines)
        by_cycle: dict[int, list[LiveHypothesis]] = {}
        for h in items:
            by_cycle.setdefault(h.cycle, []).append(h)
        for cyc in sorted(by_cycle):
            lines += [f"## Cycle {cyc}", ""]
            for h in by_cycle[cyc]:
                pack = f" pack=`{h.pack_hash[:12]}`" if h.pack_hash else ""
                ep = f" episode=`{h.episode_id}`" if h.episode_id else ""
                note = f" ({h.note})" if h.note else ""
                lines.append(f"- `{h.id}` [{h.status}]{note} {h.claim or '(no claim)'}{pack}{ep}")
            lines.append("")
        return "\n".join(lines)


class HypothesisStore:
    """Compatibility facade: the single "current" hypothesis view over a board.

    ``current`` resolves to the explicitly selected hypothesis (``current_id``),
    else the board's ``active()`` one, else a blank placeholder. ``save``
    mirrors the current hypothesis to ``hypothesis.json`` / ``hypothesis.md``
    next to state.json, as the single-hypothesis store used to.
    """

    def __init__(
        self,
        path: Path,
        board: HypothesisBoard,
        *,
        current_id: Callable[[], str | None],
        cycle: Callable[[], int],
    ) -> None:
        self.path = path
        self.md_path = path.with_suffix(".md")
        self.board = board
        self._current_id = current_id
        self._cycle = cycle
        self.save()

    @property
    def archive_path(self) -> Path:
        return self.board.index_path

    @property
    def current(self) -> LiveHypothesis:
        hyp_id = self._current_id()
        if hyp_id and self.board.has(hyp_id):
            return self.board.get(hyp_id)
        active = self.board.active(cycle=self._cycle())
        return active if active is not None else LiveHypothesis()

    @property
    def archive(self) -> list[dict[str, Any]]:
        """Closed/killed hypotheses from cycles before the current one."""
        cycle = self._cycle()
        return [
            h.to_dict()
            for h in self.board.list()
            if h.cycle < cycle and h.status in ("closed", "killed")
        ]

    def save(self) -> None:
        cur = self.current
        if cur.id:
            self.board.save(cur)
        _write_atomic(self.path, json.dumps(cur.to_dict(), indent=2, sort_keys=True))
        _write_atomic(self.md_path, cur.markdown())

    def roll(self) -> LiveHypothesis:
        """Retire earlier cycles' hypotheses on the board (see ``roll_cycle``)."""
        self.board.roll_cycle(self._cycle())
        self.save()
        return self.current

    def update(
        self,
        *,
        claim: str | None = None,
        why: str | None = None,
        falsify: str | None = None,
        status: str | None = None,
    ) -> LiveHypothesis:
        cur = self.current
        if cur.id:
            hyp = self.board.update(cur.id, claim=claim, why=why, falsify=falsify, status=status)
        else:
            hyp = self.board.open(
                claim=claim or "", why=why or "", falsify=falsify or "", cycle=self._cycle()
            )
            if status is not None:
                hyp = self.board.update(hyp.id, status=status)
        self.save()
        return hyp
