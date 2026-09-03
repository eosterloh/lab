from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import re

from lab.jobs import Job
from lab.pack import ArtifactPack


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (s[:48] or "episode")


def make_title(pack: ArtifactPack, job: Job, ev: dict[str, Any]) -> str:
    hyp = pack.hypothesis.strip().split("\n")[0][:72] or pack.trainer
    ppl = ev.get("confirm_ppl")
    src = ev.get("confirm_source") or "none"
    return f"{hyp} — {job.status}, confirm_ppl={ppl} ({src})"


def make_card(ep: dict[str, Any]) -> str:
    cfg = ep.get("config") or {}
    ev = ep.get("eval") or {}
    loss = (ev.get("loss") or {}) if isinstance(ev.get("loss"), dict) else {}
    lines = [
        f"# {ep['title']}",
        "",
        f"- id: `{ep['id']}`",
        f"- status: {ep.get('job_status')}",
        f"- confirm_ppl: {ev.get('confirm_ppl')} ({ev.get('confirm_source')})",
        f"- val_loss / val_ppl: {loss.get('val_loss')} / {loss.get('val_ppl')}",
        f"- trainer: {ep.get('trainer')}  config: `{json.dumps(cfg, sort_keys=True)}`",
        f"- pack: `{ep.get('pack_hash')}`  job: `{ep.get('job_id')}`",
        f"- parent: {ep.get('parent_checkpoint')}",
        f"- suite: {ep.get('eval_suite_id')} v{ep.get('eval_suite_version')}",
        "",
        "## Hypothesis",
        "",
        ep.get("hypothesis") or "(none)",
        "",
        "## Verdict",
        "",
        ep.get("verdict") or "(none)",
        "",
        "## Next",
        "",
        ep.get("next_hint") or "(none)",
        "",
    ]
    return "\n".join(lines)


def _verdict(job: Job, ev: dict[str, Any]) -> str:
    if job.status != "succeeded":
        return f"train {job.status}: {job.error or 'no metrics'}"
    src = ev.get("confirm_source")
    ppl = ev.get("confirm_ppl")
    if src == "trainer_val_proxy":
        return f"dummy/proxy confirm_ppl={ppl}; not a frozen-holdout win, do not promote on this."
    return f"confirm_ppl={ppl} source={src} backend={ev.get('backend')}"


def _next_hint(job: Job, ev: dict[str, Any]) -> str:
    if job.status != "succeeded":
        return "Fix the trainer failure before changing data mix."
    if ev.get("confirm_source") == "trainer_val_proxy":
        return "Run against a real checkpoint so confirm_ppl is frozen holdout, not trainer val."
    return "Compare confirm_ppl to prior episodes before promoting."


class EpisodeStore:
    """One skill-like card per finished train+eval cycle."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = root / "index.jsonl"
        if not self.index_path.exists():
            self.index_path.write_text("", encoding="utf-8")

    def summaries(self, n: int = 20, query: str | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if query and query.lower() not in json.dumps(row).lower():
                continue
            rows.append(row)
        return rows[-n:]

    def load(self, ep_id: str) -> dict[str, Any]:
        path = self.root / f"{ep_id}.json"
        if not path.is_file():
            raise FileNotFoundError(ep_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def exists_for_job(self, job_id: str) -> bool:
        return any(s.get("job_id") == job_id for s in self.summaries(n=10_000))

    def record(
        self,
        *,
        cycle: int,
        pack: ArtifactPack,
        pack_hash: str,
        job: Job,
        ev: dict[str, Any],
    ) -> dict[str, Any]:
        if self.exists_for_job(job.id):
            for s in self.summaries(n=10_000):
                if s.get("job_id") == job.id:
                    return self.load(s["id"])
        n = len(self.summaries(n=10_000)) + 1
        title = make_title(pack, job, ev)
        ep_id = f"ep-{n:04d}-{_slug(title)[:24]}"
        ep = {
            "id": ep_id,
            "title": title,
            "ts": _now(),
            "cycle": cycle,
            "hypothesis": pack.hypothesis,
            "trainer": pack.trainer,
            "config": pack.config,
            "data_manifest": pack.data_manifest,
            "budgets": {
                "max_hours": pack.budgets.max_hours,
                "max_steps": pack.budgets.max_steps,
                "max_tokens": pack.budgets.max_tokens,
            },
            "pack_hash": pack_hash,
            "parent_checkpoint": pack.parent_checkpoint,
            "eval_suite_id": pack.eval_suite_id,
            "eval_suite_version": pack.eval_suite_version,
            "job_id": job.id,
            "job_status": job.status,
            "job_error": job.error,
            "metrics": job.metrics,
            "eval": ev,
            "verdict": _verdict(job, ev),
            "next_hint": _next_hint(job, ev),
        }
        (self.root / f"{ep_id}.json").write_text(
            json.dumps(ep, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.root / f"{ep_id}.md").write_text(make_card(ep), encoding="utf-8")
        summary = {
            "id": ep_id,
            "title": title,
            "ts": ep["ts"],
            "cycle": cycle,
            "job_id": job.id,
            "job_status": job.status,
            "confirm_ppl": ev.get("confirm_ppl"),
            "confirm_source": ev.get("confirm_source"),
            "pack_hash": pack_hash,
        }
        with self.index_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, sort_keys=True) + "\n")
        return ep

    def beliefs_markdown(self) -> str:
        rows = self.summaries(n=10_000)
        lines = ["# Current beliefs", ""]
        if not rows:
            return "# Current beliefs\n\n(none yet)\n"
        numbered = [r for r in rows if r.get("confirm_ppl") is not None]
        if numbered:
            best = min(numbered, key=lambda r: float(r["confirm_ppl"]))
            lines.append(
                f"- Best confirm_ppl: **{best['confirm_ppl']}** "
                f"(`{best['id']}`: {best['title']})"
            )
        lines.append("- Episodes:")
        for r in rows[-12:]:
            lines.append(
                f"  - `{r['id']}`: {r['title']}"
            )
        lines.append("")
        return "\n".join(lines)
