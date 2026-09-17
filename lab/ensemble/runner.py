"""Run N ensembles in parallel, then commit their output to the supervisor serially.

``run_research`` only reads (and writes inside each ensemble's sandbox prefix).
``commit`` is the single writer: it opens hypotheses, freezes packs, dedupes by
pack hash, falls back to one harness-authored pack when the models produced
none, and queues the candidates as this cycle's trials.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
import json
import threading

from lab.ensemble.ensemble import Ensemble, EnsembleConfig, EnsembleResult
from lab.ensemble.prompts import VARIANT_HINTS
from lab.ensemble.roles import Roles
from lab.pack import ArtifactPack
from lab.policy import lab_pack
from lab.tracing import CHAIN, TOOL, Tracer

FALLBACK_SOURCE = "harness-fallback"


class EnsembleRunner:
    def __init__(
        self,
        roles: Roles,
        sup: Any,
        *,
        n: int,
        max_rounds: int,
        tracer: Tracer | None = None,
        variant_hints: list[str] | None = None,
    ) -> None:
        self.roles = roles
        self.sup = sup
        self.n = max(1, int(n))
        self.max_rounds = max(1, int(max_rounds))
        self.tracer = tracer or getattr(sup, "tracer", None) or Tracer(None)
        self.hints = list(variant_hints) if variant_hints else list(VARIANT_HINTS)
        self.lock = threading.Lock()

    # --------------------------------------------------------------- research

    def configs(self) -> list[EnsembleConfig]:
        return [
            EnsembleConfig(
                index=i + 1,
                max_rounds=self.max_rounds,
                variant_hint=self.hints[i % len(self.hints)],
            )
            for i in range(self.n)
        ]

    def run_research(self, obs: dict[str, Any]) -> list[EnsembleResult]:
        cfgs = self.configs()

        def one(cfg: EnsembleConfig) -> EnsembleResult:
            meta = {"cycle": obs.get("cycle"), "ensemble": cfg.index, "variant": cfg.variant_hint}
            with self.tracer.span(f"ensemble {cfg.index}", kind=CHAIN, inputs=meta, metadata=meta) as span:
                res = Ensemble(self.roles, self.sup, cfg, call_lock=self.lock, tracer=self.tracer).run(obs)
                span.set(outputs=res.compact(), error=res.error)
            return res

        with ThreadPoolExecutor(max_workers=self.n, thread_name_prefix="ensemble") as pool:
            results = list(pool.map(one, cfgs))
        results.sort(key=lambda r: r.index)
        for res in results:
            flag = f"error={res.error}" if res.error else ("pack" if res.pack else "no pack")
            print(
                f"[ensemble {res.index}] {res.rounds} round(s), {res.tool_calls} tool call(s), "
                f"{res.code_runs} code action(s), {len(res.hypotheses)} hypothesis(es): {flag}",
                flush=True,
            )
        return results

    # ----------------------------------------------------------------- commit

    def commit(self, results: list[EnsembleResult], obs: dict[str, Any]) -> dict[str, Any]:
        meta = {"cycle": obs.get("cycle"), "phase": "research"}
        with self.lock, self.tracer.span("ensemble commit", kind=CHAIN, metadata=meta) as span:
            out = self._commit_locked(results, obs)
            span.set(outputs={"candidates": len(out["candidates"]), "fallback": out["fallback"]})
            return out

    def _call(self, tool: str, args: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
        meta = {"cycle": obs.get("cycle"), "phase": "research", "tool": tool}
        with self.tracer.span(f"tool.{tool}", kind=TOOL, inputs=args, metadata=meta) as span:
            result = self.sup.call(tool, args)
            span.set(outputs=result, error=None if result.get("ok") else str(result.get("error")))
        return result

    def _commit_locked(self, results: list[EnsembleResult], obs: dict[str, Any]) -> dict[str, Any]:
        candidates: list[dict[str, str]] = []
        notes: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for res in results:
            if not res.pack:
                continue
            try:
                digest = ArtifactPack.from_dict(res.pack).digest()
            except Exception as e:
                notes.append({"ensemble": res.index, "error": f"pack invalid at commit: {e}"})
                continue
            if digest in seen_hashes:
                notes.append({"ensemble": res.index, "skipped": f"duplicate pack {digest[:12]}"})
                continue
            hyps = list(res.hypotheses) or [
                {
                    "claim": str(res.pack.get("hypothesis") or "").strip(),
                    "why": "ensemble did not state a hypothesis",
                    "falsify": "confirm_ppl does not improve",
                }
            ]
            ids: list[str] = []
            for h in hyps:
                out = self._call(
                    "write_hypothesis",
                    {
                        "claim": h.get("claim", ""),
                        "why": h.get("why", ""),
                        "falsify": h.get("falsify", ""),
                        "source": f"ensemble-{res.index}",
                    },
                    obs,
                )
                if out.get("ok") and out.get("id"):
                    ids.append(str(out["id"]))
                else:
                    notes.append({"ensemble": res.index, "error": f"write_hypothesis: {out.get('error')}"})
            if not ids:
                notes.append({"ensemble": res.index, "error": "no hypothesis could be opened; pack skipped"})
                continue
            out = self._call("write_pack", {"pack": res.pack, "hypothesis_id": ids[-1]}, obs)
            if not out.get("ok"):
                notes.append({"ensemble": res.index, "error": f"write_pack: {out.get('error')}"})
                continue
            pack_hash = str(out["pack_hash"])
            seen_hashes.add(pack_hash)
            candidates.append({"hypothesis_id": ids[-1], "pack_hash": pack_hash})

        fallback = False
        if not candidates:
            fallback = True
            print("[ensemble] fallback pack authored by harness, not the models", flush=True)
            hyp = self._call(
                "write_hypothesis",
                {
                    "claim": f"cycle {obs.get('cycle')}: harness fallback lab pack (ensembles produced no pack)",
                    "why": "no ensemble produced a valid pack this cycle",
                    "falsify": "lab job fails or confirm_ppl does not drop",
                    "source": FALLBACK_SOURCE,
                },
                obs,
            )
            if hyp.get("ok"):
                out = self._call("write_pack", {"pack": lab_pack(obs), "hypothesis_id": hyp["id"]}, obs)
                if out.get("ok"):
                    candidates.append({"hypothesis_id": str(hyp["id"]), "pack_hash": str(out["pack_hash"])})
                else:
                    notes.append({"ensemble": None, "error": f"fallback write_pack: {out.get('error')}"})
            else:
                notes.append({"ensemble": None, "error": f"fallback write_hypothesis: {hyp.get('error')}"})

        queued = self._call("queue_candidates", {"candidates": candidates}, obs) if candidates else {
            "ok": False,
            "error": "no candidates",
        }
        if not queued.get("ok"):
            notes.append({"ensemble": None, "error": f"queue_candidates: {queued.get('error')}"})

        summary = {
            "cycle": obs.get("cycle"),
            "n": self.n,
            "candidates": candidates,
            "fallback": fallback,
            "queued": queued,
            "notes": notes,
            "results": [r.to_dict() for r in results],
        }
        self._write_summary(obs, summary)
        print(
            f"[ensemble] cycle {obs.get('cycle')}: {len(candidates)} candidate(s) queued"
            + (" (harness fallback)" if fallback else ""),
            flush=True,
        )
        return {
            "candidates": candidates,
            "fallback": fallback,
            "queued": bool(queued.get("ok")),
            "notes": notes,
            "results": [r.compact() for r in results],
        }

    def _write_summary(self, obs: dict[str, Any], summary: dict[str, Any]) -> Path | None:
        run_dir = getattr(getattr(self.sup, "cfg", None), "run_dir", None)
        if run_dir is None:
            return None
        cycle = int(obs.get("cycle") or 0)
        path = Path(run_dir) / "ensembles" / f"cycle-{cycle:04d}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
        except OSError:
            return None
        return path


__all__ = ["EnsembleRunner", "FALLBACK_SOURCE"]
