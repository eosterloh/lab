"""Tracing for the harness loop.

Two sinks. A local JSONL under the run dir always records, so a run on a box
with no network still leaves a readable trace. LangSmith mirrors the same
spans when langsmith is installed and LANGSMITH_TRACING=true, which is what
gives the run a browsable tree.

Nothing here may break a run: if a sink raises, the span still closes.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import os
import time

# LangSmith run types that render usefully in the UI.
CHAIN = "chain"
LLM = "llm"
TOOL = "tool"


def langsmith_enabled() -> bool:
    """True when the user asked for LangSmith and the SDK is importable."""
    if os.environ.get("LANGSMITH_TRACING", "").strip().lower() != "true":
        return False
    try:
        import langsmith  # noqa: F401
    except Exception:
        return False
    return True


def _clip(value: Any, limit: int = 2000) -> Any:
    """Keep payloads small enough to read; traces are for humans."""
    if value is None:
        return None
    try:
        text = json.dumps(value, default=str)
    except Exception:
        text = str(value)
    if len(text) <= limit:
        try:
            return json.loads(text)
        except Exception:
            return text
    return text[: limit - 3] + "..."


class Span:
    """Handle for one traced operation. Set outputs before the block exits."""

    def __init__(self, name: str, kind: str, metadata: dict[str, Any]) -> None:
        self.name = name
        self.kind = kind
        self.metadata = dict(metadata)
        self.outputs: Any = None
        self.error: str | None = None

    def set(
        self,
        *,
        outputs: Any = None,
        error: str | None = None,
        **metadata: Any,
    ) -> None:
        if outputs is not None:
            self.outputs = outputs
        if error is not None:
            self.error = error
        self.metadata.update(metadata)


class Tracer:
    def __init__(
        self,
        path: Path | None = None,
        *,
        project: str | None = None,
        use_langsmith: bool | None = None,
    ) -> None:
        self.path = Path(path) if path else None
        self.project = project or os.environ.get("LANGSMITH_PROJECT") or "lab-harness"
        self.use_langsmith = langsmith_enabled() if use_langsmith is None else use_langsmith
        self._seq = 0
        self._open: list[int] = []
        self._cycle_stack = ExitStack()
        self._cycle: int | None = None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = CHAIN,
        inputs: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        self._seq += 1
        span_id = self._seq
        parent = self._open[-1] if self._open else None
        span = Span(name, kind, metadata or {})
        started = time.time()
        stack = ExitStack()
        run_tree = None
        if self.use_langsmith:
            try:
                from langsmith import trace as ls_trace

                run_tree = stack.enter_context(
                    ls_trace(
                        name=name,
                        run_type=kind,
                        inputs={"input": _clip(inputs)} if inputs is not None else {},
                        project_name=self.project,
                        metadata=span.metadata,
                        tags=[f"{k}:{v}" for k, v in span.metadata.items() if k in {"phase", "cycle", "tool"}],
                    )
                )
            except Exception:
                run_tree = None
        self._open.append(span_id)
        try:
            yield span
        except Exception as exc:  # the run loop decides whether to continue
            span.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._open.pop()
            elapsed_ms = round((time.time() - started) * 1000, 1)
            if run_tree is not None:
                try:
                    run_tree.end(
                        outputs={"output": _clip(span.outputs)},
                        error=span.error,
                        metadata=span.metadata,
                    )
                except Exception:
                    pass
            stack.close()
            self._write(
                {
                    "id": span_id,
                    "parent": parent,
                    "ts": round(started, 3),
                    "ms": elapsed_ms,
                    "name": name,
                    "kind": kind,
                    "ok": span.error is None,
                    "error": span.error,
                    "inputs": _clip(inputs, 600),
                    "outputs": _clip(span.outputs, 600),
                    **span.metadata,
                }
            )

    def enter_cycle(self, cycle: int) -> None:
        """Open a span per cycle so the tree groups by experiment, not by step."""
        if cycle == self._cycle:
            return
        self._cycle_stack.close()
        self._cycle_stack = ExitStack()
        self._cycle = cycle
        self._cycle_stack.enter_context(
            self.span(f"cycle {cycle}", kind=CHAIN, metadata={"cycle": cycle})
        )

    def close_cycle(self) -> None:
        self._cycle_stack.close()
        self._cycle_stack = ExitStack()
        self._cycle = None

    def _write(self, row: dict[str, Any]) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except Exception:
            pass
