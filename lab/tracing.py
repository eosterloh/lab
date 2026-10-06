"""Tracing for the harness loop.

Two sinks. A local JSONL under the run dir always records, so a run on a box
with no network still leaves a readable trace. LangSmith mirrors the same
spans when langsmith is installed and LANGSMITH_TRACING=true, which is what
gives the run a browsable tree.

Nothing here may break a run: if a sink raises, the span still closes.
Inputs and outputs are stored whole. A role prompt is the exception: the
preamble, tool catalog, and skill text are replaced with the marker
``system``, and the observation, request, and transcript stay.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import os
import threading
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


# Stored in place of a role preamble, tool catalog, or skill file. The
# observation, transcript, request, and the model's reply stay intact so a
# later flywheel can train on the turn without memorizing the harness prompt.
SYSTEM_MARKER = "system"

_ROLE_STARTS = (
    "You are the THINKER",
    "You are the TOOLER",
    "You are the CODER",
    "You are the experimenter",
    "Fill in one ArtifactPack",
)


def _looks_like_role_prompt(text: str) -> bool:
    return text.startswith(_ROLE_STARTS)


def _section(text: str, start: str, stops: tuple[str, ...]) -> str | None:
    i = text.find(start)
    if i < 0:
        return None
    rest = text[i:]
    end = len(rest)
    for stop in stops:
        j = rest.find(stop, len(start))
        if j != -1:
            end = min(end, j)
    return rest[:end].rstrip()


def redact_system(text: str) -> str:
    """Replace a harness system prompt with the marker ``system``.

    A string that is not a role prompt is returned unchanged, so tool
    results and model completions are not touched.
    """
    if not _looks_like_role_prompt(text):
        return text
    if text.startswith("You are the experimenter"):
        idx = text.find("\nphase=")
        if idx == -1:
            return SYSTEM_MARKER
        return f"{SYSTEM_MARKER}\n\n{text[idx + 1:]}"
    chunks: list[str] = []
    request = _section(text, "request: ", ("\n",))
    if request:
        chunks.append(request)
    claim = _section(text, "Your claim: ", ("\n",))
    if claim:
        chunks.append(claim)
    observation = _section(
        text,
        "observation:\n",
        ("\ntranscript (last 6 turns):", "\n# "),
    )
    if observation:
        chunks.append(observation)
    transcript = _section(text, "transcript (last 6 turns):\n", ("\n# ",))
    if transcript:
        chunks.append(transcript)
    if not chunks:
        return SYSTEM_MARKER
    return SYSTEM_MARKER + "\n\n" + "\n\n".join(chunks)


def for_trace(value: Any) -> Any:
    """JSON-ready payload. No length cap. Role prompts become ``system`` plus the turn."""
    if value is None:
        return None
    if isinstance(value, str):
        return redact_system(value)
    if isinstance(value, dict):
        return {str(k): for_trace(v) for k, v in value.items()}
    if isinstance(value, list):
        return [for_trace(v) for v in value]
    return value


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
        self._lock = threading.Lock()
        # One open-span stack per thread. Parallel ensembles each keep their own
        # nesting; a worker thread's outermost span parents to whatever the main
        # thread had open when it started (the cycle / step span).
        self._local = threading.local()
        self._main_thread = threading.get_ident()
        self._main_open: list[int] = []
        self._cycle_stack = ExitStack()
        self._cycle: int | None = None

    @property
    def _open(self) -> list[int]:
        if threading.get_ident() == self._main_thread:
            return self._main_open
        stack = getattr(self._local, "open", None)
        if stack is None:
            stack = self._local.open = []
        return stack

    def _parent(self) -> int | None:
        if self._open:
            return self._open[-1]
        if threading.get_ident() != self._main_thread and self._main_open:
            return self._main_open[-1]
        return None

    @contextmanager
    def span(
        self,
        name: str,
        *,
        kind: str = CHAIN,
        inputs: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[Span]:
        with self._lock:
            self._seq += 1
            span_id = self._seq
        parent = self._parent()
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
                        inputs={"input": for_trace(inputs)} if inputs is not None else {},
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
                        outputs={"output": for_trace(span.outputs)},
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
                    "inputs": for_trace(inputs),
                    "outputs": for_trace(span.outputs),
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
            line = json.dumps(row, default=str) + "\n"
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass
