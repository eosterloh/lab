"""Role backends for the three-role SLM ensemble.

A role (thinker / tooler / coder) is a named slot that some model fills. The
slot is described by a ``RoleSpec`` (serializable, versioned) and served by a
``Backend`` (anything with ``generate``). Models are dropped in by pointing
``LAB_<ROLE>_MODEL`` at a directory; nothing is loaded until the first call.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
import threading

from lab.config import LabConfig

ROLES = ("thinker", "tooler", "coder")

SENTINEL_REPLY = '{"done": true}'

DEFAULT_INFER_ROOT = Path.home() / "Projects" / "infer"


def env_var_for(role: str) -> str:
    return f"LAB_{role.upper()}_MODEL"


class RoleNotConfigured(RuntimeError):
    """Raised when a role is asked to generate but has no model behind it."""

    def __init__(self, role: str) -> None:
        self.role = role
        super().__init__(
            f"role '{role}' has no model; set {env_var_for(role)}=/path/to/model"
        )


class Backend(Protocol):
    def generate(self, prompt: str, *, max_new_tokens: int = 512, **kwargs: Any) -> str: ...


class NullBackend:
    """Placeholder for an unfilled role; every call raises RoleNotConfigured."""

    def __init__(self, role: str) -> None:
        self.role = role

    def generate(self, prompt: str, *, max_new_tokens: int = 512, **kwargs: Any) -> str:
        raise RoleNotConfigured(self.role)

    @property
    def loaded(self) -> bool:
        return False


class ScriptedBackend:
    """Deterministic replies for tests.

    ``replies`` is either a list (popped FIFO) or a ``callable(prompt) -> str``.
    When the list runs out the backend returns ``'{"done": true}'`` so a loop
    driven by it terminates instead of crashing. Every prompt is recorded in
    ``self.prompts``.
    """

    def __init__(self, replies: list[str] | Callable[[str], str]) -> None:
        self.prompts: list[str] = []
        self._fn: Callable[[str], str] | None
        self._queue: list[str]
        if callable(replies):
            self._fn = replies
            self._queue = []
        else:
            self._fn = None
            self._queue = list(replies)

    def generate(self, prompt: str, *, max_new_tokens: int = 512, **kwargs: Any) -> str:
        self.prompts.append(prompt)
        if self._fn is not None:
            return self._fn(prompt)
        if self._queue:
            return self._queue.pop(0)
        return SENTINEL_REPLY

    @property
    def remaining(self) -> int:
        return len(self._queue)

    @property
    def loaded(self) -> bool:
        return True


class InferBackend:
    """Backend over an ``infer`` engine, loaded lazily on first ``generate``.

    A lock serializes calls so each role owns exactly one GPU stream.
    ``enable_thinking=False`` is always passed unless the caller overrides it.
    """

    def __init__(
        self,
        model_dir: str | Path,
        infer_root: str | Path,
        device: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.infer_root = Path(infer_root).expanduser()
        self.device = device
        self._engine: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._engine is not None

    def _ensure_engine(self) -> Any:
        if self._engine is None:
            from lab import llm_policy

            self._engine = llm_policy.load_infer_engine(
                self.model_dir, self.infer_root, self.device
            )
        return self._engine

    def generate(self, prompt: str, *, max_new_tokens: int = 512, **kwargs: Any) -> str:
        kwargs.setdefault("enable_thinking", False)
        with self._lock:
            engine = self._ensure_engine()
            return engine.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)


@dataclass
class RoleSpec:
    role: str
    backend: str  # "infer" | "scripted" | "null"
    model_dir: str | None = None
    max_new_tokens: int = 512
    version: int = 1
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RoleSpec:
        return cls(
            role=str(d["role"]),
            backend=str(d.get("backend", "null")),
            model_dir=d.get("model_dir"),
            max_new_tokens=int(d.get("max_new_tokens", 512)),
            version=int(d.get("version", 1)),
            note=str(d.get("note", "")),
        )


def null_specs() -> dict[str, RoleSpec]:
    return {role: RoleSpec(role=role, backend="null") for role in ROLES}


def _is_loaded(backend: Any) -> bool:
    return bool(getattr(backend, "loaded", True))


def _build_backend(spec: RoleSpec, infer_root: str | Path | None) -> Backend:
    if spec.backend == "infer":
        if not spec.model_dir:
            raise ValueError(f"role '{spec.role}' backend 'infer' needs model_dir")
        return InferBackend(spec.model_dir, infer_root or DEFAULT_INFER_ROOT)
    if spec.backend == "null":
        return NullBackend(spec.role)
    if spec.backend == "scripted":
        return ScriptedBackend([])
    raise ValueError(f"role '{spec.role}' has unknown backend {spec.backend!r}")


class Roles:
    """Bundle of the three role backends plus their specs."""

    def __init__(self, backends: dict[str, Backend], specs: dict[str, RoleSpec]) -> None:
        missing = [r for r in ROLES if r not in backends or r not in specs]
        if missing:
            raise ValueError(f"roles missing backend or spec: {missing}")
        self._backends = dict(backends)
        self._specs = dict(specs)

    def backend(self, role: str) -> Backend:
        return self._backends[role]

    def spec(self, role: str) -> RoleSpec:
        return self._specs[role]

    def generate(self, role: str, prompt: str, **kwargs: Any) -> str:
        spec = self._specs[role]
        backend = self._backends[role]
        max_new_tokens = int(kwargs.pop("max_new_tokens", spec.max_new_tokens))
        return backend.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            role: {**self._specs[role].to_dict(), "loaded": _is_loaded(self._backends[role])}
            for role in ROLES
        }

    @classmethod
    def from_env(cls, cfg: LabConfig) -> Roles:
        infer_root = cfg.infer_root or DEFAULT_INFER_ROOT
        backends: dict[str, Backend] = {}
        specs: dict[str, RoleSpec] = {}
        for role in ROLES:
            model_dir: Path | None = getattr(cfg, f"{role}_model")
            if model_dir is not None:
                backends[role] = InferBackend(model_dir, infer_root)
                specs[role] = RoleSpec(role=role, backend="infer", model_dir=str(model_dir))
            else:
                backends[role] = NullBackend(role)
                specs[role] = RoleSpec(role=role, backend="null")
        return cls(backends, specs)

    @classmethod
    def scripted(
        cls,
        thinker: list[str] | Callable[[str], str],
        tooler: list[str] | Callable[[str], str],
        coder: list[str] | Callable[[str], str],
    ) -> Roles:
        scripts = {"thinker": thinker, "tooler": tooler, "coder": coder}
        backends: dict[str, Backend] = {
            role: ScriptedBackend(scripts[role]) for role in ROLES
        }
        specs = {role: RoleSpec(role=role, backend="scripted") for role in ROLES}
        return cls(backends, specs)

    @classmethod
    def from_specs(
        cls, specs: dict[str, RoleSpec], infer_root: str | Path | None = None
    ) -> Roles:
        backends: dict[str, Backend] = {
            role: _build_backend(spec, infer_root) for role, spec in specs.items()
        }
        return cls(backends, dict(specs))
