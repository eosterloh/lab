"""Extra research tools for the tool-calling role. Registered into actions.REGISTRY."""

from __future__ import annotations

from typing import Any, Callable

from lab.types import ToolResult

EXTRA_REGISTRY: dict[str, Callable[[Any, dict[str, Any]], ToolResult]] = {}
