"""Three-role SLM ensemble: thinker, tooler, coder."""

from lab.ensemble.ensemble import Ensemble, EnsembleConfig, EnsembleResult, FORBIDDEN_TOOLS
from lab.ensemble.policy import EnsemblePolicy
from lab.ensemble.protocol import CodeAction, Thought, ToolCall, Turn
from lab.ensemble.registry import RoleRegistry
from lab.ensemble.roles import ROLES, RoleNotConfigured, Roles, RoleSpec
from lab.ensemble.runner import EnsembleRunner

__all__ = [
    "FORBIDDEN_TOOLS",
    "ROLES",
    "CodeAction",
    "Ensemble",
    "EnsembleConfig",
    "EnsemblePolicy",
    "EnsembleResult",
    "EnsembleRunner",
    "RoleNotConfigured",
    "RoleRegistry",
    "RoleSpec",
    "Roles",
    "Thought",
    "ToolCall",
    "Turn",
]
