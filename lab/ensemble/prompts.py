"""Role prompts for the ensemble. Short, imperative, JSON-only replies.

Everything a role sees is built here so the tests can pin the wording that
matters (variant hint, sandbox prefix, tool catalog) without a model.
"""

from __future__ import annotations

from typing import Any, Iterable
import json

from lab.data_cache import DEFAULT_MIX
from lab.ensemble.protocol import Turn
from lab.types import PHASE_TOOLS, Phase

# Angles handed to parallel ensembles so they do not all try the same thing.
VARIANT_HINTS = [
    "learning rate",
    "training steps",
    "data mix",
    "batch size / sequence length",
    "width or depth (architecture; weights will not resume)",
    "replicate the best episode with a small perturbation",
]

# {tool: {"args": {name: type}, "purpose": str}} for every tool a research-phase
# ensemble may see. Optional args end in '?'. Kept in sync with lab/actions.py
# and lab/tools_extra.py by a test.
TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    # always
    "read_notebook": {"args": {"n?": "int"}, "purpose": "last n notebook entries + beliefs"},
    "write_note": {"args": {"text": "str", "kind?": "str"}, "purpose": "append a note to the notebook"},
    "write_beliefs": {"args": {"text": "str"}, "purpose": "overwrite beliefs.md"},
    "halt": {"args": {"reason": "str"}, "purpose": "stop the run (harness only)"},
    # memory
    "list_episodes": {"args": {"n?": "int", "query?": "str"}, "purpose": "past train+eval episodes (id, confirm_ppl, title)"},
    "read_episode": {"args": {"id": "str"}, "purpose": "full episode card: config, metrics, verdict"},
    "read_hypothesis": {"args": {"id?": "str"}, "purpose": "one hypothesis (default: current)"},
    "list_hypotheses": {"args": {"cycle?": "int", "status?": "str"}, "purpose": "hypothesis board rows"},
    "write_hypothesis": {"args": {"claim": "str", "why": "str", "falsify": "str", "source?": "str"}, "purpose": "open a hypothesis (runner only)"},
    # inspect
    "grep_files": {"args": {"pattern": "str", "path?": "str", "max_hits?": "int"}, "purpose": "regex search sandbox files"},
    "list_models": {"args": {}, "purpose": "model dirs under LAB_MODELS_DIR"},
    "inspect_model": {"args": {"model_dir": "str"}, "purpose": "config-only capabilities of a model dir"},
    "read_checkpoint_meta": {"args": {"path": "str"}, "purpose": "config/n_params of a .pt under run_dir (needs torch)"},
    "read_train_log": {"args": {"job_id?": "str", "tail?": "int"}, "purpose": "job dict + train.log tail"},
    "read_trace": {"args": {"cycle?": "int", "n?": "int", "kind?": "tool|llm|chain"}, "purpose": "last n trace rows"},
    "list_packs": {"args": {"n?": "int"}, "purpose": "newest packs (hash, hypothesis, config)"},
    "read_pack": {"args": {"pack_hash": "str"}, "purpose": "canonical pack json"},
    "diff_packs": {"args": {"a": "str", "b": "str"}, "purpose": "config/data diff between two packs"},
    "list_jobs": {"args": {"n?": "int"}, "purpose": "newest jobs (status, val_ppl, lr, steps)"},
    "read_skill": {"args": {"name": "str"}, "purpose": "read a skill card"},
    "list_skills": {"args": {}, "purpose": "skill card names"},
    "read_episode_metrics": {"args": {"n?": "int"}, "purpose": "confirm_ppl table + best episode"},
    "sandbox_usage": {"args": {}, "purpose": "sandbox files/bytes/cap"},
    # research extras
    "run_python": {"args": {"path": "str", "args?": "list[str]", "timeout_s?": "number"}, "purpose": "run an existing sandbox script"},
    "write_and_run": {"args": {"path": "str", "content": "str", "args?": "list[str]", "timeout_s?": "number"}, "purpose": "write a python script and run it"},
    "data_stats": {"args": {"source": "str"}, "purpose": "stats for an already-cached data source"},
    # research core
    "list_files": {"args": {"path?": "str"}, "purpose": "list sandbox files"},
    "read_file": {"args": {"path": "str"}, "purpose": "read a sandbox file"},
    "write_file": {"args": {"path": "str", "content": "str"}, "purpose": "write a sandbox file"},
    "exec": {"args": {"argv": "list[str]"}, "purpose": "run a shell command in the sandbox"},
    "web_fetch": {"args": {"url": "str", "path?": "str"}, "purpose": "fetch a URL into the sandbox"},
    "web_search": {"args": {"query": "str"}, "purpose": "web search"},
    "prefetch_data": {"args": {"source": "str", "split?": "str", "n?": "int"}, "purpose": "cache an allowlisted HF text source"},
    "write_pack": {"args": {"pack": "object", "hypothesis_id?": "str"}, "purpose": "freeze a train pack (runner only)"},
    "queue_candidates": {"args": {"candidates": "list[{hypothesis_id, pack_hash}]"}, "purpose": "plan this cycle's trials (runner only)"},
    "enter_train": {"args": {}, "purpose": "start training (harness only)"},
}


def _schema_line(name: str) -> str:
    spec = TOOL_SCHEMAS.get(name)
    if not spec:
        return f"{name} {{}}"
    args = ", ".join(f"{k}: {v}" for k, v in spec.get("args", {}).items())
    purpose = spec.get("purpose", "")
    line = f"{name} {{{args}}}"
    return f"{line} — {purpose}" if purpose else line


def tool_catalog(phase: Phase, exclude: Iterable[str] = ()) -> str:
    """One line per allowed tool in ``phase``: ``name {arg: type, ...} — purpose``."""
    skip = set(exclude)
    names = sorted(n for n in PHASE_TOOLS[phase] if n not in skip)
    return "\n".join(_schema_line(n) for n in names)


def _clip(value: Any, n: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def obs_digest(obs: dict[str, Any], max_chars: int = 3000) -> str:
    """Compact, model-facing view of the observation."""
    last_eval = obs.get("last_eval") or {}
    episodes = list(obs.get("episodes") or [])
    lines = [
        f"cycle={obs.get('cycle')} phase={obs.get('phase')} trial={obs.get('trial')}/{obs.get('trials_planned')}",
        f"last_eval.confirm_ppl={last_eval.get('confirm_ppl')} source={last_eval.get('confirm_source')}",
        f"subject_checkpoint={obs.get('subject_checkpoint')}",
        f"last_checkpoint={obs.get('last_checkpoint')}",
    ]
    scored = [e for e in episodes if e.get("confirm_ppl") is not None]
    if scored:
        best = min(scored, key=lambda e: float(e["confirm_ppl"]))
        lines.append(f"best_episode={best.get('id')} confirm_ppl={best.get('confirm_ppl')}")
    if episodes:
        lines.append("episodes (last 6):")
        for ep in episodes[-6:]:
            cfg = ep.get("config") or {}
            cfg_txt = ""
            if cfg:
                cfg_txt = f" lr={cfg.get('lr')} steps={cfg.get('steps')}"
            lines.append(
                f"  - {ep.get('id')} confirm_ppl={ep.get('confirm_ppl')} "
                f"status={ep.get('job_status')}{cfg_txt} :: {_clip(ep.get('title') or '', 100)}"
            )
    else:
        lines.append("episodes: none yet")
    hyps = list(obs.get("hypotheses") or [])
    if hyps:
        lines.append("hypotheses this cycle:")
        for h in hyps[-6:]:
            lines.append(f"  - {h.get('id')} [{h.get('status')}] {_clip(h.get('claim') or '', 120)}")
    cache = obs.get("data_cache") or {}
    cached = cache.get("files") or []
    names = [
        (c.get("path") or c.get("name") or "").rsplit("/", 1)[-1] if isinstance(c, dict) else str(c)
        for c in cached
    ]
    lines.append(f"data_cache={names[:8]} default_mix={list(DEFAULT_MIX)}")
    research = obs.get("research") or {}
    lines.append(
        f"research_budget: tool_calls={research.get('tool_calls')}/{research.get('max_tool_calls')} "
        f"seconds_left={research.get('seconds_left')}"
    )
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def transcript_tail(transcript: list[Turn], n: int = 6, clip: int = 600) -> str:
    if not transcript:
        return "(no turns yet)"
    lines = []
    for t in transcript[-n:]:
        head = f"[round {t.round}] {t.role}"
        if t.tool:
            head += f" -> {t.tool}"
        if t.role == "harness" and t.result is not None:
            # Harness turns are instructions (nudges carry a full pack
            # template); clipping them would hide what we are asking for.
            body = str(t.result.get("error") or t.result)
        else:
            body = _clip(t.result, clip) if t.result is not None else _clip(t.raw, clip)
        lines.append(f"{head}: {body}")
    return "\n".join(lines)


# A filled example is the fastest way to get a small model to emit a valid
# pack, and also the fastest way to get it to emit *that example* as its own
# experiment. Observed on Spark: Llama-3.2-3B returned the documentation's
# claim and config verbatim in both cycles while claiming a different lr.
# So every example leaves the knob under test as a placeholder: the model
# cannot copy its way to a valid pack, only to an error that names the field.
PLACEHOLDER = "<value your claim names>"

# Which config key each variant hint is asking the ensemble to move.
HINT_KNOB = {
    "learning rate": "lr",
    "training steps": "steps",
    "batch size / sequence length": "batch",
    "width or depth (architecture; weights will not resume)": "hidden",
}


def knob_for_hint(hint: str) -> str | None:
    """The config key an angle varies, or None for angles that are not one key."""
    return HINT_KNOB.get(hint)


def _has_placeholder(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("<") and value.endswith(">")


def placeholder_fields(pack: dict[str, Any]) -> list[str]:
    """Field paths in a pack still holding an unedited placeholder."""
    out = [k for k, v in pack.items() if _has_placeholder(v)]
    out += [f"config.{k}" for k, v in (pack.get("config") or {}).items() if _has_placeholder(v)]
    return sorted(out)


def pack_template(
    obs: dict[str, Any],
    hypothesis: str | None = None,
    *,
    knob: str | None = None,
) -> dict[str, Any]:
    """A pack for this run that a small model only has to finish.

    Everything is filled from the real observation except `knob` -- the one key
    this ensemble's angle exists to vary. That key is a placeholder, so the
    model must supply the number its own claim promised.
    """
    parent = obs.get("last_checkpoint") or obs.get("subject_checkpoint") or "<subject_checkpoint>"
    config: dict[str, Any] = {
        "lr": 0.001,
        "steps": 64,
        "hidden": 32,
        "layers": 1,
        "heads": 1,
        "seq_len": 32,
        "batch": 8,
    }
    if knob in config:
        config[knob] = PLACEHOLDER
    return {
        "hypothesis": hypothesis or "<one sentence: what changes and why confirm_ppl should drop>",
        "trainer": "lab",
        "config": config,
        "data_manifest": {"sources": list(DEFAULT_MIX)},
        "eval_suite_id": "core",
        "eval_suite_version": 1,
        "parent_checkpoint": parent,
        "budgets": {"max_hours": 0.1, "max_steps": 64},
    }


def pack_nudge(
    obs: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    *,
    knob: str | None = None,
) -> str:
    """Nudge text for a thinker that stated a hypothesis but emitted no pack."""
    claim = hypotheses[-1]["claim"] if hypotheses else None
    template = json.dumps(pack_template(obs, claim, knob=knob))
    lead = (
        "A hypothesis does nothing until a pack tests it. "
        if hypotheses
        else "You asked for nothing and emitted no pack. "
    )
    tail = (
        f'Replace "config.{knob}" with the number your claim names; leave every other value alone.'
        if knob
        else "Change only the values your claim names; leave every other value alone."
    )
    return lead + f'Reply again with the same JSON shape, with "pack" set to:\n{template}\n{tail}'


_THOUGHT_SHAPE = (
    '{"thought": "<short reasoning>", '
    '"hypotheses": [{"claim": "...", "why": "...", "falsify": "..."}], '
    '"need": null | {"kind": "tool", "request": "..."} | {"kind": "code", "request": "...", "path": "optional"}, '
    '"pack": null | {<pack>}, '
    '"done": false}'
)


def thinker_prompt(
    obs: dict[str, Any],
    transcript: list[Turn],
    *,
    variant_hint: str,
    skills: str,
) -> str:
    knob = knob_for_hint(variant_hint)
    example = json.dumps(pack_template(obs, knob=knob))
    knob_rule = (
        f'- "config.{knob}" above is a placeholder. Your angle is {variant_hint}, so {knob} is the one value '
        "you choose; it must be the number your claim names, and every other config value stays as given.\n"
        if knob
        else ""
    )
    return (
        "You are the THINKER of ONE of N parallel experimenters in a local ML harness.\n"
        f"Your angle: {variant_hint}. Other experimenters cover other angles; do not drift.\n"
        "You cannot call tools yourself. You may ask for a tool call "
        '("need": {"kind": "tool", "request": "<what to look up>"}) or for a script '
        '("need": {"kind": "code", "request": "<what it should compute>"}); the result '
        "appears in the transcript next turn.\n"
        'When you are ready, emit "pack": one ArtifactPack for the lab trainer, filled in for THIS run:\n'
        f"{example}\n"
        "Rules:\n"
        + knob_rule
        + "- Numbers and claims in the reference material are illustrations from other runs. Never copy one; "
        "read the episodes in your observation and choose your own.\n"
        "- parent_checkpoint = observation.last_checkpoint when set, else subject_checkpoint.\n"
        "- Keep hidden/layers/heads/seq_len identical to the parent so weights resume, unless your angle is "
        "architecture; then parent_checkpoint may be the subject and you must note that weights will not resume.\n"
        "- Write 1-3 hypotheses in total across your turns, each with claim / why / falsify.\n"
        "- A hypothesis is inert until a pack tests it: the turn that states a hypothesis should also carry "
        'its "pack", unless you first need a tool or code result.\n'
        "- Do not repeat a config that already appears in episodes.\n"
        "- budgets.max_hours <= 0.5. eval_suite_id 'core', eval_suite_version 1.\n"
        '- When you have nothing more to ask and no pack, set "done": true.\n'
        "Reply with ONE JSON object of this shape and nothing else:\n"
        f"{_THOUGHT_SHAPE}\n\n"
        f"observation:\n{obs_digest(obs)}\n\n"
        f"transcript (last 6 turns):\n{transcript_tail(transcript)}\n\n"
        f"{skills}\n"
    )


def tooler_prompt(
    request: str,
    obs: dict[str, Any],
    transcript: list[Turn],
    *,
    catalog: str,
    sandbox_prefix: str,
    skills: str,
) -> str:
    return (
        "You are the TOOLER. Fulfil this request with exactly one tool call.\n"
        f"request: {request}\n"
        'Reply with ONE JSON object and nothing else: {"tool": "<name>", "args": {...}}\n'
        f"Relative paths must start with '{sandbox_prefix}/'. Prefer read tools; "
        "use write_and_run for scripts. Never call write_pack, write_hypothesis, "
        "queue_candidates, enter_train or halt.\n"
        f"tools:\n{catalog}\n\n"
        f"observation:\n{obs_digest(obs, max_chars=1500)}\n\n"
        f"transcript (last 6 turns):\n{transcript_tail(transcript, clip=300)}\n\n"
        f"{skills}\n"
    )


def coder_prompt(
    request: str,
    obs: dict[str, Any],
    transcript: list[Turn],
    *,
    sandbox_prefix: str,
    skills: str,
    default_path: str,
) -> str:
    return (
        "You are the CODER. Write ONE python script to satisfy the request.\n"
        f"request: {request}\n"
        f'Reply with {{"path": "{sandbox_prefix}/<name>.py", "content": "<source>", "run": true, "args": []}} '
        "or a single ```python fence (then the script is saved as "
        f"{default_path} and run).\n"
        "The script must print a final JSON line with its findings. No network. <= 200 lines. "
        "Use os.environ.get('LAB_TRAIN_DEVICE', 'cpu') for any torch device. "
        "stdout is clipped to 32k chars.\n\n"
        f"observation:\n{obs_digest(obs, max_chars=1500)}\n\n"
        f"transcript (last 6 turns):\n{transcript_tail(transcript, clip=300)}\n\n"
        f"{skills}\n"
    )


__all__ = [
    "HINT_KNOB",
    "PLACEHOLDER",
    "TOOL_SCHEMAS",
    "VARIANT_HINTS",
    "coder_prompt",
    "knob_for_hint",
    "obs_digest",
    "pack_nudge",
    "pack_template",
    "placeholder_fields",
    "thinker_prompt",
    "tool_catalog",
    "tooler_prompt",
    "transcript_tail",
]
