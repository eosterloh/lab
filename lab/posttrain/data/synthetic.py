"""Synthetic tool-use, thoughts, and scripts from the live harness schema.

Positives are single calls, multi-turn episodes, and recoveries whose prompt
already contains a harness refusal. Rejections are wrong calls for DPO; they
are never written as assistant targets.
"""

from __future__ import annotations

from typing import Any
import json

from lab.ensemble.protocol import Turn
from lab.posttrain.data.examples import Example
from lab.posttrain.data.harness import (
    VARIANT_HINTS,
    allowed_tooler_tools,
    knob_for_hint,
    make_obs,
    pack_template,
    refusal_text,
    render_coder,
    render_thinker,
    render_tooler,
    schema_fields,
    skill_names,
    tool_schemas,
)

_PREFIX = "ens-1"
_HASH_A = "a1" * 32
_HASH_B = "b2" * 32


def _turn(
    role: str,
    raw: str,
    result: dict[str, Any],
    round_: int,
    *,
    tool: str | None = None,
    parsed_ok: bool = True,
) -> Turn:
    return Turn(
        role=role,
        prompt_chars=len(raw),
        raw=raw,
        parsed_ok=parsed_ok,
        result=result,
        ms=1.0,
        round=round_,
        tool=tool,
    )


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _value(tool: str, name: str, prefix: str) -> Any:
    skills = skill_names()
    if name == "path" and tool == "read_checkpoint_meta":
        return "checkpoints/job-0001.pt"
    if name == "path" and tool == "web_fetch":
        return f"{prefix}/fetch/page.txt"
    if name == "path" and tool in {"write_file", "write_and_run", "run_python"}:
        return f"{prefix}/probe.py"
    if name == "path":
        return f"{prefix}/notes.txt"
    if name == "model_dir":
        return "Qwen3-0.6B"
    if name == "pack_hash":
        return _HASH_A
    if name == "a":
        return _HASH_A
    if name == "b":
        return _HASH_B
    if name == "id" and tool == "read_episode":
        return "ep-0002"
    if name == "id":
        return "hyp-0001"
    if name == "source":
        return "hf:roneneldan/TinyStories:train:1000"
    if name == "name":
        return skills[0] if skills else "lab_harness"
    if name == "kind" and tool == "read_trace":
        return "tool"
    if name == "kind":
        return "note"
    if name == "query":
        return "TinyStories learning rate"
    if name == "pattern":
        return "confirm_ppl"
    if name == "url":
        return "https://example.com/tinystories"
    if name == "text":
        return "parent run looks undertrained at the current learning rate"
    if name == "content":
        return "import json\nprint(json.dumps({'ok': True}))\n"
    if name == "argv":
        return ["python3", f"{prefix}/probe.py"]
    if name == "args":
        return ["--steps", "5"]
    if name == "job_id":
        return "job-0001"
    if name == "split":
        return "train"
    if name == "status":
        return "open"
    if name in {"n", "tail", "max_hits", "cycle"}:
        return 5
    if name == "timeout_s":
        return 5
    return "value"


def _args(tool: str, prefix: str, *, optional: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, _typ, required in schema_fields(tool):
        if not required and not optional:
            continue
        out[name] = _value(tool, name, prefix)
    return out


def _request(tool: str, args: dict[str, Any], *, phrase: int) -> str:
    purpose = str((tool_schemas().get(tool) or {}).get("purpose") or tool)
    if not args:
        lead = "Call it with an empty args object." if phrase == 0 else "No arguments this time."
        return f"{purpose}. {lead}"
    bits = ", ".join(f"{key}={value!r}" for key, value in args.items())
    lead = "Do this now." if phrase == 0 else "One call, nothing else."
    return f"{purpose}. {lead} Use {tool} with {bits}."


def _pair(role: str, prompt: str, completion: str, *, kind: str, **kwargs: Any) -> Example:
    return Example(
        role=role,
        messages=[
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        source="synthetic",
        kind=kind,
        **kwargs,
    )


def _tooler_singles(obs: dict[str, Any]) -> list[Example]:
    out: list[Example] = []
    for tool in sorted(allowed_tooler_tools()):
        specs = [_args(tool, _PREFIX, optional=False)]
        full = _args(tool, _PREFIX, optional=True)
        if full != specs[0]:
            specs.append(full)
        for phrase, args in enumerate(specs):
            raw = _dump({"tool": tool, "args": args})
            prompt = render_tooler(_request(tool, args, phrase=phrase), [], obs=obs, prefix=_PREFIX)
            out.append(_pair("tooler", prompt, raw, kind="single", obs=obs))
    return out


def _tooler_episodes(obs: dict[str, Any]) -> list[Example]:
    """Multi-turn traces. The second user prompt contains a real refusal."""
    list_raw = _dump({"tool": "list_files", "args": {"path": _PREFIX}})
    read_raw = _dump({"tool": "read_file", "args": {"path": f"{_PREFIX}/notes.txt"}})
    grep_raw = _dump({"tool": "grep_files", "args": {"pattern": "confirm_ppl", "path": _PREFIX}})
    bad = _dump({"tool": "write_pack", "args": {"pack": {}}})
    forbidden = {"ok": False, "error": refusal_text("forbidden", "write_pack")}
    repeated = {"ok": False, "error": refusal_text("repeated")}
    steps = [
        (
            "List the files in this ensemble's sandbox.",
            list_raw,
            {"ok": True, "files": [f"{_PREFIX}/notes.txt"]},
            "list_files",
        ),
        (
            "The write_pack call was refused. Read the notes file instead.",
            read_raw,
            {"ok": True, "path": f"{_PREFIX}/notes.txt", "content": "lr 0.001"},
            "read_file",
        ),
        (
            "Do not repeat the read. Search that file for confirm_ppl.",
            grep_raw,
            {"ok": True, "hits": []},
            "grep_files",
        ),
    ]
    messages: list[dict[str, str]] = []
    transcript: list[Turn] = []
    for i, (request, raw, result, tool) in enumerate(steps, start=1):
        if i == 2:
            transcript.append(_turn("tooler", bad, forbidden, i, tool="write_pack"))
        if i == 3:
            transcript.append(_turn("tooler", read_raw, repeated, i, tool="read_file"))
        prompt = render_tooler(request, transcript, obs=obs, prefix=_PREFIX)
        messages.append({"role": "user", "content": prompt})
        messages.append({"role": "assistant", "content": raw})
        transcript.append(_turn("tooler", raw, result, i, tool=tool))
    episode = Example(role="tooler", messages=messages, source="synthetic", kind="episode", obs=obs)

    # Single-turn recovery: the only thing in the transcript is the refusal.
    recoveries = []
    cases = [
        (
            "That tool is not yours. List sandbox files instead.",
            [_turn("tooler", bad, forbidden, 1, tool="write_pack")],
            _dump({"tool": "list_files", "args": {"path": _PREFIX}}),
        ),
        (
            "Your last reply did not parse. Read the notes file.",
            [
                _turn(
                    "tooler",
                    "sure, reading it",
                    {"ok": False, "error": refusal_text("unparseable_tool")},
                    1,
                    parsed_ok=False,
                )
            ],
            read_raw,
        ),
        (
            "Search the web for TinyStories, then stop.",
            [
                _turn(
                    "tooler",
                    _dump({"tool": "halt", "args": {"reason": "stop"}}),
                    {"ok": False, "error": refusal_text("forbidden", "halt")},
                    1,
                    tool="halt",
                )
            ],
            _dump({"tool": "web_search", "args": {"query": "TinyStories dataset"}}),
        ),
    ]
    for request, prior, raw in cases:
        prompt = render_tooler(request, prior, obs=obs, prefix=_PREFIX)
        recoveries.append(_pair("tooler", prompt, raw, kind="recovery", obs=obs))
    return [episode, *recoveries]


_KNOBS = {
    "learning rate": ("lower config.lr -> 0.0005 so the run should beat the parent", 0.0005),
    "training steps": ("raise config.steps -> 128 so the run trains longer than the parent", 128),
    "batch size / sequence length": ("set config.batch -> 16 and leave sequence length alone", 16),
    "width or depth (architecture; weights will not resume)": (
        "set config.hidden -> 64; weights will not resume from the parent",
        64,
    ),
    "data mix": ("add a wikipedia slice beside TinyStories and expect a lower score than the parent", None),
    "replicate the best episode with a small perturbation": (
        "replicate the best episode and set config.lr -> 0.0012",
        0.0012,
    ),
}


def _hyp(claim: str) -> dict[str, str]:
    return {
        "claim": claim,
        "why": "the parent config is already in the episode list",
        "falsify": "the next episode does not beat the parent checkpoint",
    }


def _pack_for(obs: dict[str, Any], hint: str) -> tuple[dict[str, Any], str, str | None]:
    claim, value = _KNOBS.get(
        hint,
        (f"change the {hint} angle and expect a lower score than the parent", None),
    )
    knob = knob_for_hint(hint)
    pack = pack_template(obs, claim, knob=knob)
    config = dict(pack["config"])
    if knob is not None and value is not None:
        config[knob] = value
    elif hint.startswith("replicate") and value is not None:
        config["lr"] = value
    pack["config"] = config
    if hint == "data mix":
        pack["data_manifest"] = {
            "sources": [
                "hf:roneneldan/TinyStories:train:10000",
                "hf:wikimedia/wikipedia:train:1000",
            ]
        }
    if knob == "hidden":
        pack["parent_checkpoint"] = obs["subject_checkpoint"]
    steps = config.get("steps")
    pack["budgets"] = {"max_hours": 0.1, "max_steps": int(steps) if isinstance(steps, int) else 64}
    return pack, claim, knob


def _thought(obs: dict[str, Any], hint: str) -> list[Example]:
    pack, claim, knob = _pack_for(obs, hint)
    hyp = _hyp(claim)
    packed = _dump(
        {
            "thought": f"Test the {hint} angle with one pack.",
            "hypotheses": [hyp],
            "need": None,
            "pack": pack,
            "done": False,
        }
    )
    need_tool = _dump(
        {
            "thought": "I need the episode card before I freeze a pack.",
            "hypotheses": [],
            "need": {"kind": "tool", "request": "Read episode ep-0002 and report confirm_ppl."},
            "pack": None,
            "done": False,
        }
    )
    need_code = _dump(
        {
            "thought": "A short script should check that hidden divides by heads.",
            "hypotheses": [],
            "need": {
                "kind": "code",
                "request": "Print JSON with divisible true when 32 % 1 == 0.",
                "path": f"{_PREFIX}/divisible.py",
            },
            "pack": None,
            "done": False,
        }
    )
    # Each completion needs its own prompt. The harness prompt includes the
    # observation, so a different cycle is a different string and the three
    # labels do not collide.
    singles = (
        (packed, obs, claim),
        (need_tool, make_obs(cycle=obs["cycle"] + 1, tool_calls=0), None),
        (need_code, make_obs(cycle=obs["cycle"] + 2, tool_calls=2), None),
    )
    out = []
    for raw, row_obs, row_claim in singles:
        prompt = render_thinker([], obs=row_obs, variant_hint=hint)
        out.append(_pair("thinker", prompt, raw, kind="single", obs=row_obs, knob=knob, claim=row_claim))
    episode_obs = make_obs(cycle=obs["cycle"] + 3, tool_calls=1)
    prompt_need = render_thinker([], obs=episode_obs, variant_hint=hint)
    prompt_pack = render_thinker(
        [
            _turn(
                "thinker",
                need_tool,
                {"need": {"kind": "tool"}, "pack": False, "hypotheses": 0, "done": False, "thought": "look up"},
                1,
            )
        ],
        obs=episode_obs,
        variant_hint=hint,
    )
    episode = Example(
        role="thinker",
        messages=[
            {"role": "user", "content": prompt_need},
            {"role": "assistant", "content": need_tool},
            {"role": "user", "content": prompt_pack},
            {"role": "assistant", "content": packed},
        ],
        source="synthetic",
        kind="episode",
        obs=episode_obs,
        knob=knob,
        claim=claim,
    )
    refused = _turn(
        "harness",
        "",
        {"ok": False, "error": refusal_text("pack")},
        1,
    )
    recovery_prompt = render_thinker([refused], obs=obs, variant_hint=hint)
    recovery = _pair(
        "thinker",
        recovery_prompt,
        packed,
        kind="recovery",
        obs=obs,
        knob=knob,
        claim=claim,
    )
    # One episode and one recovery per hint is a lot of near-duplicates.
    # Callers keep singles for every hint and the episode/recovery for the first.
    out.extend([episode, recovery])
    return out


def _thinker_examples(obs: dict[str, Any]) -> list[Example]:
    out: list[Example] = []
    for i, hint in enumerate(VARIANT_HINTS):
        rows = _thought(obs, hint)
        if i == 0:
            out.extend(rows)
        else:
            # singles only: pack, need tool, need code. Drop episode and recovery.
            out.extend(rows[:3])
    done = _dump(
        {
            "thought": "The pack is already in the transcript and I have nothing else to ask.",
            "hypotheses": [],
            "need": None,
            "pack": None,
            "done": True,
        }
    )
    prior = _turn(
        "thinker",
        _dump({"thought": "pack sent", "hypotheses": [], "need": None, "pack": {"hypothesis": "sent"}, "done": False}),
        {"need": None, "pack": True, "hypotheses": 1, "done": False, "thought": "pack sent"},
        1,
        tool="pack",
    )
    prompt = render_thinker([prior], obs=obs, variant_hint=VARIANT_HINTS[0])
    out.append(_pair("thinker", prompt, done, kind="single", obs=obs))
    return out


_CODER_SCRIPTS = [
    (
        "Check that hidden 32 is divisible by heads 1 and print JSON.",
        "divisible.py",
        "import json\nhidden, heads = 32, 1\nprint(json.dumps({'ok': True, 'divisible': hidden % heads == 0, 'hidden': hidden}))\n",
        [],
    ),
    (
        "Read argv for a step count and print it as JSON. Default to 4.",
        "steps.py",
        "import json, sys\nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 4\nprint(json.dumps({'ok': True, 'steps': steps}))\n",
        ["4"],
    ),
    (
        "Report the torch device from LAB_TRAIN_DEVICE, defaulting to cpu.",
        "device.py",
        "import json, os\ndevice = os.environ.get('LAB_TRAIN_DEVICE', 'cpu')\nprint(json.dumps({'ok': True, 'device': device}))\n",
        [],
    ),
    (
        "Estimate a tiny parameter count for a 32-wide one-layer model and print JSON.",
        "params.py",
        "import json\nhidden, layers = 32, 1\nn = hidden * hidden * layers\nprint(json.dumps({'ok': True, 'n_params': n}))\n",
        [],
    ),
    (
        "Sum the numbers in argv and print JSON. Use 1 2 3 when argv is empty.",
        "sum.py",
        "import json, sys\nnums = [float(x) for x in sys.argv[1:]] or [1.0, 2.0, 3.0]\nprint(json.dumps({'ok': True, 'total': sum(nums)}))\n",
        ["1", "2", "3"],
    ),
    (
        "Round-trip a tiny train config and print the learning rate as JSON.",
        "config.py",
        "import json\ncfg = {'lr': 0.0005, 'steps': 64, 'hidden': 32, 'heads': 1}\nok = cfg['hidden'] % cfg['heads'] == 0\nprint(json.dumps({'ok': ok, 'lr': cfg['lr'], 'steps': cfg['steps']}))\n",
        [],
    ),
    (
        "Clip a note to 40 characters and print JSON.",
        "clip.py",
        "import json\ntext = 'parent run looks undertrained at the current learning rate'\nprint(json.dumps({'ok': True, 'clipped': text[:40]}))\n",
        [],
    ),
    (
        "Compare two learning rates and print which one is smaller as JSON.",
        "compare_lr.py",
        "import json\na, b = 0.001, 0.0005\nprint(json.dumps({'ok': True, 'smaller': b if b < a else a}))\n",
        [],
    ),
]


def _coder_examples(obs: dict[str, Any]) -> list[Example]:
    out: list[Example] = []
    for request, name, content, args in _CODER_SCRIPTS:
        path = f"{_PREFIX}/{name}"
        raw = _dump({"path": path, "content": content, "run": True, "args": args})
        prompt = render_coder(request, [], obs=obs, prefix=_PREFIX, default_path=path)
        out.append(_pair("coder", prompt, raw, kind="single", obs=obs, default_path=path))
    fence_path = f"{_PREFIX}/fence.py"
    fence = "```python\nimport json\nprint(json.dumps({'ok': True, 'fence': True}))\n```\n"
    fence_prompt = render_coder(
        "Print a JSON object with fence true. A python fence is enough.",
        [],
        obs=obs,
        prefix=_PREFIX,
        default_path=fence_path,
    )
    out.append(_pair("coder", fence_prompt, fence, kind="single", obs=obs, default_path=fence_path))
    bad = "Sure, here is the script in prose."
    prior = [
        _turn(
            "coder",
            bad,
            {"ok": False, "error": refusal_text("unparseable_code")},
            1,
            parsed_ok=False,
        )
    ]
    path = f"{_PREFIX}/retry.py"
    content = "import json\nprint(json.dumps({'ok': True, 'recovered': True}))\n"
    raw = _dump({"path": path, "content": content, "run": True, "args": []})
    prompt = render_coder(
        "Your last reply did not parse. Write the script as JSON and print a JSON line.",
        prior,
        obs=obs,
        prefix=_PREFIX,
        default_path=path,
    )
    out.append(_pair("coder", prompt, raw, kind="recovery", obs=obs, default_path=path))
    return out


_TOOL_NEGATIVES = [
    '{"tool": "write_pack", "args": {"pack": {}}}',
    '{"tool": "read_file", "args": {}}',
    '{"tool": "read_file", "args": {"path": "../secrets.txt"}}',
    '{"tool": "get_weather", "args": {"city": "Paris"}}',
    "not json",
    '{"tool": "read_trace", "args": {"kind": "nope"}}',
    '{"tool": "exec", "args": {"argv": "ls"}}',
    '{"tool": "exec", "args": {"argv": ["rm", "-rf", "ens-1"]}}',
    '{"tool": "web_fetch", "args": {"url": "ftp://example.com/a"}}',
    '{"tool": "read_skill", "args": {"name": "not_a_real_skill"}}',
    '{"tool": "prefetch_data", "args": {"source": "hf:not/allowed"}}',
    '{"tool": "halt", "args": {"reason": "stop"}}',
]

_THINK_NEGATIVES = [
    "I think we should lower the learning rate.",
    _dump({"thought": "", "hypotheses": [], "need": None, "pack": None, "done": False}),
    _dump(
        {
            "thought": "copied the slot",
            "hypotheses": [{"claim": "compare to confirm_ppl <ppl>", "why": "template", "falsify": "no drop"}],
            "need": None,
            "pack": None,
            "done": False,
        }
    ),
]

_CODE_NEGATIVES = [
    "Sure, I will write that script next.",
    _dump(
        {
            "path": "../escape.py",
            "content": "import json\nprint(json.dumps({'ok': True}))\n",
            "run": True,
            "args": [],
        }
    ),
    _dump(
        {
            "path": f"{_PREFIX}/net.py",
            "content": "import urllib\nimport json\nprint(json.dumps({'ok': True}))\n",
            "run": False,
            "args": [],
        }
    ),
    _dump(
        {
            "path": f"{_PREFIX}/plain.py",
            "content": "print('hello')\n",
            "run": True,
            "args": [],
        }
    ),
]


def _attach_negatives(examples: list[Example], negatives: list[str]) -> None:
    """One wrong completion per single-turn row, cycling the negative list."""
    singles = [ex for ex in examples if ex.kind == "single"]
    for i, example in enumerate(singles):
        example.rejected = negatives[i % len(negatives)]


def generate(role: str) -> list[Example]:
    obs = make_obs()
    if role == "tooler":
        rows = _tooler_singles(obs) + _tooler_episodes(obs)
        _attach_negatives(rows, _TOOL_NEGATIVES)
        return rows
    if role == "thinker":
        rows = _thinker_examples(obs)
        # A pack whose lr is still the template placeholder.
        bad_pack = pack_template(obs, knob="lr")
        _THINK = list(_THINK_NEGATIVES) + [
            _dump(
                {
                    "thought": "left the slot in",
                    "hypotheses": [_hyp("lower config.lr -> 0.0005 so the run should beat the parent")],
                    "need": None,
                    "pack": bad_pack,
                    "done": False,
                }
            )
        ]
        _attach_negatives(rows, _THINK)
        return rows
    if role == "coder":
        rows = _coder_examples(obs)
        _attach_negatives(rows, _CODE_NEGATIVES)
        return rows
    raise ValueError(f"unknown role {role!r}")
