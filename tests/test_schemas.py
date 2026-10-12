"""Tool-call schemas, and InferPolicy handing them to engines that decode under one.

The tests that need infer's matcher (or a real infer engine) import it from
LAB_INFER_ROOT and skip when it is not set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from lab.actions import REGISTRY
from lab.llm_policy import SYSTEM, InferPolicy
from lab.pack import ArtifactPack
from lab.parse import parse_tool_call
from lab.schemas import TOOL_ARGS, pack_schema, tool_call_schema, train_sources
from lab.supervisor import Supervisor
from lab.types import PHASE_TOOLS, Phase

OBS: dict[str, Any] = {
    "phase": "research",
    "cycle": 2,
    "subject_checkpoint": "subjects/tinytrain-8m",
    "last_checkpoint": "runs/x/checkpoints/job-0001.pt",
    "data_cache": {
        "files": [
            {"path": "/data/lab/roneneldan_TinyStories__train__10000__fedcba9876543210.txt"},
            {"path": "/data/lab/wikimedia_wikipedia__train__2000__0123456789abcdef.txt"},
            {"path": "/data/lab/notes.txt"},
        ]
    },
}


class RecordingEngine:
    """Scripted replies; remembers the keyword arguments of every call."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return self.replies.pop(0) if self.replies else '{"tool": "halt", "args": {}}'


class SchemaEngine(RecordingEngine):
    """Looks like an infer engine that understands json_schema."""

    def generate_json(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("InferPolicy parses the text itself")


def _tools(schema: dict[str, Any]) -> list[str]:
    return sorted(b["properties"]["tool"]["const"] for b in schema.get("anyOf", [schema]))


def test_every_tool_has_an_argument_schema() -> None:
    assert set(REGISTRY) == set(TOOL_ARGS) | {"write_pack"}


def test_each_turn_is_limited_to_the_phase_tools(sup: Supervisor) -> None:
    engine = SchemaEngine(['{"tool": "run_eval", "args": {}}'])
    obs = sup.observe()
    assert InferPolicy(engine, max_new_tokens=64).act(obs)[0] == "run_eval"
    assert _tools(engine.calls[0]["json_schema"]) == sorted(PHASE_TOOLS[Phase(obs["phase"])])


def test_a_nudge_asks_for_exactly_the_wanted_tool() -> None:
    engine = SchemaEngine(['{"tool": "write_hypothesis", "args": {"claim": "a", "why": "b", "falsify": "c"}}'])
    name, _ = InferPolicy(engine)._nudge(OBS, "Write the claim.", want="write_hypothesis")
    assert name == "write_hypothesis"
    assert _tools(engine.calls[0]["json_schema"]) == ["write_hypothesis"]


def test_engines_without_json_schema_get_the_plain_call(monkeypatch: pytest.MonkeyPatch, sup: Supervisor) -> None:
    plain = RecordingEngine(['{"tool": "run_eval", "args": {}}'])
    InferPolicy(plain).act(sup.observe())
    assert "json_schema" not in plain.calls[0]

    monkeypatch.setenv("LAB_STRUCTURED", "0")
    off = SchemaEngine(['{"tool": "run_eval", "args": {}}'])
    InferPolicy(off).act(sup.observe())
    assert "json_schema" not in off.calls[0]


def test_pack_schema_pins_what_the_harness_decides() -> None:
    props = pack_schema(OBS)["properties"]
    assert props["parent_checkpoint"] == {"const": "runs/x/checkpoints/job-0001.pt"}
    assert props["trainer"] == {"const": "lab"}
    assert props["eval_suite_id"] == {"enum": ["core"]}
    assert props["data_manifest"]["properties"]["sources"]["items"]["enum"] == [
        "hf:roneneldan/TinyStories:train:10000",
        "hf:wikimedia/wikipedia:train:2000",
    ]
    first = pack_schema({"subject_checkpoint": "subjects/s"})["properties"]
    assert first["parent_checkpoint"] == {"const": "subjects/s"}
    assert train_sources({}) == ["hf:roneneldan/TinyStories:train:10000"]


# ------------------------------------------------------------ with infer


def _infer_module(name: str) -> Any:
    root = os.environ.get("LAB_INFER_ROOT")
    if not root:
        pytest.skip("LAB_INFER_ROOT is not set")
    root = str(Path(root).expanduser().resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    return pytest.importorskip(name)


def test_schemas_compile_in_infer() -> None:
    structured = _infer_module("engine.structured")
    for phase in Phase:
        structured.as_grammar(tool_call_schema(PHASE_TOOLS[phase], OBS))


def test_the_system_prompt_example_fits_the_schema() -> None:
    structured = _infer_module("engine.structured")
    example = SYSTEM.split("- Lab pack: ", 1)[1].split("\n", 1)[0]
    call = json.loads(example.replace("<last_checkpoint or subjects/tinytrain-8m>", OBS["last_checkpoint"]))
    assert structured.matches(tool_call_schema(PHASE_TOOLS[Phase.RESEARCH], OBS), json.dumps(call))
    ArtifactPack.from_dict(call["args"]["pack"])


def test_harness_fallbacks_fit_the_schema(sup: Supervisor) -> None:
    structured = _infer_module("engine.structured")
    policy = InferPolicy(RecordingEngine([]))
    for phase in ("eval", "train"):
        obs = {**sup.observe(), "phase": phase}
        name, args = policy._fallback(obs)
        schema = tool_call_schema(PHASE_TOOLS[Phase(phase)], obs)
        assert structured.matches(schema, json.dumps({"tool": name, "args": args})), (phase, name)


def test_a_random_model_still_writes_a_legal_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A random-weight model knows nothing about JSON: whatever parses is the schema's doing."""
    _infer_module("torch")
    synth = _infer_module("engine.synth")
    from engine.agent_api import load_engine
    from engine.config import ModelConfig

    raw = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 300,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "hidden_act": "silu",
        "eos_token_id": 299,
    }
    folder = synth.write_config(tmp_path / "rand", raw)
    cfg = ModelConfig.from_pretrained(folder)
    synth.write_hf_folder(folder, cfg, synth.random_engine_weights(cfg, seed=0))
    monkeypatch.setenv("INFER_MEM_FLOOR_GB", "0")
    engine = load_engine(folder, device="cpu", dtype="float32")

    policy = InferPolicy(engine, max_new_tokens=512)
    name, args = policy._nudge(OBS, "Call write_pack now.", want="write_pack")
    assert name == "write_pack"
    pack = args["pack"]
    assert pack["parent_checkpoint"] == OBS["last_checkpoint"]
    assert pack["trainer"] == "lab"
    assert set(pack["data_manifest"]["sources"]) <= set(train_sources(OBS))

    train = {**OBS, "phase": "train"}
    raw_turn = engine.generate(
        policy._prompt(train), max_new_tokens=512, json_schema=tool_call_schema(PHASE_TOOLS[Phase.TRAIN], train)
    )
    turn = parse_tool_call(raw_turn)
    assert turn is not None and turn[0] in PHASE_TOOLS[Phase.TRAIN], raw_turn
