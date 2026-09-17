# infer_api
Load and query local HF checkpoints with the `infer` engine (~/Projects/infer on the DGX Spark). Greedy text only; no training.

## Where it lives
- Repo: `~/Projects/infer` on Spark. Package: `engine/`. Not installed as a package; put the repo root on `sys.path`.
- Shell: `cd ~/Projects/infer && source .venv/bin/activate && export PYTHONPATH=~/Projects/infer`
- Python (pattern from `lab/llm_policy.py:load_infer_engine`):
```python
import sys; sys.path.insert(0, "/home/eosterloh/Projects/infer")
from engine.agent_api import inspect_capabilities, load_engine
```
- The lab harness finds infer via `LAB_INFER_ROOT=~/Projects/infer` (used by `lab/scorer.py` for holdout scoring).
- Deps: torch, safetensors, transformers, accelerate, huggingface_hub, pillow.

## Model directory layout
`config.json` + weights (`model.safetensors`, or `model-0000N-of-000NN.safetensors` + `model.safetensors.index.json`; GGUF also accepted) + tokenizer files (`tokenizer.json` / `tokenizer_config.json` / `vocab.json` + `merges.txt`). Optional: `chat_template.jinja`, `generation_config.json` (eos ids picked up automatically). Recipe is auto-detected from `config.json`; nothing is registered.

Available under `~/models` on Spark: Qwen3.8-27B, Qwen3-0.6B, Qwen2.5-1.5B-Instruct, Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Mistral-7B-Instruct-v0.3, gemma-2-2b-it, Phi-3-mini-4k-instruct, SmolLM2-1.7B-Instruct, TinyLlama-1.1B-Chat-v1.0, Yi-1.5-6B-Chat, pythia-410m, gpt2, NVIDIA-Nemotron-3-Nano-30B-A3B-BF16.

## inspect_capabilities(model_dir) -> Capabilities
Reads `config.json` only; no weights. Raises `engine.detect.UnsupportedRecipeError` for unknown families. Frozen dataclass, `.to_dict()`:
`recipe_id, model_type, architectures, can_run, missing, dense_mlp, moe, mamba2, attention, rope, mtp, nvfp4, fp8, hybrid_pattern, num_layers, hidden_size, vocab_size, max_position_embeddings, notes`.
Check `can_run` before loading; `missing` lists advertised-but-unimplemented parts (`mtp_decode`, `vision`).

## load_engine(model_dir, *, device=None, dtype=None) -> Engine
- `device`: `"cuda"` if available else `"cpu"` when None. `dtype`: `"bfloat16" | "float16" | "float32"`, default from `config.json`.
- Loads shard by shard straight onto the device. Raises `FileNotFoundError` / `UnsupportedRecipeError`.
- `Engine` fields: `model_dir, config, model, tokenizer, capabilities, n_params`, plus `processor`, `vision_weights`, `mtp` for Qwen3.5/3.8 folders. `engine.info()` returns a dict summary.

## generate
```python
eng = load_engine("~/models/Qwen3-0.6B", device="cuda")
text = eng.generate("Reply with one word: ok", max_new_tokens=16, enable_thinking=False)
```
Signature: `generate(prompt, *, max_new_tokens=32, use_cache=True, apply_chat_template=None, enable_thinking=False, num_speculative_tokens=0) -> str`.
- Decoding is greedy argmax; stops at `eos_token_id`. There is no temperature/top-p/seed; extra kwargs are ignored by `engine.generate.generate`.
- `apply_chat_template=None`: wrap the prompt in the folder's chat template if one exists and the prompt is not already templated. `False` sends raw tokens (adds BOS).
- `enable_thinking=False` closes the Qwen `<think>` block. `lab/llm_policy.py` always calls `engine.generate(prompt, max_new_tokens=384, enable_thinking=False)`.
- `num_speculative_tokens=K` uses native MTP (Qwen3.8 only; lossless greedy). `eng.last_mtp_stats` has `rounds/drafted/accepted`.
- `eng.stream(...)` same args, yields text pieces. `eng.generate_messages(messages, ...)` for Qwen image/video via the HF processor.
- One prompt at a time (`tokens` is `[1, T]`). No batching API.

## Raw logits (how lab scores holdout PPL)
```python
ids = eng.tokenizer.encode(text, add_special_tokens=True)
tokens = torch.tensor([ids], device=eng.model.device)
logits = eng.model.forward(tokens, cache=None)   # [1, T, vocab]
```
`lab/scorer.py:InferScorer` does exactly this, then `log_softmax` and gathers targets `ids[1:]`.

## CLI
```bash
python -m engine.chat --model ~/models/Qwen3-0.6B --inspect                    # config only
python -m engine.chat --model ~/models/Llama-3.2-1B-Instruct --device cuda \
  --prompt "The capital of France is" --max-new-tokens 32
```
Flags: `--model DIR` (required), `--device cuda|cpu`, `--dtype bfloat16|float16|float32`, `--inspect`, `--prompt STR`, `--max-new-tokens N` (default 32), `--no-cache`, `--raw-prompt`, `--enable-thinking`, `--mtp-draft-tokens K`, `--skip-tokenizer-smoke`. Prints `load_ok=true`, `completion=...`, `generate_ok=true`.

## Supported recipes (engine/detect.py KNOWN_RECIPES)
`llama, mistral, qwen2, qwen3, yi, gemma, phi3, mixtral, llama4, gpt2, gpt_neox, gpt_oss, deepseek_v3, nemotron_h, qwen3_5`.
Qwen3.5/Qwen3.8 = `qwen3_5` (hybrid GDN + attention, vision, native MTP). Qwen MoE maps to `mixtral`. Rejected: `qwen3_next`, `qwen3_5_moe`. NVFP4/FP8 checkpoints are dequantized on load, not fused.

## Memory guidance
- Spark: NVIDIA GB10, ~121 GB unified memory shared by CPU and GPU (`free -g`). No separate VRAM pool.
- Qwen3.8-27B is 52 GB of bf16 safetensors on disk; loaded it takes roughly the same plus KV cache, about 55 GB (unverified exact number). Qwen3-0.6B is 1.5 GB.
- Run one big engine per process at a time. Loading a second 27B alongside the first, or alongside a training job, will exhaust memory. Free with `del eng; torch.cuda.empty_cache()`.
- `inspect_capabilities` is free; call it before `load_engine`.

## What infer does NOT do
- No training, fine-tuning, or optimizer code. Use PyTorch directly (see pytorch_transformer skill).
- No sampling: greedy only. No logprob/scoring helper; use `eng.model.forward` as above.
- No HTTP server, no batching, no multi-GPU sharding (unverified: none found in `engine/`).
- Does not read lab `checkpoint.pt` files; those are TinyGPT byte models scored by `lab/scorer.py:LabScorer`.
