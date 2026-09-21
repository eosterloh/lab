# lab_harness
Facts about the lab Eval -> Research -> Train harness: packs, jobs, metrics, checkpoints, phases, tools.

## Pack schema (`lab/pack.py:ArtifactPack`)
Exactly these fields; unknown keys are rejected. The values are shape, not content: choose your own from the observation.
```json
{"hypothesis": "<knob> <old>-><new>, vs <baseline ep> (confirm_ppl <ppl>); expect lower",
 "trainer": "lab",
 "config": {"lr": 0.001, "steps": 64, "hidden": 32, "layers": 1, "heads": 1, "seq_len": 32, "batch": 8},
 "data_manifest": {"sources": ["hf:roneneldan/TinyStories:train:10000"]},
 "eval_suite_id": "core",
 "eval_suite_version": 1,
 "parent_checkpoint": "subjects/tinytrain-8m",
 "budgets": {"max_hours": 0.1, "max_steps": 64}}
```
`parent_checkpoint` must be copied from the observation: `last_checkpoint` when set, else `subject_checkpoint`. Never invent a path.
Rules: `hypothesis` non-empty; `trainer` in `dummy | lab | tinytrain`; `config` non-empty object; `eval_suite_id` = `core`, `eval_suite_version` = 1; `parent_checkpoint` non-empty string; `budgets.max_hours` in (0, 3.5]; lab: `hidden % heads == 0`; tinytrain: `config.command` argv list required. The pack hash is sha256 of the canonical JSON; same content = same hash = same job.

## Trainers
- `dummy`: no torch, instant fake metrics. Use for wiring tests only. Its confirm_ppl is a trainer proxy, never a real result.
- `lab`: `python -m lab.train --job-dir <dir>` subprocess, TinyGPT byte model on `data.txt`. Default for real experiments.
- `tinytrain`: runs `config.command` in `LAB_TINYTRAIN_ROOT`. Only when that env var is set.

## Job dir contract (`runs/<id>/jobs/job-XXXX/`)
In: `pack.json`, `data.txt` (materialized from `data_manifest.sources`, must be >= 64 chars), optional `parent.pt` (copied from `parent_checkpoint`, else newest `checkpoints/job-*.pt`), `corpus.txt` when `builtin:tiny` is used.
Out: `train.log` (stdout+stderr), `metrics.json`, `checkpoint.pt`, `job.json` (status, error, pid).
Env is stripped: `PATH, HOME=<job dir>, PYTHONPATH=<repo>, LAB_JOB_DIR, TMPDIR, LAB_TRAIN_DEVICE (cpu|cuda), CUDA_VISIBLE_DEVICES`. cwd = job dir. Timeout = `max(30s, max_hours*3600)`. Exit != 0 or missing `metrics.json` -> `failed`, log tail in `job.error`.

## metrics.json (lab trainer)
`train_loss, val_loss, val_ppl, steps, lr, tokens_seen, n_params, hidden, layers, device, backend="lab", checkpoint, resumed, parent, train_chars`; after publish also `run_checkpoint`. `val_ppl = exp(min(val_loss, 20))`. Trainer prints `step=N train_loss=X` every 10 steps and a final JSON line.

## Checkpoints
- Job writes `jobs/job-XXXX/checkpoint.pt`; on success it is copied to `runs/<id>/checkpoints/job-XXXX.pt` and `checkpoints/latest.pt`.
- Format: `{"model": state_dict, "optim": ..., "config": {hidden, layers, heads, seq_len, vocab}}`.
- Next cycle's `parent.pt` = `parent_checkpoint` if it is a file, else newest `checkpoints/job-*.pt`. Weights resume only when all five config keys match; otherwise the trainer prints `parent architecture mismatch` and starts from scratch.
- Set `parent_checkpoint` to `observation.last_checkpoint` when present.

## confirm_ppl (the promotion metric, lower is better)
- Suite `core` v1: LM holdouts `tune / confirm / ood` (`lm_*.jsonl` under `runs/<id>/frozen_eval/`, read-only) plus HellaSwag, ARC-Easy, PIQA mini slices (reported, not gating).
- `confirm_ppl = exp(sum NLL / tokens)` over `lm_confirm.jsonl`, computed by `lab/evals.py:run_eval` with a scorer on the `.pt`: `LabScorer` for TinyGPT files (byte-level, sliding `seq_len` window), `InferScorer`/`HfScorer` for model dirs.
- Trainer `val_ppl` is one random batch of training text. It is NOT confirm_ppl. When no checkpoint loads, eval falls back to `confirm_source=trainer_val_proxy`; that never ticks `holdout_not_proxy` and must not be promoted.
- Eval result keys: `confirm_ppl, confirm_source, backend, loss.{train_loss,val_loss,val_ppl,holdout}, benchmarks, checkpoint`.

## Phase machine and legal tools (`lab/types.py`)
Cycle: `eval -> research -> train -> eval`. One tool call per turn; illegal tools are rejected.
- Always: `read_notebook, write_note, write_beliefs, halt`
- Memory (eval + research): `list_episodes, read_episode, read_hypothesis, list_hypotheses`
- Inspect, read-only (eval + research): `grep_files, list_models, inspect_model, read_checkpoint_meta, read_train_log, read_trace, list_packs, read_pack, diff_packs, list_jobs, read_skill, list_skills, read_episode_metrics, sandbox_usage`
- eval: `run_eval, read_metrics, read_samples, enter_research`
- research: `list_files, read_file, write_file, exec, run_python, write_and_run, data_stats, web_fetch, web_search, write_hypothesis, prefetch_data, write_pack, enter_train`
- train: `job_status, cancel_job, enter_eval, read_train_log, list_jobs, read_trace`
Research is budgeted (`research_max_tool_calls=50`, `research_max_seconds=600`). `enter_train` refuses without a valid pack, and while `claim_written` or `pack_ready` is still open.

## Hypothesis checklist (`lab/hypothesis.py`)
`write_hypothesis` args: `{"claim", "why", "falsify"}` only. Status: `open -> testing -> killed | closed`. Checklist ids: `claim_written` (train), `pack_ready` (train), `post_eval` (close), `episode_sealed` (close), `holdout_not_proxy` (promote).

## Episode cards (`runs/<id>/episodes/`)
One `ep-NNNN-<slug>.json` + `.md` per finished train+eval, plus `index.jsonl`. Fields: `id, title, cycle, hypothesis, trainer, config, data_manifest, budgets, pack_hash, parent_checkpoint, job_id, job_status, eval, verdict, next_hint`. `list_episodes(n, query)` returns summaries; `read_episode(id)` returns the JSON and the card. `beliefs.md` is rebuilt from the index.

## lab config keys and sane ranges
| key | default | range |
|---|---|---|
| `lr` | 3e-3 | 1e-4 .. 1e-2 |
| `steps` | 20 | 8 .. 512 (also set `budgets.max_steps`) |
| `hidden` | 32 | 16 .. 256, divisible by `heads` |
| `layers` | 1 | 1 .. 4 |
| `heads` | 1 | 1 .. 8 |
| `seq_len` | 32 | 16 .. 256 |
| `batch` | 8 | 4 .. 64 |
vocab is fixed at 256 (bytes).

## Data sources
Allowlist: `hf:roneneldan/TinyStories`, `hf:wikimedia/wikipedia`, `hf:HuggingFaceFW/fineweb-edu`. Source string `hf:<org>/<name>:<split>:<n>`; default mix `["hf:roneneldan/TinyStories:train:10000"]`. `prefetch_data {"source","split","n"}` fills `LAB_DATA_CACHE`; the train job stays offline. Frozen eval is never a train source.

## LAB_* env (README)
`LAB_TINYTRAIN_ROOT`, `LAB_INFER_ROOT`, `LAB_SUBJECT_CHECKPOINT`, `LAB_GPU_LOCK`, `LAB_ALLOW_NETWORK` (`0` = no fetch/search/Hub), `LAB_DATA_CACHE` (default `~/data/lab`), `LAB_POLICY_MODEL` (default `~/models/Qwen3.8-27B`), `LAB_TRAIN_DEVICE` (`cpu` default; `cuda`), `LAB_THINKER_MODEL / LAB_TOOLER_MODEL / LAB_CODER_MODEL / LAB_MODELS_DIR`, `LAB_ENSEMBLES`, `LAB_ENSEMBLE_ROUNDS`, `LANGSMITH_*` for tracing.

## CLI
`python -m lab run --policy lab --cycles 1 --run-dir runs/x`, `python -m lab status --run-dir runs/x`, `python -m lab promote --run-dir runs/x`, `python -m lab seal --run-dir runs/x`.
