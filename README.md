# lab — Eval → Research → Train harness

Supervisor, phase tools, notebook, GPU lock, research sandbox, dummy / **lab** / tinytrain jobs, frozen **core v1** eval, scripted control. 120B is not wired yet.

## Loop

Eval (read-only) → Research (web/code/data, budgeted) → Train (one GPU-locked commit from a hashed pack) → Eval.

`enter_train` refuses without a valid pack. Frozen eval and promotion are not policy tools.

## Eval suite (`core` v1)

Promotion metric is **confirm_ppl** (lower is better), measured on a frozen holdout the policy cannot write. Trainer loss/PPL is always logged. Benchmarks are reported, not the promotion gate — an 8M TinyStories model will sit near chance.

| Track | What | Notes |
|---|---|---|
| Trainer | `train_loss`, `val_loss`, `val_ppl` | From the job. `ppl = exp(loss)` |
| Frozen LM | tune / **confirm** / ood PPL | Confirm is held out from tune. Ood is general English, not stories |
| HellaSwag | 4-way commonsense completion | Frozen mini slice (16 items) |
| ARC-Easy | 4-way science QA | Frozen mini slice (16 items) |
| PIQA | 2-way physical commonsense | Frozen mini slice (16 items) |

If no checkpoint is loadable, holdout/benchmarks are skipped and `confirm_ppl` proxies from trainer `val_ppl` (`confirm_source=trainer_val_proxy`). Lab `.pt` checkpoints are scored by TinyGPT byte-NLL on the frozen files. HuggingFace dirs and `infer` engines still work via `LAB_INFER_ROOT`.

## Live hypothesis

`hypothesis.json` (+ `.md`) is the working theory: claim, why, falsify, status, checklist.
`enter_train` refuses while `claim_written` or `pack_ready` are open. Dummy/proxy eval does **not** tick `holdout_not_proxy`.

## Episodes

Each finished train+eval cycle writes a skill-like card under `episodes/` (`ep-*.json` + `.md`, plus `index.jsonl`). Title, hypothesis, config, pack hash, job stats, confirm PPL, verdict, next hint. `list_episodes` / `read_episode` are policy tools. Beliefs.md is rewritten from the index.

```bash
python -m lab seal --run-dir runs/nano-cycle2
```

These mini slices pin the *format and code path*. Swap in official HF splits later without changing the suite id unless the items change — then bump `eval_suite_version`.

## Run

```bash
cd ~/Projects/lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m lab run --policy dummy --cycles 1
python -m lab run --policy lab --cycles 1 --run-dir runs/lab-smoke
python -m lab run --policy scripted --cycles 2 --run-dir runs/ctrl
python -m lab run --policy interleave --cycles 2 --run-dir runs/mix
python -m lab run --policy ensemble --cycles 2 --ensembles 3 --run-dir runs/ens   # see Ensemble
python -m lab status --run-dir runs/mix
python -m lab promote --run-dir runs/mix
```

Default `--run-dir` is `runs/<timestamp>/`. Everything for a run lives there:

```
runs/<id>/
  state.json          phase, cycle, pack hash, job id
  hypothesis.json     live theory + checklist
  notebook.jsonl      append-only log
  beliefs.md          episode index
  sandbox/            research jail
  packs/<hash>/       frozen train commits
  jobs/job-XXXX/      pack.json, data.txt, train.log, metrics.json, checkpoint.pt, job.json
  checkpoints/        copies of job-XXXX.pt
  frozen_eval/        copied suite (read-only)
  episodes/           skill cards after train+eval
  hypotheses/         board: hyp-XXXX.json per hypothesis, index.jsonl, board.md
  ensembles/          cycle-XXXX.json: every ensemble's transcript, pack, candidates
  roles.json          which model filled thinker / tooler / coder (versioned)
  trace.jsonl         one row per span: run, cycle, step, tool, policy turn
```

The **lab** trainer is a sandboxed subprocess (`python -m lab.train`) with cwd = the job dir, stripped env (`HOME`/`TMPDIR` inside the job), and a wall-clock timeout from `budgets.max_hours`. Needs `pip install torch` (or `pip install -e ".[train]"`). Dummy stays the fast no-torch path.

Tests: `pytest`

## Env

| Variable | Meaning |
|---|---|
| `LAB_TINYTRAIN_ROOT` | existing tinytrain checkout; pack must set `config.command` argv |
| `LAB_INFER_ROOT` | `infer` repo for holdout/benchmark scoring |
| `LAB_SUBJECT_CHECKPOINT` | parent checkpoint dir if it should be scored |
| `LAB_GPU_LOCK` | lock file (default `/tmp/lab-gpu.lock` via CLI) |
| `LAB_ALLOW_NETWORK` | `0` disables fetch/search **and** Hub downloads on cache miss |
| `LAB_DATA_CACHE` | allowlisted HF text cache (default `~/data/lab`) |
| `LAB_POLICY_MODEL` | experimenter checkpoint (default `~/models/Qwen3.8-27B`) |
| `LAB_THINKER_MODEL` | model dir for the ensemble **thinker** role (unset = role not configured) |
| `LAB_TOOLER_MODEL` | model dir for the **tooler** role |
| `LAB_CODER_MODEL` | model dir for the **coder** role |
| `LAB_MODELS_DIR` | root that `list_models` / `inspect_model` scan (default `~/models`) |
| `LAB_ENSEMBLES` | parallel ensembles per research phase (default 3; flag `--ensembles`) |
| `LAB_ENSEMBLE_ROUNDS` | max thinker rounds per ensemble (default 12; flag `--rounds`) |
| `LANGSMITH_TRACING` | `true` mirrors spans to LangSmith (needs `pip install -e ".[trace]"`) |
| `LANGSMITH_API_KEY` | LangSmith key; without it only the local `trace.jsonl` is written |
| `LANGSMITH_PROJECT` | LangSmith project name (default `lab-harness`) |

## Tracing

Every run writes `trace.jsonl` locally, with no dependencies and no network.
Set `LANGSMITH_TRACING=true` to also mirror the tree to LangSmith. A sink that
fails never fails the run: a bad key costs stderr noise, nothing else.

```bash
pip install -e ".[trace]"
export LANGSMITH_TRACING=true LANGSMITH_API_KEY=lsv2_... LANGSMITH_PROJECT=lab-harness
python -m lab run --policy qwen --cycles 12 --run-dir runs/qwen-12

python -m lab trace-report --run-dir runs/qwen-12          # markdown digest
python -m lab trace-report --run-dir runs/qwen-12 --json    # machine-readable
```

The span tree is `lab.run → cycle N → step K → {policy.act | policy.nudge, tool.<name>}`.
Policy turns carry the prompt, the raw completion, and the tool that was parsed
out of it; tool spans carry args, the result, and the rejection reason. Cycle,
phase, and tool are attached as both metadata and tags, so in LangSmith you can
filter to one phase or one tool across a whole run.

## Ensemble

`--policy ensemble` replaces the single experimenter with N parallel three-role ensembles. Each role is a small model behind a `Roles` slot:

| Role | Does | Replies with |
|---|---|---|
| thinker | reasons about the observation, asks for tools or code, writes 1–3 hypotheses, emits the pack | `{"thought", "hypotheses": [{claim, why, falsify}], "need": null \| {"kind": "tool" \| "code", "request"}, "pack", "done"}` |
| tooler | turns a thinker request into ONE research tool call | `{"tool", "args"}` |
| coder | writes ONE python script for the sandbox (`write_and_run`) | `{"path", "content", "run", "args"}` or a ```` ```python ```` fence |

Loop per ensemble (`lab/ensemble/ensemble.py`): thinker → (tooler \| coder) → thinker, up to `--rounds` rounds, ≤10 tool calls, ≤3 nudges. Each ensemble gets an angle from `VARIANT_HINTS` (lr, steps, data mix, batch/seq_len, architecture, perturb-the-best) and a sandbox jail `sandbox/ens-<i>/`; tooler `path`/`dest` args are prefixed into it. The tooler cannot call `write_pack`, `write_hypothesis`, `queue_candidates`, `enter_*` or `halt`. Invalid packs are fed back as a tool error; repeated identical calls are refused.

Research phase (`lab/ensemble/runner.py`): N ensembles run in threads (`sup.call` is serialized by one lock), then **commit** serially: `write_hypothesis(source="ensemble-<i>")` for each hypothesis, `write_pack(hypothesis_id=…)`, dedupe by pack hash, `queue_candidates`. Every candidate becomes a **trial**: the harness trains and evaluates each `(hypothesis, pack)` in turn before the cycle number advances (`trial`/`trials_planned` in `state.json`, one episode per trial, a `cycle_summary` note at the end). If no ensemble produced a valid pack, ONE harness `lab_pack` is queued with `source="harness-fallback"` and the run prints `[ensemble] fallback pack authored by harness, not the models`.

```bash
LAB_THINKER_MODEL=~/models/Qwen3.8-4B LAB_TOOLER_MODEL=~/models/Qwen3.8-1.7B LAB_CODER_MODEL=~/models/Qwen3.8-Coder-7B \
python -m lab run --policy ensemble --cycles 4 --ensembles 3 --rounds 12 --run-dir runs/ens
python -m lab roles --run-dir runs/ens                   # roles.json as a table
python -m lab run --policy ensemble --scripted-roles roles.json --run-dir runs/demo   # offline: canned replies
```

Dropping in a model is one env var per role; nothing loads until the first `generate`. An unconfigured role raises `RoleNotConfigured` inside its ensemble (recorded as `error`, never fatal) and the startup banner lists the missing `LAB_*_MODEL` vars. `--scripted-roles` takes `{"thinker": [...], "tooler": [...], "coder": [...]}` lists of replies (FIFO, shared across ensembles; when a list runs out the role answers `{"done": true}`). Artifacts: `runs/<id>/ensembles/cycle-XXXX.json` (every transcript, pack, candidate, fallback flag), `roles.json`, `hypotheses/`. Trace spans: `ensemble <i>` → `ensemble <i> round <r>` → `ensemble <i> <role>` (llm). Default `--max-steps` is `max(60, cycles*40)` because trials add steps.

## Policy tools

- Always: `read_notebook`, `write_note`, `write_beliefs`, `halt`
- Eval: `run_eval`, `read_metrics`, `read_samples`, `enter_research`
- Research: `list_files`, `read_file`, `write_file`, `exec`, `web_fetch`, `web_search`, `prefetch_data`, `write_hypothesis`, `write_pack`, `queue_candidates`, `enter_train`, plus `run_python`, `write_and_run`, `data_stats`
- Eval + research (inspect): `list_episodes`, `read_episode`, `read_hypothesis`, `list_hypotheses`, `grep_files`, `list_models`, `inspect_model`, `read_checkpoint_meta`, `read_train_log`, `read_trace`, `list_packs`, `read_pack`, `diff_packs`, `list_jobs`, `read_skill`, `list_skills`, `read_episode_metrics`, `sandbox_usage`
- Train: `job_status`, `cancel_job`, `enter_eval`

Pack fields: hypothesis, trainer (`dummy` \| `lab` \| `tinytrain`), config, data_manifest, eval_suite_id (`core`), eval_suite_version (`1`), parent_checkpoint, budgets (`max_hours` ≤ 3.5).

Lab config keys: `lr`, `steps`, `hidden`, `layers`, `heads`, `seq_len`, `batch`. Default train mix is the allowlisted TinyStories slice, materialized into `jobs/.../data.txt` before the trainer starts. `prefetch_data` fills `LAB_DATA_CACHE`; the GPU job stays offline. Frozen eval is never a train source. Cycle N+1 copies `checkpoints/latest.pt` in as `parent.pt` so weights continue. Frozen confirm PPL is measured on the `.pt` (not trainer val). The experimenter (`--policy qwen` / `--policy nano`) defaults to Qwen3.8-27B.

Install Hub support with `pip install -e ".[data]"` (`datasets`). Cache misses fail closed when `LAB_ALLOW_NETWORK=0`.
