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

## Policy tools

- Always: `read_notebook`, `write_note`, `write_beliefs`, `halt`
- Eval: `run_eval`, `read_metrics`, `read_samples`, `enter_research`
- Research: `list_files`, `read_file`, `write_file`, `exec`, `web_fetch`, `web_search`, `prefetch_data`, `write_hypothesis`, `write_pack`, `enter_train`
- Train: `job_status`, `cancel_job`, `enter_eval`

Pack fields: hypothesis, trainer (`dummy` \| `lab` \| `tinytrain`), config, data_manifest, eval_suite_id (`core`), eval_suite_version (`1`), parent_checkpoint, budgets (`max_hours` ≤ 3.5).

Lab config keys: `lr`, `steps`, `hidden`, `layers`, `heads`, `seq_len`, `batch`. Default train mix is the allowlisted TinyStories slice, materialized into `jobs/.../data.txt` before the trainer starts. `prefetch_data` fills `LAB_DATA_CACHE`; the GPU job stays offline. Frozen eval is never a train source. Cycle N+1 copies `checkpoints/latest.pt` in as `parent.pt` so weights continue. Frozen confirm PPL is measured on the `.pt` (not trainer val). The experimenter (`--policy qwen` / `--policy nano`) defaults to Qwen3.8-27B.

Install Hub support with `pip install -e ".[data]"` (`datasets`). Cache misses fail closed when `LAB_ALLOW_NETWORK=0`.
