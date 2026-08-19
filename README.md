# lab — Eval → Research → Train harness

v0: supervisor, phase tools, notebook, GPU lock, sandbox, dummy (and optional tinytrain) jobs, frozen story eval, scripted control. No 120B yet.

## Loop

Eval (read-only) → Research (web/code/data, budgeted) → Train (one GPU-locked commit from a hashed pack) → Eval.

`enter_train` refuses without a valid pack. Frozen eval and promotion are not policy tools.

## Run

```bash
cd ~/Projects/lab
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python -m lab run --policy dummy --cycles 1
python -m lab run --policy scripted --cycles 2 --run-dir runs/ctrl
python -m lab run --policy interleave --cycles 2 --run-dir runs/mix
python -m lab status --run-dir runs/mix
python -m lab promote --run-dir runs/mix
```

Tests: `pytest`

## Env

| Variable | Meaning |
|---|---|
| `LAB_TINYTRAIN_ROOT` | existing tinytrain checkout; pack must set `config.command` argv |
| `LAB_INFER_ROOT` | `infer` repo for optional sample generation |
| `LAB_SUBJECT_CHECKPOINT` | parent checkpoint id (default `subjects/tinytrain-8m`) |
| `LAB_GPU_LOCK` | lock file (default `/tmp/lab-gpu.lock` via CLI) |
| `LAB_ALLOW_NETWORK` | `0` disables fetch/search |

## Policy tools

- Always: `read_notebook`, `write_note`, `write_beliefs`, `halt`
- Eval: `run_eval`, `read_metrics`, `read_samples`, `enter_research`
- Research: `list_files`, `read_file`, `write_file`, `exec`, `web_fetch`, `web_search`, `write_pack`, `enter_train`
- Train: `job_status`, `cancel_job`, `enter_eval`

Pack fields: hypothesis, trainer (`dummy` \| `tinytrain`), config, data_manifest, eval_suite_id, eval_suite_version, parent_checkpoint, budgets (`max_hours` ≤ 3.5).
