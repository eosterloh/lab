# sandbox_rules
Rules for files and commands in the research sandbox (`lab/sandbox.py`). Break one and the tool call is rejected.

## Paths
- Relative only. `/abs`, `~/x`, and any `..` segment are rejected ("path must be a relative sandbox path").
- cwd for every command is the sandbox root: `runs/<id>/sandbox/`. `HOME` is also set to it.
- Write with `write_file {"path": "probe.py", "content": "..."}`; subdirectories are created for you.
- `read_file` returns at most 32,000 bytes. `list_files` lists everything recursively.
- Job dirs, checkpoints, and `frozen_eval/` are outside the sandbox. To use a checkpoint, ask the tooler for its path in the tool result; do not guess `../checkpoints`.

## Limits
- Disk cap: 50 MB total for the sandbox (`sandbox_max_bytes`). Writes over the cap fail with "sandbox disk cap exceeded". Delete large outputs by overwriting with an empty string.
- Exec timeout: 30 s wall clock by default (`exec_timeout_s`); `run_python`/`write_and_run` accept `timeout_s`, capped at `LAB_EXEC_MAX_TIMEOUT_S` (300 s). A timeout returns `returncode=-1, timed_out=true`. Longer work belongs in a `lab` train job via `write_pack`.
- stdout and stderr are each cut at 32,000 chars. Print little; print the summary last.
- Research phase has a tool-call budget (50 calls, 600 s). Every failed call still counts.

## Environment is stripped
Commands run with `PATH, HOME, LANG, LC_ALL` plus `LAB_TRAIN_DEVICE` and `CUDA_VISIBLE_DEVICES` when the harness has them. Nothing else: no other `LAB_*`, no API keys.
- Read configuration from argv or a JSON file you wrote, never from env.
- Device: `os.environ.get("LAB_TRAIN_DEVICE", "cpu")`; fall back to cpu when unset or when `torch.cuda.is_available()` is false.
- `run_python` / `write_and_run` run the harness interpreter (`sys.executable`) with `PYTHONPATH=<repo root>`, so `from lab.train.loop import load_checkpoint, build_model` works there. Plain `exec ["python3", ...]` has no `PYTHONPATH`; copy the functions you need instead.

## Network
- `LAB_ALLOW_NETWORK=0`: `web_fetch`, `web_search`, and Hub downloads all fail with "network disabled". Do not retry; use cached data (`prefetch_data`, `LAB_DATA_CACHE`).
- When allowed, `web_fetch {"url", "path"}` saves under `fetch/<name>` (2 MB cap, http/https only). Never fetch frozen eval files.

## Forbidden
- Commands: `rm`, `sudo`, `chmod`, `chown`, `mkfs` as argv[0]. Shell strings are not accepted; `exec` takes an argv list: `{"argv": ["python3", "probe.py", "--steps", "10"]}`.
- Writing outside the sandbox, editing `frozen_eval`, or reading `lm_confirm.jsonl` into training data.
- Training inside `exec`. Train through a pack.

## Script hygiene
- Keep each script under ~200 lines and idempotent: rerunning overwrites its outputs, never appends.
- Take inputs from argparse with defaults (`--data data.txt`, `--out .`). Return non-zero on failure.
- Print progress sparingly (`step=N train_loss=X`), then a single final JSON line:
  `print(json.dumps({"ok": True, "n_params": n, "val_ppl": ppl}))`
  The tool result is parsed from stdout; the last JSON line is what the tooler and thinker will read.
- Catch exceptions at the top level and print `{"ok": false, "error": "..."}` so failures are also machine-readable.
- Prefer `write_and_run {"path": "probe.py", "content": "...", "args": ["--steps", "5"], "timeout_s": 60}` (write + run in one call) or `run_python {"path", "args", "timeout_s"}` over raw `exec`. Both return `returncode, stdout, stderr`. Use `exec {"argv": [...]}` only for non-Python commands (`ls`, `wc`, `head`).
- Small first: `--steps 5` and a 2 KB `data.txt` to check shapes, then scale. Sandbox exec has 30 s; a 32-hidden TinyGPT does about 50 steps/s on CPU (unverified on other hardware).
