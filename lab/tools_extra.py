"""Extra research tools for the tool-calling role. Registered into actions.REGISTRY.

Every handler is `(sup, args) -> ToolResult`, sandboxed to relative paths under
`sup.sandbox.root` (or read-only paths under `sup.cfg.run_dir`), offline-safe,
and never raises: bad arguments and runtime failures come back as `err(...)`
with a message that names the offending argument.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
import importlib
import json
import os
import re
import sys

from lab.types import ToolResult, err, ok

Args = dict[str, Any]
Handler = Callable[[Any, Args], ToolResult]

MAX_N = 500
MAX_LIST = 200
TEXT_CLIP = 300
GREP_MAX_FILE_BYTES = 2 * 1024 * 1024
TRACE_KINDS = frozenset({"tool", "llm", "chain"})


# --------------------------------------------------------------------------- helpers


def _clip(text: Any, n: int = TEXT_CLIP) -> str:
    if not isinstance(text, str):
        try:
            text = json.dumps(text, default=str, sort_keys=True)
        except Exception:
            text = str(text)
    if len(text) <= n:
        return text
    return text[: n - 3] + "..."


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")


def _int_arg(
    args: Args, key: str, default: int, *, lo: int = 1, hi: int = MAX_N
) -> tuple[int, str | None]:
    """Coerce an optional int arg, clamped into [lo, hi]. Returns (value, error)."""
    raw = args.get(key, default)
    if raw is None:
        raw = default
    if isinstance(raw, bool):
        return default, f"{key} must be an integer, got bool"
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default, f"{key} must be an integer, got {type(raw).__name__}"
    return max(lo, min(val, hi)), None


def _float_arg(args: Args, key: str) -> tuple[float | None, str | None]:
    raw = args.get(key)
    if raw is None:
        return None, None
    if isinstance(raw, bool):
        return None, f"{key} must be a number, got bool"
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None, f"{key} must be a number, got {type(raw).__name__}"
    if val <= 0:
        return None, f"{key} must be > 0"
    return val, None


def _str_arg(args: Args, key: str, *, required: bool = True, default: str = "") -> tuple[str, str | None]:
    raw = args.get(key)
    if raw is None:
        if required:
            return "", f"{key} is required (string)"
        return default, None
    if not isinstance(raw, str):
        return "", f"{key} must be a string, got {type(raw).__name__}"
    if required and not raw.strip():
        return "", f"{key} must be a non-empty string"
    return raw, None


def _str_list_arg(args: Args, key: str) -> tuple[list[str], str | None]:
    raw = args.get(key)
    if raw is None:
        return [], None
    if not isinstance(raw, list) or not all(isinstance(x, str) for x in raw):
        return [], f"{key} must be a list of strings"
    return list(raw), None


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def _infer_root(sup: Any) -> Path:
    root = getattr(sup.cfg, "infer_root", None)
    if root is None:
        root = Path.home() / "Projects" / "infer"
    return Path(root).expanduser()


def _models_dir(sup: Any) -> Path:
    raw = os.environ.get("LAB_MODELS_DIR")
    if raw:
        return Path(raw).expanduser()
    cfg_dir = getattr(sup.cfg, "models_dir", None)
    if cfg_dir:
        return Path(cfg_dir).expanduser()
    return Path("~/models").expanduser()


def _skills_module() -> tuple[Any, str | None]:
    try:
        # import_module honours sys.modules[...] = None, so tests can simulate absence.
        mod = importlib.import_module("lab.ensemble.skills")
        if not hasattr(mod, "load_skill") or not hasattr(mod, "list_skills"):
            raise ImportError("lab.ensemble.skills lacks load_skill/list_skills")
    except Exception as e:  # ImportError, or sys.modules[...] = None -> ImportError
        return None, f"skills module not available: {e}"
    return mod, None


def _skill_names(mod: Any) -> list[str]:
    try:
        return [str(s) for s in mod.list_skills()][:MAX_LIST]
    except Exception:
        return []


# --------------------------------------------------------------------------- code execution


def run_python(sup: Any, args: Args) -> ToolResult:
    """{path: str, args?: list[str], timeout_s?: number}"""
    path, e = _str_arg(args, "path")
    if e:
        return err(e)
    argv, e = _str_list_arg(args, "args")
    if e:
        return err(e)
    timeout_s, e = _float_arg(args, "timeout_s")
    if e:
        return err(e)
    try:
        return ok(**sup.sandbox.run_python(path, argv, timeout_s=timeout_s))
    except FileNotFoundError:
        return err(f"path {path!r} is not a file in the sandbox")
    except Exception as ex:
        return err(str(ex))


def write_and_run(sup: Any, args: Args) -> ToolResult:
    """{path: str, content: str, args?: list[str], timeout_s?: number}"""
    path, e = _str_arg(args, "path")
    if e:
        return err(e)
    content = args.get("content")
    if not isinstance(content, str):
        return err("content must be a string (python source)")
    argv, e = _str_list_arg(args, "args")
    if e:
        return err(e)
    timeout_s, e = _float_arg(args, "timeout_s")
    if e:
        return err(e)
    try:
        rel = sup.sandbox.write_file(path, content)
    except Exception as ex:
        return err(f"write_file failed: {ex}")
    try:
        result = sup.sandbox.run_python(rel, argv, timeout_s=timeout_s)
    except Exception as ex:
        return err(f"run failed after writing {rel}: {ex}", path=rel)
    result.pop("path", None)
    return ok(path=rel, **result)


def grep_files(sup: Any, args: Args) -> ToolResult:
    """{pattern: str, path?: str = ".", max_hits?: int = 50}"""
    pattern, e = _str_arg(args, "pattern")
    if e:
        return err(e)
    rel, e = _str_arg(args, "path", required=False, default=".")
    if e:
        return err(e)
    max_hits, e = _int_arg(args, "max_hits", 50)
    if e:
        return err(e)
    try:
        rx = re.compile(pattern)
    except re.error as ex:
        return err(f"pattern is not a valid regex: {ex}")
    try:
        files = sup.sandbox.list_files(rel or ".")
    except Exception as ex:
        return err(f"path: {ex}")
    hits: list[dict[str, Any]] = []
    scanned = 0
    skipped = 0
    for name in files:
        if len(hits) >= max_hits:
            break
        fpath = sup.sandbox.root / name
        try:
            if fpath.stat().st_size > GREP_MAX_FILE_BYTES:
                skipped += 1
                continue
            data = fpath.read_bytes()
        except OSError:
            skipped += 1
            continue
        if b"\x00" in data[:8192]:
            skipped += 1
            continue
        scanned += 1
        text = data.decode("utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append({"path": name, "line": lineno, "text": _clip(line, 200)})
                if len(hits) >= max_hits:
                    break
    return ok(
        pattern=pattern,
        hits=hits,
        n_hits=len(hits),
        truncated=len(hits) >= max_hits,
        scanned=scanned,
        skipped=skipped,
    )


# --------------------------------------------------------------------------- models / checkpoints


def list_models(sup: Any, args: Args) -> ToolResult:
    """{} -> model folders with config.json under LAB_MODELS_DIR (default ~/models)"""
    root = _models_dir(sup)
    if not root.is_dir():
        return ok(models=[], models_dir=str(root))
    models: list[dict[str, Any]] = []
    try:
        entries = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as ex:
        return err(f"cannot list {root}: {ex}")
    for d in entries:
        cfg_path = d / "config.json"
        if not cfg_path.is_file():
            continue
        model_type: Any = None
        architectures: Any = None
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict):
                model_type = cfg.get("model_type")
                architectures = cfg.get("architectures")
        except Exception:
            pass
        has_st = any(True for _ in d.glob("*.safetensors"))
        models.append(
            {
                "name": d.name,
                "path": str(d),
                "model_type": model_type,
                "architectures": architectures,
                "has_safetensors": has_st,
            }
        )
        if len(models) >= MAX_LIST:
            break
    return ok(models=models, models_dir=str(root))


def inspect_model(sup: Any, args: Args) -> ToolResult:
    """{model_dir: str} -> engine.agent_api.inspect_capabilities(model_dir).to_dict(); config only"""
    model_dir, e = _str_arg(args, "model_dir")
    if e:
        return err(e)
    path = Path(model_dir).expanduser()
    if not path.is_absolute():
        path = _models_dir(sup) / path
    if not path.is_dir():
        return err(f"model_dir {model_dir!r} is not a directory")
    if not (path / "config.json").is_file():
        return err(f"model_dir {model_dir!r} has no config.json")
    root = str(_infer_root(sup).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from engine.agent_api import inspect_capabilities  # type: ignore
    except ImportError as ex:
        return err(f"infer not available: {ex} (infer_root={root})")
    except Exception as ex:
        return err(f"infer not available: {type(ex).__name__}: {ex}")
    try:
        caps = inspect_capabilities(path).to_dict()
    except Exception as ex:
        return err(f"{type(ex).__name__}: {ex}")
    return ok(model_dir=str(path), capabilities=caps)


def read_checkpoint_meta(sup: Any, args: Args) -> ToolResult:
    """{path: str} -> config, n_params, first 20 state-dict keys; path must live under run_dir"""
    raw, e = _str_arg(args, "path")
    if e:
        return err(e)
    run_dir = Path(sup.cfg.run_dir)
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = run_dir / path
    if not _is_under(path, run_dir):
        return err(f"path must be inside run_dir ({run_dir})")
    if not path.is_file():
        return err(f"path {raw!r} is not a file")
    try:
        import torch  # type: ignore
    except ImportError:
        return err("torch not available")
    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as ex:
        return err(f"torch.load failed: {type(ex).__name__}: {ex}")
    if not isinstance(blob, dict):
        return err(f"checkpoint is a {type(blob).__name__}, expected dict")
    state = blob.get("model")
    n_params = 0
    keys: list[str] = []
    if isinstance(state, dict):
        keys = [str(k) for k in list(state.keys())[:20]]
        for v in state.values():
            numel = getattr(v, "numel", None)
            if callable(numel):
                try:
                    n_params += int(numel())
                except Exception:
                    pass
    return ok(
        path=str(path),
        bytes=path.stat().st_size,
        config=blob.get("config"),
        n_params=n_params,
        n_tensors=len(state) if isinstance(state, dict) else 0,
        keys=keys,
        top_level_keys=[str(k) for k in list(blob.keys())[:20]],
    )


# --------------------------------------------------------------------------- jobs / traces


def read_train_log(sup: Any, args: Args) -> ToolResult:
    """{job_id?: str, tail?: int = 80} -> job dict + last `tail` lines of jobs/<id>/train.log"""
    job_id, e = _str_arg(args, "job_id", required=False, default="")
    if e:
        return err(e)
    job_id = job_id or (sup.state.current_job_id or "")
    if not job_id:
        return err("job_id is required (no current job)")
    tail, e = _int_arg(args, "tail", 80)
    if e:
        return err(e)
    store = sup.jobs.store
    job_dir = Path(store.root) / job_id
    if not _is_under(job_dir, Path(store.root)) or not (job_dir / "job.json").is_file():
        return err(f"unknown job {job_id!r}")
    try:
        job = store.load(job_id).to_dict()
    except Exception as ex:
        return err(f"cannot load job {job_id!r}: {ex}")
    log_path = job_dir / "train.log"
    lines: list[str] = []
    total = 0
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        total = len(all_lines)
        lines = [_clip(x, 400) for x in all_lines[-tail:]]
    return ok(
        job_id=job_id,
        job=job,
        log_path=str(log_path),
        log_exists=log_path.is_file(),
        total_lines=total,
        lines=lines,
    )


def read_trace(sup: Any, args: Args) -> ToolResult:
    """{cycle?: int, n?: int = 50, kind?: "tool"|"llm"|"chain"} -> last n rows of run_dir/trace.jsonl"""
    n, e = _int_arg(args, "n", 50)
    if e:
        return err(e)
    cycle: int | None = None
    if args.get("cycle") is not None:
        cycle, e = _int_arg(args, "cycle", 1, lo=0, hi=10**9)
        if e:
            return err(e)
    kind = args.get("kind")
    if kind is not None:
        if not isinstance(kind, str) or kind not in TRACE_KINDS:
            return err(f"kind must be one of {sorted(TRACE_KINDS)}")
    path = Path(sup.cfg.run_dir) / "trace.jsonl"
    if not path.is_file():
        return ok(rows=[], path=str(path), total=0)
    rows: list[dict[str, Any]] = []
    total = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        total += 1
        if kind is not None and row.get("kind") != kind:
            continue
        if cycle is not None and row.get("cycle") != cycle:
            continue
        rows.append(row)
    rows = rows[-n:]
    for row in rows:
        for key in ("inputs", "outputs"):
            if row.get(key) is not None:
                row[key] = _clip(row[key], TEXT_CLIP)
    return ok(rows=rows, n=len(rows), total=total, path=str(path))


# --------------------------------------------------------------------------- data


def data_stats(sup: Any, args: Args) -> ToolResult:
    """{source: str} -> stats for an already-cached source (never downloads)"""
    from lab.data_cache import cache_file, compose_hf_source, parse_hf_source

    source, e = _str_arg(args, "source")
    if e:
        return err(e)
    src = source.strip()
    path: Path
    if src in {"builtin:tiny", "dummy://tinystories"}:
        from lab.train import BUILTIN_CORPUS

        path = Path(BUILTIN_CORPUS)
        canonical = "builtin:tiny"
    else:
        try:
            canonical = compose_hf_source({"source": src})
            spec, split, n = parse_hf_source(canonical)
        except Exception as ex:
            return err(f"source: {ex}")
        path = cache_file(Path(sup.cfg.data_cache_dir), spec, split, n)
    if not path.is_file():
        return err(f"source {canonical!r} is not cached (use prefetch_data first)", path=str(path))
    try:
        data = path.read_bytes()
    except OSError as ex:
        return err(f"cannot read {path}: {ex}")
    text_head = data[:4096].decode("utf-8", errors="replace")
    return ok(
        source=canonical,
        path=str(path),
        bytes=len(data),
        lines=data.count(b"\n"),
        unique_bytes=len(set(data)),
        sample=text_head[:TEXT_CLIP],
    )


# --------------------------------------------------------------------------- packs


def _pack_entries(sup: Any) -> list[tuple[float, Path]]:
    root = Path(sup.packs.root)
    if not root.is_dir():
        return []
    out: list[tuple[float, Path]] = []
    for p in root.iterdir():
        pj = p / "pack.json"
        if p.is_dir() and pj.is_file():
            try:
                out.append((pj.stat().st_mtime, p))
            except OSError:
                continue
    out.sort(key=lambda t: (t[0], t[1].name), reverse=True)
    return out


def list_packs(sup: Any, args: Args) -> ToolResult:
    """{n?: int = 20} -> newest-first [{pack_hash, hypothesis, trainer, config, mtime}]"""
    n, e = _int_arg(args, "n", 20)
    if e:
        return err(e)
    packs: list[dict[str, Any]] = []
    for mtime, d in _pack_entries(sup)[:n]:
        try:
            raw = json.loads((d / "pack.json").read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        hyp = str(raw.get("hypothesis") or "").strip().split("\n")[0]
        packs.append(
            {
                "pack_hash": d.name,
                "hypothesis": _clip(hyp, 200),
                "trainer": raw.get("trainer"),
                "config": raw.get("config"),
                "mtime": _iso(mtime),
            }
        )
    return ok(packs=packs, n=len(packs), current=sup.state.pack_hash)


def _load_pack(sup: Any, key: str, args: Args) -> tuple[dict[str, Any] | None, str | None]:
    digest, e = _str_arg(args, key)
    if e:
        return None, e
    digest = digest.strip()
    if not re.fullmatch(r"[0-9a-fA-F]{8,64}", digest):
        return None, f"{key} must be a pack hash (hex)"
    if not sup.packs.exists(digest):
        return None, f"unknown pack {digest!r} for {key}"
    try:
        return sup.packs.load(digest).canonical(), None
    except Exception as ex:
        return None, f"cannot load pack {digest!r}: {ex}"


def read_pack(sup: Any, args: Args) -> ToolResult:
    """{pack_hash: str} -> canonical pack"""
    pack, e = _load_pack(sup, "pack_hash", args)
    if e:
        return err(e)
    return ok(pack_hash=str(args["pack_hash"]).strip(), pack=pack)


def _dict_diff(a: Any, b: Any) -> dict[str, list[Any]]:
    a = a if isinstance(a, dict) else {"_": a}
    b = b if isinstance(b, dict) else {"_": b}
    out: dict[str, list[Any]] = {}
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            out[str(key)] = [a.get(key), b.get(key)]
    return out


def diff_packs(sup: Any, args: Args) -> ToolResult:
    """{a: str, b: str} -> {config_diff, data_diff, other_diff} for differing keys"""
    pa, e = _load_pack(sup, "a", args)
    if e:
        return err(e)
    pb, e = _load_pack(sup, "b", args)
    if e:
        return err(e)
    assert pa is not None and pb is not None
    config_diff = _dict_diff(pa.get("config"), pb.get("config"))
    data_diff = _dict_diff(pa.get("data_manifest"), pb.get("data_manifest"))
    other_a = {k: v for k, v in pa.items() if k not in {"config", "data_manifest"}}
    other_b = {k: v for k, v in pb.items() if k not in {"config", "data_manifest"}}
    other_diff = _dict_diff(other_a, other_b)
    return ok(
        a=str(args["a"]).strip(),
        b=str(args["b"]).strip(),
        config_diff=config_diff,
        data_diff=data_diff,
        other_diff=other_diff,
        same=not (config_diff or data_diff or other_diff),
    )


# --------------------------------------------------------------------------- jobs


def list_jobs(sup: Any, args: Args) -> ToolResult:
    """{n?: int = 20} -> newest-first [{id, status, backend, val_ppl, lr, steps, started_at, ended_at}]"""
    n, e = _int_arg(args, "n", 20)
    if e:
        return err(e)
    root = Path(sup.jobs.store.root)
    if not root.is_dir():
        return ok(jobs=[], n=0)
    dirs = sorted((p for p in root.glob("job-*") if (p / "job.json").is_file()), key=lambda p: p.name, reverse=True)
    jobs: list[dict[str, Any]] = []
    for d in dirs[:n]:
        try:
            raw = json.loads((d / "job.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        metrics = raw.get("metrics") or {}
        if not isinstance(metrics, dict):
            metrics = {}
        jobs.append(
            {
                "id": raw.get("id", d.name),
                "status": raw.get("status"),
                "backend": raw.get("backend"),
                "val_ppl": metrics.get("val_ppl"),
                "lr": metrics.get("lr"),
                "steps": metrics.get("steps"),
                "started_at": raw.get("started_at"),
                "ended_at": raw.get("ended_at"),
                "error": _clip(raw.get("error"), 200) if raw.get("error") else None,
            }
        )
    return ok(jobs=jobs, n=len(jobs), current=sup.state.current_job_id)


# --------------------------------------------------------------------------- skills


def read_skill(sup: Any, args: Args) -> ToolResult:
    """{name: str} -> skill text from lab.ensemble.skills.load_skill"""
    name, e = _str_arg(args, "name")
    if e:
        return err(e)
    mod, e = _skills_module()
    if e:
        return err(e)
    try:
        content = mod.load_skill(name)
    except Exception as ex:
        return err(f"unknown skill {name!r}: {ex}", skills=_skill_names(mod))
    if not isinstance(content, str):
        content = _clip(content, 32_000)
    return ok(name=name, content=content[:32_000])


def list_skills(sup: Any, args: Args) -> ToolResult:
    """{} -> skill names from lab.ensemble.skills.list_skills"""
    mod, e = _skills_module()
    if e:
        return err(e)
    try:
        skills = [str(s) for s in mod.list_skills()]
    except Exception as ex:
        return err(f"list_skills failed: {ex}")
    return ok(skills=skills[:MAX_LIST])


# --------------------------------------------------------------------------- episodes / sandbox


def read_episode_metrics(sup: Any, args: Args) -> ToolResult:
    """{n?: int = 20} -> [{id, cycle, confirm_ppl, job_status}] + best (min confirm_ppl)"""
    n, e = _int_arg(args, "n", 20)
    if e:
        return err(e)
    try:
        rows = sup.episodes.summaries(n=n)
    except Exception as ex:
        return err(f"episodes unavailable: {ex}")
    table: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for r in rows:
        ppl = r.get("confirm_ppl")
        try:
            ppl_f: float | None = float(ppl) if ppl is not None else None
        except (TypeError, ValueError):
            ppl_f = None
        table.append(
            {
                "id": r.get("id"),
                "cycle": r.get("cycle"),
                "confirm_ppl": ppl_f,
                "job_status": r.get("job_status"),
            }
        )
        if ppl_f is not None and (best is None or ppl_f < best["confirm_ppl"]):
            best = {"id": r.get("id"), "confirm_ppl": ppl_f}
    return ok(episodes=table, n=len(table), best=best)


def sandbox_usage(sup: Any, args: Args) -> ToolResult:
    """{} -> {files, bytes, cap_bytes}"""
    from lab.sandbox import dir_size

    root = Path(sup.sandbox.root)
    try:
        files = sum(1 for p in root.rglob("*") if p.is_file()) if root.exists() else 0
        size = dir_size(root)
    except OSError as ex:
        return err(f"cannot stat sandbox: {ex}")
    cap = int(sup.cfg.sandbox_max_bytes)
    return ok(
        files=files,
        bytes=size,
        cap_bytes=cap,
        free_bytes=max(0, cap - size),
        root=str(root),
    )


EXTRA_REGISTRY: dict[str, Handler] = {
    "run_python": run_python,
    "write_and_run": write_and_run,
    "grep_files": grep_files,
    "list_models": list_models,
    "inspect_model": inspect_model,
    "read_checkpoint_meta": read_checkpoint_meta,
    "read_train_log": read_train_log,
    "read_trace": read_trace,
    "data_stats": data_stats,
    "list_packs": list_packs,
    "read_pack": read_pack,
    "diff_packs": diff_packs,
    "list_jobs": list_jobs,
    "read_skill": read_skill,
    "list_skills": list_skills,
    "read_episode_metrics": read_episode_metrics,
    "sandbox_usage": sandbox_usage,
}
