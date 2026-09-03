"""Allowlisted Hugging Face text cache. Train stays offline on files."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import os
import re


@dataclass(frozen=True)
class SourceSpec:
    repo: str
    config: str | None
    default_split: str
    text_keys: tuple[str, ...] = ("text", "content", "article")
    max_n: int = 50_000


# Keys are hf:<repo> as used in packs / prefetch_data.
ALLOWLIST: dict[str, SourceSpec] = {
    "hf:roneneldan/TinyStories": SourceSpec("roneneldan/TinyStories", None, "train"),
    "hf:wikimedia/wikipedia": SourceSpec("wikimedia/wikipedia", "20231101.en", "train"),
    "hf:HuggingFaceFW/fineweb-edu": SourceSpec("HuggingFaceFW/fineweb-edu", None, "train"),
}

DEFAULT_MIX: tuple[str, ...] = (
    "hf:roneneldan/TinyStories:train:10000",
)

# First shard to materialize n rows without resolving the whole dataset.
_SHARDS: dict[str, tuple[str, str]] = {
    "roneneldan/TinyStories": ("TinyStories-valid.txt", "txt"),
    "wikimedia/wikipedia": ("20231101.en/train-00000-of-00041.parquet", "parquet"),
    "HuggingFaceFW/fineweb-edu": ("data/CC-MAIN-2013-20/train-00000-of-00014.parquet", "parquet"),
}

_HF_SRC = re.compile(
    r"^hf:([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?::([A-Za-z0-9_.-]+))?(?::(\d+))?$"
)

PKG_FROZEN_EVAL = Path(__file__).resolve().parent / "data" / "frozen_eval"


def default_cache_dir() -> Path:
    raw = os.environ.get("LAB_DATA_CACHE")
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / "data" / "lab").resolve()


def parse_hf_source(src: str) -> tuple[SourceSpec, str, int]:
    raw = src.strip()
    if "frozen_eval" in raw:
        raise ValueError("frozen_eval is not a train source")
    m = _HF_SRC.match(raw)
    if not m:
        raise ValueError(f"unrecognized source {src!r}")
    key = f"hf:{m.group(1)}"
    spec = ALLOWLIST.get(key)
    if spec is None:
        raise ValueError(f"source not on allowlist: {key}")
    split = m.group(2) or spec.default_split
    n = int(m.group(3) or 10_000)
    n = max(1, min(n, spec.max_n))
    return spec, split, n


def compose_hf_source(args: dict[str, Any]) -> str:
    source = str(args.get("source") or "").strip()
    if not source:
        raise ValueError("source is required")
    if not source.startswith("hf:"):
        source = f"hf:{source}"
    spec, split, n = parse_hf_source(source)
    if args.get("split"):
        split = str(args["split"])
    if args.get("n") is not None:
        n = max(1, min(int(args["n"]), spec.max_n))
    return f"hf:{spec.repo}:{split}:{n}"


def cache_file(cache_dir: Path, spec: SourceSpec, split: str, n: int) -> Path:
    blob = f"{spec.repo}|{spec.config or ''}|{split}|{n}".encode()
    digest = sha256(blob).hexdigest()[:16]
    safe = spec.repo.replace("/", "_")
    return cache_dir / f"{safe}__{split}__{n}__{digest}.txt"


def seed_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def seed_default_mix(cache_dir: Path, text: str | None = None) -> list[Path]:
    """Write stub files for DEFAULT_MIX so tests never hit the Hub."""
    body = text or ("The cat sat on the mat.\nA farmer planted wheat by the river.\n" * 40)
    out: list[Path] = []
    for src in DEFAULT_MIX:
        spec, split, n = parse_hf_source(src)
        out.append(seed_file(cache_file(cache_dir, spec, split, n), body))
    return out


def list_cache(cache_dir: Path) -> dict[str, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in cache_dir.glob("*.txt") if p.is_file())
    return {
        "allowlist": sorted(ALLOWLIST),
        "default_mix": list(DEFAULT_MIX),
        "cache_dir": str(cache_dir),
        "files": [{"path": str(p), "bytes": p.stat().st_size} for p in files],
        "bytes": sum(p.stat().st_size for p in files),
    }


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except (ValueError, OSError):
        return False


def assert_not_frozen(path: Path, extra_roots: list[Path] | None = None) -> None:
    roots = [PKG_FROZEN_EVAL, *(extra_roots or [])]
    resolved = path.expanduser().resolve()
    for root in roots:
        if root.is_dir() and _is_under(resolved, root):
            raise ValueError("frozen_eval is not a train source")


def ensure_hf_text(
    src: str,
    cache_dir: Path,
    *,
    allow_network: bool = True,
) -> Path:
    spec, split, n = parse_hf_source(src)
    dest = cache_file(cache_dir, spec, split, n)
    if dest.is_file() and dest.stat().st_size >= 64:
        return dest
    if not allow_network:
        raise FileNotFoundError(f"cache miss for {src} (network disabled): {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = _download_hf(spec, split, n)
    if len(text.strip()) < 64:
        raise RuntimeError(f"downloaded too little text from {src}")
    dest.write_text(text, encoding="utf-8")
    return dest


def _hf_token() -> str | None:
    raw = os.environ.get("HF_TOKEN")
    if raw:
        return raw.strip()
    path = Path.home() / ".cache" / "huggingface" / "token"
    if path.is_file():
        return path.read_text(encoding="utf-8").strip() or None
    return None


def _row_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        val = row.get(key)
        if val:
            return " ".join(str(val).split())
    return ""


def _download_hf(spec: SourceSpec, split: str, n: int) -> str:
    del split  # shard map picks the split file
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise RuntimeError("prefetch_data needs huggingface_hub (pip install 'lab[data]')") from e
    shard, kind = _SHARDS.get(spec.repo, ("", ""))
    if not shard:
        raise RuntimeError(f"no shard mapping for {spec.repo}")
    print(f"download {spec.repo} {shard}", flush=True)
    path = Path(
        hf_hub_download(
            repo_id=spec.repo,
            repo_type="dataset",
            filename=shard,
            token=_hf_token(),
        )
    )
    if kind == "txt":
        raw = path.read_text(encoding="utf-8", errors="replace")
        chunks = [c.strip() for c in raw.split("<|endoftext|>") if c.strip()]
        if len(chunks) < n:
            chunks = [c.strip() for c in raw.split("\n\n") if len(c.strip()) > 40]
        return "\n".join(chunks[:n]) + "\n"
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError as e:
        raise RuntimeError("parquet extract needs pyarrow: pip install 'lab[data]'") from e
    pf = pq.ParquetFile(path)
    cols = [k for k in spec.text_keys if k in pf.schema_arrow.names]
    if not cols:
        cols = [pf.schema_arrow.names[0]]
    lines: list[str] = []
    for batch in pf.iter_batches(batch_size=256, columns=cols):
        for row in batch.to_pylist():
            chunk = _row_text(row, spec.text_keys)
            if not chunk:
                continue
            lines.append(chunk)
            if len(lines) % 500 == 0:
                print(f"  {spec.repo} {len(lines)}/{n}", flush=True)
            if len(lines) >= n:
                return "\n".join(lines) + "\n"
    return "\n".join(lines) + "\n"


def materialize_sources(
    sources: list[str],
    cache_dir: Path,
    job_dir: Path,
    *,
    allow_network: bool = True,
    frozen_eval_dir: Path | None = None,
) -> Path:
    """Write job_dir/data.txt from allowlisted / builtin / relative sources."""
    chunks: list[str] = []
    from lab.train.loop import BUILTIN_CORPUS

    extra_frozen = [frozen_eval_dir] if frozen_eval_dir is not None else []
    for src in sources or ["builtin:tiny"]:
        if "frozen_eval" in src:
            raise ValueError("frozen_eval is not a train source")
        if src in {"builtin:tiny", "dummy://tinystories"}:
            chunks.append(BUILTIN_CORPUS.read_text(encoding="utf-8"))
            continue
        if src.startswith("hf:"):
            try:
                path = ensure_hf_text(src, cache_dir, allow_network=allow_network)
            except FileNotFoundError:
                if allow_network:
                    raise
                print(f"[data] skip uncached {src}", flush=True)
                continue
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
            continue
        path = Path(src)
        if not path.is_absolute():
            path = job_dir / src
        assert_not_frozen(path, extra_frozen)
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(chunks).strip() + "\n"
    if len(text) < 64:
        raise RuntimeError("materialized train text is too short")
    dest = job_dir / "data.txt"
    dest.write_text(text, encoding="utf-8")
    return dest
