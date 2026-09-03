"""Fill LAB_DATA_CACHE for DEFAULT_MIX. Reads HF token from the hub cache file."""

from __future__ import annotations

import os
import sys
from pathlib import Path

TOKEN_PATH = Path.home() / ".cache/huggingface/token"
if TOKEN_PATH.is_file():
    os.environ["HF_TOKEN"] = TOKEN_PATH.read_text(encoding="utf-8").strip()
os.environ.setdefault("HF_HOME", str(Path.home() / "data" / "hf"))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

sys.path.insert(0, str(Path.home() / "Projects" / "lab"))

from lab.data_cache import DEFAULT_MIX, cache_file, ensure_hf_text, list_cache, parse_hf_source  # noqa: E402


def main() -> int:
    cache = Path.home() / "data" / "lab"
    cache.mkdir(parents=True, exist_ok=True)
    print("cache", cache, flush=True)
    for src in DEFAULT_MIX:
        spec, split, n = parse_hf_source(src)
        dest = cache_file(cache, spec, split, n)
        if dest.is_file() and dest.stat().st_size >= 64:
            print("hit", src, dest.stat().st_size, flush=True)
            continue
        print("prefetch", src, flush=True)
        path = ensure_hf_text(src, cache, allow_network=True)
        print(" ok", path, path.stat().st_size, flush=True)
    print("done", list_cache(cache), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
