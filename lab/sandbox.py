from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
from typing import Any

from lab.config import LabConfig
from lab.http import Transport, UrllibTransport, assert_http_url, duckduckgo_search_url, parse_duckduckgo


FORBIDDEN_PREFIXES = ("/", "~")
FORBIDDEN_COMMANDS = frozenset({"rm", "sudo", "chmod", "chown", "mkfs"})
PASSTHROUGH_ENV = ("LAB_TRAIN_DEVICE", "CUDA_VISIBLE_DEVICES")
REPO_ROOT = Path(__file__).resolve().parents[1]


def _decode(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


class Sandbox:
    def __init__(self, cfg: LabConfig, transport: Transport | None = None) -> None:
        self.cfg = cfg
        self.root = cfg.sandbox_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.transport = transport or UrllibTransport()

    def resolve(self, rel: str) -> Path:
        if not rel or rel.startswith(FORBIDDEN_PREFIXES) or ".." in Path(rel).parts:
            raise ValueError("path must be a relative sandbox path")
        path = (self.root / rel).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("path escapes sandbox")
        return path

    def list_files(self, rel: str = ".") -> list[str]:
        path = self.resolve(rel) if rel not in {".", ""} else self.root
        if not path.exists():
            return []
        out: list[str] = []
        for p in sorted(path.rglob("*")):
            if p.is_file():
                out.append(str(p.relative_to(self.root)))
        return out

    def read_file(self, rel: str, max_bytes: int = 32_000) -> str:
        path = self.resolve(rel)
        if not path.is_file():
            raise FileNotFoundError(rel)
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")

    def write_file(self, rel: str, content: str) -> str:
        encoded = content.encode("utf-8")
        projected = dir_size(self.root) + len(encoded)
        path = self.root / rel
        if path.is_file():
            projected -= path.stat().st_size
        if projected > self.cfg.sandbox_max_bytes:
            raise ValueError("sandbox disk cap exceeded")
        dest = self.resolve(rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="utf-8")
        return str(dest.relative_to(self.root))

    def _timeout(self, timeout_s: float | None) -> float:
        cap = float(os.environ.get("LAB_EXEC_MAX_TIMEOUT_S", "300"))
        if timeout_s is None:
            return min(float(self.cfg.exec_timeout_s), cap)
        return max(0.01, min(float(timeout_s), cap))

    def _env(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        # Device policy must reach sandboxed training scripts.
        for key in PASSTHROUGH_ENV:
            val = os.environ.get(key)
            if val:
                env[key] = val
        if extra:
            env.update(extra)
        return env

    def _run(self, argv: list[str], env: dict[str, str], timeout_s: float) -> dict[str, Any]:
        try:
            proc = subprocess.run(
                argv,
                cwd=self.root,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "returncode": -1,
                "stdout": _decode(e.stdout)[:32_000],
                "stderr": f"timeout after {timeout_s:g}s",
                "timed_out": True,
            }
        return {
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[:32_000],
            "stderr": (proc.stderr or "")[:32_000],
        }

    def exec(self, argv: list[str], timeout_s: float | None = None) -> dict[str, Any]:
        if not argv or not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError("argv must be a list of strings")
        if argv[0] in FORBIDDEN_COMMANDS:
            raise ValueError(f"command not allowed: {argv[0]}")
        return self._run(argv, self._env(), self._timeout(timeout_s))

    def run_python(
        self,
        rel_path: str,
        args: list[str] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        path = self.resolve(rel_path)
        if not path.is_file():
            raise FileNotFoundError(rel_path)
        args = list(args or [])
        if not all(isinstance(x, str) for x in args):
            raise ValueError("args must be a list of strings")
        env = self._env(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(REPO_ROOT),
            }
        )
        out = self._run([sys.executable, str(path), *args], env, self._timeout(timeout_s))
        out["path"] = str(path.relative_to(self.root))
        return out

    def fetch(self, url: str, dest: str | None = None) -> dict[str, Any]:
        if not self.cfg.allow_network:
            raise ValueError("network disabled")
        assert_http_url(url)
        resp = self.transport.get(url, timeout=self.cfg.fetch_timeout_s)
        body = resp.body[: self.cfg.fetch_max_bytes]
        if dest is None:
            name = url.split("/")[-1] or "fetch.bin"
            dest = f"fetch/{name}"
        self.write_file(dest, body.decode("utf-8", errors="replace"))
        return {
            "status": resp.status,
            "bytes": len(body),
            "path": dest,
            "content_type": resp.content_type,
        }

    def search(self, query: str) -> list[dict[str, str]]:
        if not self.cfg.allow_network:
            raise ValueError("network disabled")
        if not query.strip():
            raise ValueError("query is required")
        url = duckduckgo_search_url(query.strip())
        resp = self.transport.get(url, timeout=self.cfg.search_timeout_s)
        if resp.status != 200:
            raise ValueError(f"search http {resp.status}")
        return parse_duckduckgo(resp.body)
