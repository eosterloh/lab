"""Tiny causal LM loop. Writes metrics.json + checkpoint.pt into --job-dir."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


BUILTIN_CORPUS = Path(__file__).parent / "corpus.txt"


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as e:
        raise ImportError("lab trainer needs torch: pip install torch") from e
    return torch, nn, F


def pick_device(torch: Any) -> Any:
    raw = os.environ.get("LAB_TRAIN_DEVICE", "cpu").strip().lower()
    if raw in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(nn: Any, vocab: int, hidden: int, layers: int, heads: int, seq: int) -> Any:
    class TinyGPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(vocab, hidden)
            self.pos = nn.Embedding(seq, hidden)
            enc = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=heads,
                dim_feedforward=max(hidden * 4, 32),
                dropout=0.0,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(enc, num_layers=layers)
            self.ln = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)
            self.seq = seq

        def forward(self, idx: Any) -> Any:
            import torch as _torch

            _b, t = idx.shape
            pos = _torch.arange(t, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None, :, :]
            mask = _torch.triu(_torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
            x = self.blocks(x, mask=mask)
            return self.head(self.ln(x))

    return TinyGPT()


def load_checkpoint(path: Path, device: Any | None = None) -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    torch, nn, _F = _require_torch()
    if device is None:
        device = pick_device(torch)
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(blob.get("config") or {})
    hidden = int(cfg.get("hidden", 32))
    layers = int(cfg.get("layers", 1))
    heads = int(cfg.get("heads", 1))
    seq = int(cfg.get("seq_len", 32))
    vocab = int(cfg.get("vocab", 256))
    model = build_model(nn, vocab, hidden, layers, heads, seq).to(device)
    model.load_state_dict(blob["model"])
    arch = {
        "hidden": hidden,
        "layers": layers,
        "heads": heads,
        "seq_len": seq,
        "vocab": vocab,
    }
    return model, arch, device, blob


def same_arch(cfg: dict[str, Any], hidden: int, layers: int, heads: int, seq: int, vocab: int) -> bool:
    return (
        int(cfg.get("hidden", hidden)) == hidden
        and int(cfg.get("layers", layers)) == layers
        and int(cfg.get("heads", heads)) == heads
        and int(cfg.get("seq_len", seq)) == seq
        and int(cfg.get("vocab", vocab)) == vocab
    )


def load_text(pack: dict[str, Any], job_dir: Path) -> str:
    manifest = pack.get("data_manifest") or {}
    sources = list(manifest.get("sources") or ["builtin:tiny"])
    wants_hf = any(str(s).startswith("hf:") for s in sources)
    data_file = job_dir / "data.txt"
    if data_file.is_file():
        text = data_file.read_text(encoding="utf-8", errors="replace").strip()
        if len(text) >= 64:
            return text
        if wants_hf:
            raise RuntimeError(f"{data_file} is missing or too short; JobManager must materialize HF sources")
    if wants_hf:
        raise RuntimeError(f"{data_file} is missing; JobManager must materialize HF sources")
    chunks: list[str] = []
    for src in sources:
        if src in {"builtin:tiny", "dummy://tinystories"}:
            chunks.append(BUILTIN_CORPUS.read_text(encoding="utf-8"))
            continue
        path = Path(src)
        if not path.is_absolute():
            path = job_dir / src
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    text = "\n".join(chunks).strip()
    if len(text) < 64:
        text = BUILTIN_CORPUS.read_text(encoding="utf-8")
    return text


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))


def make_batch(ids: list[int], seq: int, batch: int, torch: Any) -> tuple[Any, Any]:
    n = max(len(ids) - seq - 1, 1)
    xs = []
    ys = []
    for _ in range(batch):
        i = int(torch.randint(0, n, (1,)).item())
        span = ids[i : i + seq + 1]
        if len(span) < seq + 1:
            span = (span + [0] * (seq + 1))[: seq + 1]
        xs.append(span[:-1])
        ys.append(span[1:])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def train(job_dir: Path) -> dict[str, Any]:
    torch, nn, F = _require_torch()
    pack = json.loads((job_dir / "pack.json").read_text(encoding="utf-8"))
    cfg = pack.get("config") or {}
    budgets = pack.get("budgets") or {}
    hidden = int(cfg.get("hidden", 32))
    layers = int(cfg.get("layers", 1))
    heads = int(cfg.get("heads", 1))
    seq = int(cfg.get("seq_len", 32))
    batch = int(cfg.get("batch", 8))
    lr = float(cfg.get("lr", 3e-3))
    steps = int(cfg.get("steps", budgets.get("max_steps") or 20))
    vocab = 256

    text = load_text(pack, job_dir)
    ids = encode_bytes(text)
    device = pick_device(torch)
    model = build_model(nn, vocab, hidden, layers, heads, seq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    resumed = False
    parent_path = job_dir / "parent.pt"
    if parent_path.is_file():
        try:
            _m, arch, _d, blob = load_checkpoint(parent_path, device=device)
            if same_arch(arch, hidden, layers, heads, seq, vocab):
                model.load_state_dict(blob["model"])
                if blob.get("optim"):
                    try:
                        opt.load_state_dict(blob["optim"])
                    except Exception:
                        pass
                resumed = True
                print(f"resumed parent={parent_path}", flush=True)
            else:
                print("parent architecture mismatch; training from scratch", flush=True)
        except Exception as e:
            print(f"parent load failed ({e}); training from scratch", flush=True)
    model.train()
    last = 0.0
    tokens = 0
    for step in range(1, steps + 1):
        x, y = make_batch(ids, seq, batch, torch)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        tokens += int(x.numel())
        if step == 1 or step == steps or step % 10 == 0:
            print(f"step={step} train_loss={last:.4f}", flush=True)

    model.eval()
    with torch.inference_mode():
        x, y = make_batch(ids, seq, batch, torch)
        x, y = x.to(device), y.to(device)
        val = float(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())

    ckpt = job_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optim": opt.state_dict(),
            "config": {
                "hidden": hidden,
                "layers": layers,
                "heads": heads,
                "seq_len": seq,
                "vocab": vocab,
            },
        },
        ckpt,
    )
    metrics = {
        "backend": "lab",
        "train_loss": round(last, 4),
        "val_loss": round(val, 4),
        "val_ppl": round(math.exp(min(val, 20.0)), 4),
        "steps": steps,
        "lr": lr,
        "tokens_seen": tokens,
        "hidden": hidden,
        "layers": layers,
        "device": str(device),
        "checkpoint": str(ckpt),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "resumed": resumed,
        "parent": str(parent_path) if resumed else None,
        "train_chars": len(text),
    }
    (job_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics), flush=True)
    return metrics


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lab.train")
    p.add_argument("--job-dir", type=Path, required=True)
    args = p.parse_args(argv)
    job_dir = args.job_dir.expanduser().resolve()
    if not (job_dir / "pack.json").is_file():
        raise SystemExit(f"missing pack.json in {job_dir}")
    try:
        train(job_dir)
    except ImportError as e:
        raise SystemExit(str(e)) from e
    return 0
