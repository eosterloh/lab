"""Tiny causal LM loop. Writes metrics.json + checkpoint.pt into --job-dir."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from lab.pack import parent_plan, normalize_stage


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


# Printed during a long run. Either the step ceiling or the clock can end the job.
EVAL_EVERY = 200
# A pack may not ask for more steps than this, even if the clock is still open.
SAFETY_STEPS = 1_000_000


def step_ceiling(cfg: dict[str, Any], budgets: dict[str, Any]) -> int:
    """How many steps this job may run. The wall clock is a second stop."""
    ceiling = int(budgets.get("max_steps") or cfg.get("steps") or SAFETY_STEPS)
    if cfg.get("steps") is not None:
        ceiling = min(ceiling, int(cfg["steps"]))
    return max(min(ceiling, SAFETY_STEPS), 1)


def split_ids(ids: list[int], seq: int) -> tuple[list[int], list[int]]:
    """Last tenth of the bytes is the trainer's own val split."""
    tail = max(seq + 1, len(ids) // 10)
    if len(ids) - tail < seq + 1:
        return ids, ids
    return ids[:-tail], ids[-tail:]


def mean_loss(model: Any, ids: list[int], seq: int, batch: int, torch: Any, F: Any, vocab: int, device: Any) -> float:
    model.eval()
    total = 0.0
    n = 4
    with torch.inference_mode():
        for _ in range(n):
            x, y = make_batch(ids, seq, batch, torch)
            x, y = x.to(device), y.to(device)
            total += float(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
    model.train()
    return total / n


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
    steps = step_ceiling(cfg, budgets)
    hours = float(budgets.get("max_hours") or 84.0)
    vocab = 256
    stage = normalize_stage(pack.get("stage") or cfg.get("stage"))
    plan = parent_plan(stage, parent_exists=(job_dir / "parent.pt").is_file())

    text = load_text(pack, job_dir)
    ids = encode_bytes(text)
    train_ids, val_ids = split_ids(ids, seq)
    device = pick_device(torch)
    model = build_model(nn, vocab, hidden, layers, heads, seq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    resumed = False
    optim_resumed = False
    parent_path = job_dir / "parent.pt"
    if plan["required"] and not parent_path.is_file():
        raise RuntimeError(f"{stage} requires parent.pt; there is no checkpoint to start from")
    if plan["load_weights"] and parent_path.is_file():
        try:
            _m, arch, _d, blob = load_checkpoint(parent_path, device=device)
            if not same_arch(arch, hidden, layers, heads, seq, vocab):
                if stage == "continue":
                    print("parent architecture mismatch; training from scratch", flush=True)
                else:
                    raise RuntimeError(
                        f"{stage} parent architecture does not match this pack; refusing to start from scratch"
                    )
            else:
                model.load_state_dict(blob["model"])
                resumed = True
                if plan["load_optim"] and blob.get("optim"):
                    try:
                        opt.load_state_dict(blob["optim"])
                        optim_resumed = True
                    except Exception:
                        pass
                print(f"resumed parent={parent_path} stage={stage} optim={optim_resumed}", flush=True)
        except RuntimeError:
            raise
        except Exception as e:
            if stage != "continue":
                raise RuntimeError(f"{stage} parent load failed: {e}") from e
            print(f"parent load failed ({e}); training from scratch", flush=True)
    elif stage == "pretrain" and parent_path.is_file():
        print("pretrain ignores parent.pt", flush=True)
    model.train()
    last = 0.0
    tokens = 0
    ran = 0
    stop_reason = "max_steps"
    deadline = time.time() + hours * 3600.0
    for step in range(1, steps + 1):
        if time.time() >= deadline:
            stop_reason = "max_hours"
            break
        x, y = make_batch(train_ids, seq, batch, torch)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        tokens += int(x.numel())
        ran = step
        if step == 1 or step == steps or step % 50 == 0:
            print(f"step={step} train_loss={last:.4f}", flush=True)
        if step % EVAL_EVERY == 0:
            val_now = mean_loss(model, val_ids, seq, batch, torch, F, vocab, device)
            print(f"step={step} val_loss={val_now:.4f}", flush=True)

    val = mean_loss(model, val_ids, seq, batch, torch, F, vocab, device)
    print(f"stop={stop_reason} steps={ran}/{steps}", flush=True)

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
                "stage": stage,
            },
        },
        ckpt,
    )
    metrics = {
        "backend": "lab",
        "train_loss": round(last, 4),
        "val_loss": round(val, 4),
        "val_ppl": round(math.exp(min(val, 20.0)), 4),
        "steps": ran,
        "max_steps": steps,
        "stop_reason": stop_reason,
        "lr": lr,
        "tokens_seen": tokens,
        "hidden": hidden,
        "layers": layers,
        "device": str(device),
        "checkpoint": str(ckpt),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "resumed": resumed,
        "optim_resumed": optim_resumed,
        "stage": stage,
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
