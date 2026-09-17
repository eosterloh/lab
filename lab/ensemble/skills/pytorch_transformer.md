# pytorch_transformer
Write byte-level TinyGPT scripts whose checkpoint.pt and metrics.json match lab/train/loop.py exactly.

## Reference trainer (copy, then edit only argparse defaults)

```python
"""TinyGPT byte-level trainer. Same checkpoint/metrics layout as lab.train."""
import argparse, json, math, os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device():
    raw = os.environ.get("LAB_TRAIN_DEVICE", "cpu").strip().lower()
    if raw in {"cuda", "gpu"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(vocab, hidden, layers, heads, seq):
    # Identical to lab/train/loop.py:build_model -> identical state_dict keys.
    class TinyGPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok = nn.Embedding(vocab, hidden)
            self.pos = nn.Embedding(seq, hidden)
            enc = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=max(hidden * 4, 32),
                dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
            )
            self.blocks = nn.TransformerEncoder(enc, num_layers=layers)
            self.ln = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)
            self.seq = seq

        def forward(self, idx):
            _b, t = idx.shape
            pos = torch.arange(t, device=idx.device)
            x = self.tok(idx) + self.pos(pos)[None, :, :]
            mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), diagonal=1)
            x = self.blocks(x, mask=mask)
            return self.head(self.ln(x))

    return TinyGPT()


def make_batch(ids, seq, batch):
    n = max(len(ids) - seq - 1, 1)
    xs, ys = [], []
    for _ in range(batch):
        i = int(torch.randint(0, n, (1,)).item())
        span = ids[i : i + seq + 1]
        if len(span) < seq + 1:
            span = (span + [0] * (seq + 1))[: seq + 1]
        xs.append(span[:-1])
        ys.append(span[1:])
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.txt")
    ap.add_argument("--out", default=".")
    ap.add_argument("--parent", default="parent.pt")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    vocab = 256
    text = Path(a.data).read_text(encoding="utf-8", errors="replace")
    ids = list(text.encode("utf-8", errors="replace"))
    device = pick_device()
    model = build_model(vocab, a.hidden, a.layers, a.heads, a.seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    want = {"hidden": a.hidden, "layers": a.layers, "heads": a.heads, "seq_len": a.seq_len, "vocab": vocab}
    resumed = False
    parent = Path(a.parent)
    if parent.is_file():
        blob = torch.load(parent, map_location=device, weights_only=False)
        cfg = dict(blob.get("config") or {})
        if all(int(cfg.get(k, -1)) == v for k, v in want.items()):
            model.load_state_dict(blob["model"])
            resumed = True
        else:
            print("parent architecture mismatch; training from scratch", flush=True)
    model.train()
    last, tokens = 0.0, 0
    for step in range(1, a.steps + 1):
        x, y = make_batch(ids, a.seq_len, a.batch)
        x, y = x.to(device), y.to(device)
        logits = model(x)                                   # [batch, seq, vocab]
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.item())
        tokens += int(x.numel())
        if step == 1 or step == a.steps or step % 10 == 0:
            print(f"step={step} train_loss={last:.4f}", flush=True)
    model.eval()
    with torch.inference_mode():
        x, y = make_batch(ids, a.seq_len, a.batch)
        x, y = x.to(device), y.to(device)
        val = float(F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "config": want}, ckpt)
    metrics = {
        "backend": "lab", "train_loss": round(last, 4), "val_loss": round(val, 4),
        "val_ppl": round(math.exp(min(val, 20.0)), 4), "steps": a.steps, "lr": a.lr,
        "tokens_seen": tokens, "hidden": a.hidden, "layers": a.layers, "device": str(device),
        "checkpoint": str(ckpt), "n_params": int(sum(p.numel() for p in model.parameters())),
        "resumed": resumed, "parent": str(parent) if resumed else None, "train_chars": len(text),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
```

## Checkpoint contract (must match byte for byte)
- File: `checkpoint.pt` = `{"model": state_dict, "optim": AdamW state, "config": {...}}`.
- `config` keys: `hidden`, `layers`, `heads`, `seq_len`, `vocab` (always 256).
- state_dict keys start with `tok.`, `pos.`, `blocks.layers.N.`, `ln.`, `head.`. Renaming any module breaks `load_checkpoint`.
- `hidden % heads == 0` or `nn.TransformerEncoderLayer` raises.

## metrics.json keys
`train_loss`, `val_loss`, `val_ppl`, `steps`, `lr`, `tokens_seen`, `n_params`, plus `backend`, `hidden`, `layers`, `device`, `checkpoint`, `resumed`, `parent`, `train_chars`. `val_ppl = exp(min(val_loss, 20))`.

## Causal mask
`torch.triu(full((t, t), -inf), diagonal=1)` puts -inf strictly above the diagonal: position i sees j <= i. Build it from the actual `t`, not `seq_len`. Wrong sign or `diagonal=0` masks the token itself and loss stays near `ln(256) = 5.55`.

## norm_first
`norm_first=True` = pre-LN (`x + attn(ln(x))`). The final `self.ln` is then required before `head`. Do not add a second LayerNorm inside the block; the state_dict keys would not match.

## batch_first=True
Inputs are `[batch, seq, hidden]`. Without it the encoder expects `[seq, batch, hidden]` and the embedding sum silently trains on transposed data.

## Resume from parent.pt
Load weights only if `config` matches on all five keys (the trainer's `same_arch`). On mismatch train from scratch and print it. `optim` may be loaded too; wrap in try/except. Never resize embeddings to "make it fit".

## Count parameters
`sum(p.numel() for p in model.parameters())`. Rough size: `2*vocab*hidden + seq*hidden + layers*(12*hidden**2)`.

## Perplexity on held-out text
```python
model.eval(); ids = list(text.encode("utf-8")); total = n = 0
with torch.inference_mode():
    for i in range(1, len(ids)):
        ctx = torch.tensor([ids[max(0, i - seq):i]], device=device)
        logp = F.log_softmax(model(ctx)[0, -1].float(), -1)
        total += float(-logp[ids[i]]); n += 1
print({"nll": total / n, "ppl": math.exp(total / n), "tokens": n})
```
This is what `lab/scorer.py:LabScorer` does for `confirm_ppl`. Sliding one byte at a time is slow; keep held-out text short (a few KB).

## Common mistakes
- `F.cross_entropy(logits, y)` on 3-D logits. Use `logits.reshape(-1, vocab)` and `y.reshape(-1)`.
- Forgetting `model.eval()` and `torch.inference_mode()` before scoring (dropout is 0 here, but do it anyway).
- Feeding `t > seq_len` bytes: `self.pos` has only `seq_len` rows -> index error. Slice the context to the last `seq_len` bytes.
- Tensors on different devices: `.to(device)` every batch after `pick_device()`.
- `torch.load(..., weights_only=True)` fails on the `optim` blob; use `weights_only=False`.
- Loss not decreasing: lr too high (> 1e-2) or mask wrong. Sanity target: loss < 3.0 after ~100 steps on TinyStories bytes.

## Probe script template
```python
import json, sys, torch
from lab.train.loop import load_checkpoint
model, arch, device, blob = load_checkpoint(sys.argv[1] if len(sys.argv) > 1 else "checkpoint.pt")
n_params = sum(p.numel() for p in model.parameters())
ids = list(b"Once upon a time")
with torch.inference_mode():
    for _ in range(32):
        ctx = torch.tensor([ids[-arch["seq_len"]:]], device=device)
        ids.append(int(model(ctx)[0, -1].argmax()))
print(json.dumps({"config": arch, "n_params": n_params, "sample": bytes(ids).decode("utf-8", "replace")}))
```
`from lab.train.loop import ...` works under `run_python`/`write_and_run` (they set `PYTHONPATH` to the repo) and inside train jobs; under plain `exec` copy `build_model` into the script instead.
