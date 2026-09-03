from __future__ import annotations

from pathlib import Path
from typing import Protocol
import math
import sys


class Scorer(Protocol):
    def sequence_nll(self, text: str) -> tuple[float, int]:
        """Return (sum NLL, number of predicted tokens)."""

    def continuation_nll(self, context: str, ending: str) -> tuple[float, int]:
        """Return (sum NLL of ending tokens, token count)."""


def ppl(mean_nll: float | None) -> float | None:
    if mean_nll is None:
        return None
    return math.exp(mean_nll)


def mean_nll(parts: list[tuple[float, int]]) -> float | None:
    tokens = sum(n for _, n in parts)
    if tokens <= 0:
        return None
    return sum(s for s, _ in parts) / tokens


def _try_infer(model_dir: Path, infer_root: Path | None) -> Scorer | None:
    if infer_root is None:
        return None
    root = str(infer_root)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from engine.agent_api import load_engine  # type: ignore
    except Exception:
        return None
    try:
        engine = load_engine(model_dir)
        return InferScorer(engine)
    except Exception:
        return None


def _try_hf(model_dir: Path) -> Scorer | None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None
    try:
        tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(str(model_dir))
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        return HfScorer(model, tok, device)
    except Exception:
        return None


def load_scorer(model_dir: Path | None, infer_root: Path | None = None) -> Scorer | None:
    if model_dir is None:
        return None
    path = Path(model_dir).expanduser()
    if path.is_file():
        return _try_lab(path)
    if not path.is_dir():
        return None
    return _try_infer(path, infer_root) or _try_hf(path)


def _try_lab(path: Path) -> Scorer | None:
    try:
        from lab.train.loop import load_checkpoint

        model, arch, device, _blob = load_checkpoint(path)
        return LabScorer(model, arch, device)
    except Exception:
        return None


class InferScorer:
    def __init__(self, engine: object) -> None:
        self.engine = engine

    def sequence_nll(self, text: str) -> tuple[float, int]:
        return _nll_from_ids(self.engine, _encode(self.engine, text))

    def continuation_nll(self, context: str, ending: str) -> tuple[float, int]:
        ctx = _encode(self.engine, context, specials=False)
        full = _encode(self.engine, context + ending, specials=False)
        if len(full) <= len(ctx) or len(full) < 2:
            return _nll_from_ids(self.engine, full)
        return _nll_from_ids(self.engine, full, start=max(len(ctx) - 1, 0))


class HfScorer:
    def __init__(self, model: object, tok: object, device: str) -> None:
        self.model = model
        self.tok = tok
        self.device = device

    def sequence_nll(self, text: str) -> tuple[float, int]:
        return self._nll(text, prefix_len=0)

    def continuation_nll(self, context: str, ending: str) -> tuple[float, int]:
        ctx_ids = list(self.tok.encode(context, add_special_tokens=False))
        return self._nll(context + ending, prefix_len=max(len(ctx_ids) - 1, 0))

    def _nll(self, text: str, prefix_len: int) -> tuple[float, int]:
        import torch
        import torch.nn.functional as F

        ids = list(self.tok.encode(text, add_special_tokens=True))
        if len(ids) < 2:
            return 0.0, 0
        tokens = torch.tensor([ids], device=self.device)
        with torch.inference_mode():
            logits = self.model(tokens).logits
        logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = tokens[0, 1:]
        nll = -logp.gather(1, tgt[:, None]).squeeze(1)
        start = min(max(prefix_len, 0), nll.numel())
        sl = nll[start:]
        if sl.numel() == 0:
            return 0.0, 0
        return float(sl.sum().item()), int(sl.numel())


class LabScorer:
    """Byte-level TinyGPT scorer for lab trainer checkpoints."""

    def __init__(self, model: object, arch: dict, device: object) -> None:
        self.model = model
        self.arch = arch
        self.device = device
        self.seq = int(arch.get("seq_len") or 32)

    def sequence_nll(self, text: str) -> tuple[float, int]:
        ids = list(text.encode("utf-8", errors="replace"))
        return self._nll_range(ids, start_tgt=1)

    def continuation_nll(self, context: str, ending: str) -> tuple[float, int]:
        ctx = list(context.encode("utf-8", errors="replace"))
        full = list((context + ending).encode("utf-8", errors="replace"))
        return self._nll_range(full, start_tgt=max(len(ctx), 1))

    def _nll_range(self, ids: list[int], start_tgt: int) -> tuple[float, int]:
        import torch
        import torch.nn.functional as F

        if len(ids) < 2:
            return 0.0, 0
        total = 0.0
        count = 0
        start = min(max(start_tgt, 1), len(ids) - 1)
        for i in range(start, len(ids)):
            ctx = ids[max(0, i - self.seq) : i]
            if not ctx:
                continue
            x = torch.tensor([ctx], device=self.device)
            with torch.inference_mode():
                logits = self.model(x)
            logp = F.log_softmax(logits[0, -1].float(), dim=-1)
            total += float((-logp[ids[i]]).item())
            count += 1
        return total, count


def _encode(engine: object, text: str, specials: bool = True) -> list[int]:
    return list(engine.tokenizer.encode(text, add_special_tokens=specials))  # type: ignore[attr-defined]


def _nll_from_ids(engine: object, ids: list[int], start: int = 0) -> tuple[float, int]:
    import torch
    import torch.nn.functional as F

    if len(ids) < 2:
        return 0.0, 0
    model = engine.model  # type: ignore[attr-defined]
    tokens = torch.tensor([ids], device=model.device)
    logits = model.forward(tokens, cache=None)
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = tokens[0, 1:]
    nll = -logp.gather(1, tgt[:, None]).squeeze(1)
    start = min(max(start, 0), nll.numel())
    sl = nll[start:]
    if sl.numel() == 0:
        return 0.0, 0
    return float(sl.sum().item()), int(sl.numel())
