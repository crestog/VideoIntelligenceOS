"""
The cross-encoder reranker.

A bi-encoder (encoder.py) embeds the query and each passage separately and never
sees the two together, so "red bicycle" and "a bicycle beside a red door" look
alike to it. A cross-encoder reads (query, passage) as one sequence and attends
across the pair, which is markedly better at telling a real match from a merely
plausible one. The price is a forward pass per pair, so it never touches the
corpus — only the fused shortlist search.py hands it (config.RERANK_TOP).

bge-reranker-v2-m3 matches the bge-m3 encoder default: the same multilingual
backbone, so it ranks Hindi and Hinglish as readily as English, and the
processing plane already keeps its weights on disk. Its output is one relevance
logit per pair; search fuses the *order*, so the raw logit is enough and no
calibration into a probability is needed.

Everything degrades rather than fails, exactly like the encoder: no torch, torch
refused by a signing policy, no weights, or no GPU headroom → get_reranker()
returns None, score() returns [], and search keeps the fused hybrid order. A
reranker that will not load must never take search down with it.
"""

import os
import threading

from . import config
from .tgchannel import log

_LOCK = threading.Lock()
_RERANKER = None
_TRIED = False
_ERROR = ""


def error() -> str:
    return _ERROR


def ready() -> bool:
    return _RERANKER is not None


class _CrossEncoderReranker:
    """Preferred path. sentence-transformers ships in the image and its
    CrossEncoder handles the pair tokenisation and the classifier head."""

    kind = "sentence-transformers"

    def __init__(self, model):
        self.model = model

    def score(self, query, passages):
        pairs = [[query, p] for p in passages]
        out = self.model.predict(
            pairs, batch_size=config.EMBED_BATCH,
            convert_to_numpy=True, show_progress_bar=False)
        return [float(x) for x in out]


class _TransformersReranker:
    """Fallback without sentence-transformers: tokenise the pair as (text,
    text_pair) and read the single logit the sequence classifier emits — the
    documented bge-reranker usage."""

    kind = "transformers"

    def __init__(self, tokenizer, model, torch, device):
        self.tok = tokenizer
        self.model = model
        self.torch = torch
        self.device = device

    def score(self, query, passages):
        torch = self.torch
        out = []
        step = config.EMBED_BATCH
        for i in range(0, len(passages), step):
            batch = passages[i:i + step]
            enc = self.tok([query] * len(batch), batch, padding=True,
                           truncation=True, max_length=512, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits.view(-1).float()
            out.extend(logits.cpu().tolist())
        return out


def _pick_device(torch):
    """GPU only when a comfortable margin is free, else CPU — the same policy as
    the encoder, for the same reason: Atlas can share a card with the harvester's
    shards, and a second process grabbing VRAM is how a GPU worker dies."""
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return "cpu", None
    if config.RERANK_DEVICE == "cpu":
        return "cpu", None
    try:
        free, _total = torch.cuda.mem_get_info(0)
        if free < 2_500_000_000:        # bge-reranker-v2-m3 fp16 is ~1.1 GB
            log("reranker staying on CPU — GPU has under 2.5 GB free")
            return "cpu", None
    except Exception:
        return "cpu", None
    return "cuda", torch.float16


def get_reranker():
    """Load the reranker once. Returns None if it cannot be had."""
    global _RERANKER, _TRIED, _ERROR
    with _LOCK:
        if _RERANKER is not None or _TRIED:
            return _RERANKER
        _TRIED = True

        # Share the harvester's model cache, exactly as the encoder does.
        for var, path in (("HF_HOME", config.HF_CACHE),
                          ("SENTENCE_TRANSFORMERS_HOME", config.ST_CACHE)):
            os.environ.setdefault(var, path)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                pass
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Absent vs refused, as in the encoder: torch can be installed and still
        # blocked (Windows Smart App Control raises OSError WinError 4551 loading
        # its unsigned DLLs), which no ImportError clause would catch. Either way
        # the reranker just goes away and search keeps its fused order.
        try:
            import torch
        except ImportError as e:
            _ERROR = f"torch missing ({e})"
            log(f"reranker unavailable — {_ERROR}; results keep the fused order")
            return None
        except Exception as e:                             # noqa: BLE001
            _ERROR = (f"torch present but unusable — {type(e).__name__}: "
                      f"{str(e)[:160]}")
            log(f"reranker unavailable — {_ERROR}; results keep the fused order")
            return None
        device, dtype = _pick_device(torch)
        if device == "cpu":
            try:
                torch.set_num_threads(max(2, (os.cpu_count() or 4) - 1))
            except Exception:
                pass

        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder(config.RERANK_MODEL, device=device,
                                 max_length=512)
            if dtype is not None:
                try:
                    model.model = model.model.half()
                except Exception:
                    pass
            _RERANKER = _CrossEncoderReranker(model)
            log(f"reranker ready — {config.RERANK_MODEL} on {device} "
                f"via sentence-transformers")
            return _RERANKER
        except Exception as e:
            log(f"sentence-transformers reranker failed "
                f"({type(e).__name__}: {e}) — trying transformers directly")

        try:
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            tok = AutoTokenizer.from_pretrained(config.RERANK_MODEL)
            kw = {}
            if dtype is not None:
                kw["torch_dtype"] = dtype
            model = AutoModelForSequenceClassification.from_pretrained(
                config.RERANK_MODEL, **kw)
            model = model.to(device).eval()
            _RERANKER = _TransformersReranker(tok, model, torch, device)
            log(f"reranker ready — {config.RERANK_MODEL} on {device} "
                f"via transformers")
            return _RERANKER
        except Exception as e:
            _ERROR = f"{type(e).__name__}: {e}"
            log(f"reranker unavailable — {_ERROR}; results keep the fused order")
            return None


def score(query, passages):
    """Relevance of each passage to the query, aligned to `passages`.

    Returns [] whenever the reranker cannot run, which callers read as "no
    rerank" — never as "everything scored zero", which would bury every result.
    """
    if not passages:
        return []
    rr = get_reranker()
    if rr is None:
        return []
    try:
        return rr.score(query, list(passages))
    except Exception as e:                                 # noqa: BLE001
        log(f"rerank failed — {type(e).__name__}: {e}; keeping fused order")
        return []


def warm() -> bool:
    """Load the model and run one throwaway pair, so the first real query does
    not pay for kernel compilation and lazy weight init."""
    rr = get_reranker()
    if rr is None:
        return False
    try:
        rr.score("warm", ["warm"])
        return True
    except Exception as e:
        log(f"reranker warm-up failed — {type(e).__name__}: {e}")
        return False
