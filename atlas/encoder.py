"""
The text encoder.

One model, loaded once, used for both sides of dense retrieval. Three details
here are easy to get wrong and each one quietly wrecks accuracy rather than
raising an error:

**BGE pools the CLS token, not the mean.** Every "how to embed with
transformers" example mean-pools, and mean-pooling a BGE model produces vectors
that are *plausible* — normalised, right shape, cosine similarities in a sane
range — and materially worse at ranking. The sentence-transformers config for
this model sets `pooling_mode_cls_token: true`, so the manual fallback path
below takes `last_hidden_state[:, 0]`.

**Queries and passages may be encoded differently, per model.** The prefixes
come from the model profile in config, not from this file: bge-v1.5 wants an
asymmetric query-only instruction, e5 wants "query:"/"passage:" on both sides,
and bge-m3 (the default) wants neither. Applying the wrong prefix, or none where
one is due, costs recall silently — so both `EMBED_QUERY_PREFIX` and
`EMBED_PASSAGE_PREFIX` travel with the model and are applied on the right side
here.

**Vectors are L2-normalised on the way out.** Once every vector is unit length,
cosine similarity is a plain dot product, so the whole search reduces to one
matrix multiply. That is what makes exhaustive search fast enough to skip an ANN
index entirely.

Model choice is a *profile*, not a lone string — see the retrieval section of
config.py, where vector width and both prefixes move together. The default is
`bge-m3`: 568M parameters, 1024 dimensions, multilingual by construction,
because this archive holds English, Hindi and Hinglish and the old English-only
default (`bge-small-en`) could not place a Hindi query near the Hindi passage
that answers it — search was structurally blind to two of the three languages
it indexes. bge-m3 is also CLS-pooled, so the manual fallback above stays
correct; the e5 profiles are *mean*-pooled, so a host that dropped to the
hand-rolled `_TransformersEncoder` with an e5 model would pool the wrong token.
e5 must therefore go through the sentence-transformers path, which reads the
model's own pooling config. The cost of the bge-m3 default is a cold start that
re-encodes the whole corpus with a 568M model instead of a 33M one — minutes,
not seconds — the deliberate trade of cold-start speed for answering the query
at all.

Everything degrades rather than fails: if torch is missing, if torch is present
but the host refuses to load it, if the weights will not download, if there is no
GPU — `get_encoder()` returns None, the dense index is skipped, and search runs
lexically. A slower search is a working site; an exception at import is not.
"""

import os
import threading

from . import config
from .tgchannel import log

_LOCK = threading.Lock()
_ENCODER = None
_TRIED = False
_ERROR = ""


def error() -> str:
    return _ERROR


def ready() -> bool:
    return _ENCODER is not None


class _SentenceTransformerEncoder:
    """Preferred path. sentence-transformers already ships in the image and
    reads the model's own pooling config, so CLS pooling is handled for us."""

    kind = "sentence-transformers"

    def __init__(self, model, device: str):
        self.model = model
        self.device = device

    def encode_passages(self, texts):
        pre = config.EMBED_PASSAGE_PREFIX
        items = [pre + t for t in texts] if pre else list(texts)
        return self.model.encode(
            items, batch_size=config.EMBED_BATCH,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False)

    def encode_query(self, text: str):
        return self.model.encode(
            [config.EMBED_QUERY_PREFIX + text], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False)[0]


class _TransformersEncoder:
    """Fallback for an image without sentence-transformers. Implements BGE's
    pooling by hand — CLS, then L2 — because that is what the model was
    trained with."""

    kind = "transformers"

    def __init__(self, tokenizer, model, torch, device: str, dtype):
        self.tok = tokenizer
        self.model = model
        self.torch = torch
        self.device = device
        self.dtype = dtype

    def _encode(self, texts):
        torch = self.torch
        out = []
        step = config.EMBED_BATCH
        for i in range(0, len(texts), step):
            batch = texts[i:i + step]
            enc = self.tok(batch, padding=True, truncation=True,
                           max_length=512, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                hidden = self.model(**enc).last_hidden_state
                vec = hidden[:, 0]                      # CLS, not mean
                vec = torch.nn.functional.normalize(vec, p=2, dim=1)
            out.append(vec.float().cpu().numpy())
        import numpy as np
        return np.vstack(out) if out else np.zeros((0, config.EMBED_DIM),
                                                   dtype="float32")

    def encode_passages(self, texts):
        pre = config.EMBED_PASSAGE_PREFIX
        items = [pre + t for t in texts] if pre else list(texts)
        return self._encode(items).astype("float32")

    def encode_query(self, text: str):
        return self._encode([config.EMBED_QUERY_PREFIX + text])[0].astype("float32")


def _pick_device(torch):
    """GPU if one is free, CPU otherwise.

    Atlas can share a machine with the harvester's Qwen shards, and a second
    process grabbing VRAM is how the GPU worker dies mid-narrative. So the GPU
    is only taken when a comfortable margin is actually free — this model runs
    perfectly well on CPU, and a slow encode in the background is much better
    than an out-of-memory kill somewhere else.
    """
    if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
        return "cpu", None
    if config.EMBED_DEVICE == "cpu":
        return "cpu", None
    try:
        free, _total = torch.cuda.mem_get_info(0)
        if free < 2_500_000_000:            # bge-m3 fp16 is ~1.1 GB of weights
            log("encoder staying on CPU — GPU has under 2.5 GB free")
            return "cpu", None
    except Exception:
        return "cpu", None
    return "cuda", torch.float16


def get_encoder():
    """Load the encoder once. Returns None if it cannot be had."""
    global _ENCODER, _TRIED, _ERROR
    with _LOCK:
        if _ENCODER is not None or _TRIED:
            return _ENCODER
        _TRIED = True

        # Share the harvester's model cache when it exists, so a machine that
        # already downloaded weights does not download them again.
        for var, path in (("HF_HOME", config.HF_CACHE),
                          ("SENTENCE_TRANSFORMERS_HOME", config.ST_CACHE)):
            os.environ.setdefault(var, path)
            try:
                os.makedirs(path, exist_ok=True)
            except OSError:
                pass
        # Transformers phones home on every from_pretrained; on Kaggle that is
        # a few seconds of nothing when the weights are already local.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Guarded against `Exception`, not `ImportError`, and the two clauses say
        # different things on purpose. Torch can be installed and *forbidden*:
        # under Windows Smart App Control its unsigned DLLs raise `OSError:
        # [WinError 4551] An Application Control policy has blocked this file`
        # while loading, which no `ImportError` clause catches. With the narrower
        # guard, installing torch on such a host turned a graceful "no encoder"
        # into an unhandled exception out of `search._dense`. Absent is a pip
        # install; refused is a signing policy that no reinstall of torch will
        # satisfy — so which one it is belongs in the message, and neither one
        # belongs in the response status.
        try:
            import torch
        except ImportError as e:
            _ERROR = f"torch missing ({e})"
            log(f"encoder unavailable — {_ERROR}; search will be lexical only")
            return None
        except Exception as e:                             # noqa: BLE001
            _ERROR = (f"torch present but unusable — {type(e).__name__}: "
                      f"{str(e)[:160]}")
            log(f"encoder unavailable — {_ERROR}; search will be lexical only")
            return None

        device, dtype = _pick_device(torch)
        if device == "cpu":
            # The default thread count on Kaggle's 4-core CPU oversubscribes and
            # ends up slower than a sane fixed number.
            try:
                torch.set_num_threads(max(2, (os.cpu_count() or 4) - 1))
            except Exception:
                pass

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(config.EMBED_MODEL, device=device)
            if dtype is not None:
                try:
                    model = model.half()
                except Exception:
                    pass
            _ENCODER = _SentenceTransformerEncoder(model, device)
            log(f"encoder ready — {config.EMBED_MODEL} on {device} "
                f"via sentence-transformers")
            return _ENCODER
        except Exception as e:
            log(f"sentence-transformers path failed ({type(e).__name__}: {e}) "
                f"— trying transformers directly")

        try:
            from transformers import AutoModel, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(config.EMBED_MODEL)
            kw = {}
            if dtype is not None:
                kw["torch_dtype"] = dtype
            model = AutoModel.from_pretrained(config.EMBED_MODEL, **kw)
            model = model.to(device).eval()
            _ENCODER = _TransformersEncoder(tok, model, torch, device, dtype)
            log(f"encoder ready — {config.EMBED_MODEL} on {device} "
                f"via transformers (CLS pooling)")
            return _ENCODER
        except Exception as e:
            _ERROR = f"{type(e).__name__}: {e}"
            log(f"encoder unavailable — {_ERROR}; search will be lexical only")
            return None


def warm() -> bool:
    """Load the model and run one throwaway encode.

    Worth doing at boot rather than on the first search: the first call to a
    freshly loaded model pays for CUDA kernel compilation and lazy weight init,
    which would otherwise land on whoever types the first query.
    """
    enc = get_encoder()
    if enc is None:
        return False
    try:
        enc.encode_query("warm")
        return True
    except Exception as e:
        log(f"encoder warm-up failed — {type(e).__name__}: {e}")
        return False
