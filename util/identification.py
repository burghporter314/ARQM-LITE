"""
Requirement identification using a fine-tuned TinyBERT sequence classifier.

Loads the local BertForSequenceClassification model and runs batched inference.
The requirement label index is determined once at startup by probing the model
with a small set of unambiguous examples.
"""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_MODEL_DIR = os.path.join(
    _BASE_DIR,
    "../models/requirement_identification/bert-base-requirements-identification-generative-ai",
)
_TOKENIZER_DIR = os.path.join(
    _BASE_DIR,
    "../models/bert-base-requirements-identification-generative-ai-siamese",
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
_BATCH_SIZE = 64

# ── Singleton state ───────────────────────────────────────────────────────────
_model      = None
_tokenizer  = None
_req_label  = None   # int: label index that corresponds to "requirement"

_PROBE_REQS = [
    "The system shall validate user credentials before granting access.",
    "The application must support concurrent users without performance degradation.",
    "The software shall encrypt all data transmitted over the network.",
]


def _init() -> None:
    global _model, _tokenizer, _req_label

    if _model is not None:
        return

    print("[Identification] Loading TinyBERT classifier …")

    _tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_DIR)
    _model     = AutoModelForSequenceClassification.from_pretrained(_MODEL_DIR)
    _model.eval()

    # Determine which output index corresponds to "requirement"
    with torch.no_grad():
        enc = _tokenizer(
            _PROBE_REQS,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        preds = _model(**enc).logits.argmax(dim=-1).tolist()

    _req_label = max(set(preds), key=preds.count)
    print(f"[Identification] TinyBERT ready. Requirement label index: {_req_label}")


def identify_requirements(
    sentences: list[str],
    batch_size: int = _BATCH_SIZE,
) -> list[str]:
    """
    Return the subset of *sentences* identified as software requirements.

    Each sentence is classified by the fine-tuned TinyBERT model.
    """
    _init()

    flags: list[bool] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]

        with torch.no_grad():
            enc = _tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            preds = _model(**enc).logits.argmax(dim=-1).tolist()

        flags.extend(p == _req_label for p in preds)

    return [s for s, flag in zip(sentences, flags) if flag]
