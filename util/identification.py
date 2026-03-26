"""
Requirement identification using a fine-tuned TinyBERT sequence classifier.

Loads the local BertForSequenceClassification model and runs batched inference.
The requirement label index is determined once at startup by probing the model
with a small set of unambiguous examples.
"""

from __future__ import annotations

import os
import re

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

# ── Post-classification filter ────────────────────────────────────────────────
# Sentences where the only modal verb is a weak one (can/could/may/might)
# appearing inside a relative/subordinate clause ("which/that can be …") are
# almost always descriptive, not prescriptive.  Reject them even if TinyBERT
# classifies them as requirements.
_STRONG_MODAL_RE = re.compile(r"\b(shall|must|should|will)\b", re.IGNORECASE)
_RELATIVE_MODAL_RE = re.compile(
    r"\b(which|that)\b.{0,40}\b(can|could|may|might)\b", re.IGNORECASE
)

# Gherkin clause starters — these are test assertions, not requirements
_GHERKIN_RE = re.compile(r"^\s*(given|when|then|and|but)\b", re.IGNORECASE)

# Common document boilerplate openings that are never requirements
_BOILERPLATE_RE = re.compile(
    r"^\s*(the\s+purpose\s+of|the\s+goal\s+of|the\s+mission\s+(of|is)|"
    r"in\s+order\s+to|it\s+is\s+important\s+to|this\s+document\s+will|"
    r"the\s+following\s+is|the\s+intent\s+of)",
    re.IGNORECASE,
)


def _is_descriptive(sentence: str) -> bool:
    """Return True when the sentence is almost certainly descriptive, not a requirement."""
    if _GHERKIN_RE.match(sentence):
        return True
    if _BOILERPLATE_RE.match(sentence):
        return True
    if _STRONG_MODAL_RE.search(sentence):
        return False  # strong modal present — keep as requirement candidate
    return bool(_RELATIVE_MODAL_RE.search(sentence))


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

    return [s for s, flag in zip(sentences, flags) if flag and not _is_descriptive(s)]
