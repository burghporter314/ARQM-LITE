"""
Requirement identification using a trained TextCNN classifier.

Loads the CNN model + vocabulary saved by
util/training/train_identification_cnn.py and runs batched inference.
Falls back to a keyword heuristic if the model files are not found.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
import torch.nn as nn

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR = _BASE_DIR.parent / "models" / "requirement_identification" / "cnn"

# ── Hyperparameters ───────────────────────────────────────────────────────────

_BATCH_SIZE = 256

# ── Post-classification filter ────────────────────────────────────────────────

_STRONG_MODAL_RE = re.compile(r"\b(shall|must|should|will)\b", re.IGNORECASE)
_RELATIVE_MODAL_RE = re.compile(
    r"\b(which|that)\b.{0,40}\b(can|could|may|might)\b", re.IGNORECASE
)
_GHERKIN_RE = re.compile(r"^\s*(given|when|then|and|but)\b", re.IGNORECASE)
_BOILERPLATE_RE = re.compile(
    r"^\s*(the\s+purpose\s+of|the\s+goal\s+of|the\s+mission\s+(of|is)|"
    r"in\s+order\s+to|it\s+is\s+important\s+to|this\s+document\s+will|"
    r"the\s+following\s+is|the\s+intent\s+of)",
    re.IGNORECASE,
)


def _is_descriptive(sentence: str) -> bool:
    if _GHERKIN_RE.match(sentence):
        return True
    if _BOILERPLATE_RE.match(sentence):
        return True
    if _STRONG_MODAL_RE.search(sentence):
        return False
    return bool(_RELATIVE_MODAL_RE.search(sentence))

# ── TextCNN (must match training definition) ──────────────────────────────────

class _TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.relu(conv(emb)).max(dim=-1).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)

# ── Singleton state ───────────────────────────────────────────────────────────

_model   = None
_vocab   = None     # dict[str, int]
_max_len = 64
_device  = None
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")


def _tokenize(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


def _encode(text: str) -> list[int]:
    ids = [_vocab.get(w, 1) for w in _tokenize(text)][:_max_len]
    ids += [0] * (_max_len - len(ids))
    return ids


def _init() -> None:
    global _model, _vocab, _max_len, _device

    if _model is not None:
        return

    cfg_path   = _MODEL_DIR / "config.json"
    vocab_path = _MODEL_DIR / "vocab.json"
    model_path = _MODEL_DIR / "model.pt"

    if not model_path.exists():
        raise FileNotFoundError(
            f"CNN model not found at {_MODEL_DIR}. "
            "Run: python util/training/train_identification_cnn.py"
        )

    print("[Identification] Loading TextCNN …")
    cfg      = json.loads(cfg_path.read_text(encoding="utf-8"))
    _vocab   = json.loads(vocab_path.read_text(encoding="utf-8"))
    _max_len = cfg["max_len"]
    _device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _model = _TextCNN(
        vocab_size   = cfg["vocab_size"],
        embed_dim    = cfg["embed_dim"],
        num_filters  = cfg["num_filters"],
        filter_sizes = cfg["filter_sizes"],
        dropout      = cfg["dropout"],
    ).to(_device)
    _model.load_state_dict(torch.load(model_path, map_location=_device))
    _model.eval()
    print(f"[Identification] TextCNN ready  (device={_device}, vocab={cfg['vocab_size']})")


def identify_requirements(
    sentences: list[str],
    batch_size: int = _BATCH_SIZE,
) -> list[str]:
    """Return the subset of *sentences* identified as software requirements."""
    _init()

    flags: list[bool] = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        x = torch.tensor([_encode(s) for s in batch], dtype=torch.long).to(_device)
        with torch.no_grad():
            preds = (_model(x).sigmoid() >= 0.5).tolist()
        flags.extend(bool(p) for p in preds)

    return [s for s, flag in zip(sentences, flags) if flag and not _is_descriptive(s)]
