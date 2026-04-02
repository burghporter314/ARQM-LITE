"""
TextCNN trainer for requirement identification.

Trains a Kim (2014) TextCNN on the PURE_train.csv dataset, saves the best
checkpoint (by val F1) to models/requirement_identification/cnn/.

Usage:
    python util/training/train_identification_cnn.py
"""

from __future__ import annotations

import csv
import json
import re
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────

_ROOT       = Path(__file__).parent.parent.parent
_DATA_PATH  = _ROOT / "datasets" / "requirement_identification" / "PURE_train.csv"
_SAVE_DIR   = _ROOT / "models" / "requirement_identification" / "cnn"

# ── Hyperparameters ────────────────────────────────────────────────────────────

EMBED_DIM    = 128
NUM_FILTERS  = 128
FILTER_SIZES = [2, 3, 4, 5]
DROPOUT      = 0.5
MAX_LEN      = 64
MIN_FREQ     = 2
BATCH_SIZE   = 64
EPOCHS       = 30
LR           = 1e-3
PATIENCE     = 5        # early-stopping patience (val F1)
VAL_FRAC     = 0.15
SEED         = 42

# ── Tokeniser ─────────────────────────────────────────────────────────────────

_PUNCT_RE = re.compile(r"[^a-z0-9\s]")

def tokenize(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()

# ── Vocabulary ────────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

class Vocabulary:
    def __init__(self) -> None:
        self.w2i: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.i2w: list[str]      = [PAD_TOKEN, UNK_TOKEN]

    def build(self, texts: list[str], min_freq: int = MIN_FREQ) -> None:
        counts: Counter = Counter()
        for t in texts:
            counts.update(tokenize(t))
        for word, freq in counts.items():
            if freq >= min_freq:
                self.w2i[word] = len(self.i2w)
                self.i2w.append(word)
        print(f"[Vocab] {len(self.i2w)} tokens (min_freq={min_freq})")

    def encode(self, text: str, max_len: int = MAX_LEN) -> list[int]:
        ids = [self.w2i.get(w, 1) for w in tokenize(text)][:max_len]
        ids += [0] * (max_len - len(ids))   # pad
        return ids

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.w2i, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        vocab = cls()
        vocab.w2i = json.loads(path.read_text(encoding="utf-8"))
        vocab.i2w = [""] * len(vocab.w2i)
        for w, i in vocab.w2i.items():
            vocab.i2w[i] = w
        return vocab

# ── Dataset ───────────────────────────────────────────────────────────────────

class ReqDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], vocab: Vocabulary) -> None:
        self.ids    = [torch.tensor(vocab.encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float)

    def __len__(self)  -> int:              return len(self.labels)
    def __getitem__(self, i): return self.ids[i], self.labels[i]

# ── Model ─────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size:   int,
        embed_dim:    int   = EMBED_DIM,
        num_filters:  int   = NUM_FILTERS,
        filter_sizes: list  = FILTER_SIZES,
        dropout:      float = DROPOUT,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)   # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))               # (batch, num_filters, L)
            p = c.max(dim=-1).values                # (batch, num_filters)
            pooled.append(p)
        cat = torch.cat(pooled, dim=1)              # (batch, num_filters * n_filters)
        return self.fc(self.dropout(cat)).squeeze(1)

# ── Training ──────────────────────────────────────────────────────────────────

def _load_data() -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(_DATA_PATH, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = row["text"].strip()
            if text:
                texts.append(text)
                labels.append(1 if row["classification"].strip() == "T" else 0)
    return texts, labels


def _eval(model: TextCNN, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion  = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
            preds = (logits.sigmoid() >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.long().cpu().tolist())
    loss = total_loss / len(loader.dataset)
    f1   = f1_score(all_labels, all_preds, average="binary")
    return loss, f1


def train() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    texts, labels = _load_data()
    print(f"[Train] Loaded {len(texts)} samples  (req={sum(labels)}, non-req={len(labels)-sum(labels)})")

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=VAL_FRAC, stratify=labels, random_state=SEED
    )

    vocab = Vocabulary()
    vocab.build(tr_texts)

    tr_ds  = ReqDataset(tr_texts,  tr_labels,  vocab)
    val_ds = ReqDataset(val_texts, val_labels, vocab)
    tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model     = TextCNN(vocab_size=len(vocab.i2w)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_f1       = 0.0
    patience_left = PATIENCE

    _SAVE_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in tr_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        tr_loss = total_loss / len(tr_ds)

        val_loss, val_f1 = _eval(model, val_loader, device)
        marker = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_left = PATIENCE
            torch.save(model.state_dict(), _SAVE_DIR / "model.pt")
            vocab.save(_SAVE_DIR / "vocab.json")
            (_SAVE_DIR / "config.json").write_text(json.dumps({
                "vocab_size":   len(vocab.i2w),
                "embed_dim":    EMBED_DIM,
                "num_filters":  NUM_FILTERS,
                "filter_sizes": FILTER_SIZES,
                "dropout":      DROPOUT,
                "max_len":      MAX_LEN,
            }, indent=2), encoding="utf-8")
            marker = "  *best*"
        else:
            patience_left -= 1

        print(
            f"Epoch {epoch:3d}/{EPOCHS}  "
            f"tr_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  val_F1={val_f1:.4f}"
            f"{marker}"
        )

        if patience_left == 0:
            print(f"[Train] Early stopping at epoch {epoch}")
            break

    print(f"\n[Train] Best val F1: {best_f1:.4f}")
    print(f"[Train] Model saved to {_SAVE_DIR}")

    # Final report using best saved model
    print("\n[Train] Final evaluation on val set (best checkpoint):")
    best_model = TextCNN(vocab_size=len(vocab.i2w)).to(device)
    best_model.load_state_dict(torch.load(_SAVE_DIR / "model.pt", map_location=device))
    best_model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            preds = (best_model(x).sigmoid() >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(y.long().tolist())
    print(classification_report(all_labels, all_preds, target_names=["non-req", "req"]))


if __name__ == "__main__":
    train()
