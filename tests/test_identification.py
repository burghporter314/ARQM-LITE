"""
Requirement identification tests.

For each of the 6 labeled datasets in datasets/requirement_identification/PURE/,
run identify_requirements() (TinyBERT classifier) against ground-truth labels
and assert minimum precision, recall, and F1 thresholds.

Also tests on the full PURE_train.csv dataset.
"""
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import compute_metrics
from util.identification import identify_requirements

_ROOT        = Path(__file__).parent.parent
_DATASET_DIR = _ROOT / "datasets" / "requirement_identification" / "PURE"
_TRAIN_CSV   = _ROOT / "datasets" / "requirement_identification" / "PURE_train.csv"

_DATASET_FILES = sorted(_DATASET_DIR.glob("*.xlsx"))

# Observed across 6 per-project datasets: F1=0.660–0.727, Acc=0.543–0.657
_MIN_PRECISION = 0.50
_MIN_RECALL    = 0.80
_MIN_F1        = 0.60


@pytest.mark.parametrize("dataset_path", _DATASET_FILES, ids=[f.name for f in _DATASET_FILES])
def test_identification_metrics(dataset_path):
    df = pd.read_excel(dataset_path)
    df.columns = df.columns.str.strip()

    sentences = df["Sentence"].astype(str).tolist()
    y_true    = df["is_requirement"].astype(bool).tolist()

    identified = set(identify_requirements(sentences))
    y_pred = [s in identified for s in sentences]

    m = compute_metrics(y_true, y_pred)

    print(
        f"\n{dataset_path.name}  n={len(sentences)}  pos={sum(y_true)}"
        f"  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        f"  Acc={m['accuracy']:.3f}"
        f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}"
    )

    assert m["precision"] >= _MIN_PRECISION, (
        f"{dataset_path.name}: precision {m['precision']:.3f} < {_MIN_PRECISION}"
    )
    assert m["recall"] >= _MIN_RECALL, (
        f"{dataset_path.name}: recall {m['recall']:.3f} < {_MIN_RECALL}"
    )
    assert m["f1"] >= _MIN_F1, (
        f"{dataset_path.name}: F1 {m['f1']:.3f} < {_MIN_F1}"
    )


def test_identification_pure_train():
    """
    Evaluate on the full PURE_train.csv dataset.

    Note: TinyBERT may have been trained on a portion of this data, so
    results here may be optimistic. Use the per-project xlsx tests for
    out-of-domain performance.

    Observed: P=0.642, R=0.910, F1=0.753, Acc=0.678
    """
    df = pd.read_csv(_TRAIN_CSV)
    df.columns = df.columns.str.strip().str.lower()
    df["label"] = df["classification"].map({"T": True, "F": False})
    df = df.dropna(subset=["label"])

    sentences = df["text"].astype(str).tolist()
    y_true    = df["label"].tolist()

    identified = set(identify_requirements(sentences))
    y_pred = [s in identified for s in sentences]

    m = compute_metrics(y_true, y_pred)

    print(
        f"\nPURE_train  n={len(sentences)}  pos={sum(y_true)}"
        f"  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        f"  Acc={m['accuracy']:.3f}"
        f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}"
    )

    assert m["precision"] >= 0.58, f"PURE_train precision {m['precision']:.3f} < 0.58"
    assert m["recall"]    >= 0.85, f"PURE_train recall {m['recall']:.3f} < 0.85"
    assert m["f1"]        >= 0.70, f"PURE_train F1 {m['f1']:.3f} < 0.70"
