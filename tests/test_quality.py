"""
Quality detector tests.

For each of the 2 quality-labeled datasets in datasets/requirement_quality/PURE/,
run all four detectors against the ground-truth violation labels and assert
minimum precision, recall, and F1 thresholds per dimension.
"""
from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import compute_metrics

_DATASET_DIR = (
    Path(__file__).parent.parent / "datasets" / "requirement_quality" / "PURE"
)

# Only the two files that have the quality-violation columns
_QUALITY_DATASETS = [
    _DATASET_DIR / "dataset_1_quality_labeled.xlsx",
    _DATASET_DIR / "dataset_6_quality_labeled.xlsx",
]

_DIMENSIONS = ("ambiguity", "feasibility", "singularity", "verifiability")

# Per-dimension regression thresholds (10% margin below observed minimums across 2 datasets)
# Observed: ambiguity P=0.21–0.29 R=0.67 F1=0.32–0.41
#           feasibility P=0.15–0.20 R=0.21–0.38 F1=0.21
#           singularity P=0.55–0.63 R=0.51–0.56 F1=0.55–0.56
#           verifiability P=0.32–0.33 R=0.52 F1=0.40
_THRESHOLDS: dict[str, dict[str, float]] = {
    "ambiguity":     {"precision": 0.18, "recall": 0.60, "f1": 0.28},
    "feasibility":   {"precision": 0.12, "recall": 0.18, "f1": 0.17},
    "singularity":   {"precision": 0.50, "recall": 0.47, "f1": 0.52},
    "verifiability": {"precision": 0.28, "recall": 0.47, "f1": 0.36},
}


def _has_violation(result: object, dimension: str) -> bool:
    """Return True if the detector result indicates a violation."""
    d = result.to_dict()
    if dimension == "ambiguity":
        return bool(d.get("is_ambiguous", False))
    if dimension == "feasibility":
        return not bool(d.get("is_feasible", True))
    if dimension == "singularity":
        return not bool(d.get("is_singular", True))
    if dimension == "verifiability":
        return not bool(d.get("is_verifiable", True))
    raise ValueError(f"Unknown dimension: {dimension}")


@pytest.mark.parametrize("dataset_path", _QUALITY_DATASETS, ids=[f.name for f in _QUALITY_DATASETS])
@pytest.mark.parametrize("dimension", _DIMENSIONS)
def test_quality_metrics(
    dataset_path, dimension,
    amb_detector, feas_detector, sing_detector, verif_detector,
):
    detectors = {
        "ambiguity":     amb_detector,
        "feasibility":   feas_detector,
        "singularity":   sing_detector,
        "verifiability": verif_detector,
    }
    df = pd.read_excel(dataset_path)
    df.columns = df.columns.str.strip()

    col = f"{dimension}_violation"
    sentences = df["Sentence"].astype(str).tolist()
    y_true    = df[col].astype(bool).tolist()

    detector = detectors[dimension]
    y_pred   = [_has_violation(detector.analyze(s), dimension) for s in sentences]

    m = compute_metrics(y_true, y_pred)
    n = len(sentences)
    n_pos = sum(y_true)

    print(
        f"\n{dataset_path.name}  dim={dimension}  n={n}  pos={n_pos}"
        f"  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}"
        f"  Acc={m['accuracy']:.3f}"
        f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}"
    )

    thresh = _THRESHOLDS[dimension]
    assert m["precision"] >= thresh["precision"], (
        f"{dataset_path.name} [{dimension}]: precision {m['precision']:.3f} < {thresh['precision']}"
    )
    assert m["recall"] >= thresh["recall"], (
        f"{dataset_path.name} [{dimension}]: recall {m['recall']:.3f} < {thresh['recall']}"
    )
    assert m["f1"] >= thresh["f1"], (
        f"{dataset_path.name} [{dimension}]: F1 {m['f1']:.3f} < {thresh['f1']}"
    )
