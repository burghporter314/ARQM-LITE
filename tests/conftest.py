"""
Shared fixtures and helpers for ARQM-LITE tests.
"""
import sys
from pathlib import Path

import pytest

# Make util/training importable
_ROOT = Path(__file__).parent.parent
_TRAINING = _ROOT / "util" / "training"
for p in (str(_ROOT / "util"), str(_TRAINING)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Detector fixtures (session-scoped: load once per test run) ─────────────


@pytest.fixture(scope="session")
def amb_detector():
    from training_ambiguity import AmbiguityDetector
    return AmbiguityDetector(
        calibration_data=str(_ROOT / "calibration_data.json")
    )


@pytest.fixture(scope="session")
def feas_detector():
    from training_feasibility import FeasibilityDetector
    return FeasibilityDetector(
        calibration_data=str(_ROOT / "feasibility_calibration_data.json")
    )


@pytest.fixture(scope="session")
def sing_detector():
    from training_singularity import SingularityDetector
    return SingularityDetector(
        calibration_data=str(_TRAINING / "singularity_calibration_data.json")
    )


@pytest.fixture(scope="session")
def verif_detector():
    from training_verifiability import VerifiabilityDetector
    return VerifiabilityDetector(
        calibration_data=str(_ROOT / "verifiability_calibration_data.json")
    )


# ── Metrics helper ─────────────────────────────────────────────────────────


def compute_metrics(y_true: list[bool], y_pred: list[bool]) -> dict:
    """Return precision, recall, F1, and accuracy."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "accuracy":  accuracy,
    }
