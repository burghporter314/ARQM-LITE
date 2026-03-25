"""
Quality analysis orchestrator: initialises and runs all four detectors
(ambiguity, feasibility, singularity, verifiability) on a list of requirements.

Detectors are loaded once as a module-level singleton to avoid repeated
SentenceTransformer initialisation overhead across requests.
"""

import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
_TRAINING_DIR = Path(__file__).parent / "training"
_ROOT_DIR     = Path(__file__).parent.parent

# Make training modules importable
if str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from training_ambiguity    import AmbiguityDetector
from training_feasibility  import FeasibilityDetector
from training_singularity  import SingularityDetector
from training_verifiability import VerifiabilityDetector

# ── singleton detectors ───────────────────────────────────────────────────────
_detectors = None


def get_detectors():
    """Return (or lazily initialise) the four quality detectors."""
    global _detectors
    if _detectors is None:
        print("[Analyzer] Initialising quality detectors …")
        _detectors = (
            AmbiguityDetector(
                calibration_data=str(_ROOT_DIR / "calibration_data.json")
            ),
            FeasibilityDetector(
                calibration_data=str(_ROOT_DIR / "feasibility_calibration_data.json")
            ),
            SingularityDetector(
                calibration_data=str(_TRAINING_DIR / "singularity_calibration_data.json")
            ),
            VerifiabilityDetector(
                calibration_data=str(_ROOT_DIR / "verifiability_calibration_data.json")
            ),
        )
        print("[Analyzer] All detectors ready.")
    return _detectors


def analyze_requirements(requirements: list[str]) -> list[dict]:
    """
    Run all four detectors on each requirement sentence.

    Returns a list of result dicts, one per requirement:
        {
            "sentence":      str,
            "ambiguity":     AnalysisResult,
            "feasibility":   FeasibilityResult,
            "singularity":   SingularityResult,
            "verifiability": VerifiabilityResult,
        }
    """
    amb_det, feas_det, sing_det, verif_det = get_detectors()

    results = []
    for req in requirements:
        results.append({
            "sentence":      req,
            "ambiguity":     amb_det.analyze(req),
            "feasibility":   feas_det.analyze(req),
            "singularity":   sing_det.analyze(req),
            "verifiability": verif_det.analyze(req),
        })
    return results
