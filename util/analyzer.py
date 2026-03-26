"""
Quality analysis orchestrator: initialises and runs all four detectors
(ambiguity, feasibility, singularity, verifiability) on a list of requirements.

Detectors are loaded once as a module-level singleton to avoid repeated
SentenceTransformer initialisation overhead across requests.

RAG document context
--------------------
Pass the full document text as ``document_text`` to ``analyze_requirements()``.
The orchestrator will extract noun phrases and named entities from the document
and build a per-request DomainKnowledgeBase that suppresses false-positive
violations caused by project-specific terminology.
"""

import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
_TRAINING_DIR = Path(__file__).parent / "training"
_ROOT_DIR     = Path(__file__).parent.parent
_UTIL_DIR     = Path(__file__).parent

# Make training modules importable
for _p in (str(_TRAINING_DIR), str(_UTIL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training_ambiguity    import AmbiguityDetector
from training_feasibility  import FeasibilityDetector
from training_singularity  import SingularityDetector
from training_verifiability import VerifiabilityDetector
from domain_kb import extract_document_terms

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


def analyze_requirements(requirements: list[str], document_text: str = "") -> list[dict]:
    """
    Run all four detectors on each requirement sentence.

    Args:
        requirements:   List of requirement sentences to analyse.
        document_text:  Full text of the source document.  When provided,
                        domain terms are extracted from the document and used
                        to suppress project-specific false positives (RAG).

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

    # Build a per-request document KB by augmenting the static domain KB
    # with noun phrases extracted from the uploaded document.
    doc_kb = None
    if document_text:
        doc_terms = extract_document_terms(document_text, amb_det.nlp)
        if doc_terms:
            doc_kb = amb_det.domain_kb.augment(doc_terms)

    results = []
    for req in requirements:
        results.append({
            "sentence":      req,
            "ambiguity":     amb_det.analyze(req,  doc_kb=doc_kb),
            "feasibility":   feas_det.analyze(req, doc_kb=doc_kb),
            "singularity":   sing_det.analyze(req, doc_kb=doc_kb),
            "verifiability": verif_det.analyze(req, doc_kb=doc_kb),
        })
    return results
