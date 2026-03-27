"""
Quality analysis orchestrator: initialises and runs all four detectors
(ambiguity, feasibility, singularity, verifiability) on a list of requirements.

Detectors are loaded once as a module-level singleton to avoid repeated
SentenceTransformer initialisation overhead across requests.

RAG document context
--------------------
Pass the full document text as ``document_text`` to ``analyze_requirements()``
or ``analyze_full()``.  The orchestrator will:

  1. Parse the document once with spaCy (shared parse).
  2. Extract domain terms and named entities from the same parse.
  3. Build a per-request DomainKnowledgeBase (static + corpus + doc layers).
  4. Persist new document terms to the cross-request corpus KB.
  5. Log the detected vertical domain (healthcare, fintech, etc.) when found.

Use ``analyze_full()`` to receive both quality results and named entities in
a single call with no duplicate NLP work.
"""

import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
_TRAINING_DIR = Path(__file__).parent / "training"
_ROOT_DIR     = Path(__file__).parent.parent
_UTIL_DIR     = Path(__file__).parent

for _p in (str(_TRAINING_DIR), str(_UTIL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from training_ambiguity    import AmbiguityDetector
from training_feasibility  import FeasibilityDetector
from training_singularity  import SingularityDetector
from training_verifiability import VerifiabilityDetector
from domain_kb import (
    extract_document_terms_from_doc,
    save_corpus_terms,
    detect_domain,
)
from entity_extraction import extract_entities_from_doc

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


def analyze_full(
    requirements: list[str],
    document_text: str = "",
) -> tuple[list[dict], dict]:
    """Run all four detectors and return quality results plus named entities.

    The spaCy parse is performed once and shared between domain term extraction
    and entity extraction, eliminating a redundant NLP pass per request.

    Args:
        requirements:   List of requirement sentences to analyse.
        document_text:  Full text of the source document.  When provided,
                        domain terms are extracted and used to suppress
                        project-specific false positives (RAG).

    Returns:
        ``(results, entities)`` where:

        * *results* — list of per-requirement dicts with keys
          ``"sentence"``, ``"ambiguity"``, ``"feasibility"``,
          ``"singularity"``, ``"verifiability"``.
        * *entities* — dict from ``extract_entities_from_doc()``
          (display_label → [(text, count), …]).
    """
    amb_det, feas_det, sing_det, verif_det = get_detectors()

    doc_kb   = None
    entities = {}

    if document_text:
        # ── Detect vertical domain ─────────────────────────────────────
        domain = detect_domain(document_text)
        if domain:
            print(f"[Analyzer] Detected domain: {domain}")

        # ── Single spaCy parse shared for both purposes ────────────────
        doc = amb_det.nlp(document_text[:100_000])

        # ── Domain term extraction (uses pre-parsed doc) ───────────────
        doc_terms = extract_document_terms_from_doc(doc, document_text)
        if doc_terms:
            doc_kb = amb_det.domain_kb.with_document(doc_terms)
            # Persist novel terms to the cross-request corpus KB
            save_corpus_terms(doc_terms)

        # ── Entity extraction (reuses the same parse) ──────────────────
        entities = extract_entities_from_doc(doc)

    results = []
    for req in requirements:
        results.append({
            "sentence":      req,
            "ambiguity":     amb_det.analyze(req,  doc_kb=doc_kb),
            "feasibility":   feas_det.analyze(req, doc_kb=doc_kb),
            "singularity":   sing_det.analyze(req, doc_kb=doc_kb),
            "verifiability": verif_det.analyze(req, doc_kb=doc_kb),
        })

    return results, entities


def analyze_requirements(
    requirements: list[str],
    document_text: str = "",
) -> list[dict]:
    """Backward-compatible wrapper around ``analyze_full()``."""
    results, _ = analyze_full(requirements, document_text)
    return results
