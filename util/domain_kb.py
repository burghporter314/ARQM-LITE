"""
domain_kb.py — RAG-based domain suppression filter (three-layer architecture).

Layers (combined in max_similarity queries):
  1. Static   — generic + domain-specific technical terms, encoded once and cached
  2. Corpus   — terms accumulated across documents (persisted to corpus_domain_terms.json)
  3. Document — noun phrases extracted from the current document (per-request only)

Key improvements over v1:
  - No re-encoding of static/corpus terms on augmentation (efficiency)
  - Frequency-weighted noun chunk extraction; hapax legomena filtered out
  - Graduated suppression via max_similarity() instead of binary is_domain_term()
  - nearest_term() for enhancing violation suggestions
  - Corpus KB persists across requests (cross-document learning)
  - Domain-specific overlay files (domain_terms_healthcare.json, etc.)
  - Feedback suppressions loaded from user-reported false positives
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

_DIR = Path(__file__).parent

_DEFAULT_TERMS_FILE = _DIR / "domain_terms.json"
_CORPUS_TERMS_FILE  = _DIR / "corpus_domain_terms.json"
_FEEDBACK_FILE      = _DIR / "feedback_suppressions.json"

DEFAULT_SUPPRESSION_THRESHOLD = 0.82

_MIN_CHUNK_TOKENS = 2
_MIN_CHUNK_FREQ   = 2   # noun chunks must appear >= this many times to be indexed

_NOISE_TOKENS = {
    "shall", "must", "should", "may", "will", "can", "would", "could",
    "the", "a", "an", "this", "that", "these", "those", "its", "their",
    "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "be", "is", "are", "was", "were", "been",
    "have", "has", "had", "do", "does", "did",
    "system", "application", "service", "component", "module", "feature",
    "function", "requirement", "user", "data", "value", "item",
}

# Keywords used to auto-detect the document's vertical domain
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "healthcare": ["patient", "clinical", "diagnosis", "treatment", "hospital",
                   "medication", "physician", "ehr", "fhir", "hl7"],
    "fintech":    ["transaction", "payment", "ledger", "settlement",
                   "kyc", "aml", "credit", "swift", "banking"],
    "aviation":   ["aircraft", "runway", "altitude", "atc", "navigation",
                   "airspace", "autopilot", "avionics", "takeoff"],
    "automotive": ["vehicle", "brake", "adas", "ecu",
                   "transmission", "lidar", "powertrain", "chassis"],
    "defense":    ["weapon", "munition", "radar", "classified",
                   "nato", "mil-std", "electronic warfare"],
}

_corpus_lock = threading.Lock()


# ── Domain detection ──────────────────────────────────────────────────────────

def detect_domain(text: str) -> str | None:
    """Return the most probable vertical domain for *text*, or None."""
    lower = text[:20_000].lower()
    scores = {
        domain: sum(1 for kw in keywords if kw in lower)
        for domain, keywords in _DOMAIN_KEYWORDS.items()
    }
    best, best_score = max(scores.items(), key=lambda kv: kv[1])
    return best if best_score >= 3 else None


# ── Knowledge base ────────────────────────────────────────────────────────────

class DomainKnowledgeBase:
    """Three-layer domain suppression knowledge base.

    The static and corpus embedding arrays are stored separately from the
    document layer so that ``with_document()`` only encodes new doc terms,
    reusing all other embeddings by reference.

    Usage::

        kb = DomainKnowledgeBase.load(encoder)   # static + corpus layers
        kb = kb.with_document(doc_terms)          # + document layer (per-request)

        sim = kb.max_similarity(text)             # graduated suppression
        pair = kb.nearest_term(text)              # for suggestion enhancement
    """

    # Class-level embedding cache keyed by encoder id()
    _static_cache: dict[int, tuple[list[str], np.ndarray]] = {}

    def __init__(
        self,
        encoder: SentenceTransformer,
        *,
        static_terms: list[str],
        static_embs: np.ndarray,
        corpus_terms: list[str] | None = None,
        corpus_embs: np.ndarray | None = None,
        doc_terms: list[str] | None = None,
        doc_embs: np.ndarray | None = None,
        threshold: float = DEFAULT_SUPPRESSION_THRESHOLD,
    ) -> None:
        self._encoder     = encoder
        self.threshold    = threshold

        self._static_terms = static_terms
        self._static_embs  = static_embs

        self._corpus_terms = corpus_terms or []
        self._corpus_embs  = corpus_embs if corpus_embs is not None else self._empty()

        self._doc_terms = doc_terms or []
        self._doc_embs  = doc_embs if doc_embs is not None else self._empty()

        # Flat list for backward-compat callers
        self.terms = self._static_terms + self._corpus_terms + self._doc_terms

    def _empty(self) -> np.ndarray:
        dim = self._encoder.get_sentence_embedding_dimension()
        return np.empty((0, dim), dtype=np.float32)

    def _encode(self, terms: list[str]) -> np.ndarray:
        if not terms:
            return self._empty()
        return self._encoder.encode(
            terms, normalize_embeddings=True, show_progress_bar=False
        )

    # ── Factories ─────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        encoder: SentenceTransformer,
        path: str | Path = _DEFAULT_TERMS_FILE,
        threshold: float = DEFAULT_SUPPRESSION_THRESHOLD,
    ) -> "DomainKnowledgeBase":
        """Load static + corpus layers.

        Static terms (generic + domain overlays + feedback) are encoded once
        per encoder instance and cached at the class level.  Corpus terms are
        loaded from disk but re-encoded each startup (they can change between
        restarts).
        """
        enc_id = id(encoder)

        # ── Static layer (cached per encoder) ─────────────────────────
        if enc_id not in cls._static_cache:
            static_terms = _load_json_terms(path)

            # Include all domain-specific overlay files that exist on disk
            for overlay in sorted(Path(path).parent.glob("domain_terms_*.json")):
                static_terms = list(dict.fromkeys(
                    static_terms + _load_json_terms(overlay)
                ))

            # Include user-reported false positives in the static layer
            feedback = _load_json_terms(_FEEDBACK_FILE)
            static_terms = list(dict.fromkeys(static_terms + feedback))

            if static_terms:
                print(f"[DomainKB] Encoding {len(static_terms)} static terms …")
            embs = encoder.encode(
                static_terms, normalize_embeddings=True, show_progress_bar=False
            ) if static_terms else np.empty(
                (0, encoder.get_sentence_embedding_dimension()), dtype=np.float32
            )
            cls._static_cache[enc_id] = (static_terms, embs)
            print(f"[DomainKB] Static layer ready ({len(static_terms)} terms).")
        else:
            static_terms, embs = cls._static_cache[enc_id]

        # ── Corpus layer (persisted across requests) ───────────────────
        corpus_terms = _load_json_terms(_CORPUS_TERMS_FILE)
        corpus_embs: np.ndarray | None = None
        if corpus_terms:
            print(f"[DomainKB] Encoding {len(corpus_terms)} corpus terms …")
            corpus_embs = encoder.encode(
                corpus_terms, normalize_embeddings=True, show_progress_bar=False
            )

        return cls(
            encoder,
            static_terms=static_terms,
            static_embs=embs,
            corpus_terms=corpus_terms,
            corpus_embs=corpus_embs,
            threshold=threshold,
        )

    def with_document(self, doc_terms: list[str]) -> "DomainKnowledgeBase":
        """Return a new KB augmented with document-specific terms.

        Only *doc_terms* are encoded; static and corpus embeddings are
        reused by reference — no redundant re-encoding.
        """
        if not doc_terms:
            return self
        doc_embs = self._encode(doc_terms)
        return DomainKnowledgeBase(
            self._encoder,
            static_terms=self._static_terms,
            static_embs=self._static_embs,
            corpus_terms=self._corpus_terms,
            corpus_embs=self._corpus_embs,
            doc_terms=doc_terms,
            doc_embs=doc_embs,
            threshold=self.threshold,
        )

    def augment(self, extra_terms: list[str]) -> "DomainKnowledgeBase":
        """Backward-compatible alias for ``with_document()``."""
        return self.with_document(extra_terms)

    # ── Public API ────────────────────────────────────────────────────────

    def max_similarity(self, text: str) -> float:
        """Return the highest cosine similarity between *text* and any
        known term across all three layers."""
        emb = self._encode([text])
        best = 0.0
        for arr in (self._static_embs, self._corpus_embs, self._doc_embs):
            if arr.shape[0] > 0:
                sim = float((emb @ arr.T)[0].max())
                if sim > best:
                    best = sim
        return best

    def nearest_term(self, text: str) -> tuple[str, float] | None:
        """Return ``(term, similarity)`` of the most similar known term,
        or ``None`` if no terms are loaded."""
        emb = self._encode([text])
        best_sim  = -1.0
        best_term: str | None = None
        for arr, terms in (
            (self._static_embs, self._static_terms),
            (self._corpus_embs, self._corpus_terms),
            (self._doc_embs,    self._doc_terms),
        ):
            if arr.shape[0] == 0:
                continue
            sims = (emb @ arr.T)[0]
            idx  = int(sims.argmax())
            if float(sims[idx]) > best_sim:
                best_sim  = float(sims[idx])
                best_term = terms[idx]
        return (best_term, best_sim) if best_term is not None else None

    def is_domain_term(self, text: str) -> bool:
        """Backward-compatible binary suppression check."""
        return self.max_similarity(text) >= self.threshold


# ── Corpus persistence ────────────────────────────────────────────────────────

def save_corpus_terms(new_terms: list[str], max_terms: int = 1_000) -> None:
    """Merge *new_terms* into the persisted corpus KB, capped at *max_terms*."""
    with _corpus_lock:
        existing = _load_json_terms(_CORPUS_TERMS_FILE)
        merged   = list(dict.fromkeys(existing + new_terms))[:max_terms]
        if len(merged) > len(existing):
            _CORPUS_TERMS_FILE.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            added = len(merged) - len(existing)
            print(f"[DomainKB] Corpus KB: +{added} terms ({len(merged)} total).")


def save_feedback_term(term: str) -> bool:
    """Persist a user-reported false-positive term to the feedback file.

    Returns True if the term was newly added.  Changes take effect on the
    next server restart (the static embedding cache is not invalidated at
    runtime to avoid race conditions with in-flight requests).
    """
    with _corpus_lock:
        existing = _load_json_terms(_FEEDBACK_FILE)
        if term in existing:
            return False
        existing.append(term)
        _FEEDBACK_FILE.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"[DomainKB] Feedback saved: {term!r} (takes effect on restart).")
    return True


def _load_json_terms(path: Path | str) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return raw if isinstance(raw, list) else raw.get("terms", [])
    except Exception:
        return []


# ── Document-level term extraction ───────────────────────────────────────────

def extract_document_terms(text: str, nlp) -> list[str]:
    """Parse *text* with *nlp* and extract candidate domain terms.

    Convenience wrapper around ``extract_document_terms_from_doc`` for
    callers that do not have a pre-parsed doc.
    """
    doc = nlp(text[:100_000])
    return extract_document_terms_from_doc(doc, text)


def extract_document_terms_from_doc(doc, original_text: str = "") -> list[str]:
    """Extract candidate domain terms from a pre-parsed spaCy *doc*.

    Accepts a pre-parsed document so the NLP pass can be shared with
    entity extraction, eliminating a second full NLP pass per request.

    Improvements over v1:
    - Noun chunks frequency-filtered (only chunks appearing >= _MIN_CHUNK_FREQ
      times are indexed, eliminating hapax legomena).
    - Named entities always indexed (high-signal, no frequency filter).
    - Capitalised compound terms (AES-256, OAuth 2.0) via regex.
    """
    candidates: set[str] = set()

    # Named entities — always high signal, no frequency filter
    for ent in doc.ents:
        term = ent.text.strip()
        if _is_useful_term(term):
            candidates.add(term)

    # Noun chunks — frequency-filtered to suppress hapax legomena
    chunk_counts: Counter[str] = Counter()
    raw_chunks: dict[str, str] = {}
    for chunk in doc.noun_chunks:
        tokens = [t for t in chunk if not t.is_space]
        while tokens and tokens[0].is_stop:
            tokens = tokens[1:]
        if len(tokens) < _MIN_CHUNK_TOKENS:
            continue
        term = " ".join(t.text for t in tokens).strip()
        norm = term.lower()
        chunk_counts[norm] += 1
        raw_chunks.setdefault(norm, term)

    for norm, count in chunk_counts.items():
        if count >= _MIN_CHUNK_FREQ and _is_useful_term(raw_chunks[norm]):
            candidates.add(raw_chunks[norm])

    # Capitalised compound terms via regex (e.g. AES-256, OAuth 2.0, TLS-1.3)
    sample = original_text[:100_000] or doc.text[:100_000]
    for m in re.finditer(
        r"\b([A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)+)\b"
        r"|"
        r"\b([A-Z]{2,}(?:\s+\d[\d.]*)?)\b",
        sample,
    ):
        term = (m.group(1) or m.group(2)).strip()
        if len(term) >= 3:
            candidates.add(term)

    result = sorted(candidates)
    if result:
        print(f"[DomainKB] Extracted {len(result)} document-specific terms.")
    return result


def _is_useful_term(term: str) -> bool:
    """Return True if *term* is worth adding to the domain KB."""
    words = term.split()
    if len(words) < _MIN_CHUNK_TOKENS:
        return False
    signal_words = [w for w in words if w.lower() not in _NOISE_TOKENS]
    return len(signal_words) >= 1
