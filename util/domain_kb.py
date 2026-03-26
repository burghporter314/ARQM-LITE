"""
domain_kb.py — RAG-based domain suppression filter.

Two-layer knowledge base:
  1. Static layer  — generic technical terms loaded from domain_terms.json
  2. Document layer — noun phrases extracted from the uploaded requirements
                      document at analysis time

Before flagging a semantic violation, each detector checks both layers.
If the slot text matches a known term in either layer the violation is
suppressed, preventing project-specific vocabulary from being
misclassified as vague or problematic language.

Usage (inside a detector __init__):
    from domain_kb import DomainKnowledgeBase
    self.domain_kb = DomainKnowledgeBase.load(self.encoder)

Usage (inside analyze()):
    # build per-request KB from the document and pass it down
    doc_kb = self.domain_kb.augment(extract_document_terms(text, nlp))
    result = self._slots_to_violations(slots, sentence, doc_kb=doc_kb)

Usage (inside _slots_to_violations):
    if score >= threshold and not doc_kb.is_domain_term(text):
        violations.append(...)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    import spacy

_DEFAULT_TERMS_FILE = Path(__file__).parent / "domain_terms.json"

# Cosine-similarity threshold above which a slot value is treated as a
# known domain term and suppressed.
DEFAULT_SUPPRESSION_THRESHOLD = 0.82

# Minimum number of tokens in a noun chunk before it is treated as a
# candidate domain term worth indexing.
_MIN_CHUNK_TOKENS = 2

# Tokens that carry no domain signal on their own (used to filter chunks
# where every token is noise).
_NOISE_TOKENS = {
    "shall", "must", "should", "may", "will", "can", "would", "could",
    "the", "a", "an", "this", "that", "these", "those", "its", "their",
    "and", "or", "but", "of", "in", "on", "at", "to", "for", "with",
    "by", "from", "as", "be", "is", "are", "was", "were", "been",
    "have", "has", "had", "do", "does", "did",
    "system", "application", "service", "component", "module", "feature",
    "function", "requirement", "user", "data", "value", "item",
}


class DomainKnowledgeBase:
    """
    Encodes a corpus of known, precise domain terms and provides a fast
    similarity lookup used to suppress false-positive violations.

    Terms are encoded once at construction time; lookups re-encode only
    the query text (one forward pass per check).
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        terms: list[str],
        threshold: float = DEFAULT_SUPPRESSION_THRESHOLD,
    ) -> None:
        self._encoder  = encoder
        self.terms     = terms
        self.threshold = threshold
        if terms:
            self._embs: np.ndarray = encoder.encode(
                terms, normalize_embeddings=True, show_progress_bar=False
            )
        else:
            dim = encoder.get_sentence_embedding_dimension()
            self._embs = np.empty((0, dim), dtype=np.float32)

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        encoder: SentenceTransformer,
        path: str | Path = _DEFAULT_TERMS_FILE,
        threshold: float = DEFAULT_SUPPRESSION_THRESHOLD,
    ) -> "DomainKnowledgeBase":
        """Load static domain terms from *path* (JSON list or
        ``{"terms": [...]}``) and return a ready DomainKnowledgeBase."""
        p = Path(path)
        if not p.exists():
            print(f"[DomainKB] {p.name!r} not found — suppression disabled.")
            return cls(encoder, [], threshold)

        with open(p, encoding="utf-8") as f:
            raw = json.load(f)

        terms: list[str] = raw if isinstance(raw, list) else raw.get("terms", [])
        print(f"[DomainKB] Loaded {len(terms)} static domain terms.")
        return cls(encoder, terms, threshold)

    def augment(self, extra_terms: list[str]) -> "DomainKnowledgeBase":
        """Return a new DomainKnowledgeBase combining this KB's terms with
        *extra_terms* (e.g. noun phrases extracted from a document).

        The original KB is not modified, making this safe to call per-request
        against a shared static KB.
        """
        if not extra_terms:
            return self
        combined = list(dict.fromkeys(self.terms + extra_terms))  # dedup, preserve order
        return DomainKnowledgeBase(self._encoder, combined, self.threshold)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_domain_term(self, text: str) -> bool:
        """Return ``True`` if *text* is semantically close enough to any
        known domain term (cosine similarity >= ``self.threshold``).
        """
        if self._embs.shape[0] == 0:
            return False
        emb = self._encoder.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )
        sims = (emb @ self._embs.T)[0]
        return bool(sims.max() >= self.threshold)


# ------------------------------------------------------------------
# Document-level term extraction
# ------------------------------------------------------------------

def extract_document_terms(text: str, nlp) -> list[str]:
    """Extract candidate domain terms from a requirements document.

    Uses spaCy noun chunks and named entities to surface multi-word
    technical phrases that are specific to the project being analysed.
    These are returned as a list of strings ready to be passed to
    ``DomainKnowledgeBase.augment()``.

    Args:
        text: Full document text (pre-extracted from PDF/DOCX/TXT).
        nlp:  A loaded spaCy Language model.

    Returns:
        Deduplicated list of candidate domain terms (lower-cased).
    """
    # Process in chunks to avoid spaCy's token limit
    max_chars = 100_000
    sample = text[:max_chars]

    doc = nlp(sample)
    candidates: set[str] = set()

    # Named entities — always high-signal
    for ent in doc.ents:
        term = ent.text.strip()
        if _is_useful_term(term):
            candidates.add(term)

    # Noun chunks — multi-word compound nouns
    for chunk in doc.noun_chunks:
        # Drop determiners / leading stopwords from the chunk
        tokens = [t for t in chunk if not t.is_space]
        while tokens and tokens[0].is_stop:
            tokens = tokens[1:]
        if len(tokens) < _MIN_CHUNK_TOKENS:
            continue
        term = " ".join(t.text for t in tokens).strip()
        if _is_useful_term(term):
            candidates.add(term)

    # Also extract capitalised compound terms via regex
    # (catches things like "AES-256", "OAuth 2.0", "JWT token" that
    #  spaCy may not recognise as entities)
    for m in re.finditer(
        r"\b([A-Z][A-Za-z0-9]*(?:[-/][A-Za-z0-9]+)+)\b"   # hyphenated: AES-256, TLS-1.3
        r"|"
        r"\b([A-Z]{2,}(?:\s+\d[\d.]*)?)\b",                # acronyms: HTTP, GDPR, HTTP 200
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
    # Reject if every word is noise
    signal_words = [w for w in words if w.lower() not in _NOISE_TOKENS]
    return len(signal_words) >= 1
