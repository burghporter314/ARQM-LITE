"""
util/entity_extraction.py
=========================
Extracts named entities from full document text using spaCy NER, grouped and
counted by label.  Only entity types meaningful in an SRS/requirements context
are retained.
"""
from __future__ import annotations

import re
from collections import Counter

import spacy

# ── Labels to retain ─────────────────────────────────────────────────────────
_KEEP_LABELS: set[str] = {
    "PERSON",    # stakeholders, roles
    "ORG",       # organisations, clients, vendors
    "PRODUCT",   # named software / hardware products
    "GPE",       # countries, cities (geographic constraints)
    "LOC",       # other locations
    "LAW",       # regulations, standards, acts
    "DATE",      # dates and periods
    "TIME",      # time references
    "CARDINAL",  # bare numerals (quantitative requirements)
    "PERCENT",   # percentage values
    "MONEY",     # monetary constraints
    "QUANTITY",  # measurements with units
    "NORP",      # nationality / regulatory bodies
    "EVENT",     # named events
    "FAC",       # facilities / physical systems
}

# ── Human-readable label display names ───────────────────────────────────────
LABEL_DISPLAY: dict[str, str] = {
    "PERSON":   "People / Roles",
    "ORG":      "Organisations",
    "PRODUCT":  "Products / Systems",
    "GPE":      "Locations",
    "LOC":      "Locations",
    "LAW":      "Standards & Regulations",
    "DATE":     "Dates",
    "TIME":     "Time References",
    "CARDINAL": "Numeric Values",
    "PERCENT":  "Percentages",
    "MONEY":    "Monetary Values",
    "QUANTITY": "Quantities",
    "NORP":     "Groups / Nationalities",
    "EVENT":    "Events",
    "FAC":      "Facilities",
}

# Accent colours for each display label (used in the HTML report)
LABEL_COLORS: dict[str, str] = {
    "People / Roles":           "#7c3aed",
    "Organisations":            "#0369a1",
    "Products / Systems":       "#0891b2",
    "Locations":                "#16a34a",
    "Standards & Regulations":  "#dc2626",
    "Dates":                    "#d97706",
    "Time References":          "#b45309",
    "Numeric Values":           "#475569",
    "Percentages":              "#6d28d9",
    "Monetary Values":          "#065f46",
    "Quantities":               "#1e40af",
    "Groups / Nationalities":   "#9f1239",
    "Events":                   "#c2410c",
    "Facilities":               "#166534",
}

# ── Internal spaCy singleton ──────────────────────────────────────────────────
_nlp = None

_MAX_CHUNK = 100_000   # characters per spaCy call (avoids memory issues on large docs)
_MIN_TEXT_LEN = 2      # ignore single-character entity texts


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ── Public API ────────────────────────────────────────────────────────────────

def extract_entities(text: str) -> dict[str, list[tuple[str, int]]]:
    """Return named entities from *text* grouped by display label.

    Each value is a list of ``(entity_text, count)`` tuples sorted by
    descending count.  Labels with no entities are omitted.
    """
    nlp = _get_nlp()
    chunks = [text[i : i + _MAX_CHUNK] for i in range(0, len(text), _MAX_CHUNK)]

    counts: dict[str, Counter] = {}

    for chunk in chunks:
        doc = nlp(chunk)
        _accumulate_entities(doc, counts)

    return {
        label: sorted(counter.items(), key=lambda kv: -kv[1])
        for label, counter in counts.items()
    }


def extract_entities_from_doc(doc) -> dict[str, list[tuple[str, int]]]:
    """Return named entities from a pre-parsed spaCy *doc*.

    Use this variant when the document has already been parsed (e.g. shared
    with domain term extraction) to avoid a second full NLP pass.
    Only processes the single doc span — ensure it covers the required text
    range before calling.
    """
    counts: dict[str, Counter] = {}
    _accumulate_entities(doc, counts)
    return {
        label: sorted(counter.items(), key=lambda kv: -kv[1])
        for label, counter in counts.items()
    }


def _accumulate_entities(doc, counts: dict) -> None:
    """Accumulate entity counts from *doc* into the *counts* dict in-place."""
    for ent in doc.ents:
        if ent.label_ not in _KEEP_LABELS:
            continue
        display = LABEL_DISPLAY.get(ent.label_, ent.label_)
        norm = re.sub(r"\s+", " ", ent.text.strip())
        if len(norm) < _MIN_TEXT_LEN:
            continue
        counts.setdefault(display, Counter())[norm] += 1
