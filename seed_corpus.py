"""
seed_corpus.py
==============
Pre-populate the corpus domain KB from the PURE dataset documents.

Processes every file in datasets/docs/, extracts domain-specific noun chunks
and named entities via the same pipeline used at request time, then writes
them all to util/corpus_domain_terms.json.

Clears the existing corpus first so stale/low-quality terms are replaced.

Usage
-----
  python seed_corpus.py [docs_dir]

Default docs_dir: datasets/docs
"""
from __future__ import annotations

import sys
import json
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).parent
for p in (_ROOT, _ROOT / "util" / "training"):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from util.ingestion import extract_text
from util.domain_kb import (
    extract_document_terms_from_doc,
    save_corpus_terms,
    _CORPUS_TERMS_FILE,
)

_DOCS_DIR = _ROOT / "datasets" / "docs"
_MAX_TERMS = 5_000  # larger cap than the per-request default of 1000


def _extract_text_any(path: Path) -> str:
    """Extract text from PDF/DOC/DOCX/HTML/HTM; skip RTF."""
    ext = path.suffix.lower()
    if ext in {".pdf", ".doc", ".docx", ".txt"}:
        return extract_text(path.read_bytes(), path.name)
    if ext in {".html", ".htm"}:
        try:
            from html.parser import HTMLParser

            class _Strip(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self._parts: list[str] = []
                def handle_data(self, data):
                    self._parts.append(data)

            p = _Strip()
            p.feed(path.read_text(encoding="utf-8", errors="replace"))
            return " ".join(p._parts)
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported format: {ext}")


def main(argv: list[str]) -> None:
    docs_dir = Path(argv[1]) if len(argv) > 1 else _DOCS_DIR

    if not docs_dir.is_dir():
        print(f"[ERROR] Directory not found: {docs_dir}")
        sys.exit(1)

    files = sorted(docs_dir.iterdir())
    print(f"\nSeed Corpus — {len(files)} files in {docs_dir}")
    print("=" * 60)

    # Clear existing corpus so stale terms are replaced
    _CORPUS_TERMS_FILE.write_text(json.dumps([], indent=2), encoding="utf-8")
    print("[seed] Cleared existing corpus.\n")

    # Load NLP once (reuse the ambiguity detector's nlp model)
    print("[seed] Loading NLP model…")
    from util.analyzer import get_detectors
    amb_det, *_ = get_detectors()
    nlp = amb_det.nlp
    print("[seed] NLP ready.\n")

    all_terms: list[str] = []
    ok = skipped = failed = 0

    for path in files:
        ext = path.suffix.lower()
        if ext in {".rtf"}:
            print(f"  SKIP  {path.name}  (RTF not supported)")
            skipped += 1
            continue

        t0 = time.perf_counter()
        try:
            text = _extract_text_any(path)
            if not text.strip():
                print(f"  EMPTY {path.name}")
                skipped += 1
                continue

            doc = nlp(text[:100_000])
            terms = extract_document_terms_from_doc(doc, text)
            all_terms.extend(terms)
            elapsed = time.perf_counter() - t0
            print(f"  OK    {path.name:<45} {len(terms):>4} terms  ({elapsed:.1f}s)")
            ok += 1

        except Exception as exc:
            print(f"  FAIL  {path.name}  — {exc}")
            failed += 1

    # Deduplicate preserving order, then save
    seen: set[str] = set()
    unique: list[str] = []
    for t in all_terms:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    save_corpus_terms(unique, max_terms=_MAX_TERMS)

    print(f"\n{'=' * 60}")
    print(f"  Processed : {ok} docs")
    print(f"  Skipped   : {skipped} docs")
    print(f"  Failed    : {failed} docs")
    print(f"  Raw terms : {len(all_terms)}")
    print(f"  Unique    : {len(unique)}")
    corpus = json.loads(_CORPUS_TERMS_FILE.read_text(encoding="utf-8"))
    print(f"  Saved     : {len(corpus)} terms (cap={_MAX_TERMS})")
    print()


if __name__ == "__main__":
    main(sys.argv)
