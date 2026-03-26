"""
batch_analyze.py
================
One-off analysis script: runs the current ARQM-LITE pipeline over every PDF
in the submissions folder and writes a baseline JSON dump for qualitative
analysis and future regression comparison.

Output
------
  batch_baseline.json   — full per-requirement results for every document
  batch_summary.json    — aggregate flag-rate stats per document and overall

Usage
-----
  python batch_analyze.py [input_dir] [output_file]
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT     = Path(__file__).parent
_TRAINING = _ROOT / "util" / "training"
for p in (_ROOT, _TRAINING):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# ── Imports ───────────────────────────────────────────────────────────────────
from util.ingestion            import extract_text
from util.identification       import identify_requirements
from util.analyzer             import analyze_requirements
from requirements_preprocessor import preprocess

# ── Config ────────────────────────────────────────────────────────────────────
_DEFAULT_INPUT = (
    r"C:\Users\burgh\OneDrive - The Pennsylvania State University"
    r"\Desktop\Desktop_Temp\submissions sweng894 fa24"
)

_DIMS     = ("ambiguity", "feasibility", "singularity", "verifiability")
_VIOLS    = {"ambiguity": "spans", "feasibility": "violations",
             "singularity": "violations", "verifiability": "violations"}
_IS_GOOD  = {"ambiguity": "is_ambiguous", "feasibility": "is_feasible",
             "singularity": "is_singular", "verifiability": "is_verifiable"}
_FLAGGED  = {"ambiguity": True,           "feasibility": False,
             "singularity": False,         "verifiability": False}


def _is_flagged(result_dict: dict, dim: str) -> bool:
    key = _IS_GOOD[dim]
    val = result_dict.get(key, False)
    return val if _FLAGGED[dim] else not val


def _serialize_result(sentence: str, results: dict) -> dict:
    entry = {"sentence": sentence, "dimensions": {}}
    for dim in _DIMS:
        d    = results[dim].to_dict()
        viols_key = _VIOLS[dim]
        entry["dimensions"][dim] = {
            "flagged":    _is_flagged(d, dim),
            "violations": [
                {
                    "text":       v.get("text", ""),
                    "slot":       v.get("slot", ""),
                    "reason":     v.get("reason", ""),
                    "score":      round(v.get("score", 0.0), 4),
                    "suggestion": v.get("suggestion"),
                }
                for v in d.get(viols_key, [])
            ],
        }
    return entry


def process_document(pdf_path: Path) -> dict:
    file_bytes = pdf_path.read_bytes()
    text       = extract_text(file_bytes, pdf_path.name)
    sentences  = preprocess(text)
    reqs       = identify_requirements(sentences)
    results    = analyze_requirements(reqs, document_text=text)

    serialized = [_serialize_result(r["sentence"], r) for r in results]

    # per-doc flag counts
    counts = {dim: sum(1 for r in serialized if r["dimensions"][dim]["flagged"])
              for dim in _DIMS}

    return {
        "file":         pdf_path.name,
        "n_sentences":  len(sentences),
        "n_reqs":       len(reqs),
        "flag_counts":  counts,
        "requirements": serialized,
        "error":        None,
    }


def main(argv: list[str]) -> None:
    input_dir   = Path(argv[1]) if len(argv) > 1 else Path(_DEFAULT_INPUT)
    output_file = Path(argv[2]) if len(argv) > 2 else _ROOT / "batch_baseline.json"

    if not input_dir.is_dir():
        print(f"[ERROR] Directory not found: {input_dir}")
        sys.exit(1)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDFs found in {input_dir}")
        sys.exit(1)

    print(f"\n{'='*64}")
    print(f"  ARQM-LITE Batch Baseline  ({len(pdfs)} documents)")
    print(f"{'='*64}\n")

    # Warm detectors once
    from util.analyzer import get_detectors
    print("[Batch] Warming detectors …")
    t0 = time.perf_counter()
    get_detectors()
    print(f"[Batch] Ready in {time.perf_counter()-t0:.1f}s\n")

    all_docs: list[dict] = []
    t_start = time.perf_counter()

    for i, pdf in enumerate(pdfs, 1):
        print(f"[{i:02d}/{len(pdfs)}] {pdf.name[:60]}")
        t_doc = time.perf_counter()
        try:
            doc = process_document(pdf)
            all_docs.append(doc)
            fc = doc["flag_counts"]
            print(
                f"        {doc['n_reqs']:3d} reqs  |  "
                f"Amb {fc['ambiguity']:3d}  "
                f"Feas {fc['feasibility']:3d}  "
                f"Sing {fc['singularity']:3d}  "
                f"Veri {fc['verifiability']:3d}  |  "
                f"{time.perf_counter()-t_doc:.1f}s"
            )
        except Exception:
            err = traceback.format_exc()
            all_docs.append({
                "file": pdf.name, "n_sentences": 0, "n_reqs": 0,
                "flag_counts": {d: 0 for d in _DIMS},
                "requirements": [], "error": err,
            })
            print(f"        *** FAILED ***\n{err}")

    # ── Write baseline JSON ────────────────────────────────────────────────
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_docs, f, indent=2, ensure_ascii=False)

    # ── Summary stats ─────────────────────────────────────────────────────
    good = [d for d in all_docs if not d["error"]]
    total_reqs  = sum(d["n_reqs"] for d in good)
    total_flags = {dim: sum(d["flag_counts"][dim] for d in good) for dim in _DIMS}

    summary = {
        "n_documents": len(pdfs),
        "n_processed": len(good),
        "total_reqs":  total_reqs,
        "total_flags": total_flags,
        "flag_rate":   {dim: round(total_flags[dim] / total_reqs, 3) if total_reqs else 0
                        for dim in _DIMS},
        "documents":   [
            {"file": d["file"], "n_reqs": d["n_reqs"], "flag_counts": d["flag_counts"]}
            for d in good
        ],
    }

    summary_file = output_file.parent / (output_file.stem + "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'='*64}")
    print(f"  Corpus totals  ({total_reqs} requirements, {elapsed:.0f}s)")
    print(f"  Ambiguity    : {total_flags['ambiguity']:4d}  ({summary['flag_rate']['ambiguity']*100:.1f}%)")
    print(f"  Feasibility  : {total_flags['feasibility']:4d}  ({summary['flag_rate']['feasibility']*100:.1f}%)")
    print(f"  Singularity  : {total_flags['singularity']:4d}  ({summary['flag_rate']['singularity']*100:.1f}%)")
    print(f"  Verifiability: {total_flags['verifiability']:4d}  ({summary['flag_rate']['verifiability']*100:.1f}%)")
    print(f"\n  Baseline: {output_file}")
    print(f"  Summary : {summary_file}")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main(sys.argv)
