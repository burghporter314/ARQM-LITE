"""
run_batch.py
============
Batch-process every PDF in a directory through the ARQM-LITE pipeline and
produce one PDF quality report per document.

Pipeline per document
---------------------
  1. Extract raw text      (util.ingestion)
  2. Preprocess            (requirements_preprocessor)
  3. Identify requirements (util.identification — modal-verb filter)
  4. Analyse quality       (util.analyzer — all four detectors)
  5. Generate PDF report   (generate_quality_report)

Usage
-----
  python run_batch.py  [input_dir]  [output_dir]

Defaults:
  input_dir  = C:/Users/burgh/OneDrive - The Pennsylvania State University/
               Desktop/Desktop_Temp/submissions sweng894 fa24
  output_dir = <input_dir>/ARQM_Reports_v2
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
_TRAINING = _ROOT / "util" / "training"
for p in (_ROOT, _TRAINING):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# ── Imports ───────────────────────────────────────────────────────────────────
from util.ingestion          import extract_text
from util.identification     import identify_requirements
from util.analyzer           import analyze_requirements, get_detectors
from requirements_preprocessor import preprocess
from generate_quality_report import generate_pdf

# ── Directories ───────────────────────────────────────────────────────────────
_DEFAULT_INPUT = (
    r"C:\Users\burgh\OneDrive - The Pennsylvania State University"
    r"\Desktop\Desktop_Temp\submissions sweng894 fa24"
)

def _resolve_dirs(argv: list[str]) -> tuple[Path, Path]:
    input_dir  = Path(argv[1]) if len(argv) > 1 else Path(_DEFAULT_INPUT)
    output_dir = Path(argv[2]) if len(argv) > 2 else input_dir / "ARQM_Reports_v2"
    return input_dir, output_dir


# ── Per-document processing ───────────────────────────────────────────────────

def process_pdf(
    pdf_path:   Path,
    output_dir: Path,
    detectors:  dict,
) -> dict:
    """
    Run the full pipeline on one PDF.  Returns a summary dict.
    """
    t0 = time.perf_counter()
    stem = pdf_path.stem

    # 1. Extract text
    file_bytes = pdf_path.read_bytes()
    raw_text   = extract_text(file_bytes, pdf_path.name)

    # 2. Preprocess
    candidates = preprocess(raw_text)

    # 3. Identify requirements (modal-verb filter)
    requirements = identify_requirements(candidates)

    # 4. Analyse
    raw_results = {
        name: det.analyze_many(requirements)
        for name, det in detectors.items()
    }

    # 5. Generate PDF report
    safe_stem   = stem.replace(" ", "_").replace("/", "_")
    report_path = output_dir / f"ARQM_{safe_stem}.pdf"
    generate_pdf(raw_results, output_path=str(report_path))

    elapsed = time.perf_counter() - t0

    # ── Collect summary stats ──────────────────────────────────────────────
    n_req = len(requirements)

    def _flagged(dim_key: str, is_good_key: str, invert: bool) -> int:
        results = raw_results[dim_key]
        dicts   = [r.to_dict() for r in results]
        if invert:   # ambiguity: is_ambiguous True → flagged
            return sum(1 for d in dicts if d.get(is_good_key))
        else:         # others: is_feasible False → flagged
            return sum(1 for d in dicts if not d.get(is_good_key))

    amb  = _flagged("ambiguity",     "is_ambiguous",  invert=True)
    feas = _flagged("feasibility",   "is_feasible",   invert=False)
    veri = _flagged("verifiability", "is_verifiable", invert=False)
    sing = _flagged("singularity",   "is_singular",   invert=False)

    return {
        "file":            pdf_path.name,
        "candidates":      len(candidates),
        "requirements":    n_req,
        "ambiguity":       amb,
        "feasibility":     feas,
        "verifiability":   veri,
        "singularity":     sing,
        "total_flags":     amb + feas + veri + sing,
        "elapsed_s":       round(elapsed, 1),
        "report":          report_path.name,
        "error":           None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str]) -> None:
    input_dir, output_dir = _resolve_dirs(argv)

    if not input_dir.is_dir():
        print(f"[ERROR] Input directory not found: {input_dir}")
        sys.exit(1)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[ERROR] No PDF files found in {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*64}")
    print(f"  ARQM-LITE Batch Processor")
    print(f"{'='*64}")
    print(f"  Input  : {input_dir}")
    print(f"  Output : {output_dir}")
    print(f"  PDFs   : {len(pdfs)}")
    print(f"{'='*64}\n")

    # Load detectors once — this is the slow step (~30-60 s)
    print("[Batch] Initialising detectors …")
    t_load = time.perf_counter()
    get_detectors()   # warms the singleton in util.analyzer

    # Build a local dict that matches generate_pdf expectations
    from util.analyzer import get_detectors as _gd
    amb_det, feas_det, sing_det, verif_det = _gd()
    detectors = {
        "ambiguity":     amb_det,
        "feasibility":   feas_det,
        "verifiability": verif_det,
        "singularity":   sing_det,
    }
    print(f"[Batch] Detectors ready in {time.perf_counter()-t_load:.1f}s\n")

    summaries: list[dict] = []

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i:02d}/{len(pdfs)}] {pdf_path.name}")
        try:
            summary = process_pdf(pdf_path, output_dir, detectors)
            summaries.append(summary)
            print(
                f"       {summary['requirements']} reqs  |  "
                f"Amb {summary['ambiguity']}  "
                f"Feas {summary['feasibility']}  "
                f"Veri {summary['verifiability']}  "
                f"Sing {summary['singularity']}  |  "
                f"{summary['elapsed_s']}s  ->  {summary['report']}"
            )
        except Exception:
            err = traceback.format_exc()
            summaries.append({
                "file": pdf_path.name, "candidates": 0, "requirements": 0,
                "ambiguity": 0, "feasibility": 0,
                "verifiability": 0, "singularity": 0,
                "total_flags": 0, "elapsed_s": 0, "report": "FAILED",
                "error": err,
            })
            print(f"       *** FAILED ***\n{err}")

    # ── Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"  Results Summary")
    print(f"{'='*64}")
    hdr = f"{'File':<42} {'Reqs':>4} {'Amb':>4} {'Feas':>4} {'Veri':>4} {'Sing':>4} {'Flags':>5}"
    print(hdr)
    print("-" * len(hdr))

    total_reqs  = 0
    total_flags = 0
    for s in summaries:
        name = s["file"][:41]
        if s["error"]:
            print(f"  {name:<41} ERROR")
        else:
            print(
                f"  {name:<41} "
                f"{s['requirements']:>4} "
                f"{s['ambiguity']:>4} "
                f"{s['feasibility']:>4} "
                f"{s['verifiability']:>4} "
                f"{s['singularity']:>4} "
                f"{s['total_flags']:>5}"
            )
            total_reqs  += s["requirements"]
            total_flags += s["total_flags"]

    print("-" * len(hdr))
    print(
        f"  {'TOTAL':<41} "
        f"{total_reqs:>4}    "
        f"                   "
        f"{total_flags:>5}"
    )
    print(f"\n  Reports written to: {output_dir}\n")


if __name__ == "__main__":
    main(sys.argv)
