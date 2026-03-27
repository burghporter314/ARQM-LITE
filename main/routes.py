import base64
import csv
import io
import json
import threading as _threading
from pathlib import Path as _Path

from flask import Blueprint, render_template, request, send_file, jsonify, Response

from util.ingestion              import extract_text
from util.identification         import identify_requirements
from util.analyzer               import analyze_full
from requirements_preprocessor   import preprocess
from generate_quality_report     import generate_pdf_bytes as generate_report
from generate_html_report        import generate_html_bytes as generate_html_report

main_bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}

# ── Annotation tool helpers ───────────────────────────────────────────────────

_calibration_lock = _threading.Lock()

_ROOT = _Path(__file__).parent.parent
_CALIBRATION_FILES = {
    'ambiguity':     _ROOT / 'calibration_data.json',
    'feasibility':   _ROOT / 'feasibility_calibration_data.json',
    'verifiability': _ROOT / 'verifiability_calibration_data.json',
    'singularity':   _ROOT / 'util' / 'training' / 'singularity_calibration_data.json',
}
_PURE_TRAIN_CSV = _ROOT / 'datasets' / 'requirement_identification' / 'PURE_train.csv'

_VALID_QUALITIES = {'ambiguity', 'feasibility', 'singularity', 'verifiability'}
_VALID_SLOTS     = {'subject', 'modal', 'action', 'object', 'condition', 'qualifier'}
_VALID_SPLITS    = {'train', 'val'}


def _calibration_counts() -> dict:
    counts = {}
    for quality, path in _CALIBRATION_FILES.items():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            counts[quality] = {'train': len(data.get('train', [])),
                               'val':   len(data.get('val', []))}
        except Exception:
            counts[quality] = {'train': 0, 'val': 0}
    return counts


def _append_calibration_entry(quality: str, split: str, entry: dict) -> dict:
    """Thread-safe append to the appropriate calibration JSON. Returns updated counts."""
    path = _CALIBRATION_FILES[quality]
    with _calibration_lock:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            data = {'train': [], 'val': []}
        data.setdefault('train', [])
        data.setdefault('val', [])
        data[split].append(entry)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return {'train': len(data['train']), 'val': len(data['val'])}


@main_bp.route('/')
def home():
    return render_template('home.html')


@main_bp.route('/about')
def about():
    return "ARQM-LITE — Automated Requirement Quality Measurement"


@main_bp.route('/analyze-quality', methods=['POST'])
def analyze_quality():
    """
    POST /analyze-quality
    Form-data fields:
      file  (required) — PDF, DOCX, or TXT requirement document
    Returns a PDF quality report as a file download.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file provided. Send the document as 'file' in form-data."}), 400

    uploaded = request.files['file']
    filename  = uploaded.filename or "document"

    from pathlib import Path
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type. Accepted: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    file_bytes = uploaded.read()

    # ── 1. Extract raw text ──────────────────────────────────────────────────
    try:
        text = extract_text(file_bytes, filename)
    except Exception as exc:
        return jsonify({"error": f"Failed to extract text: {exc}"}), 422

    # ── 2. Preprocess (split tables, filter Gherkin blocks, clean boilerplate) ──
    sentences = preprocess(text)
    if not sentences:
        return jsonify({"error": "No sentences could be extracted from the document."}), 422

    # ── 3. Requirement identification ────────────────────────────────────────
    requirements = identify_requirements(sentences)
    if not requirements:
        return jsonify({
            "error": "No requirements were identified in the document. "
                     "Requirements are expected to contain modal verbs such as "
                     "'shall', 'must', 'should', 'will', 'may', or 'can'."
        }), 422

    # ── 4. Quality analysis + entity extraction (single shared NLP parse) ────
    results, entities = analyze_full(requirements, document_text=text)

    stem = Path(filename).stem
    fmt  = request.args.get("format", "pdf").lower()

    # ── 5a. HTML report ───────────────────────────────────────────────────────
    if fmt == "html":
        html_bytes = generate_html_report(results, entities=entities)
        return Response(html_bytes, mimetype="text/html")

    # ── 5b. PDF report generation ─────────────────────────────────────────────
    pdf_bytes   = generate_report(results)
    report_name = f"ARQM_Report_{stem}.pdf"

    # ── 6. Optional JSON response with highlights ─────────────────────────────
    if request.args.get("json"):
        highlights = []
        for r in results:
            entry = {"sentence": r["sentence"]}
            for dim in ("ambiguity", "feasibility", "singularity", "verifiability"):
                d = r[dim].to_dict()
                viols_key = "spans" if dim == "ambiguity" else "violations"
                entry[dim] = [
                    {
                        "text":        v["text"],
                        "highlighted": v.get("highlighted", r["sentence"]),
                        "reason":      v["reason"],
                        "score":       v["score"],
                        "suggestion":  v.get("suggestion"),
                    }
                    for v in d.get(viols_key, [])
                ]
            highlights.append(entry)

        return jsonify({
            "report_name": report_name,
            "pdf":         base64.b64encode(pdf_bytes).decode(),
            "highlights":  highlights,
        })

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=report_name,
    )


@main_bp.route('/api/feedback', methods=['POST'])
def feedback():
    """
    POST /api/feedback
    JSON body: {"term": "<violation text to suppress>"}

    Saves the term as a confirmed false positive.  It will be included in the
    static domain KB layer on the next server restart, suppressing future
    detections of semantically similar text.
    """
    data = request.get_json(silent=True) or {}
    term = (data.get("term") or "").strip()
    if not term:
        return jsonify({"error": "No term provided"}), 400
    if len(term) > 500:
        return jsonify({"error": "Term too long (max 500 chars)"}), 400

    from util.domain_kb import save_feedback_term
    added = save_feedback_term(term)
    return jsonify({
        "ok":    True,
        "term":  term,
        "added": added,
        "note":  "Suppression takes effect on next server restart.",
    })


@main_bp.route('/annotate')
def annotate():
    """Annotation tool for building calibration data from the PURE train dataset."""
    requirements = []
    try:
        with _PURE_TRAIN_CSV.open(encoding='utf-8', newline='') as f:
            for row in csv.DictReader(f):
                if row.get('classification', '').strip() == 'T':
                    text = row.get('text', '').strip()
                    if text:
                        requirements.append(text)
    except FileNotFoundError:
        pass

    return render_template('annotate.html',
                           requirements=requirements,
                           counts=_calibration_counts())


@main_bp.route('/api/annotate', methods=['POST'])
def api_annotate():
    """Save a single annotation entry to the appropriate calibration JSON."""
    data    = request.get_json(silent=True) or {}
    quality = data.get('quality', '')
    split   = data.get('split', '')
    entry   = data.get('entry', {})

    if quality not in _VALID_QUALITIES:
        return jsonify({'ok': False, 'error': 'Invalid quality attribute'}), 400
    if split not in _VALID_SPLITS:
        return jsonify({'ok': False, 'error': 'split must be train or val'}), 400
    if not isinstance(entry.get('span'), str) or not entry['span'].strip():
        return jsonify({'ok': False, 'error': 'span is required'}), 400
    if not isinstance(entry.get('sentence'), str) or not entry['sentence'].strip():
        return jsonify({'ok': False, 'error': 'sentence is required'}), 400
    if entry.get('slot') not in _VALID_SLOTS:
        return jsonify({'ok': False, 'error': 'Invalid slot'}), 400
    if entry.get('label') not in (0, 1):
        return jsonify({'ok': False, 'error': 'label must be 0 or 1'}), 400

    totals = _append_calibration_entry(quality, split, {
        'span':     entry['span'].strip(),
        'sentence': entry['sentence'].strip(),
        'slot':     entry['slot'],
        'label':    int(entry['label']),
    })
    return jsonify({'ok': True, 'total': totals})
