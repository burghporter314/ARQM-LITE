import base64
import io

from flask import Blueprint, render_template, request, send_file, jsonify

from util.ingestion          import extract_text, extract_sentences
from util.identification     import identify_requirements
from util.analyzer           import analyze_requirements
from generate_quality_report import generate_pdf_bytes as generate_report

main_bp = Blueprint('main', __name__)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc", ".txt"}


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

    # ── 2. Sentence tokenisation ─────────────────────────────────────────────
    sentences = extract_sentences(text)
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

    # ── 4. Quality analysis ──────────────────────────────────────────────────
    results = analyze_requirements(requirements)

    # ── 5. PDF report generation ─────────────────────────────────────────────
    pdf_bytes = generate_report(results)

    stem = Path(filename).stem
    report_name = f"ARQM_Report_{stem}.pdf"

    # ── 6. Optional JSON response with highlights ─────────────────────────────
    # Add ?json=1 to the request URL to receive a JSON body containing the
    # base64-encoded PDF and per-requirement highlight data instead of a direct
    # file download.
    if request.args.get("json"):
        highlights = []
        for r in results:
            entry = {"sentence": r["sentence"]}
            for dim in ("ambiguity", "feasibility", "singularity", "verifiability"):
                d = r[dim].to_dict()
                # Collect only the highlighted strings from each violation/span
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
