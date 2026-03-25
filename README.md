# ARQM-LITE

**Automated Requirement Quality Measurement — Lightweight Edition**

ARQM-LITE is a Flask-based REST API that ingests a software requirements document (PDF, DOCX, or TXT), automatically identifies requirement sentences, and analyses each one across four quality dimensions. Results are returned as a downloadable PDF report.

---

## Quality Dimensions

| Dimension | What it checks |
|---|---|
| **Ambiguity** | Vague, imprecise, or unmeasurable language (e.g. "quickly", "sufficient", "easy to use") |
| **Feasibility** | Impossible absolutes, internal contradictions, or unrealistic thresholds (e.g. "100% uptime", "zero latency") |
| **Singularity** | Requirements that bundle multiple actions, actors, or concerns that should be stated separately |
| **Verifiability** | Requirements with no testable pass/fail condition or that rely on subjective judgement |

Each detector combines rule-based heuristics with semantic prototype scoring via SentenceTransformers, with per-slot thresholds calibrated on labelled data.

---

## Project Structure

```
ARQM-LITE/
├── app.py                          # Flask application entry point
├── main/
│   ├── __init__.py
│   └── routes.py                   # POST /analyze-quality endpoint
├── templates/
│   └── home.html
├── util/
│   ├── ingestion.py                # PDF / DOCX / TXT text extraction
│   ├── identification.py           # Requirement sentence identification (spaCy)
│   ├── analyzer.py                 # Quality analysis orchestrator
│   ├── report.py                   # PDF report generation (reportlab)
│   └── training/
│       ├── training_ambiguity.py   # Ambiguity detector
│       ├── training_feasibility.py # Feasibility detector
│       ├── training_singularity.py # Singularity detector
│       └── training_verifiability.py # Verifiability detector
├── datasets/
│   ├── requirement_identification/ # Labelled datasets for identification
│   └── requirement_quality/        # Labelled datasets for quality dimensions
├── calibration_data.json           # Ambiguity threshold calibration data
├── feasibility_calibration_data.json
├── verifiability_calibration_data.json
└── util/training/singularity_calibration_data.json
```

---

## Setup

### Prerequisites

- Python 3.10+
- pip

### Install dependencies

```bash
pip install flask \
            PyMuPDF \
            python-docx \
            reportlab \
            nltk \
            spacy \
            sentence-transformers
```

```bash
python -m spacy download en_core_web_sm
```

### Run the server

```bash
python app.py
```

The server starts at `http://localhost:5050`.

> **Note:** The first request triggers lazy initialisation of all four detectors (SentenceTransformer model loading). This takes 30–60 seconds. Subsequent requests are fast.

---

## API

### `POST /analyze-quality`

Accepts a requirements document and returns a PDF quality report.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | Yes | Requirements document (`.pdf`, `.docx`, `.doc`, `.txt`) |

**Response**

On success: `application/pdf` download named `ARQM_Report_<filename>.pdf`.

On error: JSON `{ "error": "..." }` with an appropriate HTTP status code.

**Example (curl)**

```bash
curl -X POST http://localhost:5050/analyze-quality \
  -F "file=@requirements.pdf" \
  --output ARQM_Report.pdf
```

**Example (Postman)**

1. Method: `POST`, URL: `http://localhost:5050/analyze-quality`
2. Body → form-data → key `file` (type: File) → select your document
3. Send → save the response as a PDF

---

## Report Structure

The generated PDF contains:

1. **Cover page** — document name and generation timestamp
2. **Summary table** — total requirements analysed and violation count per dimension
3. **Quality dimensions** — description of each dimension
4. **Detailed analysis** — one block per identified requirement:
   - Green header: no violations found
   - Red header: one or more violations found
   - Per-dimension issues listed with a plain-English title, explanation, and suggested fix

---

## How It Works

```
Uploaded document
       │
       ▼
  Text extraction          (PyMuPDF / python-docx / UTF-8)
       │
       ▼
  Sentence tokenisation    (NLTK sent_tokenize)
       │
       ▼
  Requirement identification  (spaCy: modal + verb pattern matching)
       │
       ▼
  Quality analysis (× 4 detectors)
  ├── Ambiguity     — slot parsing + semantic prototype scoring + syntactic rules
  ├── Feasibility   — impossible-absolute rules + contradiction detection + semantic scoring
  ├── Singularity   — conjunction/compound detection + mixed-concern rules + semantic scoring
  └── Verifiability — acceptance-criteria rules + subjectivity detection + semantic scoring
       │
       ▼
  PDF report generation    (reportlab)
```

### Detectors

Each detector follows the same architecture:

1. **Slot parser** — splits the requirement into structural slots: *subject*, *modal*, *action*, *object*, *condition*, *qualifier* using spaCy dependency parsing.
2. **Rule-based detection** — high-confidence, category-specific rules (e.g. detecting "100%" for feasibility, coordinating conjunctions for singularity).
3. **Semantic prototype scoring** — each slot is compared to prototype embeddings (e.g. vague vs. precise phrases) using cosine similarity. Scores are normalised via sigmoid and compared against per-slot calibrated thresholds.
4. **Threshold calibration** — thresholds are optimised for F1 on a labelled validation split loaded from the `*_calibration_data.json` files.
