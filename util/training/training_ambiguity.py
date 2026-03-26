"""
ambiguity_detector_v2.py
========================
Improvements over v1:
  [2] Cross-encoder context scoring  — span scored WITH full sentence as context,
      violation tokens mapped back to original offsets
  [3] Structured slot parsing        — sentence split into subject / modal /
      action / object / condition / qualifier before scoring
  [6] Calibrated per-slot thresholds — learned from labelled data via
      precision-recall optimisation; falls back to defaults if no data present
"""

import re
import json
import sys
import spacy
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from domain_kb import DomainKnowledgeBase

# ─────────────────────────────────────────────
# Slot definitions
# ─────────────────────────────────────────────

# Used to divide a requirement into typical parts via NLP.
SLOTS = ("subject", "modal", "action", "object", "condition", "qualifier")

# Default thresholds per slot — overridden after calibration.
DEFAULT_SLOT_THRESHOLDS: dict[str, float] = {
    "qualifier": 0.55,
    "condition": 0.52,
    "object":    0.50,
    "action":    0.48,
    "modal":     0.45,
    "subject":   0.60,
}

# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class TokenSpan:
    start: int
    end: int
    text: str

@dataclass
class AmbiguousSpan:
    text: str
    score: float
    slot: str
    reason: str = "semantic"
    token_spans: list[TokenSpan] = field(default_factory=list)
    suggestion: Optional[str] = None

    def highlight(self, sentence: str) -> str:
        if not self.token_spans:
            return sentence
        result = sentence
        offset = 0
        for ts in sorted(self.token_spans, key=lambda x: x.start):
            s, e = ts.start + offset, ts.end + offset
            result = result[:s] + ">>" + result[s:e] + "<<" + result[e:]
            offset += 4
        return result

@dataclass
class RequirementSlots:
    """Parsed structure of a single requirement sentence."""
    subject:   Optional[str] = None
    modal:     Optional[str] = None
    action:    Optional[str] = None
    object:    Optional[str] = None
    condition: Optional[str] = None
    qualifier: Optional[str] = None
    raw:       str = ""

    def filled_slots(self) -> dict[str, str]:
        return {s: getattr(self, s) for s in SLOTS if getattr(self, s)}

@dataclass
class AnalysisResult:
    """A result representative of an individual requirement."""
    sentence: str
    slots: Optional[RequirementSlots] = None
    ambiguous_spans: list[AmbiguousSpan] = field(default_factory=list)

    @property
    def semantic_score(self) -> float:
        sem = [s.score for s in self.ambiguous_spans if s.reason == "semantic"]
        return float(np.mean(sem)) if sem else 0.0

    @property
    def syntactic_score(self) -> float:
        syn = [s.score for s in self.ambiguous_spans if s.reason == "syntactic"]
        return float(np.mean(syn)) if syn else 0.0

    @property
    def is_ambiguous(self) -> bool:
        return bool(self.ambiguous_spans)

    def __str__(self) -> str:
        _SLOT_LABELS = {
            "subject":   "subject",
            "modal":     "obligation word",
            "action":    "action phrase",
            "object":    "object",
            "condition": "condition",
            "qualifier": "qualifier",
        }

        n = len(self.ambiguous_spans)
        if self.is_ambiguous:
            status = f"AMBIGUOUS  ({n} issue{'s' if n != 1 else ''} found)"
        else:
            status = "CLEAR — no ambiguity detected"

        lines = [
            f"Requirement : {self.sentence}",
            f"Ambiguity   : {status}",
        ]

        for i, s in enumerate(self.ambiguous_spans, 1):
            slot_label = _SLOT_LABELS.get(s.slot, s.slot)

            if s.reason == "syntactic":
                sg = s.suggestion or ""
                if "passive" in sg.lower():
                    title  = "Passive voice — unclear actor"
                    detail = (f'The action phrase "{s.text}" uses passive voice without '
                              f"specifying who performs this action.")
                elif "shall" in sg.lower():
                    title  = "Ambiguous obligation level"
                    detail = (f'"{s.text}" is ambiguous — "should" can mean either mandatory '
                              f'or recommended. Use "shall" for mandatory, "may" for optional.')
                elif "condition" in sg.lower():
                    title  = "Undefined condition boundary"
                    detail = (f'The condition "{s.text}" has no defined threshold or '
                              f"boundary value (e.g. 'under load > 1 000 RPS').")
                elif "baseline" in sg.lower():
                    title  = "Comparison without a reference"
                    detail = (f'"{s.text}" is a comparative term with no stated baseline. '
                              f"State what you are comparing against.")
                else:
                    title  = f"Imprecise {slot_label}"
                    detail = (f'"{s.text}" is a gradable term with no numeric bound '
                              f"or measurable criterion.")
            else:
                title  = f"Vague {slot_label}"
                detail = (f'"{s.text}" is imprecise and cannot be objectively measured '
                          f"or tested. Replace it with a specific, quantifiable value.")

            lines.append(f"\n  Issue {i} — {title}")
            lines.append(f"  {detail}")
            if s.suggestion:
                lines.append(f"  Suggested fix: {s.suggestion}")
            highlighted = s.highlight(self.sentence)
            if highlighted != self.sentence:
                lines.append(f"  In context: {highlighted}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialisable representation of the result.

        Shape:
        {
            "sentence": str,
            "is_ambiguous": bool,
            "semantic_score": float,
            "syntactic_score": float,
            "span_count": int,
            "spans": [
                {
                    "reason": str,
                    "slot": str,
                    "text": str,
                    "score": float,
                    "suggestion": str | None,
                    "highlighted": str
                },
                ...
            ]
        }
        """
        return {
            "sentence":        self.sentence,
            "is_ambiguous":    self.is_ambiguous,
            "semantic_score":  round(self.semantic_score, 4),
            "syntactic_score": round(self.syntactic_score, 4),
            "span_count":      len(self.ambiguous_spans),
            "spans": [
                {
                    "reason":      s.reason,
                    "slot":        s.slot,
                    "text":        s.text,
                    "score":       round(s.score, 4),
                    "suggestion":  s.suggestion,
                    "highlighted": s.highlight(self.sentence),
                }
                for s in self.ambiguous_spans
            ],
        }


# ─────────────────────────────────────────────
# HTML report renderer
# ─────────────────────────────────────────────

_REASON_COLOUR: dict[str, str] = {
    "syntactic": "#0891b2",
    "semantic":  "#d97706",
}

_SLOT_COLOUR: dict[str, str] = {
    "qualifier": "#7c3aed",
    "condition": "#0891b2",
    "object":    "#dc2626",
    "action":    "#d97706",
    "modal":     "#6b7280",
    "subject":   "#6b7280",
}


def _score_bar(score: float, colour: str) -> str:
    pct = int(score * 100)
    return (
        f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px">'
        f'<div style="flex:1;height:6px;background:#e5e7eb;border-radius:3px">'
        f'<div style="width:{pct}%;height:100%;background:{colour};border-radius:3px"></div>'
        f'</div>'
        f'<span style="font-size:11px;color:#6b7280;min-width:32px">{score:.2f}</span>'
        f'</div>'
    )


def _highlight_html(highlighted: str) -> str:
    return re.sub(
        r">>(.+?)<<",
        r'<mark style="background:#fef08a;padding:1px 2px;border-radius:2px">\1</mark>',
        highlighted,
    )


def render_html(results: list[AnalysisResult], title: str = "Ambiguity Analysis") -> str:
    """
    Produce a self-contained HTML report for a list of AnalysisResult objects.

    Usage:
        results = detector.analyze_many(sentences)
        html = render_html(results, title="My Requirements Review")
        with open("report.html", "w") as f:
            f.write(html)
    """
    total   = len(results)
    flagged = sum(1 for r in results if r.is_ambiguous)
    clear   = total - flagged

    summary_html = (
        f'<div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap">'
        f'<div style="padding:12px 20px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#16a34a">{clear}</div>'
        f'<div style="font-size:12px;color:#15803d">clear</div></div>'
        f'<div style="padding:12px 20px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#d97706">{flagged}</div>'
        f'<div style="font-size:12px;color:#b45309">ambiguous</div></div>'
        f'<div style="padding:12px 20px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#334155">{total}</div>'
        f'<div style="font-size:12px;color:#64748b">total</div></div>'
        f'</div>'
    )

    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px">'
    for label, colour in [("Syntactic", "#0891b2"), ("Semantic", "#d97706")]:
        legend_html += (
            f'<span style="display:flex;align-items:center;gap:5px;font-size:12px;color:#374151">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{colour};display:inline-block"></span>'
            f'{label}</span>'
        )
    for slot, colour in _SLOT_COLOUR.items():
        if slot in ("modal", "subject"):
            continue
        legend_html += (
            f'<span style="display:flex;align-items:center;gap:5px;font-size:12px;color:#374151">'
            f'<span style="width:10px;height:10px;border-radius:2px;background:{colour};display:inline-block"></span>'
            f'slot: {slot}</span>'
        )
    legend_html += '</div>'

    cards_html = ""
    for r in results:
        border    = "#fde68a" if r.is_ambiguous else "#bbf7d0"
        badge_bg  = "#fffbeb" if r.is_ambiguous else "#f0fdf4"
        badge_fg  = "#d97706" if r.is_ambiguous else "#16a34a"
        badge_txt = (
            f"⚠ {len(r.ambiguous_spans)} issue{'s' if len(r.ambiguous_spans) != 1 else ''}"
            if r.is_ambiguous else "✓ clear"
        )

        spans_html = ""
        for s in r.ambiguous_spans:
            reason_colour = _REASON_COLOUR.get(s.reason, "#6b7280")
            slot_colour   = _SLOT_COLOUR.get(s.slot, "#6b7280")
            spans_html += (
                f'<div style="margin-top:10px;padding:10px 12px;'
                f'background:#fafafa;border-left:3px solid {reason_colour};border-radius:4px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap">'
                f'<span style="font-size:11px;font-weight:600;color:{reason_colour};'
                f'text-transform:uppercase;letter-spacing:.5px">{s.reason}</span>'
                f'<span style="font-size:11px;padding:1px 6px;border-radius:10px;'
                f'background:{slot_colour}22;color:{slot_colour};font-weight:500">slot: {s.slot}</span>'
                f'</div>'
                f'<div style="font-size:13px;color:#374151;margin-bottom:4px">'
                f'{_highlight_html(s.highlight(r.sentence))}</div>'
            )
            if s.suggestion:
                spans_html += (
                    f'<div style="font-size:12px;color:#6b7280;margin-top:4px">'
                    f'&#8594; {s.suggestion}</div>'
                )
            spans_html += _score_bar(s.score, reason_colour)
            spans_html += '</div>'

        cards_html += (
            f'<div style="margin-bottom:12px;padding:16px;border:1px solid {border};'
            f'border-radius:8px;background:#fff">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">'
            f'<div style="font-size:14px;color:#1f2937;line-height:1.5;flex:1">{r.sentence}</div>'
            f'<span style="font-size:11px;font-weight:600;padding:3px 8px;'
            f'background:{badge_bg};color:{badge_fg};border-radius:12px;white-space:nowrap">'
            f'{badge_txt}</span>'
            f'</div>'
            f'{spans_html}'
            f'</div>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
  body {{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
         margin:0;padding:32px;background:#f8fafc;color:#1f2937}}
  h1   {{font-size:20px;font-weight:600;margin:0 0 4px}}
  p    {{font-size:13px;color:#6b7280;margin:0 0 24px}}
</style>
</head>
<body>
<h1>{title}</h1>
<p>{total} requirement{'s' if total != 1 else ''} analysed</p>
{summary_html}
{legend_html}
{cards_html}
</body>
</html>"""


# ─────────────────────────────────────────────
# [3] Slot Parser
# ─────────────────────────────────────────────

class SlotParser:
    MODAL_LEMMAS = {"shall", "must", "should", "may", "will", "can", "could", "might"}
    CONDITION_MARKERS = {
        "when", "while", "if", "unless", "until", "after", "before",
        "during", "under", "in case", "provided", "assuming",
    }

    def __init__(self, nlp):
        self.nlp = nlp

    def parse(self, sentence: str) -> RequirementSlots:
        doc = self.nlp(sentence)
        slots = RequirementSlots(raw=sentence)

        root = next((t for t in doc if t.dep_ == "ROOT"), None)
        if root is None:
            return slots

        subj_token = next((t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}), None)
        if subj_token:
            subtree_tokens = sorted(subj_token.subtree, key=lambda x: x.i)
            slots.subject = " ".join(t.text for t in subtree_tokens)

        modal_token = next(
            (t for t in doc if t.lemma_.lower() in self.MODAL_LEMMAS and t.pos_ in {"AUX", "VERB"}), None
        )
        if modal_token:
            slots.modal = modal_token.text

        verb = root if root.pos_ == "VERB" else next(
            (t for t in doc if t.pos_ == "VERB" and t != modal_token), None
        )
        if verb:
            aux_tokens = [t for t in doc if t.dep_ in {"aux", "auxpass"} and t.head == verb]
            action_tokens = sorted(aux_tokens + [verb], key=lambda x: x.i)
            slots.action = " ".join(t.text for t in action_tokens)

        obj_token = next((t for t in doc if t.dep_ in {"dobj", "attr", "oprd"}), None)
        if obj_token:
            subtree_tokens = [
                t for t in sorted(obj_token.subtree, key=lambda x: x.i)
                if t.dep_ not in {"prep", "relcl", "advcl"}
            ]
            slots.object = " ".join(t.text for t in subtree_tokens)

        cond_tokens = []
        for token in doc:
            if token.dep_ in {"advcl", "prep"} and token.head.pos_ == "VERB":
                if token.text.lower() in self.CONDITION_MARKERS or any(
                    c.text.lower() in self.CONDITION_MARKERS for c in token.subtree
                ):
                    cond_tokens.extend(sorted(token.subtree, key=lambda x: x.i))
        if cond_tokens:
            seen: set[int] = set()
            ordered = []
            for t in cond_tokens:
                if t.i not in seen:
                    seen.add(t.i)
                    ordered.append(t)
            slots.condition = " ".join(t.text for t in sorted(ordered, key=lambda x: x.i))

        condition_indices = {t.i for t in (cond_tokens or [])}
        qual_tokens = [
            t for t in doc
            if t.dep_ in {"advmod", "npadvmod", "prt"}
            and t.head.pos_ in {"VERB", "ADJ"}
            and t.i not in condition_indices
            and t.text.lower() not in {"not", "also", "only", "just", "even"}
        ]
        if qual_tokens:
            slots.qualifier = " ".join(t.text for t in sorted(qual_tokens, key=lambda x: x.i))

        return slots


# ─────────────────────────────────────────────
# [6] Threshold Calibrator
# ─────────────────────────────────────────────

class ThresholdCalibrator:
    def __init__(self, encoder: SentenceTransformer, vague_embs: np.ndarray, precise_embs: np.ndarray):
        self.encoder = encoder
        self.vague_embs = vague_embs
        self.precise_embs = precise_embs

    def fit(self, data_path: str) -> dict[str, float]:
        path = Path(data_path)
        if not path.exists():
            print(f"[Calibrator] '{data_path}' not found — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        with open(path) as f:
            data = json.load(f)

        train_records = data.get("train", [])
        val_records   = data.get("val",   [])

        if not val_records:
            print("[Calibrator] No val records found — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        train_sents = {r["sentence"] for r in train_records}
        leaked = [r for r in val_records if r["sentence"] in train_sents]
        if leaked:
            print(f"[Calibrator] WARNING: {len(leaked)} val sentence(s) also appear "
                  f"in train — results may be optimistic.")

        print(f"[Calibrator] Scoring {len(val_records)} val examples...")
        contexts = [f"{r['span']} [SEP] {r['sentence']}" for r in val_records]
        embs = self.encoder.encode(contexts, normalize_embeddings=True)
        v_sims = (embs @ self.vague_embs.T).max(axis=1)
        p_sims = (embs @ self.precise_embs.T).max(axis=1)
        raw    = v_sims - p_sims
        scores = 1 / (1 + np.exp(-raw * 8))

        for rec, score in zip(val_records, scores):
            rec["pred_score"] = float(score)

        thresholds: dict[str, float] = {}
        by_slot: dict[str, list] = {}
        for rec in val_records:
            by_slot.setdefault(rec["slot"], []).append(rec)

        for slot, records in by_slot.items():
            slot_scores = np.array([r["pred_score"] for r in records])
            slot_labels = np.array([r["label"]      for r in records])
            n_vague   = int(slot_labels.sum())
            n_precise = len(slot_labels) - n_vague

            if len(records) < 4 or n_vague == 0 or n_precise == 0:
                thresholds[slot] = DEFAULT_SLOT_THRESHOLDS.get(slot, 0.50)
                print(f"  [{slot:12s}]  threshold={thresholds[slot]:.2f}  "
                      f"(default — n={len(records)}, vague={n_vague}, precise={n_precise})")
                continue

            best_f1, best_t = 0.0, DEFAULT_SLOT_THRESHOLDS.get(slot, 0.50)
            for t in np.linspace(0.30, 0.80, 51):
                preds = (slot_scores >= t).astype(int)
                tp = int(((preds == 1) & (slot_labels == 1)).sum())
                fp = int(((preds == 1) & (slot_labels == 0)).sum())
                fn = int(((preds == 0) & (slot_labels == 1)).sum())
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1   = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
                if f1 > best_f1:
                    best_f1, best_t = f1, float(t)

            thresholds[slot] = best_t
            print(f"  [{slot:12s}]  threshold={best_t:.2f}  "
                  f"val_F1={best_f1:.3f}  n={len(records)}")

        for slot in SLOTS:
            if slot not in thresholds:
                thresholds[slot] = DEFAULT_SLOT_THRESHOLDS.get(slot, 0.50)

        return thresholds


# ─────────────────────────────────────────────
# Prototype lists
# ─────────────────────────────────────────────

VAGUE_PROTOTYPES = [
    # Time vagueness
    "quickly", "soon", "fast", "promptly", "immediately", "eventually",
    "as soon as possible", "near real-time", "in a timely manner",
    "in a timely fashion", "without unnecessary delay", "often",
    "sometimes", "occasionally", "regularly", "frequently", "rarely",
    "at all times", "whenever possible",
    # Quantity / degree vagueness
    "sufficient", "adequate", "minimal", "significant", "reasonable",
    "acceptable", "appropriate", "optimal", "good enough", "fast enough",
    "many", "few", "several", "large", "small", "some", "various",
    # Quality / correctness vagueness
    "accurate", "correct", "valid", "complete", "consistent", "precise",
    "reliable", "stable", "robust", "secure", "safe", "protected",
    "user-friendly", "intuitive", "easy to use", "seamless",
    "flexible", "scalable", "maintainable", "efficient",
    # Process adverb vagueness (extremely common in SRS documents)
    "properly", "appropriately", "correctly", "suitably", "adequately",
    "gracefully", "effectively", "successfully",
    # Conditionality vagueness
    "optional", "temporary", "as needed", "where applicable",
    "if necessary", "in most cases", "under normal conditions",
    "where possible", "as appropriate", "typically", "generally",
    "usually", "normally",
    # Meta-description anchors
    "vague requirement", "imprecise wording", "uncertain specification",
    "subjective description", "approximate value", "undefined behavior",
    "requirement without measurable criteria", "ambiguous statement",
    "no measurable threshold", "cannot be objectively tested",
]

PRECISE_PROTOTYPES = [
    # Performance
    "response time under 200 milliseconds",
    "p99 latency below 500 milliseconds",
    "process at least 1000 requests per second",
    "throughput of 10000 messages per minute",
    "timeout after 30 seconds",
    # Availability and reliability
    "99.9 percent uptime SLA",
    "MTBF greater than 2000 hours",
    "maximum 5 retry attempts with exponential backoff",
    "recovery time objective of 4 hours",
    # Error and correctness
    "error rate below 0.1 percent of requests",
    "output matches expected value to 3 decimal places",
    "passes all 847 regression tests in CI pipeline",
    "fewer than 1 critical defect per 1000 lines of code",
    # Data and storage
    "store up to 10 gigabytes of data per user",
    "log entry written within 1 second of event",
    "data retained for exactly 90 days then deleted",
    "maximum file upload size of 25 megabytes",
    # Security — concrete and testable
    "AES-256 encryption applied to all data at rest",
    "all session tokens expire after 30 minutes of inactivity",
    "zero P1 or P2 findings in annual penetration test",
    "password must be at least 12 characters with one symbol",
    # Scheduling and frequency
    "scheduled weekly on Monday at 08:00 UTC",
    "notify within 60 seconds of threshold breach",
    "batch job completes within 2 hours of trigger",
    # UI and usability — concrete
    "task completed by target user in 3 steps or fewer",
    "SUS usability score of 70 or above",
    "page load time under 3 seconds on a 10 Mbps connection",
    # Resource usage
    "memory usage under 512 MB under peak load",
    "CPU utilisation below 80 percent at 1000 concurrent users",
]

NEUTRAL_SINGLE_WORDS = {
    # Structural verbs with no requirement-quality signal
    "respond", "provide", "deliver", "allow", "ensure", "handle", "support",
    "enable", "include", "contain", "display", "show", "use", "perform",
    "process", "send", "receive", "create", "update", "delete", "return",
    "call", "run", "execute", "store", "save", "load", "read", "write",
    "maintain", "manage", "monitor", "log", "notify", "trigger", "generate",
    "encrypt", "decrypt", "compress", "validate", "authenticate", "authorise",
    "offer", "print", "download", "upload", "import", "export", "parse",
    "apply", "check", "verify", "confirm", "submit", "cancel", "open", "close",
    # Domain nouns
    "system", "application", "interface", "user", "data", "file", "task",
    "report", "request", "response", "error", "event", "module", "service",
    "component", "function", "feature", "option", "setting", "value",
    "algorithm", "transmission", "usage", "generated", "restarted",
    "database", "server", "client", "api", "endpoint", "record", "entry",
    "health", "checks", "monitor", "monitoring", "job", "jobs", "worker",
    "scheduler", "pipeline", "workflow", "batch", "daemon", "agent",
    # Modals and auxiliaries
    "must", "shall", "should", "may", "will", "can",
    "would", "could", "might", "ought",
    # Determiners and quantifiers
    "all", "every", "each", "any", "some", "no", "the", "a", "an",
    # Conjunctions and prepositions
    "before", "after", "when", "while", "during", "until", "unless",
    "and", "but", "or", "so", "as", "that", "with", "for", "to", "of",
    # Light verbs
    "have", "has", "had", "want", "wants", "let", "lets",
    "get", "gets", "got", "make", "makes", "made",
    "go", "goes", "went", "do", "does", "did",
    "be", "is", "are", "was", "were", "been",
    "seem", "seems", "feel", "look", "looks",
    # BDD / Gherkin extraction artifacts
    "given", "then", "scenario", "feature", "background",
    "story", "acceptance", "criteria", "priority", "estimation",
    "description", "mapped", "requirement",
    # Adverbs and particles with no signal value in qualifier slots
    "up", "down", "out", "on", "off", "in", "back", "away", "along",
    "together", "also", "here", "there", "now", "still", "just", "even",
    # NOTE: "never", "not", "high", "low" intentionally excluded —
    # these carry meaningful signal and should be scored.
}

# Pre-compiled pattern for detecting section-number / bullet artifacts in slot
# values.  A slot that matches this is not a requirement fragment and must be
# skipped before semantic scoring.
_ARTIFACT_RE = re.compile(r"^\s*[\d\.\•\-\*]+\s*$")

REWRITE_HINTS: dict[str, str] = {
    "quickly":       "within N ms/s (e.g. under 200 ms)",
    "fast":          "within N ms/s",
    "soon":          "within N seconds/minutes",
    "promptly":      "within N seconds of trigger",
    "immediately":   "within N ms of event (e.g. <= 100 ms)",
    "eventually":    "within N hours/days of occurrence",
    "timely":        "within a defined SLA (e.g. 4-hour response window)",
    "real-time":     "latency <= N ms end-to-end",
    "often":         "at least N times per hour/day",
    "regularly":     "every N hours/days on a fixed schedule",
    "occasionally":  "no more than N times per week",
    "sometimes":     "in N% of cases, defined by condition X",
    "frequently":    "at a rate of at least N events per minute",
    "sufficient":    "at least N [units]",
    "adequate":      "meeting threshold X (e.g. >= N MB free)",
    "minimal":       "less than N [units]",
    "significant":   "greater than N% / N [units]",
    "acceptable":    "within the range [min, max] per spec",
    "appropriate":   "conforming to standard X",
    "reasonable":    "benchmarked against specification X",
    "optimal":       "maximising metric M subject to constraint C",
    "efficient":     "uses <= N% CPU and <= N MB memory",
    "high":          "severity/priority level N (e.g. P1 on a P1-P4 scale)",
    "low":           "severity/priority level N (e.g. P4 on a P1-P4 scale)",
    "user-friendly": "passes usability test per ISO 9241 with >= N% task success",
    "intuitive":     "learnable within N minutes by target user without training",
    "easy":          "completable by target user in <= N steps / N minutes",
    "robust":        "recovers from failure within N seconds",
    "reliable":      "MTBF >= N hours; MTTR <= N minutes",
    "scalable":      "supports up to N users / N requests per second",
    "optional":      "configurable via flag X; default value is [on/off]",
    "temporary":     "deleted automatically after N minutes/hours",
    "configurable":  "settable via config key X with valid range [min, max]",
    "typically":     "in >= N% of cases under condition C",
    "generally":     "in >= N% of cases; exceptions documented in spec",
    "normally":      "under standard operating conditions defined in doc X",
    "passive_no_agent":        "specify who/what performs this action",
    "missing_quantifier":      "add a numeric threshold or measurable criterion",
    "unbounded_condition":     "specify the condition boundary (e.g. 'under load > N RPS')",
    "comparative_no_baseline": "state the baseline being compared against",
    "modal_ambiguity":         "clarify obligation: 'shall' (mandatory) vs 'should' (recommended) vs 'may' (optional)",
}


def get_suggestion(span_text: str, reason: str = "semantic") -> Optional[str]:
    if reason == "syntactic":
        return REWRITE_HINTS.get(span_text)
    for word in span_text.lower().split():
        if word in REWRITE_HINTS:
            return REWRITE_HINTS[word]
    return None


def find_token_spans(span_text: str, sentence: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    pattern = re.compile(re.escape(span_text), re.IGNORECASE)
    for m in pattern.finditer(sentence):
        spans.append(TokenSpan(start=m.start(), end=m.end(), text=m.group()))
    return spans if spans else []


# ─────────────────────────────────────────────
# Syntactic rules
# ─────────────────────────────────────────────

def detect_syntactic_ambiguities(sentence: str, doc) -> list[AmbiguousSpan]:
    found: list[AmbiguousSpan] = []
    text_lower = sentence.lower()
    claimed_verbs: set[int] = set()

    has_agent = any(tok.dep_ == "agent" for tok in doc)
    if not has_agent:
        for tok in doc:
            if tok.tag_ == "VBN" and any(c.dep_ == "auxpass" for c in tok.children):
                aux_tokens = [c for c in tok.children if c.dep_ in {"auxpass", "aux"}]
                phrase_tokens = sorted(aux_tokens + [tok], key=lambda x: x.i)
                phrase = " ".join(t.text for t in phrase_tokens)
                found.append(AmbiguousSpan(
                    text=phrase, score=0.45, slot="action", reason="syntactic",
                    token_spans=find_token_spans(phrase, sentence),
                    suggestion=REWRITE_HINTS["passive_no_agent"],
                ))
                claimed_verbs.add(tok.i)

    for tok in doc:
        if tok.lower_ == "should":
            head_verb = tok.head
            if head_verb.i not in claimed_verbs:
                phrase = f"should {head_verb.text}"
                found.append(AmbiguousSpan(
                    text=phrase, score=0.38, slot="modal", reason="syntactic",
                    token_spans=find_token_spans(phrase, sentence),
                    suggestion=REWRITE_HINTS["modal_ambiguity"],
                ))

    unbounded_patterns = [
        r"\bunder\s+(high|heavy|low|peak|normal|extreme|significant|increased)\s+\w+",
        r"\bin\s+(high|heavy|low|peak|extreme)\s+(load|traffic|demand|usage|stress)",
        r"\bduring\s+(peak|high|heavy)\s+\w+",
    ]
    for pattern in unbounded_patterns:
        for m in re.finditer(pattern, text_lower):
            phrase = sentence[m.start():m.end()]
            found.append(AmbiguousSpan(
                text=phrase, score=0.48, slot="condition", reason="syntactic",
                token_spans=[TokenSpan(m.start(), m.end(), phrase)],
                suggestion=REWRITE_HINTS["unbounded_condition"],
            ))

    for tok in doc:
        if tok.tag_ in {"JJR", "RBR"}:
            has_than = any(c.lower_ == "than" for c in tok.subtree)
            if not has_than:
                left  = doc[tok.i - 1].text if tok.i > 0 else ""
                right = doc[tok.i + 1].text if tok.i + 1 < len(doc) else ""
                context = " ".join(filter(None, [left, tok.text, right])).strip()
                found.append(AmbiguousSpan(
                    text=context, score=0.50, slot="qualifier", reason="syntactic",
                    token_spans=find_token_spans(tok.text, sentence),
                    suggestion=REWRITE_HINTS["comparative_no_baseline"],
                ))

    gradable_adjectives = {
        "accurate", "timely", "reliable", "consistent", "complete", "correct",
        "precise", "stable", "secure", "available", "responsive",
        "smooth", "clear", "simple", "comprehensive", "thorough", "detailed",
        "frequent", "infrequent", "complex", "lightweight",
    }
    for tok in doc:
        if tok.lower_ in gradable_adjectives and tok.pos_ == "ADJ":
            has_num = any(c.pos_ == "NUM" or c.dep_ == "quantmod" for c in tok.subtree)
            if not has_num:
                noun = tok.head if tok.head.pos_ == "NOUN" else None
                phrase = f"{tok.text} {noun.text}" if noun else tok.text
                found.append(AmbiguousSpan(
                    text=phrase, score=0.42, slot="qualifier", reason="syntactic",
                    token_spans=find_token_spans(phrase, sentence),
                    suggestion=REWRITE_HINTS["missing_quantifier"],
                ))

    seen: set[str] = set()
    deduped: list[AmbiguousSpan] = []
    for span in found:
        if span.text not in seen:
            seen.add(span.text)
            deduped.append(span)
    return deduped


# ─────────────────────────────────────────────
# Contextual semantic scorer
# ─────────────────────────────────────────────

class ContextualSemanticScorer:
    def __init__(self, encoder: SentenceTransformer, vague_embs: np.ndarray, precise_embs: np.ndarray):
        self.encoder = encoder
        self.vague_embs = vague_embs
        self.precise_embs = precise_embs

    def score_slot(self, slot_text: str, sentence: str) -> float:
        context = f"{slot_text} [SEP] {sentence}"
        emb = self.encoder.encode([context], normalize_embeddings=True)[0]
        v = float((emb @ self.vague_embs.T).max())
        p = float((emb @ self.precise_embs.T).max())
        return float(1 / (1 + np.exp(-(v - p) * 8)))

    def score_slots_batch(self, slot_items: list[tuple[str, str]]) -> list[float]:
        contexts = [f"{span} [SEP] {sent}" for span, sent in slot_items]
        embs = self.encoder.encode(contexts, normalize_embeddings=True)
        v = (embs @ self.vague_embs.T).max(axis=1)
        p = (embs @ self.precise_embs.T).max(axis=1)
        raw = v - p
        return [float(1 / (1 + np.exp(-r * 8))) for r in raw]


# ─────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────

class AmbiguityDetector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        calibration_data: str = "calibration_data.json",
        slot_thresholds: Optional[dict[str, float]] = None,
    ):
        self.nlp = spacy.load(spacy_model)
        self.encoder = SentenceTransformer(model_name)

        self._vague_embs   = self.encoder.encode(VAGUE_PROTOTYPES,  normalize_embeddings=True)
        self._precise_embs = self.encoder.encode(PRECISE_PROTOTYPES, normalize_embeddings=True)

        self.slot_parser = SlotParser(self.nlp)
        self.scorer = ContextualSemanticScorer(self.encoder, self._vague_embs, self._precise_embs)
        self.domain_kb = DomainKnowledgeBase.load(self.encoder)

        if slot_thresholds:
            self.thresholds = slot_thresholds
        else:
            calibrator = ThresholdCalibrator(self.encoder, self._vague_embs, self._precise_embs)
            self.thresholds = calibrator.fit(calibration_data)

        print(f"[Detector] Thresholds: {self.thresholds}")

    def _slots_to_spans(self, slots: RequirementSlots, sentence: str, doc_kb: "DomainKnowledgeBase | None" = None) -> list[AmbiguousSpan]:
        filled = slots.filled_slots()
        if not filled:
            return []

        filtered = {
            slot: text for slot, text in filled.items()
            if (slot not in {"subject"}
                or not any(w.lower() in NEUTRAL_SINGLE_WORDS for w in text.split()))
            and not _ARTIFACT_RE.match(text)
            and sum(1 for w in text.split() if w.lower() not in NEUTRAL_SINGLE_WORDS) >= 2
        }
        if not filtered:
            return []

        items  = list(filtered.items())
        scores = self.scorer.score_slots_batch(
            [(text, sentence) for _, text in items]
        )

        result: list[AmbiguousSpan] = []
        for (slot, text), score in zip(items, scores):
            threshold = self.thresholds.get(slot, 0.50)
            if score >= threshold and not self.domain_kb.is_domain_term(text) \
                    and not (doc_kb and doc_kb.is_domain_term(text)):
                result.append(AmbiguousSpan(
                    text=text,
                    score=round(score, 4),
                    slot=slot,
                    reason="semantic",
                    token_spans=find_token_spans(text, sentence),
                    suggestion=get_suggestion(text, "semantic"),
                ))
        return result

    def analyze(self, sentence: str, doc_kb: "DomainKnowledgeBase | None" = None) -> AnalysisResult:
        doc   = self.nlp(sentence)
        slots = self.slot_parser.parse(sentence)

        semantic_spans  = self._slots_to_spans(slots, sentence, doc_kb=doc_kb)
        syntactic_spans = detect_syntactic_ambiguities(sentence, doc)

        seen   = {s.text.lower() for s in syntactic_spans}
        merged = list(syntactic_spans)
        for s in semantic_spans:
            if s.text.lower() not in seen:
                merged.append(s)
                seen.add(s.text.lower())

        merged.sort(key=lambda x: (0 if x.reason == "syntactic" else 1, -x.score))
        return AnalysisResult(sentence=sentence, slots=slots, ambiguous_spans=merged)

    def analyze_many(self, sentences: list[str]) -> list[AnalysisResult]:
        return [self.analyze(s) for s in sentences]


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        "The system should respond quickly and provide sufficient feedback.",
        "Ensure minimal delay in data transmission under high load.",
        "The application must support optional user notifications.",
        "Provide accurate and timely reports for performance monitoring.",
        "The interface should be intuitive and user-friendly.",
        "All critical errors must be logged immediately.",
        "The system should handle high priority tasks efficiently.",
        "Temporary files must be cleaned automatically after use.",
        "Allow configurable settings with default options for simplicity.",
        "The algorithm must deliver optimal performance while minimizing resource usage.",
        "The API must respond within 200 ms for 95% of requests under 1000 RPS.",
        "Reports must be generated faster for better user experience.",
        "The service should be restarted when memory usage is high.",
        "Data must be encrypted before transmission.",
        "The system must support downloading PDF files to disk.",
        "The system must print with 3.75 mm P.L.A. filament.",
        "The application will provide users with a recommended route based on user preferences"
    ]

    detector = AmbiguityDetector(calibration_data="calibration_data.json")
    results  = detector.analyze_many(test_sentences)

    SEP = "=" * 70
    print(SEP)
    for r in results:
        print(r)
        print("-" * 70)

    print("\nSUMMARY (by semantic score)")
    print(f"{'Sem':>5}  {'Syn':>5}  Sentence")
    for r in sorted(results, key=lambda x: -x.semantic_score):
        flag = "⚠" if r.is_ambiguous else "✓"
        print(f"{r.semantic_score:>5.3f}  {r.syntactic_score:>5.3f}  {flag}  {r.sentence[:65]}")

    # ── structured export ─────────────────────────────────────────────────────
    import json as _json
    records = [r.to_dict() for r in results]
    with open("ambiguity_results.json", "w") as f:
        _json.dump(records, f, indent=2)
    print("\nStructured results written to ambiguity_results.json")

    # ── HTML report ───────────────────────────────────────────────────────────
    html = render_html(results, title="Ambiguity Analysis — Demo")
    with open("ambiguity_report.html", "w") as f:
        f.write(html)
    print("HTML report written to ambiguity_report.html")