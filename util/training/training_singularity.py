"""
singularity_detector.py
=======================
Detects singularity violations in software requirement sentences.

Four violation categories:
  [A] Multiple actions       — sentence contains more than one discrete requirement
                                (e.g. "must validate input and log the result and notify the user")
  [B] Compound subjects      — multiple distinct actors in one requirement
                                (e.g. "the admin and the user must both confirm")
  [C] Conjunctive conditions — multiple independent trigger conditions
                                (e.g. "when A or B or C occurs, the system must...")
  [D] Mixed concerns         — functional and non-functional requirements combined
                                (e.g. "must encrypt data and respond within 200ms")

Architecture mirrors feasibility_detector.py and verifiability_detector.py:
  - Slot-based analysis via SlotParser (inlined)
  - Prototype embedding comparison (non-singular vs singular)
  - Per-slot calibrated thresholds via SingularityCalibrator
  - Rule-based detection for [A], [B], [C], [D]
  - Semantic prototype scoring fills gaps the rules miss
"""

import re
import json
import numpy as np
import spacy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Slot definitions
# ─────────────────────────────────────────────

SLOTS = ("subject", "modal", "action", "object", "condition", "qualifier")


@dataclass
class TokenSpan:
    """Character offsets of a span within the original sentence."""
    start: int
    end: int
    text: str


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


def find_token_spans(span_text: str, sentence: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    pattern = re.compile(re.escape(span_text), re.IGNORECASE)
    for m in pattern.finditer(sentence):
        spans.append(TokenSpan(start=m.start(), end=m.end(), text=m.group()))
    return spans


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
            slots.subject = " ".join(t.text for t in sorted(subj_token.subtree, key=lambda x: x.i))

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
            slots.action = " ".join(t.text for t in sorted(aux_tokens + [verb], key=lambda x: x.i))

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
            seen_idx: set[int] = set()
            ordered = []
            for t in cond_tokens:
                if t.i not in seen_idx:
                    seen_idx.add(t.i)
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
# Per-slot thresholds
# ─────────────────────────────────────────────

DEFAULT_SLOT_THRESHOLDS: dict[str, float] = {
    "qualifier": 0.55,
    "condition": 0.55,
    "object":    0.58,
    "action":    0.58,
    "modal":     0.95,
    "subject":   0.95,
}

NEUTRAL_SLOTS = {"subject", "modal"}

NEUTRAL_WORDS = {
    "respond", "provide", "deliver", "allow", "ensure", "handle", "support",
    "enable", "include", "contain", "display", "show", "use", "perform",
    "process", "send", "receive", "create", "update", "delete", "return",
    "call", "run", "execute", "store", "save", "load", "read", "write",
    "maintain", "manage", "monitor", "log", "notify", "trigger", "generate",
    "encrypt", "decrypt", "compress", "validate", "authenticate", "authorise",
    "offer", "print", "download", "upload", "import", "export", "parse",
    "system", "application", "interface", "user", "data", "file", "task",
    "report", "request", "response", "error", "event", "module", "service",
    "component", "function", "feature", "option", "setting", "value",
    "must", "shall", "should", "may", "will", "can",
    "all", "every", "each", "any", "some", "no", "the", "a", "an",
    "before", "after", "when", "while", "during", "until", "unless",
    "writing", "reading", "sending", "receiving", "processing",
    "disk", "memory", "network", "database", "cache", "queue",
    "transmission", "storage", "retrieval", "delivery",
}


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class SingularityViolation:
    """A single detected singularity violation within a requirement sentence."""
    text: str
    score: float                           # [0, 1] confidence
    slot: str
    reason: str = "semantic"              # "multiple_actions" | "compound_subject"
                                           # | "conjunctive_condition" | "mixed_concerns" | "semantic"
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
class SingularityResult:
    """Analysis result for a single requirement sentence."""
    sentence: str
    slots: Optional[RequirementSlots] = None
    violations: list[SingularityViolation] = field(default_factory=list)

    @property
    def semantic_score(self) -> float:
        sem = [v.score for v in self.violations if v.reason == "semantic"]
        return float(np.mean(sem)) if sem else 0.0

    @property
    def rule_score(self) -> float:
        rule = [v.score for v in self.violations if v.reason != "semantic"]
        return float(np.mean(rule)) if rule else 0.0

    @property
    def max_score(self) -> float:
        return max((v.score for v in self.violations), default=0.0)

    @property
    def is_singular(self) -> bool:
        return not bool(self.violations)

    def __str__(self) -> str:
        status = "✓  SINGULAR" if self.is_singular else "✗  NON-SINGULAR"
        lines  = [f"{self.sentence}", f"   {status}"]

        for v in self.violations:
            label = {
                "multiple_actions":      "multiple actions",
                "compound_subject":      "compound subject",
                "conjunctive_condition": "conjunctive condition",
                "mixed_concerns":        "mixed concerns",
                "semantic":              "semantic",
            }.get(v.reason, v.reason)

            lines.append(f"   [{label}]  '{v.text}'")
            if v.suggestion:
                lines.append(f"   → {v.suggestion}")
            highlighted = v.highlight(self.sentence)
            if highlighted != self.sentence:
                lines.append(f"   ↳ {highlighted}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialisable representation of the result.

        Shape:
        {
            "sentence": str,
            "is_singular": bool,
            "max_score": float,
            "violation_count": int,
            "violations": [
                {
                    "reason": str,
                    "slot": str,
                    "text": str,
                    "score": float,
                    "suggestion": str | None,
                    "highlighted": str         # sentence with >>flagged<< markers
                },
                ...
            ]
        }
        """
        return {
            "sentence":        self.sentence,
            "is_singular":     self.is_singular,
            "max_score":       round(self.max_score, 4),
            "violation_count": len(self.violations),
            "violations": [
                {
                    "reason":      v.reason,
                    "slot":        v.slot,
                    "text":        v.text,
                    "score":       round(v.score, 4),
                    "suggestion":  v.suggestion,
                    "highlighted": v.highlight(self.sentence),
                }
                for v in self.violations
            ],
        }


# ─────────────────────────────────────────────
# HTML report renderer
# ─────────────────────────────────────────────

_REASON_COLOUR: dict[str, str] = {
    "multiple_actions":      "#d97706",   # amber
    "compound_subject":      "#7c3aed",   # purple
    "conjunctive_condition": "#0891b2",   # cyan
    "mixed_concerns":        "#dc2626",   # red
    "semantic":              "#6b7280",   # gray
}

_REASON_LABEL: dict[str, str] = {
    "multiple_actions":      "Multiple actions",
    "compound_subject":      "Compound subject",
    "conjunctive_condition": "Conjunctive condition",
    "mixed_concerns":        "Mixed concerns",
    "semantic":              "Semantic",
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
    """Convert >>text<< markers to <mark> spans."""
    return re.sub(
        r">>(.+?)<<",
        r'<mark style="background:#fef08a;padding:1px 2px;border-radius:2px">\1</mark>',
        highlighted,
    )


def render_html(results: list["SingularityResult"], title: str = "Singularity Analysis") -> str:
    """
    Produce a self-contained HTML report for a list of SingularityResult objects.

    Usage:
        results = detector.analyze_many(sentences)
        html = render_html(results)
        with open("report.html", "w") as f:
            f.write(html)
    """
    total     = len(results)
    flagged   = sum(1 for r in results if not r.is_singular)
    singular  = total - flagged

    # ── summary bar ──────────────────────────────────────────────────────────
    summary_html = (
        f'<div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap">'
        f'<div style="padding:12px 20px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#16a34a">{singular}</div>'
        f'<div style="font-size:12px;color:#15803d">singular</div></div>'
        f'<div style="padding:12px 20px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#dc2626">{flagged}</div>'
        f'<div style="font-size:12px;color:#b91c1c">non-singular</div></div>'
        f'<div style="padding:12px 20px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#334155">{total}</div>'
        f'<div style="font-size:12px;color:#64748b">total</div></div>'
        f'</div>'
    )

    # ── per-result cards ──────────────────────────────────────────────────────
    cards_html = ""
    for r in results:
        border = "#fca5a5" if not r.is_singular else "#bbf7d0"
        badge_bg = "#fef2f2" if not r.is_singular else "#f0fdf4"
        badge_fg = "#dc2626" if not r.is_singular else "#16a34a"
        badge_text = "NON-SINGULAR" if not r.is_singular else "SINGULAR"

        violations_html = ""
        for v in r.violations:
            colour = _REASON_COLOUR.get(v.reason, "#6b7280")
            label  = _REASON_LABEL.get(v.reason, v.reason)
            violations_html += (
                f'<div style="margin-top:10px;padding:10px 12px;'
                f'background:#fafafa;border-left:3px solid {colour};border-radius:4px">'
                f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
                f'<span style="font-size:11px;font-weight:600;color:{colour};'
                f'text-transform:uppercase;letter-spacing:.5px">{label}</span>'
                f'<span style="font-size:11px;color:#9ca3af">slot: {v.slot}</span>'
                f'</div>'
                f'<div style="font-size:13px;color:#374151;margin-bottom:4px">'
                f'{_highlight_html(v.highlight(r.sentence))}</div>'
            )
            if v.suggestion:
                violations_html += (
                    f'<div style="font-size:12px;color:#6b7280;margin-top:4px">'
                    f'&#8594; {v.suggestion}</div>'
                )
            violations_html += _score_bar(v.score, colour)
            violations_html += '</div>'

        cards_html += (
            f'<div style="margin-bottom:12px;padding:16px;border:1px solid {border};'
            f'border-radius:8px;background:#fff">'
            f'<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:12px">'
            f'<div style="font-size:14px;color:#1f2937;line-height:1.5;flex:1">{r.sentence}</div>'
            f'<span style="font-size:11px;font-weight:600;padding:3px 8px;'
            f'background:{badge_bg};color:{badge_fg};border-radius:12px;white-space:nowrap">'
            f'{badge_text}</span>'
            f'</div>'
            f'{violations_html}'
            f'</div>'
        )

    # ── legend ────────────────────────────────────────────────────────────────
    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px">'
    for reason, colour in _REASON_COLOUR.items():
        legend_html += (
            f'<span style="display:flex;align-items:center;gap:5px;font-size:12px;color:#374151">'
            f'<span style="width:10px;height:10px;border-radius:50%;'
            f'background:{colour};display:inline-block"></span>'
            f'{_REASON_LABEL[reason]}</span>'
        )
    legend_html += '</div>'

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
# Prototype lists
# ─────────────────────────────────────────────

NON_SINGULAR_PROTOTYPES = [
    # Multiple actions
    "must validate input and encrypt data and log the result",
    "shall authenticate and authorise and audit the request",
    "must save the file and notify the user and update the index",
    "should check permissions and apply rate limiting and return a response",
    "must parse the request and validate the schema and store the record",
    # Compound subjects
    "the admin and the user must both confirm",
    "the client and the server shall negotiate",
    "users and administrators and auditors must all have access",
    "the frontend and the backend must both validate",
    # Conjunctive conditions
    "when login fails or timeout occurs or session expires",
    "if the user is unauthenticated or unauthorised or suspended",
    "when memory is high or CPU is high or disk is full",
    "when request is malformed or missing or duplicate",
    # Mixed concerns
    "must encrypt all data and respond within 200ms",
    "shall authenticate the user and maintain 99.9% uptime",
    "must validate input and use no more than 256MB of memory",
    "should log errors and complete within 100 milliseconds",
    # General non-singularity signal
    "multiple requirements in one sentence",
    "compound requirement with multiple obligations",
    "several distinct concerns combined",
    "more than one testable condition",
]

SINGULAR_PROTOTYPES = [
    # Single clear action
    "the system must encrypt all data before writing to disk",
    "the API must respond within 200ms for 95% of requests",
    "the service must return HTTP 400 for invalid input",
    "the user must confirm their email address before logging in",
    "the job must run every Monday at 08:00 UTC",
    # Single subject
    "the payment service must process refunds within 3 business days",
    "the admin must approve requests before they are published",
    "the load balancer must distribute traffic across all healthy nodes",
    # Single condition
    "when the session expires the user must be redirected to the login page",
    "if input is invalid the system must return a 400 error",
    "after payment is confirmed the order must be created",
    # Single concern
    "the system must support PDF file downloads",
    "the cache must invalidate entries after 24 hours",
    "the log must record the user ID timestamp and action",
    # General singularity signal
    "single atomic requirement", "one testable obligation",
    "single actor single action", "one clear success condition",
]


# ─────────────────────────────────────────────
# Functional vs non-functional keyword sets
# (used by [D] mixed concerns detection)
# ─────────────────────────────────────────────

_FUNCTIONAL_VERBS = {
    "validate", "authenticate", "authorise", "authorize", "encrypt", "decrypt",
    "parse", "store", "save", "load", "read", "write", "send", "receive",
    "create", "update", "delete", "fetch", "query", "insert", "render",
    "display", "submit", "upload", "download", "import", "export", "generate",
    "notify", "redirect", "reject", "accept", "approve", "cancel", "publish",
    "log", "audit", "sign", "verify", "hash", "compress", "decompress",
}

# Non-functional indicators: numeric performance/quality constraints
_NFR_PATTERN = re.compile(
    r"\b(?:"
    r"\d+\s*(?:ms|milliseconds?|seconds?|minutes?|hours?)|"   # time bounds
    r"\d+\s*%\s*(?:uptime|availability|accuracy|success)|"    # percentage SLAs
    r"\d+\s*(?:mb|gb|kb|bytes?|rps|tps|req(?:uests?)?\s*per)|"  # resource/rate
    r"(?:p\d{2}|percentile|latency|throughput|uptime|availability|"
    r"memory\s+usage|cpu\s+usage|error\s+rate|response\s+time)"
    r")\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# [A] Multiple actions detector
# ─────────────────────────────────────────────

# Patterns that signal more than one discrete action in a single sentence.
# These look for coordinated verb phrases (V and V, V and V and V) after a modal.
_MA_RAW: list[tuple[str, str]] = [
    (
        r"\b(?:must|shall|should|will)\b"
        r"(?:\s+\w+){0,3}\s+\w+(?:ing|ed|e)?"
        r"(?:\s+(?:and|,))"
        r"(?:\s+\w+){0,3}\s+(?:and|,)\s+"
        r"(?:\w+(?:ing|e|ed)?)",
        "Sentence contains multiple distinct obligations; split into one requirement per action",
    ),
]

# More targeted patterns for common multi-action structures
_MA_TARGETED: list[tuple[str, str]] = [
    (
        r"\b(?:must|shall|should|will)\s+"
        r"(?:\w+\s+){0,4}\w+"
        r"(?:\s*,\s*(?:must|shall|should|will)\s+|\s+and\s+(?:also\s+)?(?:must|shall|should|will)\s+|\s+and\s+(?:also\s+)?)"
        r"(?:\w+\s+){0,4}\w+",
        "Sentence contains more than one modal obligation; split into separate requirements",
    ),
    (
        r"\b(?:must|shall|should|will)\s+(?:\w+\s+){1,5}(?:and|,)\s+(?:\w+\s+){1,5}(?:and|,)\s+\w+",
        "Three or more coordinated actions detected; each should be its own requirement",
    ),
]

# Simpler, more reliable conjunction-counting approach
def detect_multiple_actions(sentence: str, doc) -> list[SingularityViolation]:
    """
    Flag sentences where a modal governs more than one coordinated verb phrase.

    Strategy: count root-level conjoined verbs (dep_ == 'conj' on a VERB whose
    head is also a VERB). Two or more conjuncts on the same head = multiple actions.
    Also catches comma-separated modal clauses via regex fallback.
    """
    found: list[SingularityViolation] = []

    # spaCy-based: find verbs conjoined to the root verb
    root_verb = next((t for t in doc if t.dep_ == "ROOT" and t.pos_ == "VERB"), None)
    if root_verb:
        conjuncts = [t for t in doc if t.dep_ == "conj" and t.head == root_verb and t.pos_ == "VERB"]
        if len(conjuncts) >= 1:
            # Build a phrase spanning the root and all conjuncts
            all_verbs = [root_verb] + conjuncts
            phrases = " and ".join(t.text for t in sorted(all_verbs, key=lambda x: x.i))
            spans = []
            for v in all_verbs:
                for m in re.finditer(re.escape(v.text), sentence[v.idx:v.idx + len(v.text) + 1]):
                    spans.append(TokenSpan(v.idx, v.idx + len(v.text), v.text))
            found.append(SingularityViolation(
                text=phrases,
                score=0.88,
                slot="action",
                reason="multiple_actions",
                token_spans=spans,
                suggestion="Split into one requirement per action: each 'must X' becomes its own sentence",
            ))

    # Regex fallback: repeated modals in one sentence ("must X and must Y")
    repeated_modal = re.search(
        r"\b(must|shall|should|will)\b.{3,60}\b(must|shall|should|will)\b",
        sentence, re.IGNORECASE
    )
    if repeated_modal and not found:
        phrase = sentence[repeated_modal.start():repeated_modal.end()]
        found.append(SingularityViolation(
            text=phrase,
            score=0.90,
            slot="action",
            reason="multiple_actions",
            token_spans=[TokenSpan(repeated_modal.start(), repeated_modal.end(), phrase)],
            suggestion="Multiple modal clauses in one sentence; split into separate requirements",
        ))

    return found


# ─────────────────────────────────────────────
# [B] Compound subject detector
# ─────────────────────────────────────────────

def detect_compound_subjects(sentence: str, doc) -> list[SingularityViolation]:
    """
    Flag sentences where two or more distinct actors share one modal obligation.

    Uses spaCy to find conjoined nominal subjects (dep_ == 'conj' on a token
    whose head has dep_ in {nsubj, nsubjpass}), then falls back to regex for
    'X and Y must' patterns the parser might miss.
    """
    found: list[SingularityViolation] = []

    # spaCy: find subject conjuncts
    subj = next((t for t in doc if t.dep_ in {"nsubj", "nsubjpass"}), None)
    if subj:
        conjuncts = [t for t in doc if t.dep_ == "conj" and t.head == subj]
        if conjuncts:
            all_subj = [subj] + conjuncts
            phrase = " and ".join(t.text for t in sorted(all_subj, key=lambda x: x.i))
            spans = [TokenSpan(t.idx, t.idx + len(t.text), t.text) for t in all_subj]
            found.append(SingularityViolation(
                text=phrase,
                score=0.85,
                slot="subject",
                reason="compound_subject",
                token_spans=spans,
                suggestion="Assign each actor their own requirement; different actors may have different obligations",
            ))

    # Regex fallback: "the X and the Y must" or "X and Y shall"
    if not found:
        m = re.search(
            r"\b(?:the\s+)?\w+\s+and\s+(?:the\s+)?\w+\s+(?:must|shall|should|will|both)\b",
            sentence, re.IGNORECASE,
        )
        if m:
            phrase = sentence[m.start():m.end()]
            found.append(SingularityViolation(
                text=phrase,
                score=0.85,
                slot="subject",
                reason="compound_subject",
                token_spans=[TokenSpan(m.start(), m.end(), phrase)],
                suggestion="Assign each actor their own requirement; different actors may have different obligations",
            ))

    return found


# ─────────────────────────────────────────────
# [C] Conjunctive condition detector
# ─────────────────────────────────────────────

# Trigger words that mark the start of a condition clause
_CONDITION_STARTERS = r"(?:when|if|after|before|unless|until|while|whenever|in\s+the\s+event\s+that)"

def detect_conjunctive_conditions(sentence: str) -> list[SingularityViolation]:
    """
    Flag sentences where a condition clause contains multiple independent
    triggers joined by 'and' or 'or'.

    Examples:
      "when login fails or the session expires or the token is revoked"
      "if the user is unauthenticated or unauthorised"
    """
    found: list[SingularityViolation] = []

    # Find condition clauses that contain 'or' or 'and' joining sub-conditions
    pattern = re.compile(
        _CONDITION_STARTERS +
        r"\s+.{3,80}?\s+(?:or|and)\s+.{3,60}?"
        r"(?=\s*,|\s+(?:must|shall|should|will|the\s+\w+)|$)",
        re.IGNORECASE,
    )
    seen: set[str] = set()
    for m in pattern.finditer(sentence):
        phrase = sentence[m.start():m.end()].strip()
        if phrase.lower() in seen:
            continue
        # Filter: must contain at least one 'or'/'and' that splits genuine sub-conditions
        # (avoids flagging "when X is valid and stored" where "and" is not a condition join)
        conjuncts = re.findall(r"\b(?:or|and)\b", phrase, re.IGNORECASE)
        if len(conjuncts) < 1:
            continue
        seen.add(phrase.lower())
        found.append(SingularityViolation(
            text=phrase,
            score=0.83,
            slot="condition",
            reason="conjunctive_condition",
            token_spans=[TokenSpan(m.start(), m.end(), phrase)],
            suggestion="Split into one requirement per trigger condition; each scenario may require different behaviour",
        ))

    return found


# ─────────────────────────────────────────────
# [D] Mixed concerns detector
# ─────────────────────────────────────────────

def detect_mixed_concerns(sentence: str) -> list[SingularityViolation]:
    """
    Flag sentences that combine a functional obligation with a non-functional
    constraint (performance, availability, resource usage) in the same sentence.

    Detection: a functional verb appears AND a non-functional numeric/quality
    pattern appears in the same sentence, joined by a conjunction.
    """
    found: list[SingularityViolation] = []
    lower = sentence.lower()

    has_functional = any(v in lower for v in _FUNCTIONAL_VERBS)
    nfr_match = _NFR_PATTERN.search(sentence)

    if not has_functional or not nfr_match:
        return found

    # Require a conjunction between the functional and NFR parts — avoids
    # flagging sentences like "must encrypt data within 50ms" where the NFR
    # directly qualifies the single action rather than adding a second concern.
    conjunction = re.search(r"\b(?:and|,\s*and|while\s+also|as\s+well\s+as|in\s+addition)\b", sentence, re.IGNORECASE)
    if not conjunction:
        return found

    # Build a phrase spanning from just before the conjunction to the NFR match
    phrase_start = max(0, conjunction.start() - 30)
    phrase_end   = min(len(sentence), nfr_match.end() + 10)
    phrase = sentence[phrase_start:phrase_end].strip(" ,")

    # Find the functional verb for the other highlight
    func_match = next(
        (re.search(r"\b" + re.escape(v) + r"\b", sentence, re.IGNORECASE)
         for v in _FUNCTIONAL_VERBS if v in lower),
        None,
    )

    spans = [TokenSpan(nfr_match.start(), nfr_match.end(), nfr_match.group())]
    if func_match:
        spans.insert(0, TokenSpan(func_match.start(), func_match.end(), func_match.group()))

    found.append(SingularityViolation(
        text=phrase,
        score=0.82,
        slot="action",
        reason="mixed_concerns",
        token_spans=spans,
        suggestion=(
            "Separate the functional requirement from the non-functional constraint; "
            "each should be independently testable and traceable"
        ),
    ))

    return found


# ─────────────────────────────────────────────
# Contextual semantic scorer
# ─────────────────────────────────────────────

class ContextualSingularityScorer:
    """
    Scores (slot_text, sentence) pairs against non-singular/singular prototype
    embeddings. Span is contextualised as "slot_text [SEP] sentence".
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        non_singular_embs: np.ndarray,
        singular_embs: np.ndarray,
    ):
        self.encoder           = encoder
        self.non_singular_embs = non_singular_embs
        self.singular_embs     = singular_embs

    def score_slots_batch(self, slot_items: list[tuple[str, str]]) -> list[float]:
        contexts = [f"{span} [SEP] {sent}" for span, sent in slot_items]
        embs = self.encoder.encode(contexts, normalize_embeddings=True)
        ns   = (embs @ self.non_singular_embs.T).max(axis=1)
        s    = (embs @ self.singular_embs.T).max(axis=1)
        raw  = ns - s
        return [float(1 / (1 + np.exp(-r * 8))) for r in raw]


# ─────────────────────────────────────────────
# Threshold calibrator
# ─────────────────────────────────────────────

class SingularityCalibrator:
    """
    Learns per-slot thresholds from labelled singularity data by maximising
    F1 on the validation split.

    Calibration data format (JSON):
    {
        "train": [{"span": "...", "sentence": "...", "slot": "...", "label": 0}, ...],
        "val":   [...]
    }
    label=1 means the span represents a singularity violation.
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        non_singular_embs: np.ndarray,
        singular_embs: np.ndarray,
    ):
        self.encoder           = encoder
        self.non_singular_embs = non_singular_embs
        self.singular_embs     = singular_embs

    def fit(self, data_path: str) -> dict[str, float]:
        path = Path(data_path)
        if not path.exists():
            print(f"[SingularityCalibrator] '{data_path}' not found — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        with open(path) as f:
            data = json.load(f)

        train_records = data.get("train", [])
        val_records   = data.get("val",   [])

        if not val_records:
            print("[SingularityCalibrator] No val records — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        train_sents = {r["sentence"] for r in train_records}
        leaked = [r for r in val_records if r["sentence"] in train_sents]
        if leaked:
            print(
                f"[SingularityCalibrator] WARNING: {len(leaked)} val sentence(s) "
                f"also in train — results may be optimistic."
            )

        print(f"[SingularityCalibrator] Scoring {len(val_records)} val examples...")
        contexts = [f"{r['span']} [SEP] {r['sentence']}" for r in val_records]
        embs = self.encoder.encode(contexts, normalize_embeddings=True)
        ns   = (embs @ self.non_singular_embs.T).max(axis=1)
        s    = (embs @ self.singular_embs.T).max(axis=1)
        raw  = ns - s
        scores = 1 / (1 + np.exp(-raw * 8))

        for rec, score in zip(val_records, scores):
            rec["pred_score"] = float(score)

        thresholds: dict[str, float] = {}
        by_slot: dict[str, list] = {}
        for rec in val_records:
            by_slot.setdefault(rec["slot"], []).append(rec)

        for slot, records in by_slot.items():
            slot_scores    = np.array([r["pred_score"] for r in records])
            slot_labels    = np.array([r["label"]      for r in records])
            n_non_singular = int(slot_labels.sum())
            n_singular     = len(slot_labels) - n_non_singular

            if len(records) < 4 or n_non_singular == 0 or n_singular == 0:
                thresholds[slot] = DEFAULT_SLOT_THRESHOLDS.get(slot, 0.55)
                print(
                    f"  [{slot:12s}]  threshold={thresholds[slot]:.2f}  "
                    f"(default — n={len(records)}, "
                    f"non_singular={n_non_singular}, singular={n_singular})"
                )
                continue

            best_f1, best_t = 0.0, DEFAULT_SLOT_THRESHOLDS.get(slot, 0.55)
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
            print(
                f"  [{slot:12s}]  threshold={best_t:.2f}  "
                f"val_F1={best_f1:.3f}  n={len(records)}"
            )

        for slot in SLOTS:
            if slot not in thresholds:
                thresholds[slot] = DEFAULT_SLOT_THRESHOLDS.get(slot, 0.55)

        return thresholds


# ─────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────

class SingularityDetector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        calibration_data: str = "singularity_calibration_data.json",
        slot_thresholds: Optional[dict[str, float]] = None,
    ):
        self.nlp     = spacy.load(spacy_model)
        self.encoder = SentenceTransformer(model_name)

        self._non_singular_embs = self.encoder.encode(NON_SINGULAR_PROTOTYPES, normalize_embeddings=True)
        self._singular_embs     = self.encoder.encode(SINGULAR_PROTOTYPES,     normalize_embeddings=True)

        self.slot_parser = SlotParser(self.nlp)
        self.scorer      = ContextualSingularityScorer(
            self.encoder, self._non_singular_embs, self._singular_embs
        )

        if slot_thresholds:
            self.thresholds = slot_thresholds
        else:
            calibrator = SingularityCalibrator(
                self.encoder, self._non_singular_embs, self._singular_embs
            )
            self.thresholds = calibrator.fit(calibration_data)

        print(f"[SingularityDetector] Thresholds: {self.thresholds}")

    def _slots_to_violations(
        self, slots: RequirementSlots, sentence: str
    ) -> list[SingularityViolation]:
        filled = slots.filled_slots()
        if not filled:
            return []

        filtered = {
            slot: text for slot, text in filled.items()
            if slot not in NEUTRAL_SLOTS
            and not all(w.lower() in NEUTRAL_WORDS for w in text.split())
        }
        if not filtered:
            return []

        items  = list(filtered.items())
        scores = self.scorer.score_slots_batch(
            [(text, sentence) for _, text in items]
        )

        result: list[SingularityViolation] = []
        for (slot, text), score in zip(items, scores):
            threshold = self.thresholds.get(slot, 0.55)
            if score >= threshold:
                result.append(SingularityViolation(
                    text=text,
                    score=round(score, 4),
                    slot=slot,
                    reason="semantic",
                    token_spans=find_token_spans(text, sentence),
                    suggestion=None,
                ))
        return result

    def analyze(self, sentence: str) -> SingularityResult:
        doc   = self.nlp(sentence)
        slots = self.slot_parser.parse(sentence)

        rule_violations: list[SingularityViolation] = []
        rule_violations.extend(detect_multiple_actions(sentence, doc))
        rule_violations.extend(detect_compound_subjects(sentence, doc))
        rule_violations.extend(detect_conjunctive_conditions(sentence))
        rule_violations.extend(detect_mixed_concerns(sentence))

        semantic_violations = self._slots_to_violations(slots, sentence)

        seen: set[str] = {v.text.lower() for v in rule_violations}
        merged = list(rule_violations)
        for v in semantic_violations:
            if v.text.lower() not in seen:
                merged.append(v)
                seen.add(v.text.lower())

        merged.sort(key=lambda x: (0 if x.reason != "semantic" else 1, -x.score))

        return SingularityResult(sentence=sentence, slots=slots, violations=merged)

    def analyze_many(self, sentences: list[str]) -> list[SingularityResult]:
        return [self.analyze(s) for s in sentences]


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        # [A] Multiple actions
        "The system must validate input and encrypt the data and log the result.",
        "The service must authenticate the user and apply rate limiting and return a token.",
        "The API must parse the request, validate the schema, and store the record.",
        "The system must back up the database and notify the admin.",
        # [B] Compound subjects
        "The admin and the user must both confirm the deletion.",
        "The frontend and the backend must validate the input.",
        "Users and administrators must have access to the audit log.",
        # [C] Conjunctive conditions
        "When login fails or the session expires or the token is revoked, the user must be redirected.",
        "If the user is unauthenticated or unauthorised, the system must return HTTP 403.",
        "When memory usage is high or CPU usage is high, the service must scale out.",
        # [D] Mixed concerns
        "The system must encrypt all data and respond within 200ms.",
        "The service must authenticate the user and maintain 99.9% uptime.",
        "The API must validate the token and use no more than 256MB of memory.",
        # Clean — should pass
        "The API must respond within 200ms for 95% of requests under 1000 RPS.",
        "The system must encrypt all data before writing to disk.",
        "The service must return HTTP 400 for invalid input.",
        "When the session expires, the user must be redirected to the login page.",
        "The system must support downloading PDF files to disk.",
    ]

    detector = SingularityDetector(calibration_data="singularity_calibration_data.json")
    results  = detector.analyze_many(test_sentences)

    SEP = "=" * 70
    print(SEP)
    for r in results:
        print(r)
        print()

    print(SEP)
    print("SUMMARY")
    print(f"  {'':3}  {'score':>5}  sentence")
    print(f"  {'─'*3}  {'─'*5}  {'─'*60}")
    for r in sorted(results, key=lambda x: -x.max_score):
        flag = "✗" if not r.is_singular else "✓"
        score_str = f"{r.max_score:.2f}" if r.max_score > 0 else "   — "
        print(f"  {flag}    {score_str}  {r.sentence[:65]}")

    # ── structured export ─────────────────────────────────────────────────────
    import json as _json
    records = [r.to_dict() for r in results]
    with open("singularity_results.json", "w") as f:
        _json.dump(records, f, indent=2)
    print("\nStructured results written to singularity_results.json")

    # ── HTML report ───────────────────────────────────────────────────────────
    html = render_html(results, title="Singularity Analysis — Demo")
    with open("singularity_report.html", "w") as f:
        f.write(html)
    print("HTML report written to singularity_report.html")