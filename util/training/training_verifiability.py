"""
verifiability_detector.py
=========================
Detects verifiability violations in software requirement sentences.

Four violation categories:
  [A] No acceptance criteria   — requirement gives no testable pass/fail condition
                                  (e.g. "the system must be easy to use")
  [B] Subjective success       — success condition depends on human judgement
                                  (e.g. "users should be satisfied", "looks professional")
  [C] Missing actor/trigger    — unclear who or what initiates or observes verification
                                  (e.g. "errors must be handled appropriately")
  [D] Untestable negative      — absolute prohibition with no measurable bound
                                  (e.g. "the system must never lose data")

Architecture mirrors feasibility_detector.py:
  - Slot-based analysis via SlotParser (inlined)
  - Prototype embedding comparison (unverifiable vs verifiable)
  - Per-slot calibrated thresholds via VerifiabilityCalibrator
  - Rule-based detection for [A], [B], [C], [D]
  - Semantic prototype scoring fills gaps the rules miss
"""

import re
import json
import numpy as np
import spacy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
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
    "action":    0.60,
    "modal":     0.95,
    "subject":   0.95,
}

# Slots and words that carry no verifiability signal
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
class VerifiabilityViolation:
    """A single detected verifiability violation within a requirement sentence."""
    text: str
    score: float                           # [0, 1] confidence
    slot: str
    reason: str = "semantic"              # "no_acceptance_criteria" | "subjective_success"
                                           # | "missing_actor" | "untestable_negative" | "semantic"
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
class VerifiabilityResult:
    """Analysis result for a single requirement sentence."""
    sentence: str
    slots: Optional[RequirementSlots] = None
    violations: list[VerifiabilityViolation] = field(default_factory=list)

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
    def is_verifiable(self) -> bool:
        return not bool(self.violations)

    def __str__(self) -> str:
        status = "✓  VERIFIABLE" if self.is_verifiable else "✗  UNVERIFIABLE"
        lines  = [f"{self.sentence}", f"   {status}"]

        for v in self.violations:
            label = {
                "no_acceptance_criteria": "no acceptance criteria",
                "subjective_success":     "subjective success",
                "missing_actor":          "missing actor",
                "untestable_negative":    "untestable negative",
                "semantic":               "semantic",
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
            "is_verifiable": bool,
            "max_score": float,
            "violation_count": int,
            "violations": [
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
            "is_verifiable":   self.is_verifiable,
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
    "no_acceptance_criteria": "#d97706",
    "subjective_success":     "#7c3aed",
    "missing_actor":          "#0891b2",
    "untestable_negative":    "#dc2626",
    "semantic":               "#6b7280",
}

_REASON_LABEL: dict[str, str] = {
    "no_acceptance_criteria": "No acceptance criteria",
    "subjective_success":     "Subjective success",
    "missing_actor":          "Missing actor",
    "untestable_negative":    "Untestable negative",
    "semantic":               "Semantic",
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


def render_html(results: list[VerifiabilityResult], title: str = "Verifiability Analysis") -> str:
    """
    Produce a self-contained HTML report for a list of VerifiabilityResult objects.

    Usage:
        results = detector.analyze_many(sentences)
        html = render_html(results, title="My Requirements Review")
        with open("report.html", "w") as f:
            f.write(html)
    """
    total      = len(results)
    flagged    = sum(1 for r in results if not r.is_verifiable)
    verifiable = total - flagged

    summary_html = (
        f'<div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap">'
        f'<div style="padding:12px 20px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#16a34a">{verifiable}</div>'
        f'<div style="font-size:12px;color:#15803d">verifiable</div></div>'
        f'<div style="padding:12px 20px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#dc2626">{flagged}</div>'
        f'<div style="font-size:12px;color:#b91c1c">unverifiable</div></div>'
        f'<div style="padding:12px 20px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#334155">{total}</div>'
        f'<div style="font-size:12px;color:#64748b">total</div></div>'
        f'</div>'
    )

    legend_html = '<div style="display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px">'
    for reason, colour in _REASON_COLOUR.items():
        legend_html += (
            f'<span style="display:flex;align-items:center;gap:5px;font-size:12px;color:#374151">'
            f'<span style="width:10px;height:10px;border-radius:50%;background:{colour};display:inline-block"></span>'
            f'{_REASON_LABEL[reason]}</span>'
        )
    legend_html += '</div>'

    cards_html = ""
    for r in results:
        border   = "#fecaca" if not r.is_verifiable else "#bbf7d0"
        badge_bg = "#fef2f2" if not r.is_verifiable else "#f0fdf4"
        badge_fg = "#dc2626" if not r.is_verifiable else "#16a34a"
        badge_txt = (
            f"✗ {len(r.violations)} issue{'s' if len(r.violations) != 1 else ''}"
            if not r.is_verifiable else "✓ verifiable"
        )

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
            f'{badge_txt}</span>'
            f'</div>'
            f'{violations_html}'
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
# Prototype lists
# ─────────────────────────────────────────────

UNVERIFIABLE_PROTOTYPES = [
    # No acceptance criteria
    "easy to use", "user-friendly", "intuitive", "simple", "straightforward",
    "pleasant", "enjoyable", "nice", "good", "better", "improved",
    "as needed", "appropriately", "suitably", "where necessary",
    "in a reasonable manner", "to a satisfactory level",
    # Subjective success
    "users should be satisfied", "users will be happy", "meets user expectations",
    "looks professional", "feels responsive", "seems fast", "appears correct",
    "stakeholders are pleased", "customers are happy", "management approves",
    "positive user experience", "good user experience", "great experience",
    # Missing actor / passive unverifiable
    "handled appropriately", "dealt with correctly", "managed properly",
    "errors are handled", "exceptions are managed", "issues are resolved",
    "problems are addressed", "failures are dealt with",
    # Untestable negatives
    "must never lose", "must never fail", "must never crash",
    "shall not lose data", "should not have bugs", "must not have errors",
    "will never be unavailable", "must never be slow",
    # General unverifiability signal
    "unverifiable requirement", "no test criteria", "subjective requirement",
    "unmeasurable condition", "no observable outcome", "cannot be tested",
    "vague acceptance condition", "no pass fail criterion",
]

VERIFIABLE_PROTOTYPES = [
    # Clear acceptance criteria with numbers
    "response time under 200 milliseconds", "error rate below 0.1 percent",
    "99.9 percent uptime SLA", "process at least 1000 requests per second",
    "task completed within 3 steps", "measured by automated test suite",
    "verified by integration test", "confirmed by unit test passing",
    # Observable, objective outcomes
    "returns HTTP 200 status code", "log entry written within 1 second",
    "user redirected to dashboard on success", "alert sent within 60 seconds",
    "field highlighted in red when invalid", "button disabled after submission",
    # Clear actor and trigger
    "when user clicks submit", "after payment is confirmed",
    "when CPU usage exceeds 80 percent", "triggered by scheduled job at 03:00 UTC",
    "verified by QA engineer using test script",
    # Testable negatives with bounds
    "MTBF greater than 1000 hours", "maximum 1 data loss event per year",
    "fewer than 5 unhandled exceptions per day",
    "zero data loss under normal operating conditions as defined in test plan",
    # General verifiability signal
    "specific measurable acceptance criterion", "automated test verifies outcome",
    "observable system behaviour", "quantified pass fail threshold",
    "defined test procedure", "auditable log entry produced",
]


# ─────────────────────────────────────────────
# [A] No acceptance criteria rules
# ─────────────────────────────────────────────

# Phrases that indicate a requirement has no testable pass/fail condition.
# Each entry: (compiled_regex, suggestion)
_NAC_RAW: list[tuple[str, str]] = [
    (
        r"\b(?:as\s+needed|as\s+required|where\s+(?:necessary|applicable|appropriate))\b",
        "Replace '{phrase}' with a specific trigger condition (e.g. 'when request queue exceeds N')",
    ),
    (
        r"\b(?:appropriately|properly|correctly|suitably|adequately)\b",
        "'{phrase}' has no testable meaning; define what correct behaviour looks like (e.g. 'returns error code 400 for invalid input')",
    ),
    (
        r"\b(?:to\s+(?:a\s+)?(?:satisfactory|acceptable|adequate|sufficient)\s+(?:level|degree|standard|extent))\b",
        "Replace with a measurable threshold (e.g. 'task success rate >= 95% in usability test')",
    ),
    (
        r"\b(?:in\s+a\s+(?:reasonable|timely|appropriate|suitable)\s+(?:manner|way|fashion|timeframe))\b",
        "Replace with a defined SLA or numeric bound (e.g. 'within 500ms')",
    ),
    (
        r"\b(?:handle[sd]?|deal[st]?\s+with|manage[sd]?)\s+(?:errors?|exceptions?|failures?|faults?|problems?|issues?)\s+(?:appropriately|properly|correctly|gracefully|adequately)\b",
        "Define the observable outcome: which error codes are returned, what is logged, and what the user sees",
    ),
    (
        r"\b(?:best\s+(?:effort|practice|endeavour)|reasonable\s+effort)\b",
        "'{phrase}' is not testable; specify a minimum measurable obligation",
    ),
]

NO_ACCEPTANCE_CRITERIA_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), s) for p, s in _NAC_RAW
]


# ─────────────────────────────────────────────
# [B] Subjective success rules
# ─────────────────────────────────────────────

_SUBJ_RAW: list[tuple[str, str]] = [
    (
        r"\b(?:user[s\s-]*(?:friendly|centric|focused|oriented))\b",
        "Define a measurable usability criterion (e.g. 'target user completes task in <= 3 steps without training')",
    ),
    (
        r"\b(?:intuitive(?:ly)?|easy\s+to\s+(?:use|learn|understand|navigate|read))\b",
        "Define a testable learnability target (e.g. 'new user completes primary task within 5 minutes without documentation')",
    ),
    (
        r"\b(?:(?:look[s\s]+|appear[s\s]+|seem[s\s]+|feel[s\s]+)(?:professional|clean|modern|polished|responsive|fast|good|nice|pleasant))\b",
        "Subjective aesthetic — replace with an objective criterion or remove (e.g. passes accessibility audit WCAG 2.1 AA)",
    ),
    (
        r"\b(?:(?:positive|good|great|excellent|high.quality|improved)\s+(?:user\s+)?experience)\b",
        "Define how experience is measured (e.g. SUS score >= 70, NPS >= 40, task success rate >= 90%)",
    ),
    (
        r"\b(?:(?:users?|customers?|stakeholders?|management|clients?)\s+(?:are|will\s+be|should\s+be|must\s+be)\s+(?:satisfied|happy|pleased|content|delighted))\b",
        "Replace with a measurable satisfaction metric (e.g. CSAT >= 4.0/5.0, NPS >= 40)",
    ),
    (
        r"\b(?:meet[s\s]+(?:user|customer|stakeholder)\s+expectations?)\b",
        "Define what expectations means: which acceptance test, which persona, which scenario",
    ),
    (
        r"\b(?:(?:visually|aesthetically)\s+(?:appealing|pleasing|attractive|consistent|clean))\b",
        "Subjective visual criterion — define an objective standard (e.g. passes design review checklist, brand guidelines version N)",
    ),
]

SUBJECTIVE_SUCCESS_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), s) for p, s in _SUBJ_RAW
]


# ─────────────────────────────────────────────
# [C] Missing actor / trigger rules
# ─────────────────────────────────────────────

_ACT_RAW: list[tuple[str, str]] = [
    (
        r"\b(?:(?:errors?|exceptions?|failures?|faults?)\s+(?:must|shall|should|will|are)\s+be\s+(?:handled|caught|managed|dealt\s+with|addressed|resolved))\b",
        "Specify who handles it, what they do, and what the observable outcome is (e.g. 'service returns HTTP 500 and logs stack trace within 1 second')",
    ),
    (
        r"\b(?:(?:issues?|problems?|incidents?)\s+(?:must|shall|should|will|are)\s+be\s+(?:resolved|addressed|escalated|managed))\b",
        "Define resolution: by whom, within what timeframe, and what the confirmed state looks like",
    ),
    (
        r"\b(?:(?:data|records?|entries?)\s+(?:must|shall|should|will)\s+be\s+(?:validated|verified|checked|confirmed))\b",
        "Specify: what validation rules apply, what happens on failure, and what tool/test confirms this",
    ),
    (
        r"\b(?:(?:notifications?|alerts?|messages?)\s+(?:must|shall|should|will)\s+be\s+(?:sent|delivered|issued|generated))\b",
        "Specify who receives it, within what timeframe, and via what channel (e.g. 'email to on-call engineer within 60 seconds of threshold breach')",
    ),
    (
        r"\b(?:(?:access|permissions?|authorisation)\s+(?:must|shall|should|will)\s+be\s+(?:controlled|managed|enforced|restricted))\b",
        "Define who enforces access control, what roles exist, and how access decisions are audited",
    ),
]

MISSING_ACTOR_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), s) for p, s in _ACT_RAW
]


# ─────────────────────────────────────────────
# [D] Untestable negative rules
# ─────────────────────────────────────────────

_NEG_RAW: list[tuple[str, str]] = [
    (
        r"\b(?:(?:must|shall|will|should)\s+never\s+(?:lose|drop|delete|corrupt|overwrite)\s+(?:data|records?|messages?|events?|files?))\b",
        "Absolute data-loss prohibition is untestable; replace with a measurable bound (e.g. 'zero data loss under conditions defined in DR test plan' or 'RPO <= 1 hour')",
    ),
    (
        r"\b(?:(?:must|shall|will|should)\s+never\s+(?:fail|crash|go\s+down|become\s+unavailable|be\s+unavailable))\b",
        "Replace with a testable reliability target (e.g. MTBF >= 2000 hours, availability >= 99.9%)",
    ),
    (
        r"\b(?:(?:must|shall|will|should)\s+never\s+(?:expose|leak|disclose|reveal)\s+(?:sensitive|private|personal|confidential|user)\s+\w+)\b",
        "Define what 'never expose' means in test terms (e.g. 'penetration test finds zero P1/P2 data exposure vulnerabilities')",
    ),
    (
        r"\b(?:(?:must|shall|will|should)\s+(?:not|never)\s+(?:have|contain|include|produce)\s+(?:any\s+)?(?:bugs?|defects?|errors?|vulnerabilities?))\b",
        "Zero-defect clauses cannot be verified; replace with a defect density target (e.g. 'fewer than 1 critical defect per 1000 LOC at release')",
    ),
    (
        r"\b(?:(?:must|shall|will|should)\s+never\s+(?:be\s+)?(?:slow|unresponsive|delayed|blocked|unavailable))\b",
        "Replace with a testable SLA (e.g. 'p99 response time < 2s under peak load defined in load test scenario')",
    ),
    (
        r"\b(?:under\s+(?:no|any)\s+circumstances?\s+(?:should|shall|must|will)\s+(?:the\s+)?\w+\s+(?:lose|fail|crash|expose|leak))\b",
        "Absolute prohibition is untestable; quantify with a maximum failure rate or worst-case bound",
    ),
]

UNTESTABLE_NEGATIVE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), s) for p, s in _NEG_RAW
]


# ─────────────────────────────────────────────
# Rule detectors
# ─────────────────────────────────────────────

def _apply_rules(
    sentence: str,
    rules: list[tuple[re.Pattern, str]],
    reason: str,
    slot: str,
    score: float,
) -> list[VerifiabilityViolation]:
    """Generic rule application — shared by all four rule-based detectors."""
    found: list[VerifiabilityViolation] = []
    seen: set[str] = set()

    for pattern, suggestion_tmpl in rules:
        for m in pattern.finditer(sentence):
            phrase = sentence[m.start():m.end()]
            if phrase.lower() in seen:
                continue
            seen.add(phrase.lower())
            suggestion = suggestion_tmpl.replace("{phrase}", phrase)
            found.append(VerifiabilityViolation(
                text=phrase,
                score=score,
                slot=slot,
                reason=reason,
                token_spans=[TokenSpan(m.start(), m.end(), phrase)],
                suggestion=suggestion,
            ))

    # Keep only longest match when one phrase is a substring of another
    deduped: list[VerifiabilityViolation] = []
    for v in found:
        shadowed = any(
            v.text.lower() in other.text.lower() and v.text.lower() != other.text.lower()
            for other in found
        )
        if not shadowed:
            deduped.append(v)
    return deduped


def detect_no_acceptance_criteria(sentence: str) -> list[VerifiabilityViolation]:
    return _apply_rules(sentence, NO_ACCEPTANCE_CRITERIA_RULES, "no_acceptance_criteria", "qualifier", 0.88)


def detect_subjective_success(sentence: str) -> list[VerifiabilityViolation]:
    return _apply_rules(sentence, SUBJECTIVE_SUCCESS_RULES, "subjective_success", "object", 0.85)


def detect_missing_actor(sentence: str) -> list[VerifiabilityViolation]:
    return _apply_rules(sentence, MISSING_ACTOR_RULES, "missing_actor", "action", 0.82)


def detect_untestable_negatives(sentence: str) -> list[VerifiabilityViolation]:
    return _apply_rules(sentence, UNTESTABLE_NEGATIVE_RULES, "untestable_negative", "action", 0.90)


# ─────────────────────────────────────────────
# Contextual semantic scorer
# ─────────────────────────────────────────────

class ContextualVerifiabilityScorer:
    """
    Scores (slot_text, sentence) pairs against unverifiable/verifiable prototype
    embeddings. Span is contextualised as "slot_text [SEP] sentence".
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        unverifiable_embs: np.ndarray,
        verifiable_embs: np.ndarray,
    ):
        self.encoder           = encoder
        self.unverifiable_embs = unverifiable_embs
        self.verifiable_embs   = verifiable_embs

    def score_slots_batch(self, slot_items: list[tuple[str, str]]) -> list[float]:
        contexts = [f"{span} [SEP] {sent}" for span, sent in slot_items]
        embs = self.encoder.encode(contexts, normalize_embeddings=True)
        unv  = (embs @ self.unverifiable_embs.T).max(axis=1)
        ver  = (embs @ self.verifiable_embs.T).max(axis=1)
        raw  = unv - ver
        return [float(1 / (1 + np.exp(-r * 8))) for r in raw]


# ─────────────────────────────────────────────
# Threshold calibrator
# ─────────────────────────────────────────────

class VerifiabilityCalibrator:
    """
    Learns per-slot thresholds from labelled verifiability data by maximising
    F1 on the validation split.

    Calibration data format (JSON):
    {
        "train": [{"span": "...", "sentence": "...", "slot": "...", "label": 0}, ...],
        "val":   [...]
    }
    label=1 means the span represents a verifiability violation.
    """

    def __init__(
        self,
        encoder: SentenceTransformer,
        unverifiable_embs: np.ndarray,
        verifiable_embs: np.ndarray,
    ):
        self.encoder           = encoder
        self.unverifiable_embs = unverifiable_embs
        self.verifiable_embs   = verifiable_embs

    def fit(self, data_path: str) -> dict[str, float]:
        path = Path(data_path)
        if not path.exists():
            print(f"[VerifiabilityCalibrator] '{data_path}' not found — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        with open(path) as f:
            data = json.load(f)

        train_records = data.get("train", [])
        val_records   = data.get("val",   [])

        if not val_records:
            print("[VerifiabilityCalibrator] No val records — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        train_sents = {r["sentence"] for r in train_records}
        leaked = [r for r in val_records if r["sentence"] in train_sents]
        if leaked:
            print(
                f"[VerifiabilityCalibrator] WARNING: {len(leaked)} val sentence(s) "
                f"also in train — results may be optimistic."
            )

        print(f"[VerifiabilityCalibrator] Scoring {len(val_records)} val examples...")
        contexts = [f"{r['span']} [SEP] {r['sentence']}" for r in val_records]
        embs = self.encoder.encode(contexts, normalize_embeddings=True)
        unv  = (embs @ self.unverifiable_embs.T).max(axis=1)
        ver  = (embs @ self.verifiable_embs.T).max(axis=1)
        raw  = unv - ver
        scores = 1 / (1 + np.exp(-raw * 8))

        for rec, score in zip(val_records, scores):
            rec["pred_score"] = float(score)

        thresholds: dict[str, float] = {}
        by_slot: dict[str, list] = {}
        for rec in val_records:
            by_slot.setdefault(rec["slot"], []).append(rec)

        for slot, records in by_slot.items():
            slot_scores     = np.array([r["pred_score"] for r in records])
            slot_labels     = np.array([r["label"]      for r in records])
            n_unverifiable  = int(slot_labels.sum())
            n_verifiable    = len(slot_labels) - n_unverifiable

            if len(records) < 4 or n_unverifiable == 0 or n_verifiable == 0:
                thresholds[slot] = DEFAULT_SLOT_THRESHOLDS.get(slot, 0.55)
                print(
                    f"  [{slot:12s}]  threshold={thresholds[slot]:.2f}  "
                    f"(default — n={len(records)}, "
                    f"unverifiable={n_unverifiable}, verifiable={n_verifiable})"
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

class VerifiabilityDetector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        calibration_data: str = "verifiability_calibration_data.json",
        slot_thresholds: Optional[dict[str, float]] = None,
    ):
        self.nlp     = spacy.load(spacy_model)
        self.encoder = SentenceTransformer(model_name)

        self._unverifiable_embs = self.encoder.encode(UNVERIFIABLE_PROTOTYPES, normalize_embeddings=True)
        self._verifiable_embs   = self.encoder.encode(VERIFIABLE_PROTOTYPES,   normalize_embeddings=True)

        self.slot_parser = SlotParser(self.nlp)
        self.scorer      = ContextualVerifiabilityScorer(
            self.encoder, self._unverifiable_embs, self._verifiable_embs
        )

        if slot_thresholds:
            self.thresholds = slot_thresholds
        else:
            calibrator = VerifiabilityCalibrator(
                self.encoder, self._unverifiable_embs, self._verifiable_embs
            )
            self.thresholds = calibrator.fit(calibration_data)

        print(f"[VerifiabilityDetector] Thresholds: {self.thresholds}")

    def _slots_to_violations(
        self, slots: RequirementSlots, sentence: str
    ) -> list[VerifiabilityViolation]:
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

        result: list[VerifiabilityViolation] = []
        for (slot, text), score in zip(items, scores):
            threshold = self.thresholds.get(slot, 0.55)
            if score >= threshold:
                result.append(VerifiabilityViolation(
                    text=text,
                    score=round(score, 4),
                    slot=slot,
                    reason="semantic",
                    token_spans=find_token_spans(text, sentence),
                    suggestion=None,
                ))
        return result

    def analyze(self, sentence: str) -> VerifiabilityResult:
        slots = self.slot_parser.parse(sentence)

        rule_violations: list[VerifiabilityViolation] = []
        rule_violations.extend(detect_no_acceptance_criteria(sentence))
        rule_violations.extend(detect_subjective_success(sentence))
        rule_violations.extend(detect_missing_actor(sentence))
        rule_violations.extend(detect_untestable_negatives(sentence))

        semantic_violations = self._slots_to_violations(slots, sentence)

        seen: set[str] = {v.text.lower() for v in rule_violations}
        merged = list(rule_violations)
        for v in semantic_violations:
            if v.text.lower() not in seen:
                merged.append(v)
                seen.add(v.text.lower())

        merged.sort(key=lambda x: (0 if x.reason != "semantic" else 1, -x.score))

        return VerifiabilityResult(sentence=sentence, slots=slots, violations=merged)

    def analyze_many(self, sentences: list[str]) -> list[VerifiabilityResult]:
        return [self.analyze(s) for s in sentences]


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        # [A] No acceptance criteria
        "The system must handle errors appropriately.",
        "The module must manage exceptions as needed.",
        "The service must respond to failures in a reasonable manner.",
        "The component must deal with edge cases properly.",
        # [B] Subjective success
        "The interface must be intuitive and easy to use.",
        "The dashboard should look professional and visually appealing.",
        "The application must provide a positive user experience.",
        "Users must be satisfied with the checkout flow.",
        "The UI should feel responsive and clean.",
        # [C] Missing actor / trigger
        "Errors must be handled by the system.",
        "Notifications must be sent when a threshold is breached.",
        "Data must be validated before processing.",
        "Access must be controlled at the service boundary.",
        # [D] Untestable negatives
        "The system must never lose user data.",
        "The service shall never crash in production.",
        "The platform must not have any security vulnerabilities.",
        "The API must never be slow under load.",
        # Clean — should pass
        "The API must respond within 200ms for 95% of requests under 1000 RPS.",
        "The service must return HTTP 400 with a validation error body when input is malformed.",
        "The system must send an alert email to on-call within 60 seconds of a P1 incident.",
        "The login endpoint must reject requests with an invalid token and return HTTP 401.",
        "The system must support downloading PDF files to disk.",
    ]

    detector = VerifiabilityDetector(calibration_data="verifiability_calibration_data.json")
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
        flag = "✗" if not r.is_verifiable else "✓"
        score_str = f"{r.max_score:.2f}" if r.max_score > 0 else "   — "
        print(f"  {flag}    {score_str}  {r.sentence[:65]}")

    # ── structured export ─────────────────────────────────────────────────────
    import json as _json
    with open("verifiability_results.json", "w") as f:
        _json.dump([r.to_dict() for r in results], f, indent=2)
    print("\nStructured results written to verifiability_results.json")

    # ── HTML report ───────────────────────────────────────────────────────────
    with open("verifiability_report.html", "w") as f:
        f.write(render_html(results, title="Verifiability Analysis — Demo"))
    print("HTML report written to verifiability_report.html")