"""
feasibility_detector.py
=======================
Detects feasibility violations in software requirement sentences.

Three violation categories:
  [A] Impossible absolutes    — physically unachievable constraints
                                (100% uptime, zero latency, never fail)
  [B] Internal contradictions — mutually exclusive terms within one requirement
                                (synchronous and non-blocking)
  [C] Unrealistic thresholds  — numeric values outside known feasibility bounds
                                (sub-1ms network latency, uptime > 99.999%)

Architecture mirrors ambiguity_detector_v2.py:
  - Slot-based analysis via SlotParser (inlined)
  - Prototype embedding comparison (infeasible vs feasible)
  - Per-slot calibrated thresholds via FeasibilityCalibrator
  - Rule-based detection for [A], [B], and [C]
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
# Per-slot thresholds
# ─────────────────────────────────────────────

DEFAULT_SLOT_THRESHOLDS: dict[str, float] = {
    "qualifier": 0.55,
    "condition": 0.65,
    "object":    0.60,
    "action":    0.70,
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
class FeasibilityViolation:
    """A single detected feasibility violation within a requirement sentence."""
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
class FeasibilityResult:
    """Analysis result for a single requirement sentence."""
    sentence: str
    slots: Optional[RequirementSlots] = None
    violations: list[FeasibilityViolation] = field(default_factory=list)

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
    def is_feasible(self) -> bool:
        return not bool(self.violations)

    def __str__(self) -> str:
        n = len(self.violations)
        status = "FEASIBLE — no issues detected" if self.is_feasible else f"INFEASIBLE  ({n} issue{'s' if n != 1 else ''} found)"
        lines = [f"Requirement : {self.sentence}", f"Feasibility : {status}"]

        _TITLES = {
            "impossible_absolute":    "Impossible absolute",
            "internal_contradiction": "Internal contradiction",
            "unrealistic_threshold":  "Unrealistic threshold",
            "semantic":               "Potentially infeasible",
        }
        _DETAILS = {
            "impossible_absolute":    lambda t: f'"{t}" cannot be achieved in practice. No real system can guarantee this constraint.',
            "internal_contradiction": lambda t: f'"{t}" contains mutually exclusive terms that cannot both hold simultaneously.',
            "unrealistic_threshold":  lambda t: f'"{t}" exceeds known engineering limits. Use a threshold that can realistically be met.',
            "semantic":               lambda t: f'"{t}" resembles a constraint that is difficult or impossible to satisfy in practice.',
        }

        for i, v in enumerate(self.violations, 1):
            title  = _TITLES.get(v.reason, v.reason.replace("_", " ").title())
            detail = _DETAILS.get(v.reason, lambda t: f'"{t}"')(v.text)
            lines.append(f"\n  Issue {i} — {title}")
            lines.append(f"  {detail}")
            if v.suggestion:
                lines.append(f"  Suggested fix: {v.suggestion}")
            highlighted = v.highlight(self.sentence)
            if highlighted != self.sentence:
                lines.append(f"  In context: {highlighted}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """
        Serialisable representation of the result.

        Shape:
        {
            "sentence": str,
            "is_feasible": bool,
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
            "is_feasible":     self.is_feasible,
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
    "impossible_absolute":    "#dc2626",
    "internal_contradiction": "#7c3aed",
    "unrealistic_threshold":  "#d97706",
    "semantic":               "#6b7280",
}

_REASON_LABEL: dict[str, str] = {
    "impossible_absolute":    "Impossible absolute",
    "internal_contradiction": "Internal contradiction",
    "unrealistic_threshold":  "Unrealistic threshold",
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


def render_html(results: list[FeasibilityResult], title: str = "Feasibility Analysis") -> str:
    """
    Produce a self-contained HTML report for a list of FeasibilityResult objects.

    Usage:
        results = detector.analyze_many(sentences)
        html = render_html(results, title="My Requirements Review")
        with open("report.html", "w") as f:
            f.write(html)
    """
    total    = len(results)
    flagged  = sum(1 for r in results if not r.is_feasible)
    feasible = total - flagged

    summary_html = (
        f'<div style="display:flex;gap:16px;margin-bottom:24px;flex-wrap:wrap">'
        f'<div style="padding:12px 20px;background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#16a34a">{feasible}</div>'
        f'<div style="font-size:12px;color:#15803d">feasible</div></div>'
        f'<div style="padding:12px 20px;background:#fef2f2;border:1px solid #fecaca;border-radius:8px">'
        f'<div style="font-size:24px;font-weight:600;color:#dc2626">{flagged}</div>'
        f'<div style="font-size:12px;color:#b91c1c">infeasible</div></div>'
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
        border   = "#fecaca" if not r.is_feasible else "#bbf7d0"
        badge_bg = "#fef2f2" if not r.is_feasible else "#f0fdf4"
        badge_fg = "#dc2626" if not r.is_feasible else "#16a34a"
        badge_txt = (
            f"✗ {len(r.violations)} issue{'s' if len(r.violations) != 1 else ''}"
            if not r.is_feasible else "✓ feasible"
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

INFEASIBLE_PROTOTYPES = [
    "100% uptime", "zero downtime", "always available", "never fails",
    "zero latency", "instantaneous response", "instant processing",
    "unlimited throughput", "infinite scalability", "zero errors",
    "100% accuracy", "perfect reliability", "zero packet loss",
    "zero memory usage", "process infinite requests", "never crashes",
    "100% availability", "absolute guarantee of delivery",
    "real-time with zero delay", "no performance overhead",
    "synchronous and non-blocking", "stateless and maintain session",
    "real-time and batch processed", "encrypted and stored in plaintext",
    "sequential and parallel execution", "single-threaded and concurrent",
    "immutable and writable", "lossless and lossy",
    "physically impossible requirement", "unachievable constraint",
    "perfect performance", "simultaneous strong consistency and high availability",
]

FEASIBLE_PROTOTYPES = [
    "99.9% uptime SLA", "99.99% availability", "four nines availability",
    "maximum 8.7 hours downtime per year", "five nines 99.999% uptime",
    "planned maintenance window of 4 hours per month",
    "response time under 200 milliseconds", "p99 latency below 500ms",
    "round-trip latency under 50ms on LAN", "sub-second response time",
    "median latency of 20ms", "95th percentile latency under 300ms",
    "process 1000 requests per second", "500 transactions per second",
    "throughput of 10000 messages per minute",
    "memory usage under 512 MB", "CPU usage below 80 percent",
    "error rate below 0.1 percent", "maximum 5 seconds recovery time",
    "200ms latency at 1000 requests per second",
    "achievable performance target", "measurable and testable SLA",
    "realistic throughput requirement with defined hardware",
]


# ─────────────────────────────────────────────
# Contradiction pairs
# ─────────────────────────────────────────────

CONTRADICTION_PAIRS: list[tuple[frozenset[str], frozenset[str], str]] = [
    (
        frozenset({"synchronous", "sync"}),
        frozenset({"non-blocking", "nonblocking", "asynchronous", "async"}),
        "A call cannot be both synchronous (caller blocks) and non-blocking (caller continues); choose one model",
    ),
    (
        frozenset({"blocking"}),
        frozenset({"non-blocking", "nonblocking"}),
        "Blocking and non-blocking are mutually exclusive I/O models",
    ),
    (
        frozenset({"stateless"}),
        frozenset({"stateful", "maintain session", "maintain state", "session state", "persistent session"}),
        "Stateless and stateful are mutually exclusive; a stateless service cannot maintain session state",
    ),
    (
        frozenset({"real-time", "realtime", "real time"}),
        frozenset({"batch", "batch processing", "batch processed", "offline processing"}),
        "Real-time and batch processing are conflicting execution models; split into two separate requirements",
    ),
    (
        frozenset({"strongly consistent", "strong consistency", "linearizable"}),
        frozenset({"eventually consistent", "eventual consistency", "highly available"}),
        "The CAP theorem precludes simultaneous strong consistency and high availability under network partitions",
    ),
    (
        frozenset({"immutable"}),
        frozenset({"mutable", "writable", "modifiable", "updateable", "editable"}),
        "Immutable and mutable are contradictory; data cannot be both unchangeable and writable",
    ),
    (
        frozenset({"encrypted", "always encrypted"}),
        frozenset({"plaintext", "unencrypted", "in the clear", "clear text"}),
        "Data cannot be simultaneously encrypted and stored as plaintext",
    ),
    (
        frozenset({"lossless"}),
        frozenset({"lossy", "with loss", "compressed with loss"}),
        "Lossless and lossy are mutually exclusive compression strategies",
    ),
    (
        frozenset({"single-threaded", "single threaded", "single thread"}),
        frozenset({"multi-threaded", "multithreaded", "concurrent", "parallel execution"}),
        "Single-threaded and multi-threaded execution are mutually exclusive",
    ),
    (
        frozenset({"read-only", "readonly"}),
        frozenset({"writable", "read-write", "mutable"}),
        "Read-only and writable access are mutually exclusive",
    ),
    (
        frozenset({"atomic"}),
        frozenset({"non-atomic", "partial update", "partial write"}),
        "Atomic and non-atomic operations are mutually exclusive",
    ),
]


# ─────────────────────────────────────────────
# Impossible absolute patterns
# ─────────────────────────────────────────────

_ABS_RAW: list[tuple[str, str]] = [
    (
        r"\b100\s*%\s*(?:uptime|availability|reliable|reliability|success\s*rate|delivery\s*guarantee)",
        "100% {metric} is physically unachievable; use 99.9% (three nines) to 99.999% (five nines)",
    ),
    (
        r"\bzero\s+(?:latency|delay|response\s*time|lag)\b",
        "Zero latency is physically impossible; the speed of light alone imposes a minimum floor",
    ),
    (
        r"\b0\s*ms\s*(?:latency|response|delay)\b",
        "0ms latency is physically impossible; specify a realistic budget (e.g. < 10ms on LAN)",
    ),
    (
        r"\bzero\s+(?:downtime|outages?|failures?|errors?|defects?|bugs?)\b",
        "Zero {metric} is an absolute no real system can guarantee; specify a maximum rate or MTBF",
    ),
    (
        r"\bnever\s+(?:fail|crash|timeout|drop|lose|miss|become\s+unavailable)\b",
        "'{phrase}' is an absolute; replace with a maximum failure rate (e.g. MTBF >= N hours)",
    ),
    (
        r"\balways\s+(?:available|respond|succeed|be\s+available|be\s+up|be\s+online)\b",
        "'{phrase}' implies 100% availability; replace with an explicit SLA percentage",
    ),
    (
        r"\b(?:instantaneous|instant)\s+(?:response|processing|delivery|access|retrieval|startup)\b",
        "Instantaneous {metric} ignores physical transmission time, OS scheduling, and processing overhead",
    ),
    (
        r"\b(?:unlimited|infinite|unbounded)\s+(?:throughput|capacity|storage|bandwidth|scalability|concurrent\s+users?|connections?)\b",
        "'{phrase}' is unbounded; all real systems have resource limits — specify a maximum supported value",
    ),
    (
        r"\breal[-\s]?time\s+(?:with\s+)?(?:zero|no)\s+(?:delay|latency|lag|overhead)\b",
        "Real-time with zero delay is physically impossible; define an explicit latency budget (e.g. < 100ms)",
    ),
]

IMPOSSIBLE_ABSOLUTE_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), s) for p, s in _ABS_RAW
]


# ─────────────────────────────────────────────
# Numeric threshold rules
# ─────────────────────────────────────────────

@dataclass
class NumericRule:
    name: str
    value_pattern: str
    context_pattern: str
    is_violation: Callable[[float], bool]
    suggestion: str
    reason: str = "unrealistic_threshold"
    score: float = 0.85


NUMERIC_RULES: list[NumericRule] = [
    NumericRule(
        name="uptime_exactly_100",
        value_pattern=r"\b(100)\s*%",
        context_pattern=r"\b(?:uptime|availability|available|reliable|reliability)\b",
        is_violation=lambda v: v >= 100.0,
        suggestion="100% uptime is unachievable; use 99.9% (three nines, ~8.7h/yr downtime) to 99.999% (five nines, ~5min/yr)",
        reason="impossible_absolute",
        score=0.95,
    ),
    NumericRule(
        name="uptime_beyond_five_nines",
        value_pattern=r"\b(99\.(?:9{4,}|99\d*))\s*%",
        context_pattern=r"\b(?:uptime|availability|available)\b",
        is_violation=lambda v: v > 99.999,
        suggestion="Six nines (99.9999%) implies < 32 seconds downtime/year — verify your infrastructure can sustain this before committing",
        reason="unrealistic_threshold",
        score=0.75,
    ),
    NumericRule(
        name="sub_1ms_network_latency",
        value_pattern=r"\b(\d+(?:\.\d+)?)\s*ms\b",
        context_pattern=r"\b(?:response|latency|round.?trip|api|request|network|e2e|end.to.end)\b",
        is_violation=lambda v: 0 < v < 1.0,
        suggestion="Sub-millisecond network latency is only achievable for local in-process calls; LAN round-trips are typically 1–5ms, WAN 10–300ms",
        reason="unrealistic_threshold",
        score=0.80,
    ),
    NumericRule(
        name="zero_time_operation",
        value_pattern=r"\b(0(?:\.0+)?)\s*(?:ms|milliseconds?|μs|microseconds?|ns|nanoseconds?)\b",
        context_pattern=r"\b(?:response|latency|delay|time|round.?trip)\b",
        is_violation=lambda v: v == 0.0,
        suggestion="Zero-time operations are physically impossible; specify the minimum acceptable latency",
        reason="impossible_absolute",
        score=0.95,
    ),
]


def _parse_number(s: str) -> Optional[float]:
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def detect_unrealistic_thresholds(sentence: str) -> list[FeasibilityViolation]:
    found: list[FeasibilityViolation] = []
    seen: set[str] = set()

    for rule in NUMERIC_RULES:
        if not re.search(rule.context_pattern, sentence, re.IGNORECASE):
            continue
        for m in re.finditer(rule.value_pattern, sentence, re.IGNORECASE):
            val = _parse_number(m.group(1))
            if val is None or not rule.is_violation(val):
                continue
            phrase = sentence[m.start():m.end()]
            if phrase.lower() in seen:
                continue
            seen.add(phrase.lower())
            found.append(FeasibilityViolation(
                text=phrase, score=rule.score, slot="qualifier", reason=rule.reason,
                token_spans=[TokenSpan(m.start(), m.end(), phrase)],
                suggestion=rule.suggestion,
            ))

    latency_m = re.search(r"\b(\d+(?:\.\d+)?)\s*ms\b", sentence, re.IGNORECASE)
    rps_m = re.search(
        r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:rps|requests?\s*per\s*second|req/s|tps|transactions?\s*per\s*second)\b",
        sentence, re.IGNORECASE,
    )
    if latency_m and rps_m:
        latency = _parse_number(latency_m.group(1))
        rps     = _parse_number(rps_m.group(1))
        if latency is not None and rps is not None and latency < 10.0 and rps > 100_000:
            phrase = f"{latency_m.group()} at {rps_m.group()}"
            if phrase.lower() not in seen:
                seen.add(phrase.lower())
                found.append(FeasibilityViolation(
                    text=phrase, score=0.82, slot="condition", reason="unrealistic_threshold",
                    token_spans=[
                        TokenSpan(latency_m.start(), latency_m.end(), latency_m.group()),
                        TokenSpan(rps_m.start(), rps_m.end(), rps_m.group()),
                    ],
                    suggestion=(
                        f"Achieving {latency}ms latency at {rps:,.0f} RPS requires "
                        f"specialised infrastructure; verify with load testing and "
                        f"clarify whether this applies to p50 or p99 latency"
                    ),
                ))

    return found


def detect_impossible_absolutes(sentence: str) -> list[FeasibilityViolation]:
    found: list[FeasibilityViolation] = []
    seen: set[str] = set()

    for pattern, suggestion_tmpl in IMPOSSIBLE_ABSOLUTE_RULES:
        for m in pattern.finditer(sentence):
            phrase = sentence[m.start():m.end()]
            if phrase.lower() in seen:
                continue
            seen.add(phrase.lower())
            suggestion = (
                suggestion_tmpl
                .replace("{phrase}", phrase)
                .replace("{metric}", phrase.split()[-1] if phrase else "metric")
            )
            found.append(FeasibilityViolation(
                text=phrase, score=0.90, slot="qualifier", reason="impossible_absolute",
                token_spans=[TokenSpan(m.start(), m.end(), phrase)],
                suggestion=suggestion,
            ))

    deduped: list[FeasibilityViolation] = []
    for v in found:
        shadowed = any(
            v.text.lower() in other.text.lower() and v.text.lower() != other.text.lower()
            for other in found
        )
        if not shadowed:
            deduped.append(v)
    return deduped


def detect_internal_contradictions(sentence: str) -> list[FeasibilityViolation]:
    found: list[FeasibilityViolation] = []
    lower = sentence.lower()
    claimed: list[tuple[int, int]] = []

    def find_term(term: str) -> Optional[tuple[int, int]]:
        for m in re.finditer(re.escape(term), lower):
            s, e = m.start(), m.end()
            if not any(cs <= s and e <= ce for cs, ce in claimed):
                return s, e
        return None

    for set_a, set_b, suggestion in CONTRADICTION_PAIRS:
        pos_a, match_a = None, None
        for term in sorted(set_a, key=len, reverse=True):
            pos = find_term(term)
            if pos:
                pos_a, match_a = pos, term
                break

        if not match_a:
            continue

        claimed.append(pos_a)

        pos_b, match_b = None, None
        for term in sorted(set_b, key=len, reverse=True):
            pos = find_term(term)
            if pos:
                pos_b, match_b = pos, term
                break

        if not match_b:
            claimed.remove(pos_a)
            continue

        claimed.append(pos_b)

        spans = [
            TokenSpan(pos_a[0], pos_a[1], sentence[pos_a[0]:pos_a[1]]),
            TokenSpan(pos_b[0], pos_b[1], sentence[pos_b[0]:pos_b[1]]),
        ]
        found.append(FeasibilityViolation(
            text=f"{match_a} … {match_b}", score=0.92, slot="action",
            reason="internal_contradiction", token_spans=spans, suggestion=suggestion,
        ))

    return found


# ─────────────────────────────────────────────
# Contextual semantic scorer
# ─────────────────────────────────────────────

class ContextualFeasibilityScorer:
    def __init__(self, encoder: SentenceTransformer, infeasible_embs: np.ndarray, feasible_embs: np.ndarray):
        self.encoder         = encoder
        self.infeasible_embs = infeasible_embs
        self.feasible_embs   = feasible_embs

    def score_slots_batch(self, slot_items: list[tuple[str, str]]) -> list[float]:
        contexts = [f"{span} [SEP] {sent}" for span, sent in slot_items]
        embs = self.encoder.encode(contexts, normalize_embeddings=True)
        inf  = (embs @ self.infeasible_embs.T).max(axis=1)
        feas = (embs @ self.feasible_embs.T).max(axis=1)
        raw  = inf - feas
        return [float(1 / (1 + np.exp(-r * 8))) for r in raw]


# ─────────────────────────────────────────────
# Threshold calibrator
# ─────────────────────────────────────────────

class FeasibilityCalibrator:
    def __init__(self, encoder: SentenceTransformer, infeasible_embs: np.ndarray, feasible_embs: np.ndarray):
        self.encoder         = encoder
        self.infeasible_embs = infeasible_embs
        self.feasible_embs   = feasible_embs

    def fit(self, data_path: str) -> dict[str, float]:
        path = Path(data_path)
        if not path.exists():
            print(f"[FeasibilityCalibrator] '{data_path}' not found — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        with open(path) as f:
            data = json.load(f)

        train_records = data.get("train", [])
        val_records   = data.get("val",   [])

        if not val_records:
            print("[FeasibilityCalibrator] No val records — using defaults.")
            return dict(DEFAULT_SLOT_THRESHOLDS)

        train_sents = {r["sentence"] for r in train_records}
        leaked = [r for r in val_records if r["sentence"] in train_sents]
        if leaked:
            print(f"[FeasibilityCalibrator] WARNING: {len(leaked)} val sentence(s) also in train — results may be optimistic.")

        print(f"[FeasibilityCalibrator] Scoring {len(val_records)} val examples...")
        contexts = [f"{r['span']} [SEP] {r['sentence']}" for r in val_records]
        embs  = self.encoder.encode(contexts, normalize_embeddings=True)
        inf   = (embs @ self.infeasible_embs.T).max(axis=1)
        feas  = (embs @ self.feasible_embs.T).max(axis=1)
        raw   = inf - feas
        scores = 1 / (1 + np.exp(-raw * 8))

        for rec, score in zip(val_records, scores):
            rec["pred_score"] = float(score)

        thresholds: dict[str, float] = {}
        by_slot: dict[str, list] = {}
        for rec in val_records:
            by_slot.setdefault(rec["slot"], []).append(rec)

        for slot, records in by_slot.items():
            slot_scores  = np.array([r["pred_score"] for r in records])
            slot_labels  = np.array([r["label"]      for r in records])
            n_infeasible = int(slot_labels.sum())
            n_feasible   = len(slot_labels) - n_infeasible

            if len(records) < 4 or n_infeasible == 0 or n_feasible == 0:
                thresholds[slot] = DEFAULT_SLOT_THRESHOLDS.get(slot, 0.50)
                print(f"  [{slot:12s}]  threshold={thresholds[slot]:.2f}  (default — n={len(records)}, infeasible={n_infeasible}, feasible={n_feasible})")
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
            print(f"  [{slot:12s}]  threshold={best_t:.2f}  val_F1={best_f1:.3f}  n={len(records)}")

        for slot in SLOTS:
            if slot not in thresholds:
                thresholds[slot] = DEFAULT_SLOT_THRESHOLDS.get(slot, 0.50)

        return thresholds


# ─────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────

class FeasibilityDetector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        spacy_model: str = "en_core_web_sm",
        calibration_data: str = "feasibility_calibration_data.json",
        slot_thresholds: Optional[dict[str, float]] = None,
    ):
        self.nlp     = spacy.load(spacy_model)
        self.encoder = SentenceTransformer(model_name)

        self._infeasible_embs = self.encoder.encode(INFEASIBLE_PROTOTYPES, normalize_embeddings=True)
        self._feasible_embs   = self.encoder.encode(FEASIBLE_PROTOTYPES,   normalize_embeddings=True)

        self.slot_parser = SlotParser(self.nlp)
        self.scorer      = ContextualFeasibilityScorer(self.encoder, self._infeasible_embs, self._feasible_embs)

        if slot_thresholds:
            self.thresholds = slot_thresholds
        else:
            calibrator = FeasibilityCalibrator(self.encoder, self._infeasible_embs, self._feasible_embs)
            self.thresholds = calibrator.fit(calibration_data)

        print(f"[FeasibilityDetector] Thresholds: {self.thresholds}")

    def _slots_to_violations(self, slots: RequirementSlots, sentence: str) -> list[FeasibilityViolation]:
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
        scores = self.scorer.score_slots_batch([(text, sentence) for _, text in items])

        result: list[FeasibilityViolation] = []
        for (slot, text), score in zip(items, scores):
            threshold = self.thresholds.get(slot, 0.60)
            if score >= threshold:
                result.append(FeasibilityViolation(
                    text=text, score=round(score, 4), slot=slot, reason="semantic",
                    token_spans=find_token_spans(text, sentence), suggestion=None,
                ))
        return result

    def analyze(self, sentence: str) -> FeasibilityResult:
        slots = self.slot_parser.parse(sentence)

        rule_violations: list[FeasibilityViolation] = []
        rule_violations.extend(detect_impossible_absolutes(sentence))
        rule_violations.extend(detect_internal_contradictions(sentence))
        rule_violations.extend(detect_unrealistic_thresholds(sentence))

        semantic_violations = self._slots_to_violations(slots, sentence)

        seen: set[str] = {v.text.lower() for v in rule_violations}
        merged = list(rule_violations)
        for v in semantic_violations:
            if v.text.lower() not in seen:
                merged.append(v)
                seen.add(v.text.lower())

        merged.sort(key=lambda x: (0 if x.reason != "semantic" else 1, -x.score))
        return FeasibilityResult(sentence=sentence, slots=slots, violations=merged)

    def analyze_many(self, sentences: list[str]) -> list[FeasibilityResult]:
        return [self.analyze(s) for s in sentences]


# ─────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    test_sentences = [
        "The system must provide 100% uptime at all times.",
        "The API must respond with zero latency.",
        "The service must never fail under any circumstances.",
        "The platform must always be available without exception.",
        "The pipeline must process data with zero downtime.",
        "The system must offer unlimited throughput.",
        "The cache must provide instantaneous retrieval.",
        "The service must be synchronous and non-blocking.",
        "The component must be stateless and maintain session state.",
        "The system must process requests in real-time and via batch processing.",
        "All data must be encrypted and stored in plaintext for auditing.",
        "The module must be immutable and allow updates from admin users.",
        "The API must respond within 0.1ms for all requests.",
        "The system must maintain 99.9999% availability.",
        "The service must process 1,000,000 requests per second with 1ms latency.",
        "The API must respond within 200ms for 95% of requests under 1000 RPS.",
        "The service must maintain 99.9% uptime with planned maintenance windows.",
        "The system must support downloading PDF files to disk.",
        "The system must encrypt all data before writing to disk.",
    ]

    detector = FeasibilityDetector(calibration_data="feasibility_calibration_data.json")
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
        flag = "✗" if not r.is_feasible else "✓"
        score_str = f"{r.max_score:.2f}" if r.max_score > 0 else "   — "
        print(f"  {flag}    {score_str}  {r.sentence[:65]}")

    # ── structured export ─────────────────────────────────────────────────────
    import json as _json
    records = [r.to_dict() for r in results]
    with open("feasibility_results.json", "w") as f:
        _json.dump(records, f, indent=2)
    print("\nStructured results written to feasibility_results.json")

    # ── HTML report ───────────────────────────────────────────────────────────
    html = render_html(results, title="Feasibility Analysis — Demo")
    with open("feasibility_report.html", "w") as f:
        f.write(html)
    print("HTML report written to feasibility_report.html")