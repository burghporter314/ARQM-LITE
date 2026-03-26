"""
PDF report generation for ARQM-LITE quality analysis results.
Uses reportlab to produce a structured, human-readable quality report.
"""

import io
import re
from datetime import datetime

from reportlab.lib         import colors
from reportlab.lib.enums   import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles  import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units   import inch
from reportlab.platypus    import (
    HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

# ── palette ───────────────────────────────────────────────────────────────────
_TITLE_BG          = colors.HexColor("#1A237E")
_HEADER_BG         = colors.HexColor("#283593")
_AMBIGUITY_CLR     = colors.HexColor("#E65100")
_FEASIBILITY_CLR   = colors.HexColor("#B71C1C")
_SINGULARITY_CLR   = colors.HexColor("#1B5E20")
_VERIFIABILITY_CLR = colors.HexColor("#0D47A1")
_OK_CLR            = colors.HexColor("#2E7D32")
_VIOLATION_CLR     = colors.HexColor("#C62828")
_ROW_ALT           = colors.HexColor("#F5F5F5")


def _hex(clr: colors.Color) -> str:
    return f"{int(clr.red*255):02X}{int(clr.green*255):02X}{int(clr.blue*255):02X}"


# ── noise filter ──────────────────────────────────────────────────────────────
# Single structural words that are almost always false positives when flagged
# in isolation by the semantic scorer.
_NOISE_WORDS = {
    "shall", "must", "should", "will", "may", "can", "could", "might",
    "have", "has", "had", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "let", "get", "make", "set", "use",
    "removed", "added", "updated", "deleted", "changed",
}

_SECTION_ID_RE = re.compile(r'^\d+(\.\d+)*\.?$')   # "1.3", "2.1.4", "3."


def _is_noise(span) -> bool:
    """Return True for spans that are structural artifacts, not real violations."""
    t = span.text.strip().lower()
    if len(t) <= 2:
        return True
    if t in _NOISE_WORDS:
        return True
    # Section identifiers like "1.3" or "2.1"
    if _SECTION_ID_RE.match(t):
        return True
    # Single-word modals even with surrounding whitespace
    if t.split()[0] in _NOISE_WORDS and len(t.split()) == 1:
        return True
    return False


# ── human-readable violation descriptions ─────────────────────────────────────

_SLOT_LABELS = {
    "subject":   "subject",
    "modal":     "obligation word",
    "action":    "action phrase",
    "object":    "object",
    "condition": "condition",
    "qualifier": "qualifier",
}


def _violation_text(span, dimension: str) -> tuple[str, str]:
    """
    Return (short_title, one-sentence explanation) for a violation span.
    Both strings are intended to be read by a non-technical reviewer.
    """
    text       = span.text
    slot_label = _SLOT_LABELS.get(span.slot, span.slot)
    suggestion = getattr(span, "suggestion", None) or ""

    if dimension == "Ambiguity":
        if span.reason == "syntactic":
            if "passive" in suggestion.lower():
                return (
                    "Passive voice — unclear actor",
                    f'The action phrase "{text}" does not name who performs this action.',
                )
            if "shall" in suggestion.lower():
                return (
                    "Ambiguous obligation level",
                    f'"should" is ambiguous — use "shall" for mandatory '
                    f'requirements and "may" for optional ones.',
                )
            if "condition" in suggestion.lower():
                return (
                    "Undefined condition boundary",
                    f'The condition "{text}" has no defined threshold '
                    f"(e.g. specify exact load or frequency values).",
                )
            if "baseline" in suggestion.lower():
                return (
                    "Comparison without a reference",
                    f'"{text}" is a comparison with no stated baseline.',
                )
            return (
                f"Imprecise {slot_label}",
                f'"{text}" is a gradable term with no numeric bound '
                f"or measurable criterion.",
            )
        # semantic
        return (
            f"Vague {slot_label}",
            f'"{text}" is imprecise — replace it with a specific, '
            f"quantifiable value that can be objectively tested.",
        )

    if dimension == "Feasibility":
        _map = {
            "impossible_absolute": (
                "Impossible absolute",
                f'"{text}" cannot be achieved in practice. '
                f"No real system can guarantee this constraint.",
            ),
            "internal_contradiction": (
                "Internal contradiction",
                f'"{text}" contains mutually exclusive terms '
                f"that cannot both hold at the same time.",
            ),
            "unrealistic_threshold": (
                "Unrealistic threshold",
                f'"{text}" exceeds known engineering limits. '
                f"Use a value that is achievable.",
            ),
        }
        if span.reason in _map:
            return _map[span.reason]
        return (
            "Potentially infeasible",
            f'"{text}" resembles a constraint that is difficult or '
            f"impossible to satisfy in practice.",
        )

    if dimension == "Singularity":
        _map = {
            "multiple_actions": (
                "Multiple actions bundled together",
                f'"{text}" combines more than one discrete obligation. '
                f"Each action should be its own requirement.",
            ),
            "compound_subject": (
                "Multiple actors in one requirement",
                f'"{text}" names more than one actor. '
                f"Give each actor its own requirement.",
            ),
            "conjunctive_condition": (
                "Multiple trigger conditions",
                f'"{text}" lists several independent triggers, '
                f"creating ambiguity about which scenario applies.",
            ),
            "mixed_concerns": (
                "Mixed functional and non-functional concerns",
                f'"{text}" combines a functional obligation with a '
                f"non-functional constraint — separate them.",
            ),
        }
        if span.reason in _map:
            return _map[span.reason]
        return (
            "Potentially non-singular",
            f'"{text}" may address multiple independent concerns. '
            f"Consider splitting it into separate requirements.",
        )

    if dimension == "Verifiability":
        _map = {
            "no_acceptance_criteria": (
                "No acceptance criteria",
                f'"{text}" has no testable pass/fail condition. '
                f"A tester cannot determine whether this requirement is met.",
            ),
            "subjective_success": (
                "Subjective success condition",
                f'"{text}" relies on human judgement rather than '
                f"objective measurement. Add a concrete criterion.",
            ),
            "missing_actor": (
                "Missing actor or trigger",
                f'"{text}" does not specify who or what performs '
                f"or observes the verification step.",
            ),
            "untestable_negative": (
                "Untestable absolute prohibition",
                f'"{text}" is an absolute prohibition with no measurable '
                f"bound. Restate it as a positive, quantified constraint.",
            ),
        }
        if span.reason in _map:
            return _map[span.reason]
        return (
            "Potentially unverifiable",
            f'"{text}" lacks clear, objective acceptance criteria.',
        )

    return (span.reason.replace("_", " ").title(), f'"{text}"')


def _select_display_spans(spans, dimension: str) -> list[tuple[str, str, str]]:
    """
    Filter *spans* to remove noise and return a deduplicated list of
    ``(title, detail, fix)`` triples ready for rendering.

    *fix* is the span's suggestion text (empty string if absent).
    """
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str, str]] = []
    for span in spans:
        if _is_noise(span):
            continue
        title, detail = _violation_text(span, dimension)
        key = (title, detail)
        if key in seen:
            continue
        seen.add(key)
        fix = getattr(span, "suggestion", None) or ""
        result.append((title, detail, fix))
    return result


# ── styles ────────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ARQMTitle", parent=base["Title"],
            fontSize=26, textColor=_TITLE_BG,
            spaceAfter=6, alignment=TA_CENTER,
        ),
        "subtitle": ParagraphStyle(
            "ARQMSubtitle", parent=base["Normal"],
            fontSize=11, textColor=colors.HexColor("#555555"),
            spaceAfter=20, alignment=TA_CENTER,
        ),
        "section": ParagraphStyle(
            "ARQMSection", parent=base["Heading1"],
            fontSize=13, textColor=_HEADER_BG,
            spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ARQMBody", parent=base["Normal"],
            fontSize=9, spaceAfter=3, leading=13,
        ),
        "sentence": ParagraphStyle(
            "ARQMSentence", parent=base["Normal"],
            fontSize=9, spaceAfter=4, leading=13,
            leftIndent=8, rightIndent=8,
            textColor=colors.HexColor("#333333"),
        ),
        "issue_title": ParagraphStyle(
            "ARQMIssueTitle", parent=base["Normal"],
            fontSize=9, leading=13, leftIndent=12,
            spaceBefore=4,
        ),
        "issue_detail": ParagraphStyle(
            "ARQMIssueDetail", parent=base["Normal"],
            fontSize=8, leading=12, leftIndent=24, spaceAfter=1,
            textColor=colors.HexColor("#444444"),
        ),
        "issue_fix": ParagraphStyle(
            "ARQMIssueFix", parent=base["Normal"],
            fontSize=8, leading=12, leftIndent=24, spaceAfter=3,
            textColor=colors.HexColor("#1B5E20"),
        ),
    }


# ── dimension metadata ────────────────────────────────────────────────────────

_DIM_GOOD = {
    "Ambiguity":     lambda r: not r.is_ambiguous,
    "Feasibility":   lambda r: r.is_feasible,
    "Singularity":   lambda r: r.is_singular,
    "Verifiability": lambda r: r.is_verifiable,
}

_DIM_SPANS_ATTR = {
    "Ambiguity":     "ambiguous_spans",
    "Feasibility":   "violations",
    "Singularity":   "violations",
    "Verifiability": "violations",
}

_DIM_COLOR = {
    "Ambiguity":     _AMBIGUITY_CLR,
    "Feasibility":   _FEASIBILITY_CLR,
    "Singularity":   _SINGULARITY_CLR,
    "Verifiability": _VERIFIABILITY_CLR,
}


# ── public API ────────────────────────────────────────────────────────────────

def generate_report(results: list[dict], filename: str = "document") -> bytes:
    """
    Build a PDF quality report from the list of per-requirement analysis dicts
    returned by ``util.analyzer.analyze_requirements``.
    Returns the PDF content as bytes.
    """
    buf    = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch,  bottomMargin=0.75 * inch,
    )
    doc.build(_build_story(results, filename, styles))
    return buf.getvalue()


# ── story builders ────────────────────────────────────────────────────────────

def _build_story(results, filename, styles):
    story = []

    # cover
    story.append(Spacer(1, 0.7 * inch))
    story.append(Paragraph("ARQM-LITE Quality Report", styles["title"]))
    story.append(Paragraph(
        f"Document: <b>{filename}</b><br/>"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        styles["subtitle"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=_TITLE_BG))
    story.append(Spacer(1, 0.25 * inch))

    story.extend(_summary_section(results, styles))
    story.append(Spacer(1, 0.15 * inch))
    story.extend(_legend_section(styles))
    story.append(PageBreak())

    story.append(Paragraph("Detailed Analysis", styles["section"]))
    story.append(Spacer(1, 0.08 * inch))

    for idx, result in enumerate(results, 1):
        story.extend(_requirement_block(idx, result, styles))

    return story


def _summary_section(results, styles):
    total  = len(results)
    counts = {
        dim: sum(1 for r in results if not _DIM_GOOD[dim](r[dim.lower()]))
        for dim in ("Ambiguity", "Feasibility", "Singularity", "Verifiability")
    }
    total_dim = sum(counts.values())

    story = [Paragraph("Summary", styles["section"])]
    data  = [
        ["Metric", "Count"],
        ["Requirements analysed",      str(total)],
        ["Ambiguity violations",        str(counts["Ambiguity"])],
        ["Feasibility violations",      str(counts["Feasibility"])],
        ["Singularity violations",      str(counts["Singularity"])],
        ["Verifiability violations",    str(counts["Verifiability"])],
        ["Total dimension violations",  str(total_dim)],
    ]

    dim_colors = [_AMBIGUITY_CLR, _FEASIBILITY_CLR, _SINGULARITY_CLR, _VERIFIABILITY_CLR]
    tbl = Table(data, colWidths=[3.8 * inch, 1.2 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _TITLE_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 10),
        ("ALIGN",         (1, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, _ROW_ALT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ] + [
        ("TEXTCOLOR",  (1, 2 + i), (1, 2 + i), c) for i, c in enumerate(dim_colors)
    ] + [
        ("FONTNAME",   (1, 2 + i), (1, 2 + i), "Helvetica-Bold") for i in range(4)
    ]))
    story.append(tbl)
    return story


def _legend_section(styles):
    story = [
        Spacer(1, 0.15 * inch),
        Paragraph("Quality Dimensions", styles["section"]),
    ]
    data = [
        ["Dimension",     "What it checks"],
        ["Ambiguity",     "Vague, imprecise, or unmeasurable language that prevents a clear understanding of the requirement"],
        ["Feasibility",   "Impossible absolutes (e.g. 100% uptime), internal contradictions, or unrealistic numeric thresholds"],
        ["Singularity",   "Requirements that bundle multiple actions, actors, or concerns that should be stated separately"],
        ["Verifiability", "Requirements that have no testable pass/fail condition or rely on subjective judgement"],
    ]
    tbl = Table(data, colWidths=[1.4 * inch, 5.6 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.white, _ROW_ALT]),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ] + [
        ("TEXTCOLOR", (0, 1 + i), (0, 1 + i), c)
        for i, c in enumerate([_AMBIGUITY_CLR, _FEASIBILITY_CLR,
                                _SINGULARITY_CLR, _VERIFIABILITY_CLR])
    ] + [
        ("FONTNAME",  (0, 1 + i), (0, 1 + i), "Helvetica-Bold") for i in range(4)
    ]))
    story.append(tbl)
    return story


def _requirement_block(idx: int, result: dict, styles) -> list:
    sentence = result["sentence"]

    good = {
        dim: _DIM_GOOD[dim](result[dim.lower()])
        for dim in ("Ambiguity", "Feasibility", "Singularity", "Verifiability")
    }
    violation_count = sum(1 for v in good.values() if not v)
    has_violations  = violation_count > 0

    req_label    = f"REQ-{idx:03d}"
    status_label = f"{violation_count} violation(s)" if has_violations else "No violations"
    header_bg    = _VIOLATION_CLR if has_violations else _OK_CLR
    short_sent   = sentence if len(sentence) <= 100 else sentence[:97] + "…"

    hdr_tbl = Table(
        [[req_label, short_sent, status_label]],
        colWidths=[0.75 * inch, 4.8 * inch, 1.45 * inch],
    )
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), header_bg),
        ("TEXTCOLOR",     (0, 0), (-1, -1), colors.white),
        ("FONTNAME",      (0, 0), (0,  0),  "Helvetica-Bold"),
        ("FONTNAME",      (2, 0), (2,  0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (2, 0), (2,  0),  "RIGHT"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]))

    story = [hdr_tbl]

    if not has_violations:
        story.append(Spacer(1, 0.06 * inch))
        return story

    # full sentence (italic)
    story.append(Paragraph(f"<i>{sentence}</i>", styles["sentence"]))

    # per-dimension violations
    for dim in ("Ambiguity", "Feasibility", "Singularity", "Verifiability"):
        if good[dim]:
            continue

        res_obj = result[dim.lower()]
        spans   = getattr(res_obj, _DIM_SPANS_ATTR[dim], [])
        clr     = _DIM_COLOR[dim]
        hex_clr = _hex(clr)

        visible_issues = _select_display_spans(spans, dim)
        if not visible_issues:
            continue

        # Dimension header
        story.append(Paragraph(
            f"<font color='#{hex_clr}'><b>{dim}</b></font>",
            styles["body"],
        ))

        for title, detail, fix in visible_issues:
            story.append(Paragraph(
                f"<font color='#{hex_clr}'>&#8227; <b>{title}</b></font>",
                styles["issue_title"],
            ))
            story.append(Paragraph(detail, styles["issue_detail"]))
            if fix:
                story.append(Paragraph(
                    f"<i>Fix: {fix}</i>",
                    styles["issue_fix"],
                ))

    story.append(Spacer(1, 0.1 * inch))
    return story
