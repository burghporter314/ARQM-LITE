"""
generate_quality_report.py
==========================
Standalone PDF report generator for the ARQM-LITE requirements quality system.

Consumes the structured output of the four quality detectors (Ambiguity,
Feasibility, Verifiability, Singularity) and produces a multi-section PDF.

Public API
----------
    generate_pdf(results, output_path="quality_report.pdf")

``results`` must be a dict with keys "ambiguity", "feasibility",
"verifiability", "singularity", each mapping to a list of detector result
objects (AnalysisResult / FeasibilityResult / VerifiabilityResult /
SingularityResult) — one per requirement sentence.  Each object must expose
a ``to_dict()`` method that returns a serialisable dict.

Run directly for a self-contained demo:

    python generate_quality_report.py
"""

from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.graphics.shapes import Drawing, Line, Rect
from reportlab.graphics.shapes import String as GStr
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Dimension registry ────────────────────────────────────────────────────────
# Order determines section order in the report.
#   (key, display name, hex colour, one-sentence definition)
_DIMS: list[tuple[str, str, str, str]] = [
    (
        "ambiguity",
        "Ambiguity",
        "#d97706",
        "Flags vague or imprecise language that cannot be objectively tested "
        "or measured.",
    ),
    (
        "feasibility",
        "Feasibility",
        "#dc2626",
        "Flags physically impossible constraints, internal contradictions, "
        "or unrealistic numeric thresholds.",
    ),
    (
        "verifiability",
        "Verifiability",
        "#2563eb",
        "Flags requirements that have no testable pass/fail condition or "
        "rely on subjective judgement.",
    ),
    (
        "singularity",
        "Singularity",
        "#7c3aed",
        "Flags requirements that bundle more than one distinct concern, "
        "actor, or action.",
    ),
]

# Pre-computed lookup tables
_DIM_NAME:  dict[str, str]          = {k: n  for k, n, _, _  in _DIMS}
_DIM_HEX:   dict[str, str]          = {k: h  for k, _, h, _  in _DIMS}
_DIM_COLOR: dict[str, colors.Color] = {k: colors.HexColor(h) for k, _, h, _ in _DIMS}
_DIM_DEF:   dict[str, str]          = {k: d  for k, _, _, d  in _DIMS}

# In to_dict() output, the boolean "is good" field name differs per dimension.
_IS_GOOD_KEY: dict[str, str] = {
    "ambiguity":     "is_ambiguous",   # True  → flagged (inverted)
    "feasibility":   "is_feasible",    # False → flagged
    "verifiability": "is_verifiable",  # False → flagged
    "singularity":   "is_singular",    # False → flagged
}

# The violations / spans list key in to_dict() output.
_VIOLS_KEY: dict[str, str] = {
    "ambiguity":     "spans",
    "feasibility":   "violations",
    "verifiability": "violations",
    "singularity":   "violations",
}

# ── Page geometry ─────────────────────────────────────────────────────────────
_PAGE_W, _PAGE_H = letter
_MARGIN = 0.75 * inch
_BODY_W = _PAGE_W - 2 * _MARGIN  # ≈ 504 pts / 7.0 inch

# ── Dict access helpers ───────────────────────────────────────────────────────

def _is_flagged(dim_key: str, d: dict) -> bool:
    """True when the to_dict() result has at least one violation."""
    val = d.get(_IS_GOOD_KEY[dim_key])
    return bool(val) if dim_key == "ambiguity" else not bool(val)


def _violations(dim_key: str, d: dict) -> list[dict]:
    """Return the violation / span list from a to_dict() result."""
    return d.get(_VIOLS_KEY[dim_key], [])


def _max_score(dim_key: str, d: dict) -> float:
    if dim_key == "ambiguity":
        return max((s["score"] for s in d.get("spans", [])), default=0.0)
    return d.get("max_score", 0.0)


# ── Text helpers ──────────────────────────────────────────────────────────────

_HIGHLIGHT_RE = re.compile(r">>(.*?)<<")


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _tint_hex(hex_color: str, opacity: float = 0.22) -> str:
    """
    Blend *hex_color* with white at *opacity* and return the resulting hex.
    Used to produce a light background tint for inline highlights.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    tr = int(r * opacity + 255 * (1 - opacity))
    tg = int(g * opacity + 255 * (1 - opacity))
    tb = int(b * opacity + 255 * (1 - opacity))
    return f"#{tr:02x}{tg:02x}{tb:02x}"


def _parse_highlighted(text: str, hl_color: str = "#d97706") -> str:
    """
    Convert ``>>span<<`` markers from Violation.highlight() into ReportLab
    Paragraph markup with a coloured background highlight.

    Highlighted spans receive:
      • a light tint of the dimension colour as their background (backColor)
      • the dimension colour itself as the foreground text colour
      • bold weight

    Everything outside the markers is XML-escaped plain text.
    """
    bg = _tint_hex(hl_color)
    parts = _HIGHLIGHT_RE.split(text)
    out: list[str] = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            out.append(_xml_escape(part))
        else:
            out.append(
                f"<font backColor='{bg}' color='{hl_color}'>"
                f"<b>{_xml_escape(part)}</b>"
                f"</font>"
            )
    return "".join(out)


def _truncate(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[: n - 1] + "\u2026"


def _render_req(highlighted: str, sentence: str, hl_color: str,
                max_plain: int = 120) -> str:
    """
    Return Paragraph markup for a requirement cell with inline highlighting.

    Flagged spans (marked ``>>…<<``) are rendered with a coloured background
    highlight and bold text.  If the plain text fits within *max_plain*
    characters the full sentence is shown; otherwise it is truncated while
    preserving any highlighted spans that fall inside the truncation window.
    """
    plain = _HIGHLIGHT_RE.sub(r"\1", highlighted)
    if len(plain) <= max_plain:
        return _parse_highlighted(highlighted, hl_color)

    # Sentence is too long — iterate over segments, keeping highlights that
    # fit within the character budget and truncating at the boundary.
    bg = _tint_hex(hl_color)
    parts = _HIGHLIGHT_RE.split(highlighted)
    # split gives: [plain0, span0, plain1, span1, …]

    out: list[str] = []
    plain_count = 0
    done = False

    for i, part in enumerate(parts):
        if done:
            break
        remaining = max_plain - plain_count
        if remaining <= 0:
            out.append("\u2026")
            done = True
            break

        if i % 2 == 0:                    # non-highlighted segment
            if len(part) <= remaining:
                out.append(_xml_escape(part))
                plain_count += len(part)
            else:
                out.append(_xml_escape(part[:remaining]) + "\u2026")
                done = True
        else:                              # highlighted span
            span_text = part[:remaining]
            suffix = "\u2026" if len(part) > remaining else ""
            out.append(
                f"<font backColor='{bg}' color='{hl_color}'>"
                f"<b>{_xml_escape(span_text)}{suffix}</b>"
                f"</font>"
            )
            plain_count += len(span_text)
            if suffix:
                done = True

    return "".join(out)


# ── Styles ────────────────────────────────────────────────────────────────────

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    def S(name: str, parent: str = "Normal", **kw) -> ParagraphStyle:
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "title": S(
            "QRTitle", "Title",
            fontSize=22, textColor=colors.HexColor("#1e293b"),
            spaceAfter=6, alignment=TA_CENTER,
        ),
        "subtitle": S(
            "QRSubtitle",
            fontSize=10, textColor=colors.HexColor("#64748b"),
            spaceAfter=0, alignment=TA_CENTER,
        ),
        "section_def": S(
            "QRSecDef",
            fontSize=9, textColor=colors.HexColor("#64748b"),
            spaceAfter=5, leftIndent=2,
        ),
        "body": S(
            "QRBody",
            fontSize=10, spaceAfter=4, leading=14,
        ),
        "body_sm": S(
            "QRBodySm",
            fontSize=9, spaceAfter=3, leading=13,
        ),
        "tbl_hdr": S(
            "QRTblHdr",
            fontSize=9, textColor=colors.white, leading=12,
        ),
        "tbl_cell": S(
            "QRTblCell",
            fontSize=9, leading=12, spaceAfter=0,
        ),
        "tbl_cell_sm": S(
            "QRTblCellSm",
            fontSize=8, leading=11, spaceAfter=0,
        ),
        "clean_item": S(
            "QRCleanItem",
            fontSize=9, leading=12, spaceAfter=2, leftIndent=4,
            textColor=colors.HexColor("#374151"),
        ),
        "appendix_item": S(
            "QRAppItem",
            fontSize=9, leading=13, spaceAfter=4,
        ),
        "tile_label": S(
            "QRTileLabel",
            fontSize=8, textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER, spaceAfter=0,
        ),
    }


# ── Score bar (Drawing) ───────────────────────────────────────────────────────

def _score_bar(score: float, bar_color: colors.Color,
               width: float = 52, bar_h: float = 7) -> Drawing:
    """Return a Drawing with a filled-rect score bar and a % label below it."""
    total_h = bar_h + 11          # bar + text
    d = Drawing(width, total_h)
    # Background track
    d.add(Rect(0, 11, width, bar_h,
               fillColor=colors.HexColor("#e5e7eb"), strokeColor=None))
    # Filled portion
    filled = max(score * width, 0.0)
    if filled > 0:
        d.add(Rect(0, 11, filled, bar_h,
                   fillColor=bar_color, strokeColor=None))
    # Percentage text centred below bar
    d.add(GStr(width / 2, 1, f"{score * 100:.0f}%",
               textAnchor="middle", fontSize=7,
               fillColor=colors.HexColor("#374151")))
    return d


# ── Cover page ────────────────────────────────────────────────────────────────

def _cover_page(dim_stats: dict[str, dict], total: int,
                styles: dict) -> list:
    story: list = []

    story.append(Spacer(1, 0.55 * inch))

    story.append(Paragraph("Software Requirements", styles["title"]))
    story.append(Paragraph("Quality Report", styles["title"]))
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        f"\u2003\u00b7\u2003"
        f"{total} requirement{'s' if total != 1 else ''} analysed",
        styles["subtitle"],
    ))
    story.append(Spacer(1, 0.22 * inch))
    story.append(HRFlowable(
        width="100%", thickness=1.5,
        color=colors.HexColor("#cbd5e1"),
    ))
    story.append(Spacer(1, 0.28 * inch))

    # ── Metric tiles (4 dimension tiles) ──────────────────────────────────────
    story.extend(_metric_tiles(dim_stats, total))
    story.append(Spacer(1, 0.32 * inch))

    # ── Stacked bar label ─────────────────────────────────────────────────────
    bar_label_style = ParagraphStyle(
        "QRBarLbl", parent=styles["tile_label"],
        spaceAfter=5,
    )
    story.append(Paragraph(
        "Requirements Flagged per Dimension", bar_label_style,
    ))
    story.append(_quality_profile_bar(dim_stats, total, _BODY_W))

    return story


def _metric_tiles(dim_stats: dict[str, dict], total: int) -> list:
    """
    Four metric tiles laid out as a 6-row × 4-column flat table.
    Each column = one dimension.
    """
    tile_w = _BODY_W / 4

    rows_data: list[list] = [[], [], [], [], [], []]

    for key, name, hex_color, _ in _DIMS:
        stats   = dim_stats[key]
        flagged = stats["flagged"]
        passing = total - flagged
        rate    = (passing / total * 100) if total else 0.0

        def P(markup: str, size: int = 9, align: int = TA_CENTER,
              color: str = "#374151") -> Paragraph:
            st = ParagraphStyle(
                f"_t_{key}_{size}",
                parent=getSampleStyleSheet()["Normal"],
                fontSize=size, alignment=align,
                textColor=colors.HexColor(color),
                spaceAfter=0, leading=size + 3,
            )
            return Paragraph(markup, st)

        rows_data[0].append(P(f"<b>{name}</b>", size=9, color=hex_color))
        rows_data[1].append(P(f"<b>{passing}</b>", size=20, color="#15803d"))
        rows_data[2].append(P("passing", size=8, color="#6b7280"))
        rows_data[3].append(P(f"<b>{flagged}</b>", size=16, color=hex_color))
        rows_data[4].append(P("flagged", size=8, color="#6b7280"))
        rows_data[5].append(P(f"<b>{rate:.0f}%</b> pass rate", size=9))

    tbl = Table(rows_data, colWidths=[tile_w] * 4)
    tbl.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("BACKGROUND",    (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        # Outer box
        ("BOX",           (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        # Vertical tile separators
        ("LINEAFTER",     (0, 0), (0, -1), 1, colors.HexColor("#e2e8f0")),
        ("LINEAFTER",     (1, 0), (1, -1), 1, colors.HexColor("#e2e8f0")),
        ("LINEAFTER",     (2, 0), (2, -1), 1, colors.HexColor("#e2e8f0")),
        # Subtle row lines inside tiles
        ("LINEBELOW",     (0, 0), (-1, 0), 0.5, colors.HexColor("#e2e8f0")),
        ("LINEBELOW",     (0, 2), (-1, 2), 0.5, colors.HexColor("#e2e8f0")),
        ("LINEBELOW",     (0, 4), (-1, 4), 0.5, colors.HexColor("#e2e8f0")),
        # Extra top padding on dimension name row
        ("TOPPADDING",    (0, 0), (-1, 0), 8),
        ("BOTTOMPADDING", (0, 5), (-1, 5), 8),
    ]))
    return [tbl]


def _quality_profile_bar(dim_stats: dict[str, dict], total: int,
                          width: float) -> Drawing:
    """
    Horizontal stacked bar: 4 equal-width segments, one per dimension.
    Each segment is filled proportionally to the flagged rate for that dimension.
    Unfilled portion shown in light grey.
    """
    bar_h    = 24
    label_h  = 13
    gap      = 3
    total_h  = bar_h + gap + label_h
    seg_w    = width / 4

    d = Drawing(width, total_h)

    for i, (key, name, hex_color, _) in enumerate(_DIMS):
        x     = i * seg_w
        stats = dim_stats[key]
        frac  = (stats["flagged"] / total) if total else 0.0

        # Background (unfilled)
        d.add(Rect(x, label_h + gap, seg_w, bar_h,
                   fillColor=colors.HexColor("#f1f5f9"),
                   strokeColor=colors.HexColor("#e2e8f0"),
                   strokeWidth=0.5))

        # Filled portion
        filled_w = frac * seg_w
        if filled_w > 0.5:
            d.add(Rect(x, label_h + gap, filled_w, bar_h,
                       fillColor=colors.HexColor(hex_color),
                       strokeColor=None))

        # Percentage text — white only when the bar fill covers the text
        # centre (≥ 50 % of segment width); dark otherwise so it is always
        # legible against the grey track background.
        pct_text = f"{frac * 100:.0f}%"
        text_x   = x + seg_w / 2
        text_y   = label_h + gap + (bar_h / 2) - 4
        text_clr = colors.white if frac >= 0.5 else colors.HexColor("#374151")
        d.add(GStr(text_x, text_y, pct_text,
                   textAnchor="middle", fontSize=9, fillColor=text_clr))

        # Dimension label below bar
        d.add(GStr(text_x, 1, name,
                   textAnchor="middle", fontSize=8,
                   fillColor=colors.HexColor(hex_color)))

        # Right-edge separator (except after last segment)
        if i < 3:
            d.add(Line(x + seg_w, label_h + gap,
                       x + seg_w, label_h + gap + bar_h,
                       strokeColor=colors.HexColor("#cbd5e1"),
                       strokeWidth=0.7))

    return d


# ── Violations table ──────────────────────────────────────────────────────────

def _build_combined_highlighted(viols: list[dict], sentence: str) -> str:
    """
    Return *sentence* with >>…<< markers around EVERY occurrence of EVERY
    violation text.  This ensures the requirement cell in the violations table
    highlights all flagged spans—not just the first violation's span and not
    just the first occurrence of each repeated term.
    """
    spans: list[tuple[int, int]] = []
    for v in viols:
        vtext = v.get("text", "")
        if not vtext:
            continue
        try:
            pat = re.compile(re.escape(vtext), re.IGNORECASE)
        except re.error:
            continue
        for m in pat.finditer(sentence):
            spans.append((m.start(), m.end()))

    if not spans:
        # Nothing matched by text search; fall back to pre-built field
        return viols[0].get("highlighted", sentence) if viols else sentence

    # Sort and merge overlapping / adjacent spans
    spans.sort()
    merged: list[list[int]] = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    result: list[str] = []
    prev = 0
    for start, end in merged:
        result.append(sentence[prev:start])
        result.append(">>" + sentence[start:end] + "<<")
        prev = end
    result.append(sentence[prev:])
    return "".join(result)


# Column widths: Requirement, Violation Type, Flagged Span, Score, Suggestion
_VIOLS_COL_W = [
    2.80 * inch,   # Requirement (with highlighted spans)
    1.25 * inch,   # Violation type
    0.90 * inch,   # Flagged span
    0.75 * inch,   # Score bar
    1.30 * inch,   # Suggestion
]


def _violations_table(flagged_dicts: list[dict], dim_key: str,
                       styles: dict) -> Table:
    """
    Build the per-section violations table.

    ``flagged_dicts`` is a list of to_dict() results that have at least one
    violation for *dim_key*.
    """
    dim_color = _DIM_COLOR[dim_key]
    hex_color = _DIM_HEX(dim_key)

    def hdr(text: str) -> Paragraph:
        return Paragraph(text, styles["tbl_hdr"])

    headers = [
        hdr("Requirement"),
        hdr("Violation Type"),
        hdr("Flagged Span"),
        hdr("Score"),
        hdr("Suggestion"),
    ]

    rows: list[list] = [headers]

    for d in flagged_dicts:
        sentence = d["sentence"]
        viols    = _violations(dim_key, d)

        # Deduplicate by (reason, text)
        seen: set[tuple] = set()
        uniq: list[dict] = []
        for v in viols:
            k = (v["reason"], v["text"])
            if k not in seen:
                seen.add(k)
                uniq.append(v)

        if not uniq:
            continue

        # Build one combined highlighted string covering all violations so the
        # requirement cell shows every flagged span at once.
        combined_hl = _build_combined_highlighted(uniq, sentence)

        for j, v in enumerate(uniq):
            # ── Requirement cell ──
            if j == 0:
                req_markup = _render_req(combined_hl, sentence, f"#{hex_color}", 90)
            else:
                # Subsequent violations for same requirement: blank to avoid
                # repetition; indent signals continuation.
                req_markup = (
                    f"<font color='#9ca3af'>\u21b3 (continued)</font>"
                )

            req_cell = Paragraph(req_markup, styles["tbl_cell_sm"])

            # ── Violation type ──
            reason_label = v["reason"].replace("_", " ").title()
            reason_cell  = Paragraph(
                f"<font color='#{hex_color}'>{_xml_escape(reason_label)}</font>",
                styles["tbl_cell_sm"],
            )

            # ── Flagged span ──
            span_markup = (
                f"<b><font color='#{hex_color}'>"
                f"{_xml_escape(_truncate(v['text'], 32))}"
                f"</font></b>"
            )
            span_cell = Paragraph(span_markup, styles["tbl_cell_sm"])

            # ── Score bar ──
            score_cell = _score_bar(v["score"], dim_color, width=50, bar_h=7)

            # ── Suggestion ──
            sugg = v.get("suggestion") or "\u2014"
            sugg_cell = Paragraph(
                f"<i>{_xml_escape(_truncate(sugg, 60))}</i>",
                ParagraphStyle(
                    f"_sugg_{hex_color}",
                    parent=styles["tbl_cell_sm"],
                    textColor=colors.HexColor("#374151"),
                ),
            )

            rows.append([req_cell, reason_cell, span_cell, score_cell, sugg_cell])

    if len(rows) == 1:
        # Header only — nothing to show (should not happen)
        return Table(rows, colWidths=_VIOLS_COL_W)

    tbl = Table(rows, colWidths=_VIOLS_COL_W, repeatRows=1)

    # Blend dim_color at 15% opacity on white for header bg → light tint
    r = dim_color.red   * 0.15 + 0.85
    g = dim_color.green * 0.15 + 0.85
    b = dim_color.blue  * 0.15 + 0.85
    hdr_tint = colors.Color(r, g, b)

    n = len(rows)
    cmd: list[tuple] = [
        # Header row
        ("BACKGROUND",    (0, 0), (-1, 0), dim_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0), colors.white),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 9),
        # Body
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ]
    # Alternating row shading
    for row_idx in range(1, n):
        if row_idx % 2 == 0:
            cmd.append(
                ("BACKGROUND", (0, row_idx), (-1, row_idx),
                 colors.HexColor("#f8fafc"))
            )

    tbl.setStyle(TableStyle(cmd))
    return tbl


def _DIM_HEX(dim_key: str) -> str:  # noqa: N802
    """Return the 6-char lowercase hex string for a dimension colour."""
    c = _DIM_COLOR[dim_key]
    return f"{int(c.red * 255):02x}{int(c.green * 255):02x}{int(c.blue * 255):02x}"


# ── Dimension section ─────────────────────────────────────────────────────────

def _dimension_section(dim_key: str, dim_dicts: list[dict],
                        styles: dict) -> list:
    """Build all flowables for one quality-dimension section."""
    name      = _DIM_NAME[dim_key]
    dim_color = _DIM_COLOR[dim_key]
    hex_color = _DIM_HEX(dim_key)
    definition = _DIM_DEF[dim_key]

    total     = len(dim_dicts)
    flagged   = [d for d in dim_dicts if     _is_flagged(dim_key, d)]
    passing   = [d for d in dim_dicts if not _is_flagged(dim_key, d)]
    n_flagged = len(flagged)
    pass_rate = ((total - n_flagged) / total * 100) if total else 0.0

    story: list = []

    # ── Section header bar ────────────────────────────────────────────────────
    hdr_tbl = Table(
        [[Paragraph(
            f"<font color='white'><b>{name}</b></font>",
            ParagraphStyle(
                f"_sec_hdr_{dim_key}",
                parent=styles["body"],
                fontSize=14, textColor=colors.white,
                spaceAfter=0,
            ),
        )]],
        colWidths=[_BODY_W],
    )
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), dim_color),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 0.06 * inch))

    # ── Definition ────────────────────────────────────────────────────────────
    story.append(Paragraph(definition, styles["section_def"]))

    # ── Per-section summary ───────────────────────────────────────────────────
    story.append(Paragraph(
        f"{total} requirement{'s' if total != 1 else ''} analysed\u2003\u00b7\u2003"
        f"<font color='#{hex_color}'><b>{n_flagged} flagged</b></font>"
        f"\u2003\u00b7\u2003"
        f"<b>{pass_rate:.0f}%</b> pass rate",
        styles["body_sm"],
    ))
    story.append(Spacer(1, 0.1 * inch))

    # ── Violations table ──────────────────────────────────────────────────────
    if flagged:
        story.append(Paragraph(
            f"<font color='#{hex_color}'><b>Flagged Requirements ({n_flagged})</b></font>",
            styles["body_sm"],
        ))
        story.append(Spacer(1, 0.04 * inch))
        story.append(_violations_table(flagged, dim_key, styles))
        story.append(Spacer(1, 0.14 * inch))

    # ── Clean requirements ────────────────────────────────────────────────────
    if passing:
        story.append(Paragraph(
            f"<font color='#15803d'><b>"
            f"Requirements Passing This Check ({len(passing)})"
            f"</b></font>",
            styles["body_sm"],
        ))
        story.append(Spacer(1, 0.04 * inch))

        for d in passing:
            story.append(Paragraph(
                f"<font color='#15803d'>\u2713</font>\u2002"
                f"{_xml_escape(_truncate(d['sentence'], 120))}",
                styles["clean_item"],
            ))
        story.append(Spacer(1, 0.08 * inch))

    return story


# ── Appendix ──────────────────────────────────────────────────────────────────

def _appendix(all_sentences: list[str], styles: dict) -> list:
    story: list = [PageBreak()]

    story.append(Paragraph(
        "Appendix \u2014 Full Requirement List",
        ParagraphStyle(
            "QRAppTitle",
            parent=styles["body"],
            fontSize=14, textColor=colors.HexColor("#1e293b"),
            spaceBefore=0, spaceAfter=6,
        ),
    ))
    story.append(HRFlowable(
        width="100%", thickness=1,
        color=colors.HexColor("#cbd5e1"),
    ))
    story.append(Spacer(1, 0.06 * inch))
    story.append(Paragraph(
        "All requirements as analysed, in order. "
        "Use the REQ-NNN identifiers to cross-reference flagged items above.",
        styles["section_def"],
    ))
    story.append(Spacer(1, 0.08 * inch))

    for i, sent in enumerate(all_sentences, 1):
        story.append(Paragraph(
            f"<b>REQ-{i:03d}</b>\u2002{_xml_escape(sent)}",
            styles["appendix_item"],
        ))

    return story


# ── Public API ────────────────────────────────────────────────────────────────

def generate_pdf(
    results: dict[str, list],
    output_path: str = "quality_report.pdf",
) -> None:
    """
    Build a PDF quality report and write it to *output_path*.

    Parameters
    ----------
    results : dict
        Keys: "ambiguity", "feasibility", "verifiability", "singularity".
        Values: lists of detector result objects (one per requirement), each
        exposing a ``to_dict()`` method.
    output_path : str
        Destination file path.
    """
    dim_keys = [k for k, *_ in _DIMS]

    # Convert result objects → plain dicts
    dicts: dict[str, list[dict]] = {
        k: [r.to_dict() for r in results[k]]
        for k in dim_keys
    }

    total          = len(dicts[dim_keys[0]])
    all_sentences  = [d["sentence"] for d in dicts["ambiguity"]]

    # Per-dimension summary stats
    dim_stats: dict[str, dict] = {}
    for k in dim_keys:
        n_flagged    = sum(1 for d in dicts[k] if _is_flagged(k, d))
        dim_stats[k] = {"flagged": n_flagged, "passing": total - n_flagged}

    styles = _build_styles()
    story: list = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.extend(_cover_page(dim_stats, total, styles))
    story.append(PageBreak())

    # ── Quality dimension sections ────────────────────────────────────────────
    for i, (dim_key, *_) in enumerate(_DIMS):
        story.extend(_dimension_section(dim_key, dicts[dim_key], styles))
        if i < len(_DIMS) - 1:
            story.append(PageBreak())

    # ── Appendix ──────────────────────────────────────────────────────────────
    story.extend(_appendix(all_sentences, styles))

    # ── Render PDF ────────────────────────────────────────────────────────────
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_MARGIN,  bottomMargin=_MARGIN,
        title="Software Requirements Quality Report",
        author="ARQM-LITE",
    )
    doc.build(story)
    print(f"[generate_quality_report] Report saved -> {output_path}")


def generate_pdf_bytes(results: list[dict]) -> bytes:
    """
    Convenience wrapper for the Flask route.

    Accepts the list-of-per-requirement-dicts produced by
    ``util.analyzer.analyze_requirements`` (each dict has keys
    "sentence", "ambiguity", "feasibility", "singularity", "verifiability"),
    reshapes it into the per-dimension dict expected by ``generate_pdf``,
    renders the PDF into an in-memory buffer, and returns the raw bytes.
    """
    import io as _io

    dim_keys = [k for k, *_ in _DIMS]

    # Transpose: list-of-requirement-dicts → dict-of-dimension-lists
    dim_results: dict[str, list] = {k: [r[k] for r in results] for k in dim_keys}

    # Convert result objects → plain dicts
    dicts: dict[str, list[dict]] = {
        k: [r.to_dict() for r in dim_results[k]]
        for k in dim_keys
    }

    total         = len(dicts[dim_keys[0]])
    all_sentences = [d["sentence"] for d in dicts["ambiguity"]]

    dim_stats: dict[str, dict] = {}
    for k in dim_keys:
        n_flagged    = sum(1 for d in dicts[k] if _is_flagged(k, d))
        dim_stats[k] = {"flagged": n_flagged, "passing": total - n_flagged}

    styles = _build_styles()
    story: list = []

    story.extend(_cover_page(dim_stats, total, styles))
    story.append(PageBreak())

    for i, (dim_key, *_) in enumerate(_DIMS):
        story.extend(_dimension_section(dim_key, dicts[dim_key], styles))
        if i < len(_DIMS) - 1:
            story.append(PageBreak())

    story.extend(_appendix(all_sentences, styles))

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        leftMargin=_MARGIN, rightMargin=_MARGIN,
        topMargin=_MARGIN,  bottomMargin=_MARGIN,
        title="Software Requirements Quality Report",
        author="ARQM-LITE",
    )
    doc.build(story)
    return buf.getvalue()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Path setup (mirrors util/analyzer.py) ────────────────────────────────
    _ROOT      = Path(__file__).parent
    _TRAIN_DIR = _ROOT / "util" / "training"

    for p in (_TRAIN_DIR, _ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    from training_ambiguity     import AmbiguityDetector      # noqa: E402
    from training_feasibility   import FeasibilityDetector    # noqa: E402
    from training_singularity   import SingularityDetector    # noqa: E402
    from training_verifiability import VerifiabilityDetector  # noqa: E402

    # ── Representative test sentences ────────────────────────────────────────
    # Mix of clean requirements and requirements that exercise every violation
    # category across all four dimensions.
    sentences = [
        # --- Ambiguity ---
        # should → ambiguous modal; "quickly" → vague qualifier
        "The system should respond quickly under heavy load.",
        # Passive voice without agent
        "All user data shall be processed appropriately.",
        # Comparison without baseline
        "The new interface shall be faster and more efficient.",
        # Clean — precise and unambiguous
        "The system shall return a response within 200 ms for 99% of requests.",

        # --- Feasibility ---
        # Impossible absolute: 100% uptime + zero latency
        "The platform shall guarantee 100% uptime and zero latency at all times.",
        # Internal contradiction: synchronous + non-blocking
        "The API shall process requests synchronously and be fully non-blocking.",
        # Unrealistic threshold: sub-1 ms network round-trip
        "The service shall complete every network round-trip within 0.1 ms.",
        # Clean — realistic constraint
        "The system shall handle up to 1 000 concurrent users with a p95 "
        "response time under 500 ms.",

        # --- Verifiability ---
        # Subjective success: "satisfied"
        "Users must be satisfied with the overall look and feel of the interface.",
        # No acceptance criteria: "appropriately"
        "The application shall handle errors appropriately.",
        # Untestable negative: "must never lose data"
        "The system must never lose any user data under any circumstances.",
        # Missing actor: passive obligation
        "Access to sensitive records must be controlled.",
        # Clean — specific, testable criterion
        "The login endpoint shall return HTTP 401 within 300 ms when "
        "credentials are invalid.",

        # --- Singularity ---
        # Multiple actions: validate AND encrypt AND log
        "The system shall validate user input and encrypt sensitive fields "
        "and log all access attempts.",
        # Compound subject: admin and user
        "The admin and the end user shall both confirm the transaction "
        "before it is committed.",
        # Conjunctive conditions: three triggers
        "When the session expires or the token is revoked or the user logs "
        "out, the system shall clear all cached credentials.",
        # Mixed concerns: functional + non-functional in one sentence
        "The payment service shall encrypt card data and respond within 50 ms.",
        # Clean — single, focused requirement
        "The system shall authenticate users via OAuth 2.0.",
    ]

    print(f"[main] Initialising detectors …")
    detectors = {
        "ambiguity":     AmbiguityDetector(
            calibration_data=str(_ROOT / "calibration_data.json")
        ),
        "feasibility":   FeasibilityDetector(
            calibration_data=str(_ROOT / "feasibility_calibration_data.json")
        ),
        "verifiability": VerifiabilityDetector(
            calibration_data=str(_ROOT / "verifiability_calibration_data.json")
        ),
        "singularity":   SingularityDetector(
            calibration_data=str(_TRAIN_DIR / "singularity_calibration_data.json")
        ),
    }

    print("[main] Running analysis …")
    run_results = {
        name: det.analyze_many(sentences)
        for name, det in detectors.items()
    }

    print("[main] Building PDF …")
    generate_pdf(run_results, output_path="quality_report.pdf")
