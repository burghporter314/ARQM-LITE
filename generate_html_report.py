"""
generate_html_report.py
=======================
Generates a self-contained HTML quality report for the ARQM-LITE system.

Public API mirrors generate_quality_report.py:

    generate_html_bytes(results: list[dict]) -> bytes

``results`` is the list-of-per-requirement-dicts produced by
``util.analyzer.analyze_requirements`` — each dict has keys
"sentence", "ambiguity", "feasibility", "singularity", "verifiability".

The returned bytes are UTF-8 encoded HTML with all CSS and JS inlined,
suitable for direct browser delivery or saving as a .html file.
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any


def _highlight_span_in_sentence(span: str, sentence: str, color: str) -> str:
    """Return the full sentence with *span* highlighted.

    If *span* is found verbatim (case-insensitive) it is wrapped in a
    coloured <mark>.  When the span cannot be located (phantom span) the
    full sentence is returned un-highlighted so the user always has context.
    """
    escaped_sentence = html.escape(sentence)
    if not span:
        return escaped_sentence
    try:
        pattern = re.compile(re.escape(span), re.IGNORECASE)
        m = pattern.search(sentence)
        if m:
            before = html.escape(sentence[:m.start()])
            mid    = html.escape(sentence[m.start():m.end()])
            after  = html.escape(sentence[m.end():])
            return (f'{before}'
                    f'<mark style="background:{color}22;border-bottom:2px solid {color};'
                    f'border-radius:2px;padding:0 1px">{mid}</mark>'
                    f'{after}')
    except re.error:
        pass
    # Phantom span — show the plain sentence as context
    return escaped_sentence

# ── Dimension registry (mirrors generate_quality_report.py) ──────────────────
_DIMS: list[tuple[str, str, str, str]] = [
    ("ambiguity",     "Ambiguity",     "#d97706",
     "Flags vague or imprecise language that cannot be objectively tested or measured."),
    ("feasibility",   "Feasibility",   "#dc2626",
     "Flags physically impossible constraints, internal contradictions, or unrealistic numeric thresholds."),
    ("verifiability", "Verifiability", "#2563eb",
     "Flags requirements that have no testable pass/fail condition or rely on subjective judgement."),
    ("singularity",   "Singularity",   "#7c3aed",
     "Flags requirements that bundle more than one distinct concern, actor, or action."),
]

_DIM_NAME  = {k: n for k, n, _, _  in _DIMS}
_DIM_HEX   = {k: h for k, _, h, _  in _DIMS}
_DIM_DEF   = {k: d for k, _, _, d  in _DIMS}

_IS_GOOD_KEY = {
    "ambiguity":     "is_ambiguous",
    "feasibility":   "is_feasible",
    "verifiability": "is_verifiable",
    "singularity":   "is_singular",
}
_VIOLS_KEY = {
    "ambiguity":     "spans",
    "feasibility":   "violations",
    "verifiability": "violations",
    "singularity":   "violations",
}

_HIGHLIGHT_RE = re.compile(r">>(.*?)<<")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_flagged(dim_key: str, d: dict) -> bool:
    val = d.get(_IS_GOOD_KEY[dim_key])
    return bool(val) if dim_key == "ambiguity" else not bool(val)


def _violations(dim_key: str, d: dict) -> list[dict]:
    return d.get(_VIOLS_KEY[dim_key], [])


def _tint_hex(hex_color: str, opacity: float = 0.25) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r * opacity + 255 * (1 - opacity)),
        int(g * opacity + 255 * (1 - opacity)),
        int(b * opacity + 255 * (1 - opacity)),
    )


def _render_highlighted(text: str, color: str) -> str:
    """Convert >>span<< markers to <mark> tags with inline colour."""
    bg   = _tint_hex(color)
    parts = _HIGHLIGHT_RE.split(text)
    out: list[str] = []
    for i, part in enumerate(parts):
        escaped = html.escape(part)
        if i % 2 == 0:
            out.append(escaped)
        else:
            out.append(
                f'<mark style="background:{bg};color:{color};'
                f'font-weight:600;border-radius:3px;padding:0 2px">'
                f'{escaped}</mark>'
            )
    return "".join(out)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --amb:  #d97706;
  --feas: #dc2626;
  --ver:  #2563eb;
  --sing: #7c3aed;
  --bg:   #f8fafc;
  --card: #ffffff;
  --border: #e2e8f0;
  --text: #1e293b;
  --muted: #64748b;
  --radius: 8px;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.6;
}

/* ── Top bar ── */
#topbar {
  position: sticky;
  top: 0;
  z-index: 100;
  background: #1e293b;
  color: #f1f5f9;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 24px;
  height: 52px;
  box-shadow: 0 2px 8px rgba(0,0,0,.25);
}
#topbar h1 { font-size: 15px; font-weight: 700; letter-spacing: .4px; flex: 1; }
#topbar .ts { font-size: 11px; color: #94a3b8; white-space: nowrap; }

/* ── Search bar ── */
#search-bar {
  background: #0f172a;
  padding: 10px 24px;
  display: flex;
  align-items: center;
  gap: 10px;
  border-bottom: 1px solid #334155;
  position: sticky;
  top: 52px;
  z-index: 99;
}
#search-input {
  flex: 1;
  max-width: 480px;
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid #475569;
  background: #1e293b;
  color: #f1f5f9;
  font-size: 13px;
  outline: none;
}
#search-input:focus { border-color: #3b82f6; }
#search-input::placeholder { color: #64748b; }
#search-count { font-size: 12px; color: #94a3b8; min-width: 80px; }

/* ── Dimension nav tabs ── */
#dim-nav {
  display: flex;
  gap: 4px;
  padding: 0 24px;
  background: #0f172a;
  border-bottom: 1px solid #334155;
  overflow-x: auto;
}
.dim-tab {
  padding: 8px 16px;
  border: none;
  background: transparent;
  color: #94a3b8;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  border-bottom: 3px solid transparent;
  white-space: nowrap;
  transition: color .15s, border-color .15s;
}
.dim-tab:hover  { color: #f1f5f9; }
.dim-tab.active { color: #f1f5f9; border-bottom-color: var(--tab-color, #3b82f6); }

/* ── Main layout ── */
main { max-width: 1200px; margin: 0 auto; padding: 24px 24px 48px; }

/* ── Summary banner ── */
#summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
  margin-bottom: 28px;
}
.stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  border-left: 4px solid var(--accent);
}
.stat-card .label { font-size: 11px; font-weight: 700; text-transform: uppercase;
                    letter-spacing: .6px; color: var(--muted); }
.stat-card .counts { display: flex; align-items: baseline; gap: 8px; }
.stat-card .big    { font-size: 28px; font-weight: 800; color: var(--accent); }
.stat-card .of     { font-size: 13px; color: var(--muted); }
.stat-card .bar    { height: 5px; background: #e2e8f0; border-radius: 3px; overflow: hidden; }
.stat-card .bar-fill { height: 100%; border-radius: 3px; background: var(--accent); }

/* ── Section ── */
.dim-section { display: none; }
.dim-section.active { display: block; }

.section-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.section-header .dot {
  width: 14px; height: 14px; border-radius: 50%;
  background: var(--accent); flex-shrink: 0;
}
.section-header h2 { font-size: 18px; font-weight: 700; }
.section-header .def { color: var(--muted); font-size: 13px; }

/* ── Requirement cards ── */
.req-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 10px;
  overflow: hidden;
  transition: box-shadow .15s;
}
.req-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,.08); }
.req-card.hidden { display: none; }

.req-header {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 16px;
  cursor: pointer;
  user-select: none;
}
.req-num {
  font-size: 11px;
  font-weight: 700;
  color: var(--muted);
  min-width: 36px;
  padding-top: 2px;
}
.req-sentence { flex: 1; font-size: 13px; line-height: 1.55; }
.req-badge {
  font-size: 11px;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: 99px;
  white-space: nowrap;
  flex-shrink: 0;
}
.badge-flagged { background: #fee2e2; color: #991b1b; }
.badge-pass    { background: #dcfce7; color: #166534; }
.chevron { color: var(--muted); font-size: 12px; padding-top: 3px; transition: transform .2s; }
.req-card.open .chevron { transform: rotate(90deg); }

.req-body {
  display: none;
  padding: 0 16px 14px 64px;
  border-top: 1px solid var(--border);
}
.req-card.open .req-body { display: block; }

.violation {
  margin-top: 10px;
  padding: 10px 14px;
  border-radius: 6px;
  border-left: 3px solid var(--accent);
  background: var(--bg);
  font-size: 13px;
}
.violation .viol-text { font-weight: 600; margin-bottom: 4px; }
.violation .viol-meta { font-size: 11px; color: var(--muted); display: flex; gap: 12px; }
.violation .viol-suggestion { margin-top: 6px; font-size: 12px; color: #475569;
                               padding: 6px 10px; background: #f1f5f9; border-radius: 4px; }

/* ── Appendix drill-down ── */
#appendix { margin-top: 32px; }
#appendix > h2 { font-size: 16px; font-weight: 700; margin-bottom: 12px; }

/* Level 1 — requirement row */
.app-req {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 6px;
  overflow: hidden;
}
.app-req-header {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  padding: 10px 14px;
  cursor: pointer;
  user-select: none;
}
.app-req-header:hover { background: #f8fafc; }
.app-req-num { font-size: 11px; font-weight: 700; color: var(--muted); min-width: 32px; padding-top: 2px; }
.app-req-text { flex: 1; font-size: 13px; line-height: 1.5; }
.app-flag-badges { display: flex; gap: 4px; flex-shrink: 0; flex-wrap: wrap; align-items: center; }
.app-flag-badge {
  font-size: 10px; font-weight: 700;
  padding: 1px 7px; border-radius: 99px;
  white-space: nowrap;
}
.app-pass-badge { font-size: 11px; color: #86efac; }
.app-chevron { color: var(--muted); font-size: 11px; padding-top: 3px; transition: transform .2s; flex-shrink: 0; }
.app-req.open > .app-req-header .app-chevron { transform: rotate(90deg); }

/* Level 2 — dimension groups */
.app-dims { display: none; border-top: 1px solid var(--border); }
.app-req.open > .app-dims { display: block; }

.app-dim {
  border-bottom: 1px solid var(--border);
}
.app-dim:last-child { border-bottom: none; }

.app-dim-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 14px 8px 48px;
  cursor: pointer;
  user-select: none;
  font-size: 13px;
}
.app-dim-header:hover { background: #f8fafc; }
.app-dim-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
.app-dim-name { font-weight: 600; flex: 1; }
.app-dim-count { font-size: 11px; color: var(--muted); }
.app-dim-pass { font-size: 11px; color: #16a34a; }
.app-dim-chevron { color: var(--muted); font-size: 11px; transition: transform .2s; }
.app-dim.open > .app-dim-header .app-dim-chevron { transform: rotate(90deg); }

/* Level 3 — individual violations */
.app-viols { display: none; padding: 6px 14px 10px 68px; background: var(--bg); }
.app-dim.open > .app-viols { display: block; }

.app-viol {
  margin-top: 8px;
  padding: 10px 12px;
  border-radius: 6px;
  border-left: 3px solid var(--viol-color, #94a3b8);
  background: var(--card);
  font-size: 13px;
}
.app-viol-sentence { margin-bottom: 5px; line-height: 1.55; }
.app-viol-meta { font-size: 11px; color: var(--muted); display: flex; gap: 12px; }
.app-viol-sugg { margin-top: 6px; font-size: 12px; color: #475569;
                 padding: 5px 9px; background: #f1f5f9; border-radius: 4px; }
.app-no-issues { font-size: 12px; color: #16a34a; padding: 6px 0; }

/* ── Paginator ── */
.paginator {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  padding: 16px 0 4px;
}
.pager-btn {
  padding: 5px 14px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--card);
  color: var(--text);
  font-size: 12px;
  cursor: pointer;
  transition: background .15s;
}
.pager-btn:hover:not(:disabled) { background: var(--bg); border-color: var(--accent); color: var(--accent); }
.pager-btn:disabled { opacity: .35; cursor: default; }
.pager-info { font-size: 12px; color: var(--muted); min-width: 100px; text-align: center; }

/* ── Search highlights ── */
.search-match { background: #fef08a; border-radius: 2px; }

/* ── Scrollbar ── */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
"""

# ── JavaScript ────────────────────────────────────────────────────────────────

_JS = """
// ── Tab switching ──
function activateTab(dimKey) {
  document.querySelectorAll('.dim-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.dim-section').forEach(s => s.classList.remove('active'));
  document.querySelector('.dim-tab[data-dim="' + dimKey + '"]').classList.add('active');
  document.getElementById('sec-' + dimKey).classList.add('active');
  currentDim = dimKey;
  var q = document.getElementById('search-input').value;
  if (q) { applySearch(q); } else { initPage(dimKey); }
}

// ── Card expand/collapse ──
document.querySelectorAll('.req-header').forEach(h => {
  h.addEventListener('click', () => h.closest('.req-card').classList.toggle('open'));
});

// ── Pagination ──
var PAGE_SIZE  = 15;
var pageState  = {};  // { dimKey: currentPage }

function allCards(dimKey) {
  return Array.from(document.querySelectorAll('#cards-' + dimKey + ' .req-card'));
}

function initPage(dimKey) {
  if (!pageState[dimKey]) pageState[dimKey] = 1;
  renderPage(dimKey);
}

function renderPage(dimKey) {
  var searching = document.getElementById('search-input').value.trim() !== '';
  var cards  = allCards(dimKey);
  var pager  = document.getElementById('pager-' + dimKey);
  var pinfo  = document.getElementById('pinfo-' + dimKey);

  if (searching) {
    // Search active — show all matching cards, hide paginator
    pager.style.display = 'none';
    return;
  }

  pager.style.display = cards.length > PAGE_SIZE ? 'flex' : 'none';
  var page   = pageState[dimKey] || 1;
  var total  = Math.ceil(cards.length / PAGE_SIZE) || 1;
  page = Math.min(Math.max(page, 1), total);
  pageState[dimKey] = page;

  var start = (page - 1) * PAGE_SIZE;
  var end   = start + PAGE_SIZE;

  cards.forEach(function(c, idx) {
    c.style.display = (idx >= start && idx < end) ? '' : 'none';
  });

  pinfo.textContent = 'Page ' + page + ' of ' + total + ' (' + cards.length + ' flagged)';
  pager.querySelector('.pager-btn:first-child').disabled = page <= 1;
  pager.querySelector('.pager-btn:last-child').disabled  = page >= total;
}

function changePage(dimKey, delta) {
  pageState[dimKey] = (pageState[dimKey] || 1) + delta;
  renderPage(dimKey);
  document.getElementById('sec-' + dimKey).scrollIntoView({behavior:'smooth', block:'start'});
}

// ── Search / filter ──
var currentDim = document.querySelector('.dim-tab.active').dataset.dim;

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&'); }

function stripMark(node) {
  node.querySelectorAll('.search-match').forEach(m => {
    m.replaceWith(document.createTextNode(m.textContent));
  });
  node.normalize();
}

function highlightText(node, re) {
  if (node.nodeType === 3) {
    var m, text = node.nodeValue, frag = document.createDocumentFragment(), last = 0;
    re.lastIndex = 0;
    while ((m = re.exec(text)) !== null) {
      frag.appendChild(document.createTextNode(text.slice(last, m.index)));
      var span = document.createElement('span');
      span.className = 'search-match';
      span.textContent = m[0];
      frag.appendChild(span);
      last = m.index + m[0].length;
    }
    if (last > 0) { frag.appendChild(document.createTextNode(text.slice(last))); node.parentNode.replaceChild(frag, node); }
  } else if (node.nodeType === 1 && !['SCRIPT','STYLE'].includes(node.tagName)) {
    Array.from(node.childNodes).forEach(c => highlightText(c, re));
  }
}

function applySearch(query) {
  var section = document.getElementById('sec-' + currentDim);
  var cards   = section.querySelectorAll('.req-card');
  var count   = 0;
  var q       = query.trim().toLowerCase();

  cards.forEach(card => {
    stripMark(card);
    if (!q) {
      card.classList.remove('hidden');
      card.style.display = '';
      return;
    }
    var text = card.textContent.toLowerCase();
    if (text.includes(q)) {
      card.classList.remove('hidden');
      card.style.display = '';
      highlightText(card, new RegExp(escapeRe(query), 'gi'));
      count++;
    } else {
      card.classList.add('hidden');
      card.style.display = 'none';
    }
  });

  // Show/hide paginator based on whether search is active
  renderPage(currentDim);

  var el = document.getElementById('search-count');
  if (q) {
    el.textContent = count + ' match' + (count !== 1 ? 'es' : '');
  } else {
    el.textContent = cards.length + ' requirement' + (cards.length !== 1 ? 's' : '');
    renderPage(currentDim);
  }
}

document.getElementById('search-input').addEventListener('input', function() {
  applySearch(this.value);
});

// ── Appendix drill-down toggles ──
document.querySelectorAll('.app-req-header').forEach(function(h) {
  h.addEventListener('click', function() {
    h.closest('.app-req').classList.toggle('open');
  });
});

document.querySelectorAll('.app-dim-header').forEach(function(h) {
  h.addEventListener('click', function(e) {
    e.stopPropagation();
    h.closest('.app-dim').classList.toggle('open');
  });
});

// Initialise pagination for all dims, activate first tab
['ambiguity','feasibility','verifiability','singularity'].forEach(function(d) { initPage(d); });
activateTab(currentDim);
"""


# ── HTML builder ──────────────────────────────────────────────────────────────

def _summary_cards(dim_stats: dict[str, dict], total: int) -> str:
    cards: list[str] = []
    for dim_key, dim_name, color, _ in _DIMS:
        flagged  = dim_stats[dim_key]["flagged"]
        pct      = round(flagged / total * 100) if total else 0
        cards.append(f"""
        <div class="stat-card" style="--accent:{color}">
          <div class="label">{html.escape(dim_name)}</div>
          <div class="counts">
            <span class="big">{flagged}</span>
            <span class="of">/ {total} flagged</span>
          </div>
          <div class="bar"><div class="bar-fill" style="width:{pct}%"></div></div>
        </div>""")
    return "\n".join(cards)


def _requirement_cards(dim_key: str, dicts: list[dict], color: str) -> str:
    """Render only flagged requirements as collapsed cards with violations inside."""
    cards: list[str] = []
    card_idx = 0  # 0-based index within flagged cards, used for pagination

    for i, d in enumerate(dicts, 1):
        flagged = _is_flagged(dim_key, d)
        if not flagged:
            continue

        sentence = d.get("sentence", "")
        viols    = _violations(dim_key, d)

        # Build highlighted sentence header
        combined_hl = sentence
        for v in viols:
            hl = v.get("highlighted", "")
            if ">>" in hl:
                combined_hl = hl
                break
        rendered_sentence = _render_highlighted(combined_hl, color)

        # Violation detail rows
        viol_html = ""
        for v in viols:
            vtext_raw = v.get("text", "")
            vreason   = html.escape(v.get("reason", ""))
            vscore    = v.get("score", 0.0)
            vsugg     = v.get("suggestion") or ""
            hl        = v.get("highlighted", "")
            if ">>" in hl:
                rendered_v = _render_highlighted(hl, color)
            else:
                rendered_v = _highlight_span_in_sentence(vtext_raw, sentence, color)
            sugg_block = (f'<div class="viol-suggestion">💡 {html.escape(vsugg)}</div>'
                          if vsugg else "")
            viol_html += f"""
            <div class="violation" style="--accent:{color}">
              <div class="viol-text">{rendered_v}</div>
              <div class="viol-meta">
                <span>reason: {vreason}</span>
                <span>score: {vscore:.2f}</span>
              </div>
              {sugg_block}
            </div>"""

        cards.append(f"""
      <div class="req-card" data-dim="{dim_key}" data-card-idx="{card_idx}">
        <div class="req-header">
          <span class="req-num">#{i}</span>
          <span class="req-sentence">{rendered_sentence}</span>
          <span class="chevron">&#9658;</span>
        </div>
        <div class="req-body">{viol_html}</div>
      </div>""")
        card_idx += 1

    if not cards:
        return '<p style="color:#64748b;padding:16px">No violations detected in this dimension.</p>'
    return "\n".join(cards)


def _dim_sections(dicts: dict[str, list[dict]]) -> str:
    sections: list[str] = []
    for i, (dim_key, dim_name, color, definition) in enumerate(_DIMS):
        active    = "active" if i == 0 else ""
        n_flagged = sum(1 for d in dicts[dim_key] if _is_flagged(dim_key, d))
        sections.append(f"""
    <section class="dim-section {active}" id="sec-{dim_key}" style="--accent:{color}">
      <div class="section-header">
        <div class="dot"></div>
        <h2>{html.escape(dim_name)}</h2>
        <span class="def">{html.escape(definition)}</span>
      </div>
      <div class="cards-container" id="cards-{dim_key}">
        {_requirement_cards(dim_key, dicts[dim_key], color)}
      </div>
      <div class="paginator" id="pager-{dim_key}" data-dim="{dim_key}" data-total="{n_flagged}">
        <button class="pager-btn" onclick="changePage('{dim_key}',-1)">&#8592; Prev</button>
        <span class="pager-info" id="pinfo-{dim_key}"></span>
        <button class="pager-btn" onclick="changePage('{dim_key}',1)">Next &#8594;</button>
      </div>
    </section>""")
    return "\n".join(sections)


def _appendix(all_sentences: list[str], dicts: dict[str, list[dict]]) -> str:
    req_items: list[str] = []

    for i, sent in enumerate(all_sentences, 1):

        # ── Build per-dimension violation data ───────────────────────────────
        dim_html_parts: list[str] = []
        flag_badges:    list[str] = []
        any_flagged = False

        for dim_key, dim_name, color, _ in _DIMS:
            d       = dicts[dim_key][i - 1]
            flagged = _is_flagged(dim_key, d)
            viols   = _violations(dim_key, d)

            if flagged:
                any_flagged = True
                n = len(viols)
                flag_badges.append(
                    f'<span class="app-flag-badge" '
                    f'style="background:{color}22;color:{color}">'
                    f'{html.escape(dim_name[:3])} {n}</span>'
                )

            # Level 3 — individual violations
            viol_items: list[str] = []
            for v in viols:
                vtext_raw = v.get("text", "")
                hl        = v.get("highlighted", "")
                vreason   = html.escape(v.get("reason", ""))
                vscore    = v.get("score", 0.0)
                vsugg     = v.get("suggestion") or ""

                if ">>" in hl:
                    rendered = _render_highlighted(hl, color)
                else:
                    rendered = _highlight_span_in_sentence(vtext_raw, sent, color)

                sugg_block = (
                    f'<div class="app-viol-sugg">💡 {html.escape(vsugg)}</div>'
                    if vsugg else ""
                )
                viol_items.append(
                    f'<div class="app-viol" style="--viol-color:{color}">'
                    f'<div class="app-viol-sentence">{rendered}</div>'
                    f'<div class="app-viol-meta">'
                    f'<span>reason: {vreason}</span>'
                    f'<span>score: {vscore:.2f}</span>'
                    f'</div>'
                    f'{sugg_block}'
                    f'</div>'
                )

            viols_html = (
                "".join(viol_items) if viol_items
                else '<p class="app-no-issues">✓ No issues detected.</p>'
            )

            # Level 2 — dimension header
            count_label = (
                f'<span class="app-dim-count">{len(viols)} violation{"s" if len(viols) != 1 else ""}</span>'
                if flagged
                else '<span class="app-dim-pass">✓ Pass</span>'
            )
            chevron = f'<span class="app-dim-chevron">&#9658;</span>' if flagged else ''

            dim_html_parts.append(f"""
          <div class="app-dim">
            <div class="app-dim-header">
              <span class="app-dim-dot" style="background:{color}"></span>
              <span class="app-dim-name">{html.escape(dim_name)}</span>
              {count_label}
              {chevron}
            </div>
            <div class="app-viols">{viols_html}</div>
          </div>""")

        # ── Level 1 — requirement row ─────────────────────────────────────
        badges_html = (
            "".join(flag_badges) if flag_badges
            else '<span class="app-pass-badge">✓</span>'
        )

        req_items.append(f"""
      <div class="app-req">
        <div class="app-req-header">
          <span class="app-req-num">#{i}</span>
          <span class="app-req-text">{html.escape(sent)}</span>
          <span class="app-flag-badges">{badges_html}</span>
          <span class="app-chevron">&#9658;</span>
        </div>
        <div class="app-dims">{"".join(dim_html_parts)}
        </div>
      </div>""")

    return f"""
  <section id="appendix">
    <h2>All Requirements</h2>
    {"".join(req_items)}
  </section>"""


def _dim_tabs() -> str:
    tabs: list[str] = []
    for i, (dim_key, dim_name, color, _) in enumerate(_DIMS):
        active = "active" if i == 0 else ""
        tabs.append(
            f'<button class="dim-tab {active}" data-dim="{dim_key}" '
            f'style="--tab-color:{color}" onclick="activateTab(\'{dim_key}\')">'
            f'{html.escape(dim_name)}</button>'
        )
    return "\n".join(tabs)


def _build_html(dicts: dict[str, list[dict]], total: int) -> str:
    dim_stats: dict[str, dict] = {}
    for dim_key, *_ in _DIMS:
        n_flagged        = sum(1 for d in dicts[dim_key] if _is_flagged(dim_key, d))
        dim_stats[dim_key] = {"flagged": n_flagged, "passing": total - n_flagged}

    all_sentences = [d["sentence"] for d in dicts["ambiguity"]]
    timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ARQM-LITE Quality Report — {timestamp}</title>
  <style>{_CSS}</style>
</head>
<body>

<div id="topbar">
  <h1>ARQM-LITE — Software Requirements Quality Report</h1>
  <span class="ts">Generated {timestamp} &nbsp;·&nbsp; {total} requirements</span>
</div>

<div id="search-bar">
  <input id="search-input" type="search" placeholder="Search requirements…" autocomplete="off">
  <span id="search-count"></span>
</div>

<div id="dim-nav">
  {_dim_tabs()}
</div>

<main>
  <div id="summary">
    {_summary_cards(dim_stats, total)}
  </div>

  {_dim_sections(dicts)}

  {_appendix(all_sentences, dicts)}
</main>

<script>{_JS}</script>
</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────────

def generate_html_bytes(results: list[dict]) -> bytes:
    """
    Accepts the list-of-per-requirement-dicts produced by
    ``util.analyzer.analyze_requirements``, builds a self-contained HTML
    quality report, and returns the raw UTF-8 bytes.
    """
    dim_keys = [k for k, *_ in _DIMS]

    dicts: dict[str, list[dict]] = {
        k: [r[k].to_dict() for r in results]
        for k in dim_keys
    }
    total = len(results)
    return _build_html(dicts, total).encode("utf-8")
