var document={querySelector:function(){return {dataset:{dim:"ambiguity"},classList:{add:function(){},remove:function(){}},addEventListener:function(){},textContent:""}},querySelectorAll:function(){return {forEach:function(){}}},getElementById:function(){return null}};var SOFT_PROTOTYPES={ambiguity:[],feasibility:[],singularity:[],verifiability:[]};
// ── Tab switching ──
function activateTab(dimKey) {
  document.querySelectorAll('.dim-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.dim-section').forEach(s => s.classList.remove('active'));
  document.querySelector('.dim-tab[data-dim="' + dimKey + '"]').classList.add('active');
  document.getElementById('sec-' + dimKey).classList.add('active');
  currentDim = dimKey;
  var q = document.getElementById('search-input').value;
  if (q) { applySearch(q); } else { initPage(dimKey); }
  if (typeof softActive !== 'undefined' && softActive) { applySoftMatch(); }
}

// ── Card expand/collapse ──
document.querySelectorAll('.req-header').forEach(h => {
  h.addEventListener('click', () => h.closest('.req-card').classList.toggle('open'));
});

// ── Pagination ──
var PAGE_SIZE  = 15;
var pageState  = {};  // { dimKey: currentPage }

function allCards(dimKey) {
  // Exclude soft-only (passing) cards — pagination only applies to hard-flagged cards
  return Array.from(document.querySelectorAll('#cards-' + dimKey + ' .req-card:not([data-soft-only])'));
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

function escapeRe(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

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

// ── Soft Match ──────────────────────────────────────────────────────────────
var softActive = false;
var DIM_COLORS = {
  ambiguity:     '#d97706',
  feasibility:   '#dc2626',
  verifiability: '#2563eb',
  singularity:   '#7c3aed'
};
var DIM_NAMES = {
  ambiguity:     'Amb',
  feasibility:   'Feas',
  verifiability: 'Ver',
  singularity:   'Sing'
};

function stripSoftMark(node) {
  node.querySelectorAll('.soft-match').forEach(function(m) {
    m.replaceWith(document.createTextNode(m.textContent));
  });
  node.normalize();
}

// Build a regex for a prototype phrase with word boundaries where applicable.
function buildSoftRe(pattern) {
  var esc = pattern.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  var pre = /^\w/.test(pattern) ? '\\b' : '';
  var suf = /\w$/.test(pattern) ? '\\b' : '';
  return new RegExp(pre + esc + suf, 'gi');
}

function softMatchesText(text, patterns) {
  return patterns.some(function(p) {
    try { return buildSoftRe(p).test(text); }
    catch(e) { return false; }
  });
}

function highlightSoftNode(node, patterns) {
  if (node.nodeType === 3) {
    var text = node.nodeValue;
    var best = null;
    patterns.forEach(function(p) {
      try {
        var re = buildSoftRe(p);
        var m = re.exec(text);
        if (m && (best === null || m.index < best.idx ||
            (m.index === best.idx && m[0].length > best.len))) {
          best = { idx: m.index, len: m[0].length };
        }
      } catch(e) {}
    });
    if (!best) return;
    var frag = document.createDocumentFragment();
    if (best.idx > 0) frag.appendChild(document.createTextNode(text.slice(0, best.idx)));
    var span = document.createElement('span');
    span.className = 'soft-match';
    span.textContent = text.slice(best.idx, best.idx + best.len);
    frag.appendChild(span);
    var restNode = document.createTextNode(text.slice(best.idx + best.len));
    frag.appendChild(restNode);
    node.parentNode.replaceChild(frag, node);
    highlightSoftNode(restNode, patterns);
  } else if (node.nodeType === 1 && !['SCRIPT','STYLE','MARK'].includes(node.tagName)) {
    Array.from(node.childNodes).forEach(function(c) { highlightSoftNode(c, patterns); });
  }
}

// Build a flat list of all prototype terms across all dimensions (for appendix)
function allSoftProtos() {
  return ['ambiguity','feasibility','singularity','verifiability'].reduce(function(acc, d) {
    return acc.concat(SOFT_PROTOTYPES[d] || []);
  }, []);
}

function applySoftMatch() {
  // Strip any existing soft highlights first (idempotent — safe to call on tab switch)
  document.querySelectorAll('.req-sentence, .app-req-text').forEach(function(el) {
    stripSoftMark(el);
  });

  // ── Dimension card sections ──
  var dims = ['ambiguity','feasibility','verifiability','singularity'];
  dims.forEach(function(dim) {
    var prototypes = (SOFT_PROTOTYPES[dim] || []);
    if (!prototypes.length) return;

    var softCount = 0;
    var cards = document.querySelectorAll('#cards-' + dim + ' .req-card');

    cards.forEach(function(card) {
      var sentEl = card.querySelector('.req-sentence');
      if (!sentEl) return;
      var matched = softMatchesText(sentEl.textContent, prototypes);

      if (matched) highlightSoftNode(sentEl, prototypes);

      if (card.dataset.softOnly === 'true') {
        if (matched) {
          card.style.display = '';
          card.dataset.softMatched = 'true';
          var badge = card.querySelector('.req-badge');
          if (badge) { badge.textContent = 'Soft Match'; badge.className = 'req-badge badge-soft'; }
          softCount++;
        } else {
          card.style.display = 'none';
          delete card.dataset.softMatched;
        }
      }
    });

    var countEl = document.getElementById('soft-count-' + dim);
    if (countEl) countEl.textContent = softCount > 0 ? ' +' + softCount + ' soft' : '';
  });

  // ── Appendix: per-requirement, per-dimension — all three levels ──
  var dims = ['ambiguity','feasibility','verifiability','singularity'];
  document.querySelectorAll('.app-req').forEach(function(reqEl) {
    var sentEl = reqEl.querySelector('.app-req-text');
    if (!sentEl) return;
    var sentText = sentEl.textContent;

    var matchedDims = [];
    dims.forEach(function(dim) {
      var protos = SOFT_PROTOTYPES[dim] || [];
      if (!protos.length) return;
      if (!softMatchesText(sentText, protos)) return;

      var dimEl = reqEl.querySelector('.app-dim[data-dim="' + dim + '"]');
      var hardFlagged = dimEl && dimEl.dataset.flagged === 'true';
      if (!hardFlagged) matchedDims.push(dim);

      if (dimEl) {
        dimEl.dataset.softMatched = 'true';

        // Level 2: mark passing dim header with a ~Soft label
        if (!hardFlagged) {
          var passEl = dimEl.querySelector('.app-dim-pass');
          if (passEl && !passEl.querySelector('.soft-dim-label')) {
            var lbl = document.createElement('span');
            lbl.className = 'soft-dim-label';
            lbl.style.cssText = 'color:' + DIM_COLORS[dim] + ';margin-left:6px;font-size:10px;'
              + 'border:1px dashed ' + DIM_COLORS[dim] + ';border-radius:99px;'
              + 'padding:0 5px;font-weight:700;';
            lbl.textContent = '~Soft';
            passEl.appendChild(lbl);
          }
        }

        // Level 3: highlight soft terms inside violation sentences for this dim
        dimEl.querySelectorAll('.app-viol-sentence').forEach(function(vEl) {
          if (softMatchesText(vEl.textContent, protos)) {
            highlightSoftNode(vEl, protos);
          }
        });

        // Level 3 (passing dim): inject a soft-match block showing the requirement
        // sentence with matched terms highlighted
        if (!hardFlagged) {
          var violsEl = dimEl.querySelector('.app-viols');
          if (violsEl && !violsEl.querySelector('[data-soft-injected]')) {
            var infoDiv = document.createElement('div');
            infoDiv.className = 'app-viol';
            infoDiv.dataset.softInjected = 'true';
            infoDiv.style.cssText = '--viol-color:' + DIM_COLORS[dim]
              + ';border-left:3px dashed ' + DIM_COLORS[dim] + ';margin-top:8px;';
            var sentSpan = document.createElement('div');
            sentSpan.className = 'app-viol-sentence';
            sentSpan.textContent = sentEl.textContent;
            var metaSpan = document.createElement('div');
            metaSpan.className = 'app-viol-meta';
            metaSpan.style.color = DIM_COLORS[dim];
            metaSpan.textContent = '~ soft match — no hard violation detected';
            infoDiv.appendChild(sentSpan);
            infoDiv.appendChild(metaSpan);
            violsEl.appendChild(infoDiv);
            highlightSoftNode(sentSpan, protos);
          }
        }
      }
    });

    // Level 1: highlight soft terms in requirement text (all dims combined)
    var allProtos = allSoftProtos();
    if (softMatchesText(sentText, allProtos)) {
      highlightSoftNode(sentEl, allProtos);
    }

    // Level 1: inject per-dimension soft badges into .app-flag-badges
    if (matchedDims.length) {
      var badgesEl = reqEl.querySelector('.app-flag-badges');
      if (badgesEl) {
        matchedDims.forEach(function(dim) {
          var span = document.createElement('span');
          span.className = 'app-flag-badge';
          span.dataset.softBadge = dim;
          span.style.background = DIM_COLORS[dim] + '22';
          span.style.color = DIM_COLORS[dim];
          span.style.border = '1px dashed ' + DIM_COLORS[dim];
          span.textContent = '~' + DIM_NAMES[dim];
          badgesEl.appendChild(span);
        });
      }
    }
  });
}

function clearSoftMatch() {
  document.querySelectorAll('.req-card[data-soft-only="true"]').forEach(function(card) {
    card.style.display = 'none';
    delete card.dataset.softMatched;
    var badge = card.querySelector('.req-badge');
    if (badge) { badge.textContent = 'Pass'; badge.className = 'req-badge badge-pass'; }
  });
  // Level 1 card sentences + appendix requirement texts
  document.querySelectorAll('.req-sentence, .app-req-text').forEach(function(el) {
    stripSoftMark(el);
  });
  // Level 3: strip soft marks from violation sentences
  document.querySelectorAll('.app-viol-sentence').forEach(function(el) {
    stripSoftMark(el);
  });
  // Level 2: remove ~Soft labels from passing dim headers
  document.querySelectorAll('.soft-dim-label').forEach(function(el) { el.remove(); });
  // Level 3: remove injected soft-match info blocks
  document.querySelectorAll('[data-soft-injected]').forEach(function(el) { el.remove(); });
  // Level 1: remove soft badges injected into appendix rows
  document.querySelectorAll('.app-flag-badge[data-soft-badge]').forEach(function(el) {
    el.remove();
  });
  // Clear soft-matched markers on app-dim rows
  document.querySelectorAll('.app-dim[data-soft-matched]').forEach(function(el) {
    delete el.dataset.softMatched;
  });
  document.querySelectorAll('[id^="soft-count-"]').forEach(function(el) { el.textContent = ''; });
}

function toggleSoftMatch() {
  softActive = !softActive;
  var btn = document.getElementById('soft-btn');
  if (softActive) {
    btn.classList.add('active');
    applySoftMatch();
  } else {
    btn.classList.remove('active');
    clearSoftMatch();
  }
}

// ── Feedback (false-positive suppression) ────────────────────────────────────
function sendFeedback(btn) {
  if (btn.classList.contains('fb-done')) return;
  var term = btn.getAttribute('data-term');
  btn.disabled = true;
  btn.textContent = '…';
  fetch('/api/feedback', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({term: term})
  }).then(function(r) {
    if (r.ok) {
      btn.textContent = 'Suppressed';
      btn.classList.add('fb-done');
      btn.title = 'Marked as false positive — takes effect on next server restart';
    } else {
      btn.textContent = 'Error';
      btn.disabled = false;
    }
  }).catch(function() {
    btn.textContent = '✗';
    btn.disabled = false;
  });
}
