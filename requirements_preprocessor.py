"""
requirements_preprocessor.py
=============================
Cleans raw text extracted from PDF/DOCX requirement documents before the text
is passed to the quality detectors.

Public API
----------
    def preprocess(raw_text: str) -> list[str]:
        ...

The function applies the following pipeline, in order:
  1. Strip section-number / REQ-NNN prefixes from each line.
  2. Strip editorial annotation artifacts: (removed), (TBD), etc.
  3. Split bullet lists into one item per candidate.
  4. Extract Description sentences and Given/When/Then clauses from user-story
     blocks; discard metadata fields.
  5. Strip residual metadata tokens left after splitting.
  6. Reject candidates with fewer than 6 words or fewer than 30 characters.
  7. Deduplicate (case-insensitive), keeping first occurrence.
"""

from __future__ import annotations

import re

# ── Compiled patterns ─────────────────────────────────────────────────────────

# Leading section numbers: "1.", "1.1", "6.2.3", optionally preceded by REQ-NNN
_SECTION_PREFIX_RE = re.compile(
    r"^\s*(?:REQ-\d+\s*)?(?:\d+\.)*\d+\s+",
    re.IGNORECASE,
)

# Editorial annotations that should be removed wherever they appear
_ANNOTATION_RE = re.compile(
    r"\(\s*(?:removed|deleted|TBD|TODO|N/A|reserved|intentionally\s+blank)\s*\)"
    r"|\[\s*(?:removed|deleted|TBD|TODO|N/A|reserved)\s*\]",
    re.IGNORECASE,
)

# Bullet separators (inline or newline)
_BULLET_SPLIT_RE = re.compile(r"\s+•\s+|\n\s*•\s*")

# User story block markers (non-capturing, used to split the block)
_US_FIELD_RE = re.compile(
    r"(?:User\s+Story|Mapped\s+Requirement|Priority|Estimation|Description"
    r"|Acceptance\s+Criteria)\s*:",
    re.IGNORECASE,
)

# Gherkin clause starters (each becomes an individual candidate)
_GHERKIN_SPLIT_RE = re.compile(
    r"(?<!\w)(?:Given|When|Then|And|But)(?=\s)",
    re.IGNORECASE,
)

# Residual metadata tokens to strip from the start / anywhere in a candidate
_METADATA_TOKEN_RE = re.compile(
    r"^\s*(?:Priority|Estimation|Description|Mapped\s+Requirement|User\s+Story)\s*:\s*",
    re.IGNORECASE,
)

# Bare numeric / bullet artifacts left after splitting (standalone numbers,
# section IDs, or strings made entirely of digits, dots, bullets, dashes)
_BARE_ARTIFACT_RE = re.compile(r"^\s*[\d\.\-\*•]+\s*$")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_user_story_block(text: str) -> bool:
    """Return True if *text* looks like a user-story metadata block."""
    markers = re.findall(_US_FIELD_RE, text)
    return len(markers) >= 2


def _extract_from_user_story(block: str) -> list[str]:
    """
    Pull out only the meaningful content from a user-story block:
      - The Description sentence (the "As a … I want … so that …" sentence).
      - Each individual Given / When / Then clause.

    Everything else (User Story name, Mapped Requirement, Priority,
    Estimation field values) is discarded.
    """
    candidates: list[str] = []

    # ── Extract Description ───────────────────────────────────────────────────
    desc_match = re.search(
        r"Description\s*:\s*(.+?)(?=\s*(?:Acceptance\s+Criteria|$))",
        block,
        re.IGNORECASE | re.DOTALL,
    )
    if desc_match:
        desc_text = desc_match.group(1).strip()
        # A description may itself contain multiple sentences; split on "."
        # but keep the delimiter so we don't lose final punctuation.
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", desc_text)
        candidates.extend(s.strip() for s in sentences if s.strip())

    # ── Extract Acceptance Criteria Given/When/Then ───────────────────────────
    ac_match = re.search(
        r"Acceptance\s+Criteria\s*:\s*(.+)",
        block,
        re.IGNORECASE | re.DOTALL,
    )
    if ac_match:
        ac_text = ac_match.group(1).strip()
        # Split on Gherkin keywords; reconstruct each clause with its keyword
        parts = re.split(r"\b(Given|When|Then|And|But)\b", ac_text, flags=re.IGNORECASE)
        # parts = ['', 'Given', 'I am ...', 'When', '...', ...]
        i = 1
        while i < len(parts) - 1:
            keyword = parts[i].strip()
            clause  = parts[i + 1].strip()
            if keyword and clause:
                candidates.append(f"{keyword} {clause}")
            i += 2

    return [c for c in candidates if c]


def _strip_line_prefix(line: str) -> str:
    """Remove leading section-number / REQ prefix from a single line."""
    return _SECTION_PREFIX_RE.sub("", line, count=1)


def _strip_annotations(text: str) -> str:
    return _ANNOTATION_RE.sub("", text).strip()


def _strip_metadata_tokens(text: str) -> str:
    """Remove leading metadata tokens left after field splitting."""
    text = _METADATA_TOKEN_RE.sub("", text).strip()
    return text


def _clean_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def _is_trivial(text: str) -> bool:
    """True if the candidate is too short to be a meaningful requirement."""
    cleaned = _clean_whitespace(text)
    words   = cleaned.split()
    return len(words) < 6 or len(cleaned) < 30


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess(raw_text: str) -> list[str]:
    """
    Clean *raw_text* extracted from a requirements document and return a list
    of individual requirement strings ready to pass to the quality detectors.

    Steps applied in order:
        1. Strip section-number prefixes (per line).
        2. Strip editorial annotation artifacts.
        3. Split bullet lists into individual items.
        4. Extract Description / Given/When/Then from user-story blocks.
        5. Strip residual metadata tokens.
        6. Reject short / trivial candidates (< 6 words or < 30 chars).
        7. Deduplicate (case-insensitive), keeping first occurrence.
    """
    candidates: list[str] = []

    # ── Step 1 & 2: per-line prefix stripping and annotation removal ──────────
    lines: list[str] = raw_text.splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        line = _strip_line_prefix(line)
        line = _strip_annotations(line)
        line = _clean_whitespace(line)
        if line:
            cleaned_lines.append(line)

    # Rejoin for block-level processing
    rejoined = "\n".join(cleaned_lines)

    # ── Step 3: split bullet lists ────────────────────────────────────────────
    # First check inline bullets then newline bullets
    bullet_chunks: list[str] = []
    for chunk in rejoined.split("\n"):
        if _BULLET_SPLIT_RE.search(chunk):
            parts = _BULLET_SPLIT_RE.split(chunk)
            bullet_chunks.extend(p.strip() for p in parts if p.strip())
        else:
            bullet_chunks.append(chunk)

    # ── Step 4: user story extraction ────────────────────────────────────────
    # Group contiguous lines back into blocks for user-story detection.
    # A user-story block spans from "User Story:" until the next "User Story:"
    # or end of text.
    full_text = "\n".join(bullet_chunks)

    # Split on user-story block boundaries
    us_block_re = re.compile(r"(?=(?:User\s+Story|Acceptance\s+Criteria)\s*:)", re.IGNORECASE)
    blocks = us_block_re.split(full_text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if _is_user_story_block(block):
            candidates.extend(_extract_from_user_story(block))
        else:
            # Not a user-story block — split into individual lines / sentences
            for line in block.splitlines():
                line = line.strip()
                if line:
                    candidates.append(line)

    # ── Step 5: strip residual metadata tokens ────────────────────────────────
    cleaned: list[str] = []
    for c in candidates:
        c = _strip_metadata_tokens(c)
        c = _clean_whitespace(c)
        # Drop bare artifact strings (lone numbers, bullets, section IDs)
        if _BARE_ARTIFACT_RE.match(c):
            continue
        if c:
            cleaned.append(c)

    # ── Step 6: reject short / trivial candidates ─────────────────────────────
    non_trivial = [c for c in cleaned if not _is_trivial(c)]

    # ── Step 7: deduplicate (case-insensitive, first occurrence wins) ──────────
    seen: set[str] = set()
    result: list[str] = []
    for c in non_trivial:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            result.append(c)

    return result


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    SAMPLE = """
1.1 Users can create an account • Users can view products • Users can change
their password • Users can delete their account

2.1 User Story: User Registration Mapped Requirement: 1.1 Priority: High
Estimation: 2 Description: As a user, I want to register an account so that I
can access personalised features.
Acceptance Criteria: Given I am on the registration page When I submit valid
credentials Then my account shall be created and I shall be redirected to the
dashboard

2.2 (removed) The system shall validate all input fields before submission.

3.1 The application shall respond to all API requests within 500 ms under
normal operating conditions.

6.2.3 REQ-004 The platform shall support two-factor authentication for all
administrator accounts.

(TBD) Access control policies must be reviewed annually.

2.1 User Story: Product Listing Mapped Requirement: 1.3 Priority: Medium
Estimation: 3 Description: As a seller, I want to list my products so that
buyers can discover them.
Acceptance Criteria: Given I am logged in as a seller When I navigate to the
listing page Then I should be able to add a new product listing

Short line.

1.3
• • • 2
"""

    print("=" * 60)
    print("Preprocessor self-test")
    print("=" * 60)
    results = preprocess(SAMPLE)
    for i, r in enumerate(results, 1):
        print(f"  [{i:02d}] {r}")
    print(f"\n{len(results)} candidate(s) extracted.")
