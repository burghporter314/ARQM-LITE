# ARQM-LITE Corpus Analysis — Qualitative Observations

**Corpus:** 19 documents · 2,238 requirements · SWENG 894 FA24
**Date:** 2026-03-26
**Baseline file:** `batch_baseline.json`

---

## Overall Flag Rates (Baseline)

| Dimension     | Flags | Rate   |
|---------------|-------|--------|
| Ambiguity     | 1,130 | 50.5%  |
| Singularity   |   974 | 43.5%  |
| Verifiability |   466 | 20.8%  |
| Feasibility   |   402 | 18.0%  |

These rates are inflated by the structural issues documented below.

---

## Observation 1 — Gherkin / Acceptance Criteria blocks are not filtered

**Impact:** High. Affects all four dimensions.

Many SRS documents include user stories with Gherkin-style acceptance criteria.
The entire block — `Given`, `When`, `Then` lines — gets identified as a requirement
because of modals like `should` in `Then I should be able to...`.
These are test assertions, not requirements.

**Examples:**
```
"Acceptance Criteria\nGiven I have objects created\nWhen I open the object page\n
Then I should be able to assign that object a category"

"Acceptance Criteria: Given I am an admin\nWhen a user signs up for an account\n
Then I should be able to ensure they are verified"
```

The `should` in `Then I should be able to...` is Gherkin idiom meaning "the expected
outcome is" — not an ambiguous obligation level. It triggers the syntactic `should`
ambiguity rule **41 times** across the corpus. Every Gherkin block also inflates
singularity counts (coordinated verbs across Given/When/Then) and verifiability counts.

**Fix direction:** Strip or pre-filter Gherkin blocks (`Given...When...Then...`) before
analysis, or add them as a recognised document structure in the preprocessor.

---

## Observation 2 — PDF table extraction produces multi-requirement "sentences"

**Impact:** High. Root cause of ~35–45% of all phantom spans.

When requirement tables are extracted from PDF, entire table rows — sometimes multiple
rows — are merged into one sentence. The slot parser then spans clause boundaries,
producing violation text that doesn't exist verbatim in the sentence.

**Phantom span counts:**
- Ambiguity:     698 phantoms
- Singularity:   899 phantoms
- Verifiability: 330 phantoms
- Feasibility:   311 phantoms

**Examples:**
```
Sentence: "ID\nDescription\nFR-1\nThe system shall persist user data\nFR-2\n
           The system shall allow the user to define maintainable objects"

Phantom span (singularity): 'persist and allow'
Phantom span (ambiguity):   'will be used'
```

**Fix direction:** The sentence tokeniser / preprocessor needs to detect and split
merged-table and Gherkin structures before they reach the detectors.

---

## Observation 3 — Feasibility flags standard CRUD action verbs

**Impact:** High. Confirmed by user.

The feasibility action-slot detector flags essentially any infinitive verb as
potentially infeasible. Every example in the top action-slot violations is a
completely routine software operation:

| Flagged span   | Example sentence                                                               |
|----------------|--------------------------------------------------------------------------------|
| `to view`      | "Administrators must be able to **view** data analytics for each question pool" |
| `to access`    | "Users should be able to **access** the application from any modern web browser" |
| `to search`    | "HAM Candidates must be able to **search** for and register for an exam session" |
| `to assign`    | "He will be able to **assign** attributes to the object"                        |
| `to edit`      | "The user will be able to **edit** an existing object"                          |
| `to categorize`| "A user will be able to **categorize** objects"                                 |
| `to release`   | "Volunteer Examiners must be able to **release** exams to HAM Candidates"       |

The feasibility action threshold is only `0.32` — far too low for action slots.
Infeasibility should only trigger on explicitly impossible constraints:
physics violations (zero latency, 100% uptime, infinite throughput),
absolute guarantees ("shall never fail"), or unmeasurable superlatives
("best UI ever", "most intuitive experience").

**Fix direction:**
1. Add a large set of CRUD / standard-action verbs to the domain KB.
2. Re-calibrate the action slot threshold upward significantly.
3. Consider a separate "impossible constraint" rule-based check (e.g. patterns like
   `100%`, `zero`, `infinite`, `never fail`, `always`, `instantly`) rather than
   relying on semantic similarity for the action slot.

---

## Observation 4 — Feasibility flags document boilerplate and mission statements

**Impact:** Medium. Compound with Observation 1.

Sentences from introductory sections that contain weak modals (`will`, `can`) pass
through identification and get analysed as requirements:

| Flagged span   | Example sentence                                                        |
|----------------|-------------------------------------------------------------------------|
| `to outline`   | "The purpose of this document is **to outline** the requirements..."    |
| `to develop`   | "The goal of the project is **to develop** a web-based game..."         |
| `to achieve`   | "In order **to achieve** the mission statement..."                      |
| `to note`      | "It is important **to note** that there are no permissions..."          |
| `to strengthen`| "The goal is **to strengthen** my technical project management..."      |

**Fix direction:** Extend `_is_descriptive()` in `util/identification.py` to cover
common boilerplate opening patterns: sentences starting with `The purpose of`,
`The goal of`, `The mission of`, `In order to`, `It is important to`.

---

## Observation 5 — Singularity `multiple_actions` over-fires on boilerplate

**Impact:** Medium.

The `multiple_actions` rule correctly targets requirements like
"The system shall log **and** alert the user."
But it also fires on coordinated verbs in non-requirement sentences:

| Flagged span         | Example sentence                                                               |
|----------------------|--------------------------------------------------------------------------------|
| `give and scope`     | "This document will **give** a system overview, the products **scope**..."     |
| `defined and discussed`| "The functional requirements will be **defined** and...will be **discussed**" |
| `persist and allow`  | Table row merging FR-1 and FR-2                                                |
| `documented and automated` | "The system level tests are fully **documented** and **automated**"      |

The last example (`documented and automated`) is a "Definition of Done" statement —
coordinated past-participles used as predicative adjectives, which is not a
singularity violation.

**Fix direction:** Exclude past participles used as predicative adjectives (passives)
from the `multiple_actions` rule. Also, non-requirement boilerplate should be
pre-filtered (see Obs. 1, 4).

---

## Observation 6 — Verifiability correctly catches subjective terms

**Impact:** Low — this dimension is largely working correctly.

Top verifiability flags are genuinely untestable and appropriate:
- `intuitive` (11 occurrences) — "The interface shall be **intuitive**"
- `user-friendly` (4 occurrences)
- `easy to use` (3 occurrences)
- `as needed` (4 occurrences) — genuinely vague condition
- `properly` (3 occurrences) — borderline
- `correctly` (5 occurrences) — borderline; acceptable when acceptance criterion defined elsewhere

`correctly` and `properly` are borderline — they can be acceptable when the acceptance
criterion is defined elsewhere. No immediate action needed here.

---

## Observation 7 — Ambiguity rate of 50.5% is too high to be actionable

**Impact:** High — usability concern.

If half of all requirements are flagged as ambiguous, practitioners will dismiss the
tool. Inflation sources:
- Gherkin `should` (see Obs. 1)
- Phantom spans from PDF tables (see Obs. 2)
- The `should` syntactic rule firing on `should be able to`, a common accepted pattern

`should be able to` in a requirement is **less ambiguous** than bare `should` — it
defines an actor capability. The syntactic rule should not fire when `should` is
followed by `be able to`.

**Fix direction:** Add a `should be able to` exclusion to the syntactic ambiguity
rule for the `should` modal.

---

## Summary and Priority

| # | Observation                                       | Dimensions          | Priority |
|---|---------------------------------------------------|---------------------|----------|
| 1 | Gherkin blocks not filtered                       | All                 | High     |
| 2 | PDF table merge → phantom spans                   | All                 | High     |
| 3 | CRUD verbs flagged as infeasible                  | Feasibility         | High     |
| 7 | `should be able to` triggers ambiguity rule       | Ambiguity           | High     |
| 4 | Mission/boilerplate sentences analysed            | Feasibility, Sing.  | Medium   |
| 5 | `multiple_actions` fires on boilerplate           | Singularity         | Medium   |
| 6 | Subjective terms correctly flagged (working)      | Verifiability       | —        |

---

## Control Methodology

- **Frozen baseline:** `batch_baseline.json` — never overwritten.
- **Regression gate:** `tests/test_prototype_alignment.py` must maintain ≥80% pass rate
  across all 110 hand-labelled cases after every change.
- **Delta reporting:** After each change round, re-run `batch_analyze.py` and diff
  flag counts against baseline to measure Δ false positives removed and Δ regressions.
- **Human spot-check:** Any dimension whose flag rate changes by ±10% triggers a
  10-example user review before committing.
