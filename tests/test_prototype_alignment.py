"""
Prototype alignment tests.

For each quality dimension, verify that the detector correctly classifies
sentences derived directly from the prototype lists in each training module:
  - Sentences built from violation-class prototypes  → should be flagged
  - Sentences built from clean-class / neutral-word contexts → should NOT be flagged

Required: >= 80% pass rate per dimension.
"""
from __future__ import annotations
from pathlib import Path
import sys

import pytest

# ── Path bootstrap (mirrors conftest.py) ───────────────────────────────────
_ROOT     = Path(__file__).parent.parent
_TRAINING = _ROOT / "util" / "training"
for _p in (str(_ROOT / "util"), str(_TRAINING)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_MIN_PASS_RATE = 0.80


# ── Helpers ─────────────────────────────────────────────────────────────────

def _is_violated(result, dimension: str) -> bool:
    """Return True if the detector flagged a violation for the given dimension."""
    d = result.to_dict()
    if dimension == "ambiguity":
        return bool(d.get("is_ambiguous", False))
    if dimension == "feasibility":
        return not bool(d.get("is_feasible", True))
    if dimension == "singularity":
        return not bool(d.get("is_singular", True))
    if dimension == "verifiability":
        return not bool(d.get("is_verifiable", True))
    raise ValueError(f"Unknown dimension: {dimension!r}")


def _run_cases(
    cases: list[tuple[str, bool]],
    detector,
    dimension: str,
    label: str,
) -> None:
    """
    Run every (sentence, expected) case through the detector, print any
    failures, then assert that the overall pass rate is >= _MIN_PASS_RATE.
    """
    correct = 0
    failures: list[tuple[str, bool, bool]] = []

    for sentence, expected in cases:
        predicted = _is_violated(detector.analyze(sentence), dimension)
        if predicted == expected:
            correct += 1
        else:
            failures.append((sentence, expected, predicted))

    total     = len(cases)
    pass_rate = correct / total

    if failures:
        print(f"\n[{label}] Failures ({len(failures)}/{total}):")
        for sentence, expected, predicted in failures:
            exp_str  = "violation" if expected  else "clean"
            pred_str = "violation" if predicted else "clean"
            print(f"  FAIL  expected={exp_str:9s}  got={pred_str:9s}  '{sentence[:90]}'")

    assert pass_rate >= _MIN_PASS_RATE, (
        f"[{label}] pass rate {pass_rate:.1%} ({correct}/{total}) "
        f"— required >= {_MIN_PASS_RATE:.0%}"
    )


# ── Test cases ──────────────────────────────────────────────────────────────
# Each entry: (sentence, expect_violation: bool)
#   True  = the detector SHOULD flag at least one violation
#   False = the detector should NOT flag any violation

# ── Ambiguity ───────────────────────────────────────────────────────────────

_AMB_CASES: list[tuple[str, bool]] = [
    # Positive – passive voice without agent (syntactic rule)
    ("The service must be restarted.",                                          True),
    ("Requests shall be processed.",                                            True),
    ("Sessions shall be managed.",                                              True),
    ("Queries must be optimised.",                                              True),
    ("The worker pool shall be scaled.",                                        True),
    # Positive – 'should' modal ambiguity (syntactic rule)
    ("The system should respond reliably.",                                     True),
    ("The service should be available.",                                        True),
    # Positive – gradable adjective without measurable bound (syntactic rule)
    ("The API must be reliable.",                                               True),
    ("The service must be consistent.",                                         True),
    # Positive – unbounded load condition (syntactic rule)
    ("The system must respond gracefully under high load.",                     True),
    ("Caching shall be enabled during peak traffic.",                           True),
    # Positive – vague multi-word object / condition (semantic scoring)
    ("The scheduler must allocate adequate resources.",                         True),
    ("The system must provide reasonable defaults.",                            True),
    ("The system shall retry if necessary.",                                    True),
    ("The cache shall be flushed under normal conditions.",                     True),
    # Negative – precise, active-voice sentences (must NOT be flagged)
    ("The API must respond within 200ms for 95% of requests.",                 False),
    ("The service must achieve 99.9% uptime with planned maintenance windows.",False),
    ("The system must process at least 1000 requests per second.",             False),
    ("The system shall use AES-256 encryption for all data at rest.",          False),
    ("All session tokens must expire after 30 minutes of inactivity.",        False),
    ("The system shall send a confirmation email within 2 minutes.",           False),
    ("The export must include all fields defined in schema v2.3.",             False),
    ("The service shall encrypt data using AES-256-GCM before storage.",      False),
    ("The client shall retry with exponential backoff up to 3 times.",        False),
    ("Health checks must run at least once per hour.",                         False),
    ("The system must lock the account after 5 failed login attempts.",        False),
    ("The search must return the top 10 ranked results.",                      False),
    ("The system must perform backups every 24 hours at 02:00 UTC.",          False),
    ("The authentication service must validate tokens within 50ms.",          False),
    ("Alerts must fire when the error rate exceeds 5% over 60 seconds.",       False),
]

# ── Feasibility ─────────────────────────────────────────────────────────────

_FEAS_CASES: list[tuple[str, bool]] = [
    # Positive – impossible absolutes (regex rule)
    ("The system must provide 100% uptime at all times.",                      True),
    ("The API must respond with zero latency.",                                True),
    ("The service must never fail under any circumstances.",                   True),
    ("The system must offer unlimited throughput.",                            True),
    ("The classifier must achieve 100% accuracy on all inputs.",              True),
    ("The compiler must produce zero errors for all valid inputs.",            True),
    ("The pipeline must process data with zero downtime.",                     True),
    ("The cache must provide instantaneous retrieval.",                        True),
    ("The system must always be available without exception.",                 True),
    # Positive – internal contradictions (hardcoded pair matching)
    ("The service must be synchronous and non-blocking.",                      True),
    ("The component must be stateless and maintain session state.",            True),
    ("All data must be encrypted and stored in plaintext for auditing.",      True),
    ("The encoder must support lossless and lossy compression simultaneously.",True),
    ("The audit log must be immutable and writable by the compliance team.",   True),
    # Negative – realistic SLA constraints (must NOT be flagged)
    ("The service must maintain 99.9% uptime with planned maintenance windows.",False),
    ("The API must respond within 200ms for 95% of requests.",                False),
    ("The service must process 1000 requests per second.",                    False),
    ("The payment service must maintain an error rate below 0.1%.",           False),
    ("The service must achieve 99.99% availability excluding maintenance.",   False),
    ("The system must encrypt all data using AES-256.",                       False),
    ("Critical incidents must be resolved within 4 hours of detection.",     False),
    ("The client must attempt maximum 5 retries before failing.",             False),
    ("The search index must return results within 1 second.",                 False),
    ("The system must support downloading PDF files to disk.",                False),
    ("The sensor module must have an MTBF greater than 10000 hours.",        False),
]

# ── Singularity ─────────────────────────────────────────────────────────────

_SING_CASES: list[tuple[str, bool]] = [
    # Positive – multiple coordinated actions (spaCy conj + regex)
    ("The system must validate input and encrypt the data and log the result.", True),
    ("The service must authenticate the user and apply rate limiting and return a token.", True),
    ("The system must back up the database and notify the admin.",             True),
    ("The pipeline must compress the payload and encrypt it before sending.",  True),
    ("The compliance module must audit all access events and report weekly.",  True),
    # Positive – compound subjects (spaCy conj + regex fallback)
    ("The admin and the user must both confirm the deletion.",                 True),
    ("The frontend and the backend must validate the input.",                  True),
    ("The admin and the operator must approve changes before deployment.",     True),
    ("Users and administrators must have access to the audit log.",            True),
    # Positive – conjunctive conditions (multiple triggers in one clause)
    ("When login fails or the session expires, the user must be redirected.",  True),
    ("If the user is unauthenticated or unauthorised, the system must return HTTP 403.", True),
    ("If the request is missing or malformed or duplicate, the system must reject it.", True),
    # Positive – mixed functional + non-functional concerns
    ("The system must encrypt all data and respond within 200ms.",             True),
    ("The service must authenticate the user and maintain 99.9% uptime.",     True),
    # Negative – single obligation, actor, and condition (must NOT be flagged)
    ("The system must encrypt all data before writing to disk.",               False),
    ("The API must respond within 200ms for 95% of requests.",                False),
    ("The service must return HTTP 400 for invalid input.",                   False),
    ("When the session expires, the user must be redirected to the login page.", False),
    ("The scheduler must retry failed jobs up to 3 times.",                   False),
    ("All session tokens must expire after 30 minutes of inactivity.",        False),
    ("The application must comply with GDPR Article 17 right to erasure.",    False),
    ("After payment is confirmed, the order must be created within 5 seconds.", False),
    ("The system must retain audit logs for exactly 90 days.",                False),
    ("The API must reject requests that do not include a valid bearer token.", False),
    ("The form must display an inline error message when a required field is empty.", False),
]

# ── Verifiability ────────────────────────────────────────────────────────────

_VERIF_CASES: list[tuple[str, bool]] = [
    # Positive – no acceptance criteria (NAC rules)
    ("The system must handle errors appropriately.",                           True),
    ("The module must manage exceptions as needed.",                           True),
    ("The service must respond to failures in a reasonable manner.",          True),
    ("The component must deal with edge cases properly.",                      True),
    ("The importer must handle large files to a satisfactory level.",         True),
    ("The sync service must deliver messages on a best effort basis.",         True),
    # Positive – subjective success criteria
    ("The interface must be intuitive and easy to use.",                       True),
    ("The dashboard must look professional and be visually appealing.",        True),
    ("Users must be satisfied with the checkout flow.",                        True),
    ("The application must provide a positive user experience.",               True),
    # Positive – untestable absolute negatives
    ("The service must never crash in production.",                            True),
    ("The system must never lose user data.",                                  True),
    ("The platform must not have any security vulnerabilities.",               True),
    ("The release must not have any bugs.",                                    True),
    # Positive – missing actor / verification mechanism
    ("Errors must be handled by the system.",                                  True),
    # Negative – concrete, measurable requirements (must NOT be flagged)
    ("The API must respond within 200ms for 95% of requests.",                False),
    ("The service must return HTTP 400 with a validation error body when input is malformed.", False),
    ("The system must send an alert within 60 seconds of a P1 incident.",     False),
    ("The search endpoint must return results within 500ms for 99% of requests.", False),
    ("The service must produce fewer than 5 unhandled exceptions per day.",   False),
    ("The health endpoint must return HTTP 200 when all dependencies are reachable.", False),
    ("The wizard must achieve a task success rate of 90% in usability testing.", False),
    ("An alert must fire when CPU exceeds 80% for 5 consecutive minutes.",    False),
    ("A log entry must be written within 1 second of each authentication event.", False),
    ("The API must return HTTP 422 and a field-level error map for schema violations.", False),
    ("The system must encrypt all PII fields before storage.",                False),
    ("The login endpoint must reject requests with an invalid token and return HTTP 401.", False),
    ("The payment gateway must maintain an error rate below 0.1% of transactions.", False),
    ("The sensor module must have an MTBF greater than 10000 hours.",        False),
    ("Input must be sanitised before persisting to the database.",            False),
]


# ── Test functions ──────────────────────────────────────────────────────────

def test_ambiguity_prototype_alignment(amb_detector):
    """
    Ambiguity detector: prototype-aligned sentences must be classified
    correctly at >= 80% pass rate (15 violation + 15 clean = 30 cases).
    """
    _run_cases(_AMB_CASES, amb_detector, "ambiguity", "Ambiguity")


def test_feasibility_prototype_alignment(feas_detector):
    """
    Feasibility detector: prototype-aligned sentences must be classified
    correctly at >= 80% pass rate (14 violation + 11 clean = 25 cases).
    """
    _run_cases(_FEAS_CASES, feas_detector, "feasibility", "Feasibility")


def test_singularity_prototype_alignment(sing_detector):
    """
    Singularity detector: prototype-aligned sentences must be classified
    correctly at >= 80% pass rate (14 violation + 11 clean = 25 cases).
    """
    _run_cases(_SING_CASES, sing_detector, "singularity", "Singularity")


def test_verifiability_prototype_alignment(verif_detector):
    """
    Verifiability detector: prototype-aligned sentences must be classified
    correctly at >= 80% pass rate (15 violation + 15 clean = 30 cases).
    """
    _run_cases(_VERIF_CASES, verif_detector, "verifiability", "Verifiability")
