"""
Requirement identification: filter sentences that are likely software requirements
using spaCy pattern matching for modal + verb constructs.
"""

import spacy
from spacy.matcher import Matcher

MODAL_KEYWORDS = ["shall", "must", "should", "will", "may", "can"]

_nlp_instance = None


def _get_nlp():
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = spacy.load("en_core_web_sm")
    return _nlp_instance


def identify_requirements(sentences: list[str], nlp=None) -> list[str]:
    """
    Filter sentences to those that contain a modal verb followed by a main verb,
    which is the canonical form of a software requirement.

    Returns the subset of input sentences that are identified as requirements.
    """
    if nlp is None:
        nlp = _get_nlp()

    matcher = Matcher(nlp.vocab)

    # Pattern A: modal immediately followed by a verb
    matcher.add("MODAL_VERB", [[
        {"LOWER": {"IN": MODAL_KEYWORDS}},
        {"POS": "VERB", "OP": "+"},
    ]])

    # Pattern B: subject + modal + verb (the system shall provide …)
    matcher.add("SUBJ_MODAL_VERB", [[
        {"POS": {"IN": ["NOUN", "PROPN"]}, "OP": "+"},
        {"LOWER": {"IN": MODAL_KEYWORDS}},
        {"POS": "VERB", "OP": "+"},
    ]])

    requirements = []
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 4:
            continue
        doc = nlp(sentence)
        if matcher(doc):
            requirements.append(sentence)

    return requirements
