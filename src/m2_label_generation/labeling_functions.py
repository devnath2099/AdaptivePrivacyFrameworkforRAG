"""Dimension-specific labeling functions (LFs).

Each LF is a small, semantically-motivated rule that consumes a
record's evidence bundle E_A = (eta, delta, rho, epsilon) and either
votes for a class index or ABSTAINs. LFs never directly decide the
final privacy label -- that is the job of the Snorkel generative model
in `generative_model.py`. These LF sets are a project-specific design
decision, not taken verbatim from any paper.

Term-based LFs share a common factory (`_make_term_lf`) to avoid
duplicating near-identical "does the text contain any of these terms"
logic; entity/regex-based LFs remain explicit since each inspects a
different evidence field.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from m1_data_integration.schemas import UnifiedRecord

from .taxonomy import ABSTAIN, DimensionSpec

LabelingFunction = Callable[[UnifiedRecord, DimensionSpec], int]

_MEDICAL_TERMS = {"diagnosis", "diagnosed", "patient", "doctor", "treatment", "medication",
                   "symptom", "disease", "prescribed", "hospital", "blood", "fever", "pain"}
_FINANCIAL_TERMS = {"account", "balance", "invest", "mortgage", "salary", "credit", "debit",
                     "bank", "loan", "routing", "income", "tax", "fund"}
_CREDENTIAL_TERMS = {"password", "ssn", "social security", "pin", "otp", "login", "credential"}
_PERSONAL_INFO_TERMS = {"my name", "i am", "my address", "i live at", "my age", "my id"}
_IDENTITY_TERMS = {"who is", "who am i", "identify", "real name", "identity"}
_MULTIHOP_TERMS = {"also", "same person", "both", "the founder of", "which university"}
_CROSSDOC_TERMS = {"cross-reference", "linking", "combined with", "together with"}
_REID_TERMS = {"re-identify", "reidentify", "de-anonymize", "deanonymize", "trace back to"}


def _text_of(record: UnifiedRecord) -> str:
    return record.normalized_text.lower()


def _has_any(text: str, terms: set) -> bool:
    return any(t in text for t in terms)


def _entity_labels(record: UnifiedRecord) -> List[str]:
    return [e["label"] for e in record.evidence.entities] if record.evidence else []


def _regex_hit(record: UnifiedRecord, pattern_name: str) -> bool:
    return bool(record.evidence and record.evidence.regex_matches.get(pattern_name))


def _make_term_lf(terms: set, target_label: str, name: str,
                   require_question: bool = False) -> LabelingFunction:
    """Factory for 'text contains any of these terms -> vote for target_label'."""
    def _lf(record: UnifiedRecord, spec: DimensionSpec) -> int:
        text = _text_of(record)
        if require_question and "?" not in record.query_text:
            return ABSTAIN
        if _has_any(text, terms):
            return spec.label_index(target_label)
        return ABSTAIN
    _lf.__name__ = name
    return _lf


# ---------------------------------------------------------------------------
# entity_tags (multi-label, binary per type)
# ---------------------------------------------------------------------------

def _make_entity_presence_lf(entity_type: str, target_labels: tuple, name: str) -> LabelingFunction:
    def _lf(record: UnifiedRecord, spec: DimensionSpec) -> int:
        labels = _entity_labels(record)
        return 1 if any(l in target_labels for l in labels) else ABSTAIN
    _lf.__name__ = name
    return _lf


def _make_regex_presence_lf(pattern_names: tuple, name: str) -> LabelingFunction:
    def _lf(record: UnifiedRecord, spec: DimensionSpec) -> int:
        return 1 if any(_regex_hit(record, p) for p in pattern_names) else ABSTAIN
    _lf.__name__ = name
    return _lf


ENTITY_TAG_LFS: Dict[str, List[LabelingFunction]] = {
    "has_person": [
        _make_entity_presence_lf("PERSON", ("PERSON", "PROPN_HEURISTIC"), "lf_has_person"),
    ],
    "has_organization": [
        _make_entity_presence_lf("ORG", ("ORG",), "lf_has_organization"),
    ],
    "has_location": [
        _make_entity_presence_lf("location", ("GPE", "LOC"), "lf_has_location"),
    ],
    "has_contact_identifier": [
        _make_regex_presence_lf(("email",), "lf_has_email"),
        _make_regex_presence_lf(("phone",), "lf_has_phone"),
        _make_regex_presence_lf(("account_number",), "lf_has_account_number"),
    ],
}


# ---------------------------------------------------------------------------
# sensitivity: {low, medium, high}
# ---------------------------------------------------------------------------

def lf_sensitivity_high_entity_and_account(record, spec: DimensionSpec) -> int:
    if _entity_labels(record) and _regex_hit(record, "account_number"):
        return spec.label_index("high")
    return ABSTAIN


def lf_sensitivity_default_low(record, spec: DimensionSpec) -> int:
    text = _text_of(record)
    no_signal = not (_has_any(text, _MEDICAL_TERMS) or _has_any(text, _FINANCIAL_TERMS)
                      or _has_any(text, _CREDENTIAL_TERMS) or (record.evidence and record.evidence.regex_matches))
    return spec.label_index("low") if record.evidence and no_signal else ABSTAIN


SENSITIVITY_LFS: List[LabelingFunction] = [
    _make_term_lf(_MEDICAL_TERMS, "high", "lf_sensitivity_medical_terms"),
    _make_term_lf(_FINANCIAL_TERMS, "medium", "lf_sensitivity_financial_terms"),
    _make_term_lf(_CREDENTIAL_TERMS, "high", "lf_sensitivity_credential_terms"),
    lf_sensitivity_high_entity_and_account,
    lambda record, spec: spec.label_index("medium") if _regex_hit(record, "money") else ABSTAIN,
    lambda record, spec: (spec.label_index("medium")
                           if _regex_hit(record, "email") or _regex_hit(record, "phone") else ABSTAIN),
    lf_sensitivity_default_low,
]
SENSITIVITY_LFS[4].__name__ = "lf_sensitivity_numeric_financial_pattern"
SENSITIVITY_LFS[5].__name__ = "lf_sensitivity_contact_identifier_present"


# ---------------------------------------------------------------------------
# intent: {general_information, personal_information_request,
#          medical_information_request, financial_information_request,
#          identity_related_request}
# ---------------------------------------------------------------------------

def lf_intent_general_question(record, spec: DimensionSpec) -> int:
    text = _text_of(record)
    starts_wh = text.startswith(("what", "when", "where", "who", "how", "which"))
    no_sensitive_terms = not (_has_any(text, _MEDICAL_TERMS) or _has_any(text, _FINANCIAL_TERMS)
                               or _has_any(text, _PERSONAL_INFO_TERMS))
    return spec.label_index("general_information") if starts_wh and no_sensitive_terms else ABSTAIN


def _lf_intent_domain(domain: str, target_label: str, name: str) -> LabelingFunction:
    def _lf(record, spec: DimensionSpec) -> int:
        return spec.label_index(target_label) if record.domain == domain else ABSTAIN
    _lf.__name__ = name
    return _lf


INTENT_LFS: List[LabelingFunction] = [
    _make_term_lf(_PERSONAL_INFO_TERMS, "personal_information_request", "lf_intent_personal_info"),
    _make_term_lf(_MEDICAL_TERMS, "medical_information_request", "lf_intent_medical", require_question=True),
    _make_term_lf(_FINANCIAL_TERMS, "financial_information_request", "lf_intent_financial",
                  require_question=True),
    _make_term_lf(_IDENTITY_TERMS, "identity_related_request", "lf_intent_identity"),
    lf_intent_general_question,
    _lf_intent_domain("medical", "medical_information_request", "lf_intent_domain_medical"),
    _lf_intent_domain("financial", "financial_information_request", "lf_intent_domain_financial"),
]


# ---------------------------------------------------------------------------
# disclosure_scope: {single_document, multi_document}
# ---------------------------------------------------------------------------

def lf_disclosure_multiple_entities(record, spec: DimensionSpec) -> int:
    if record.evidence and len(record.evidence.entities) >= 3:
        return spec.label_index("multi_document")
    return ABSTAIN


def lf_disclosure_single_short_query(record, spec: DimensionSpec) -> int:
    if len(record.query_text.split()) <= 15 and record.domain != "multi_hop_qa":
        return spec.label_index("single_document")
    return ABSTAIN


def lf_disclosure_default_single(record, spec: DimensionSpec) -> int:
    if record.domain in ("general_qa", "medical", "financial"):
        return spec.label_index("single_document")
    return ABSTAIN


DISCLOSURE_SCOPE_LFS: List[LabelingFunction] = [
    _lf_intent_domain("multi_hop_qa", "multi_document", "lf_disclosure_multihop_domain"),
    _make_term_lf(_MULTIHOP_TERMS, "multi_document", "lf_disclosure_multihop_terms"),
    _make_term_lf(_CROSSDOC_TERMS, "multi_document", "lf_disclosure_crossdoc_terms"),
    lf_disclosure_multiple_entities,
    lf_disclosure_single_short_query,
    lf_disclosure_default_single,
]


# ---------------------------------------------------------------------------
# threat_content (multi-label, binary per category): re_identification,
# attribute_inference, membership_inference
# ---------------------------------------------------------------------------

_ATTR_SEEKING_TERMS = {"salary", "age", "income", "condition", "diagnosis", "address",
                        "phone number", "email address", "date of birth", "dob"}
_MEMBERSHIP_TERMS = {"in the dataset", "in your database", "in the records", "used for training",
                      "part of the corpus", "included in", "member of"}
_EXISTENCE_TERMS = {"is there", "does", "was", "were", "have", "has", "had"}


def _make_threat_presence_lf(terms: set, name: str) -> LabelingFunction:
    def _lf(record: UnifiedRecord, spec: DimensionSpec) -> int:
        return 1 if _has_any(_text_of(record), terms) else ABSTAIN
    _lf.__name__ = name
    return _lf


def lf_threat_reid_person_and_location(record, spec: DimensionSpec) -> int:
    labels = _entity_labels(record)
    return 1 if "PERSON" in labels and any(l in ("GPE", "LOC") for l in labels) else ABSTAIN


def lf_threat_reid_identifier(record, spec: DimensionSpec) -> int:
    return 1 if any(_regex_hit(record, p) for p in ("email", "phone", "account_number", "ssn")) else ABSTAIN


def lf_threat_attr_entity_and_attribute(record, spec: DimensionSpec) -> int:
    text = _text_of(record)
    has_entity = bool(record.evidence and record.evidence.entities)
    seeks_attribute = _has_any(text, _ATTR_SEEKING_TERMS)
    return 1 if has_entity and seeks_attribute else ABSTAIN


def lf_threat_attr_possessive(record, spec: DimensionSpec) -> int:
    text = _text_of(record)
    possessive = any(p in text for p in ("'s", " his ", " her ", " their "))
    return 1 if possessive and _has_any(text, _ATTR_SEEKING_TERMS) else ABSTAIN


def lf_threat_member_existence(record, spec: DimensionSpec) -> int:
    text = _text_of(record)
    return 1 if (_has_any(text, _EXISTENCE_TERMS) and _has_any(text, _MEMBERSHIP_TERMS)) else ABSTAIN


def lf_threat_member_dataset_ref(record, spec: DimensionSpec) -> int:
    text = _text_of(record)
    return 1 if _has_any(text, _MEMBERSHIP_TERMS) else ABSTAIN


THREAT_CONTENT_LFS: Dict[str, List[LabelingFunction]] = {
    "re_identification": [
        _make_threat_presence_lf(_REID_TERMS, "lf_threat_reid_terms"),
        lf_threat_reid_person_and_location,
        lf_threat_reid_identifier,
    ],
    "attribute_inference": [
        lf_threat_attr_entity_and_attribute,
        lf_threat_attr_possessive,
    ],
    "membership_inference": [
        lf_threat_member_existence,
        lf_threat_member_dataset_ref,
    ],
}


DIMENSION_LFS: Dict[str, List[LabelingFunction]] = {
    "sensitivity": SENSITIVITY_LFS,
    "intent": INTENT_LFS,
    "disclosure_scope": DISCLOSURE_SCOPE_LFS,
}