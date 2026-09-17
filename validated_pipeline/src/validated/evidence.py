"""Auditable query-only regex/spaCy evidence; not a trained PHI detector."""
import re

PATTERNS = {
    'email': r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',
    'phone': r'(?<!\w)(?:\+\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]\d{3}[ -]\d{4}\b',
    'identifier': r'\b(?:ssn|patient id|account number|account no|iban|mrn)\s*[:=#-]?\s*[A-Z0-9][A-Z0-9 -]{3,25}\b',
    'credential': r'\b(?:password|api[ _-]?key|secret|otp|pin)\s*(?:is|:|=)\s*\S+',
    'date': r'\b(?:\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b',
    'medical_result': r'\b(?:hba1c|test results?|diagnos(?:is|ed)|blood sugar|symptoms?|prescription)\b',
    'personal_reference': r'\b(?:my|mine|our|i have|i am|i feel|my husband|my wife|my child)\b',
}


def extract(record, nlp=None, doc=None):
    text = record['query_normalized_text']
    result = {'input_field': 'query_normalized_text', 'backend': 'spacy+regex' if nlp is not None or doc is not None else 'regex_only_explicit',
              'entities': [], 'patterns': {}}
    for name, pattern in PATTERNS.items():
        result['patterns'][name] = [{'text': m.group(), 'start': m.start(), 'end': m.end()}
                                    for m in re.finditer(pattern, text, re.I)]
    if nlp is not None or doc is not None:
        result['entities'] = [{'text': e.text, 'label': e.label_, 'start': e.start_char, 'end': e.end_char}
                              for e in (doc if doc is not None else nlp(text)).ents]
    return result


def extract_all(records, nlp):
    docs = nlp.pipe((r['query_normalized_text'] for r in records), batch_size=64) if nlp is not None else iter([None] * len(records))
    for i, (r, doc) in enumerate(zip(records, docs), 1):
        r['evidence'] = extract(r, doc=doc)
        if i % 5000 == 0:
            print(f'M1 query evidence: {i}/{len(records)}', flush=True)


def extractor(config):
    if not config['require_spacy']:
        return None
    import spacy
    # Missing model is fatal in the default profile; never heuristic NER fallback.
    return spacy.load(config['spacy_model'], disable=['parser', 'lemmatizer'])
