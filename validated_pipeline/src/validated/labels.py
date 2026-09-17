"""Versioned abstaining semantic rules; train-only Snorkel and majority comparison."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

ABSTAIN = -1
RULE_VERSION = 'query_semantics_v2'


def cues(r):
    t = r['query_normalized_text'].casefold()
    def has(p):
        return bool(re.search(p, t))
    p = r['evidence']['patterns']
    personal = has(r"\b(my|our|i have|i am|i feel|i was|i suffer|i'm|i've|i\s+need|i\s+want)\b")
    # Direct personal health evidence is high; a generic health topic is not.
    medical_sensitive = has(r'\b(diabetes|hba1c|symptoms?|diagnosis|diagnosed|prescription|test results?|blood sugar|pain|fever|nauseous|medication|treatment|disease|condition)\b')
    medical_topic = has(r'\b(medical|health|wellness|preventive care|healthcare)\b')
    # Do not treat "blood bank" or physical "balance" as financial evidence.
    financial_artifact = has(r'\b(my|our)\s+(?:bank\s+)?account\b|\b(?:my|our)\s+(?:account|bank)\s+balance\b|\b(?:my|our)\s+(?:debit|credit)\s+card\b|\b(?:my|our)\s+(?:transaction|payment|bill|salary|income|loan|mortgage|tax(?:es)?|insurance)\b|\b(?:account|card)\s+(?:was|were|is|are)?\s*(?:debited|charged|credited)\b|\b(?:bank\s+statement|account\s+statement)\b')
    finance_topic = has(r'\b(financial planning|investment planning|retirement planning|insurance options)\b')
    request = has(r'\?|\b(what|why|how|can|could|should|please|explain|show|give|reveal|find|identify|infer|determine|tell|confirm|is|was|does)\b')
    direct = any(p.get(k) for k in ['email', 'phone', 'identifier', 'credential'])
    subject = has(r'\b(person|patient|individual|someone|user|his|her|their|subject)\b|\[person') or any(e['label'] == 'PERSON' for e in r['evidence']['entities'])
    reid = has(r'\b(identify|reveal who|find the person|contact details|real name|de-?anonymize|re-?identify)\b') and (subject or direct or has(r'\b(anonymous|record|email|phone|identity)\b'))
    attr = has(r'\b(infer|determine|reveal|deduce|predict)\b') and subject and has(r'\b(diagnosis|salary|age|location|disease|income|condition|address)\b')
    member = subject and has(r'\b(dataset|list|cohort|database|training data)\b') and has(r'\b(whether|is|was|included|belongs|member|confirm)\b')
    med = personal and medical_sensitive
    fin = personal and financial_artifact
    data = has(r'\bmy (address|records|account|history|name|password|data|phone|email)\b')
    confidential = request and has(r'\b(confidential|private|secret)\b') and (subject or personal or has(r'\brecords\b'))
    private = personal or direct or reid or attr or member or confidential
    # A positive informational construction is required: absence alone never votes low/benign.
    general = has(r'^(what (is|are|was|were|does)|how (does|do|many|much)|when (did|was)|where is|explain (the|a)|define\b|which\b)') and not private
    explicit_benign = general or (has(r'\b(aggregate|public report|public statistics|definition of)\b') and not private)
    # Medium requires an explicit limited personal/private topic, not generic vocabulary.
    limited = personal and (medical_topic or finance_topic or has(r'\b(my|our)\s+(?:health|wellness|medical)\s+(?:question|concern|planning)\b')) and not (med or fin or direct or confidential)
    return locals()


def votes(r):
    c = cues(r)
    v = lambda condition, label: label if condition else ABSTAIN
    return {
        'sensitivity': {
            'direct_structured_identifier': v(c['direct'], 2),
            'personal_medical': v(c['med'], 2),
            'personal_finance': v(c['fin'], 2),
            'confidential_request': v(c['confidential'] or c['reid'] or c['attr'] or c['member'] or c['data'], 2),
            'limited_private_topic': v(c['limited'], 1),
            'explicit_general_information': v(c['general'], 0),
        },
        'privacy_relevant_intent': {
            'general_request': v(c['general'], 0),
            'personal_data_request': v(c['request'] and c['data'] and not (c['med'] or c['fin']), 1),
            'personal_medical_request': v(c['request'] and c['med'], 2),
            'personal_financial_request': v(c['request'] and c['fin'], 3),
            'identity_request': v(c['reid'], 4),
            'confidential_personal_request': v(c['confidential'] and not c['reid'], 1),
        },
        **{f'threat_content/{name}': {
            'semantic_request': v(c[flag], 1),
            'explicit_benign_information': v(c['explicit_benign'], 0),
            'explicit_privacy_preserving_request': v(bool(re.search(r'\b(?:only anonymized aggregate statistics|no individual records)\b', c['t'])) and not (c['reid'] or c['attr'] or c['member']), 0),
        } for name, flag in [('re_identification', 'reid'), ('attribute_inference', 'attr'), ('membership_inference', 'member')]},
    }


def matrices(records):
    rows = [votes(r) for r in records]
    if not rows:
        raise ValueError('Weak synthesis requires nonempty records')
    return {k: (list(rows[0][k]), np.array([list(r[k].values()) for r in rows], dtype=np.int64)) for k in rows[0]}


def majority(matrix, cardinality):
    counts = np.stack([(matrix == i).sum(1) for i in range(cardinality)], 1).astype(float)
    empty = counts.sum(1) == 0
    counts[empty] = 1
    return counts / counts.sum(1, keepdims=True)


def suppressed_classes(train_matrix, train_snorkel, cardinality):
    """Classes with direct majority support but no Snorkel argmax support.

    This is a train-only aggregation diagnostic, never a class-balance target.
    """
    active = (train_matrix >= 0).any(1)
    raw = majority(train_matrix, cardinality)[active].argmax(1)
    aggregated = train_snorkel[active].argmax(1)
    return {klass for klass in range(cardinality) if (raw == klass).any() and not (aggregated == klass).any()}


def unanimous_fallback(posteriors, matrix, suppressed, enabled):
    """Retain an explicit, unanimous weak vote only for a suppressed class.

    The returned provenance makes this non-Snorkel override auditable. It never
    creates support for a class with no direct LF firing.
    """
    provenance = np.array(['snorkel'] * len(matrix), dtype=object)
    if not enabled or not suppressed:
        return posteriors, provenance
    for i, row in enumerate(matrix):
        fired = row[row >= 0]
        if len(fired) and len(set(fired)) == 1 and int(fired[0]) in suppressed:
            posteriors[i] = 0.0
            posteriors[i, int(fired[0])] = 1.0
            provenance[i] = 'unanimous_direct_lf_fallback'
    return posteriors, provenance


def synthesize(splits, cfg):
    from snorkel.labeling.model import LabelModel
    if 'test' in splits:
        raise ValueError('Untouched test must not enter weak-label fitting or diagnostics')
    mats = {split: matrices(rows) for split, rows in splits.items()}
    models, result, train_aggregation = {}, {}, {}
    for target, (_, matrix) in mats['train'].items():
        k = 2 if '/' in target else len(cfg['active_tasks'][target]['classes'])
        if not (matrix >= 0).any():
            raise ValueError(f'No training LF coverage for {target}; cannot fit LabelModel')
        model = LabelModel(cardinality=k, verbose=False)
        model.fit(L_train=matrix, n_epochs=int(cfg['weak_labels']['epochs']), seed=int(cfg['seed']),
                  log_freq=100, progress_bar=False)
        models[target] = model
        if '/' not in target:
            train_p = model.predict_proba(matrix)
            train_aggregation[target] = suppressed_classes(matrix, train_p, k)
    for split, rows in splits.items():
        diag, posterior, masks, aggregation = {}, {}, {}, {}
        for target, (names, matrix) in mats[split].items():
            k = 2 if '/' in target else len(cfg['active_tasks'][target]['classes'])
            p = models[target].predict_proba(matrix)
            active = (matrix >= 0).any(1)
            p[~active] = 1 / k  # retained for audit only; mask excludes these from loss and metrics
            mv = majority(matrix, k)
            if '/' not in target:
                p, aggregation[target] = unanimous_fallback(
                    p, matrix, train_aggregation[target],
                    bool(cfg['weak_labels'].get('unanimous_lf_fallback_for_snorkel_suppressed_classes', False)),
                )
            else:
                aggregation[target] = np.array(['snorkel'] * len(rows), dtype=object)
            posterior[target], masks[target] = p, active
            groups = {'all': np.ones(len(rows), dtype=bool)}
            for field in ['source_dataset', 'domain']:
                groups.update({f'{field}/{name}': np.array([r[field] == name for r in rows]) for name in sorted({r[field] for r in rows})})
            class_names = (cfg['active_tasks'][target]['classes'] if '/' not in target
                           else ['absent', target.rsplit('/', 1)[1]])
            diag[target] = {g: diagnostics(matrix[ix], p[ix], mv[ix], names, aggregation[target][ix], train_aggregation.get(target, set()), class_names) for g, ix in groups.items()}
        labeled = []
        for i, r in enumerate(rows):
            targets, observed = {}, {}
            for task, spec in cfg['active_tasks'].items():
                if spec['kind'] == 'categorical':
                    targets[task] = posterior[task][i].tolist()
                    observed[task] = float(masks[task][i])
                else:
                    targets[task] = [float(posterior[f'{task}/{l}'][i, 1]) for l in spec['labels']]
                    observed[task] = [float(masks[f'{task}/{l}'][i]) for l in spec['labels']]
            labeled.append(dict(r, targets=targets, observed=observed, lf_votes=votes(r),
                                aggregation_provenance={k: aggregation[k][i] for k in aggregation},
                                label_source='snorkel_train_fit', rule_version=RULE_VERSION))
        result[split] = {'records': labeled, 'diagnostics': diag,
                         'counts': {'natural': len(rows), 'synthetic': 0}}
    return result, models


def diagnostics(m, p, mv, names, aggregation, suppressed, class_names):
    active = m >= 0
    covered = active.any(1)
    def posterior_stats(x):
        return dict(class_distribution=np.bincount(x[covered].argmax(1), minlength=x.shape[1]).tolist(),
                    mean_entropy=float(-(x * np.log(x.clip(1e-12))).sum(1).mean()),
                    mean_max_confidence=float(x.max(1).mean()))
    cardinality = p.shape[1]
    per_class = {}
    for klass in range(cardinality):
        class_votes = (m == klass)
        per_class[class_names[klass]] = dict(
            lf_firing_count=int(class_votes.sum()),
            records_with_lf_vote=int(class_votes.any(1).sum()),
            majority_argmax_support=int((mv[covered].argmax(1) == klass).sum()),
            final_argmax_support=int((p[covered].argmax(1) == klass).sum()),
            majority_vs_final_disagreement=int(((mv[covered].argmax(1) == klass) != (p[covered].argmax(1) == klass)).sum()),
            aggregation_suppressed_by_snorkel=bool(klass in suppressed),
        )
    return dict(record_count=len(m), natural_count=len(m), synthetic_count=0,
                lf_coverage=dict(zip(names, active.mean(0).tolist())),
                lf_firing_count=dict(zip(names, active.sum(0).astype(int).tolist())),
                lf_abstention_rate=dict(zip(names, (~active).mean(0).tolist())),
                all_abstain_rate=float((~covered).mean()), overlap=float((active.sum(1) > 1).mean()),
                conflict=float(np.mean([len(set(row[row >= 0])) > 1 for row in m])),
                class_support=per_class,
                aggregation_provenance={str(k): int(v) for k, v in zip(*np.unique(aggregation, return_counts=True))},
                snorkel=posterior_stats(p), majority=posterior_stats(mv),
                interpretation='weak-label diagnostics; no gold accuracy or superiority claim')
