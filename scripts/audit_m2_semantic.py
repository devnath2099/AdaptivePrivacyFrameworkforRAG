import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

np.random.seed(42)

# Load all weak label files
weak_dir = Path('outputs/m2_weak_labels')
weak_data = {}
for f in sorted(weak_dir.glob('*_weak_labels.jsonl')):
    dim = f.name.replace('_weak_labels.jsonl', '')
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rid = rec['record_id']
            if rid not in weak_data:
                weak_data[rid] = {}
            weak_data[rid][dim] = (rec['label_names'], rec['probabilities'])

# Load M1 train records
records = []
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

# Index by domain
by_domain = defaultdict(list)
for r in records:
    by_domain[r['domain']].append(r)

print(f"Records by domain: {[(d, len(v)) for d, v in by_domain.items()]}")
print()

# Sample 25 per domain with fixed seed
sample_size = 25
selected = []
for domain in sorted(by_domain.keys()):
    domain_records = [r for r in by_domain[domain] if r['record_id'] in weak_data]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(domain_records), size=min(sample_size, len(domain_records)), replace=False)
    for i in idx:
        selected.append(domain_records[i])

print(f"Selected {len(selected)} records")
print()

# --- Distribution analysis ---
intent_by_domain = defaultdict(Counter)
sensitivity_by_domain = defaultdict(Counter)
disclosure_by_domain = defaultdict(Counter)

# --- Semantic failure detection ---
failures = []

for r in selected:
    rid = r['record_id']
    domain = r['domain']
    wd = weak_data.get(rid, {})
    query = r.get('query_text', '')[:150]
    context = r.get('context_text', '')[:150]

    dims = {}
    for dim in ['sensitivity', 'intent', 'disclosure_scope', 'entity_tags', 'threat_content']:
        if dim in wd:
            label_names, probs = wd[dim]
            probs = np.array(probs)
            top_idx = int(np.argmax(probs))
            dims[dim] = (label_names[top_idx], float(probs[top_idx]), label_names, probs.tolist())
        else:
            dims[dim] = (None, 0.0, [], [])

    # Collect distributions
    if dims['intent'][0]:
        intent_by_domain[domain][dims['intent'][0]] += 1
    if dims['sensitivity'][0]:
        sensitivity_by_domain[domain][dims['sensitivity'][0]] += 1
    if dims['disclosure_scope'][0]:
        disclosure_by_domain[domain][dims['disclosure_scope'][0]] += 1

    # Check for semantic contradictions
    text = (query + ' ' + context).lower()
    intent_label = dims['intent'][0]
    sensitivity_label = dims['sensitivity'][0]

    # Medical intent on non-medical queries
    if domain != 'medical' and intent_label == 'medical_information_request':
        failures.append({
            'rid': rid, 'domain': domain,
            'issue': 'medical_information_request on non-medical domain',
            'query': query, 'intent': intent_label, 'intent_prob': dims['intent'][1],
            'sensitivity': sensitivity_label, 'sens_prob': dims['sensitivity'][1]
        })

    # Financial intent on non-financial queries (excluding financial domain)
    if domain not in ('financial',) and intent_label == 'financial_information_request':
        if 'tariff' not in text and 'tax' not in text and 'stock' not in text and 'trade' not in text:
            failures.append({
                'rid': rid, 'domain': domain,
                'issue': 'financial_information_request on non-financial domain',
                'query': query, 'intent': intent_label, 'intent_prob': dims['intent'][1],
                'sensitivity': sensitivity_label, 'sens_prob': dims['sensitivity'][1]
            })

    # Identity request on general Q&A
    if domain == 'general_qa' and intent_label == 'identity_related_request':
        failures.append({
            'rid': rid, 'domain': domain,
            'issue': 'identity_related_request on general_qa',
            'query': query, 'intent': intent_label, 'intent_prob': dims['intent'][1],
            'sensitivity': sensitivity_label, 'sens_prob': dims['sensitivity'][1]
        })

    # Sensitivity high but intent general_information (contradiction: general info should be low sens)
    if sensitivity_label == 'high' and intent_label == 'general_information':
        failures.append({
            'rid': rid, 'domain': domain,
            'issue': 'high sensitivity + general_information intent',
            'query': query, 'intent': intent_label, 'intent_prob': dims['intent'][1],
            'sensitivity': sensitivity_label, 'sens_prob': dims['sensitivity'][1]
        })

    # Medical query but sensitivity low (should be at least medium)
    if domain == 'medical' and sensitivity_label == 'low' and intent_label == 'medical_information_request':
        failures.append({
            'rid': rid, 'domain': domain,
            'issue': 'medical query with low sensitivity',
            'query': query, 'intent': intent_label, 'intent_prob': dims['intent'][1],
            'sensitivity': sensitivity_label, 'sens_prob': dims['sensitivity'][1]
        })

    # HotpotQA multi-hop getting medical intent
    if domain == 'multi_hop_qa' and intent_label == 'medical_information_request':
        failures.append({
            'rid': rid, 'domain': domain,
            'issue': 'multi_hop_qa getting medical_information_request',
            'query': query, 'intent': intent_label, 'intent_prob': dims['intent'][1],
            'sensitivity': sensitivity_label, 'sens_prob': dims['sensitivity'][1]
        })

    # Threat content: entity_tags show organization but query is clearly not about PII/organizations
    if domain in ('general_qa', 'multi_hop_qa'):
        if dims['entity_tags'][0] == 'has_organization' and dims['entity_tags'][1] > 0.99:
            failures.append({
                'rid': rid, 'domain': domain,
                'issue': 'has_organization=0.99+ on non-organization query',
                'query': query,
                'entity_tags': dims['entity_tags']
            })

# --- Print distributions ---
print("=" * 80)
print("1. INTENT TOP-LABEL DISTRIBUTION BY DOMAIN")
print("=" * 80)
for domain in sorted(intent_by_domain.keys()):
    total = sum(intent_by_domain[domain].values())
    print(f"\n{domain} ({total} records):")
    for label, count in intent_by_domain[domain].most_common():
        print(f"  {label}: {count} ({100*count/total:.1f}%)")

print("\n")
print("=" * 80)
print("2. SENSITIVITY TOP-LABEL DISTRIBUTION BY DOMAIN")
print("=" * 80)
for domain in sorted(sensitivity_by_domain.keys()):
    total = sum(sensitivity_by_domain[domain].values())
    print(f"\n{domain} ({total} records):")
    for label, count in sensitivity_by_domain[domain].most_common():
        print(f"  {label}: {count} ({100*count/total:.1f}%)")

print("\n")
print("=" * 80)
print("3. DISCLOSURE_SCOPE TOP-LABEL DISTRIBUTION BY DOMAIN")
print("=" * 80)
for domain in sorted(disclosure_by_domain.keys()):
    total = sum(disclosure_by_domain[domain].values())
    print(f"\n{domain} ({total} records):")
    for label, count in disclosure_by_domain[domain].most_common():
        print(f"  {label}: {count} ({100*count/total:.1f}%)")

# --- Print failures ---
print("\n")
print("=" * 80)
print(f"4. CLEAR SEMANTIC FAILURES (found {len(failures)})")
print("=" * 80)
for i, f in enumerate(failures[:10], 1):
    print(f"\n--- Failure {i} ---")
    print(f"  Record: {f['rid']}")
    print(f"  Domain: {f['domain']}")
    print(f"  Issue: {f['issue']}")
    print(f"  Query: {f['query']}")
    if 'intent' in f:
        print(f"  Intent: {f['intent']} (prob={f['intent_prob']:.4f})")
    if 'sensitivity' in f:
        print(f"  Sensitivity: {f['sensitivity']} (prob={f['sens_prob']:.4f})")
    if 'entity_tags' in f:
        et_name, et_prob, et_names, et_probs = f['entity_tags']
        print(f"  Entity: {et_name} (prob={et_prob:.4f})")

# --- Estimate systematic vs isolated ---
print("\n")
print("=" * 80)
print("5. ESTIMATE: ISOLATED OR SYSTEMATIC?")
print("=" * 80)
domain_failure_counts = defaultdict(int)
for f in failures:
    domain_failure_counts[f['domain']] += 1
print(f"\nTotal failures: {len(failures)} out of {len(selected)} sampled records")
for domain in sorted(domain_failure_counts.keys()):
    domain_total = min(sample_size, len([r for r in by_domain[domain] if r['record_id'] in weak_data]))
    print(f"  {domain}: {domain_failure_counts[domain]}/{domain_total} ({100*domain_failure_counts[domain]/domain_total:.1f}%)")

# Check if the same type of failure repeats
failure_types = Counter(f['issue'] for f in failures)
print(f"\nFailure type counts:")
for ft, count in failure_types.most_common():
    print(f"  {ft}: {count}")
