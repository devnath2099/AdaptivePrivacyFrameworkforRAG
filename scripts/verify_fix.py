import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

np.random.seed(42)

# Load weak labels
weak_dir = Path('outputs/m2_weak_labels')
weak_data = {}
for f in sorted(weak_dir.glob('*_weak_labels.jsonl')):
    dim = f.name.replace('_weak_labels.jsonl', '')
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            rec = json.loads(line)
            rid = rec['record_id']
            if rid not in weak_data: weak_data[rid] = {}
            weak_data[rid][dim] = (rec['label_names'], rec['probabilities'])

# Load M1 records
records = []
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        records.append(json.loads(line))

by_domain = defaultdict(list)
for r in records:
    by_domain[r['domain']].append(r)

# Sample 25 per domain
sample_size = 25
selected = []
for domain in sorted(by_domain.keys()):
    domain_records = [r for r in by_domain[domain] if r['record_id'] in weak_data]
    rng = np.random.default_rng(42)
    idx = rng.choice(len(domain_records), size=min(sample_size, len(domain_records)), replace=False)
    for i in idx:
        selected.append(domain_records[i])

# === DISTRIBUTIONS ===
print("=" * 80)
print("INTENT TOP-LABEL DISTRIBUTION BY DOMAIN (after fix)")
print("=" * 80)
intent_by_domain = defaultdict(Counter)
for r in selected:
    rid = r['record_id']
    wd = weak_data.get(rid, {})
    if 'intent' in wd:
        names, probs = wd['intent']
        probs = np.array(probs)
        top = names[int(np.argmax(probs))]
        intent_by_domain[r['domain']][top] += 1
for domain in sorted(intent_by_domain.keys()):
    total = sum(intent_by_domain[domain].values())
    print(f"  {domain}: {dict(intent_by_domain[domain])}")

print("\nSENSITIVITY TOP-LABEL BY DOMAIN")
for domain in sorted(by_domain.keys()):
    domain_records = [r for r in selected if r['domain'] == domain]
    sens = Counter()
    for r in domain_records:
        rid = r['record_id']
        if rid in weak_data and 'sensitivity' in weak_data[rid]:
            names, probs = weak_data[rid]['sensitivity']
            probs = np.array(probs)
            top = names[int(np.argmax(probs))]
            sens[top] += 1
    print(f"  {domain}: {dict(sens)}")

print("\nDISCLOSURE_SCOPE TOP-LABEL BY DOMAIN")
for domain in sorted(by_domain.keys()):
    domain_records = [r for r in selected if r['domain'] == domain]
    disc = Counter()
    for r in domain_records:
        rid = r['record_id']
        if rid in weak_data and 'disclosure_scope' in weak_data[rid]:
            names, probs = weak_data[rid]['disclosure_scope']
            probs = np.array(probs)
            top = names[int(np.argmax(probs))]
            disc[top] += 1
    print(f"  {domain}: {dict(disc)}")

# === HAS_ORGANIZATION POSITIVE RATE ===
print("\n" + "=" * 80)
print("HAS_ORGANIZATION POSITIVE RATE BY DOMAIN")
print("=" * 80)
for domain in sorted(by_domain.keys()):
    domain_records = [r for r in selected if r['domain'] == domain]
    org_pos = 0
    org_total = 0
    for r in domain_records:
        rid = r['record_id']
        if rid in weak_data and 'entity_tags' in weak_data[rid]:
            names, probs = weak_data[rid]['entity_tags']
            if 'has_organization' in names:
                org_idx = names.index('has_organization')
                if probs[org_idx] > 0.5:
                    org_pos += 1
                org_total += 1
    print(f"  {domain}: {org_pos}/{org_total} ({100*org_pos/org_total:.1f}%)")

# === 10 SAMPLED SEMANTIC EXAMPLES ===
print("\n" + "=" * 80)
print("10 SAMPLED SEMANTIC EXAMPLES (across domains)")
print("=" * 80)
# Pick 2 from each domain
shown = 0
for domain in sorted(by_domain.keys()):
    domain_records = [r for r in selected if r['domain'] == domain]
    for r in domain_records:
        if shown >= 10:
            break
        rid = r['record_id']
        wd = weak_data.get(rid, {})
        query = r.get('query_text', '')[:80]
        dims = {}
        for dim in ['sensitivity', 'intent', 'disclosure_scope', 'entity_tags', 'threat_content']:
            if dim in wd:
                names, probs = wd[dim]
                probs = np.array(probs)
                top = names[int(np.argmax(probs))]
                dims[dim] = f"{top}({probs.max():.3f})"
        print(f"  [{domain}] {query}")
        print(f"    intent={dims.get('intent','?')} sens={dims.get('sensitivity','?')} disc={dims.get('disclosure_scope','?')}")
        if 'entity_tags' in dims:
            et = wd['entity_tags']
            names, probs = et
            org_idx = names.index('has_organization') if 'has_organization' in names else -1
            org_p = probs[org_idx] if org_idx >= 0 else -1
            print(f"    org_pos={org_p:.4f} entities={dims.get('entity_tags','?')}")
        shown += 1
    if shown >= 10:
        break

# === CHECK FAILURES ===
print("\n" + "=" * 80)
print("SEMANTIC FAILURES REMAINING")
print("=" * 80)
failures = []
for r in selected:
    rid = r['record_id']
    wd = weak_data.get(rid, {})
    domain = r['domain']
    query = r.get('query_text', '').lower()
    
    # Check intent
    if 'intent' in wd:
        names, probs = wd['intent']
        probs = np.array(probs)
        top = names[int(np.argmax(probs))]
        # Medical intent on non-medical
        if domain != 'medical' and top == 'medical_information_request':
            failures.append(f"  {rid}: {domain} -> medical_information_request | query={r.get('query_text','')[:60]}")
        # Financial intent on non-financial (with domain shortcut removed, this should be fixed)
        if domain not in ('financial',) and top == 'financial_information_request':
            has_fin = any(t in query for t in ['account', 'balance', 'invest', 'mortgage', 'salary', 'credit', 'debit', 'bank', 'loan', 'routing', 'income', 'tax', 'fund'])
            if not has_fin:
                failures.append(f"  {rid}: {domain} -> financial_information_request | query={r.get('query_text','')[:60]}")
    
    # Check has_organization on general_qa
    if domain == 'general_qa' and 'entity_tags' in wd:
        names, probs = wd['entity_tags']
        if 'has_organization' in names:
            org_idx = names.index('has_organization')
            if probs[org_idx] > 0.99:
                failures.append(f"  {rid}: {domain} -> has_organization={probs[org_idx]:.4f} | query={r.get('query_text','')[:60]}")

print(f"Total failures: {len(failures)}")
for f in failures[:15]:
    print(f)
