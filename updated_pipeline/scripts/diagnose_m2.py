import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

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

# Load M1 train records
records = []
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        records.append(json.loads(line))

# Load LF matrices to understand vote patterns
et_lf = np.load('outputs/m2_lf_matrices/entity_tags_lambda_matrix.npy')
et_names = ['has_person_pos', 'has_person_neg', 'has_org_pos', 'has_org_neg',
            'has_loc_pos', 'has_loc_neg', 'has_contact_pos', 'has_contact_neg',
            'contact_other_pos', 'contact_other_neg']

tc_lf = np.load('outputs/m2_lf_matrices/threat_content_lambda_matrix.npy')
tc_names = ['reid_0', 'reid_1', 'reid_2', 'reid_3', 'attr_0', 'attr_1', 'attr_2',
            'member_0', 'member_1', 'member_2']

# ====== INTENT DIAGNOSIS ======
print("=" * 80)
print("INTENT DIAGNOSIS: Domain shortcut LFs")
print("=" * 80)

# Find the domain shortcut LFs in the LF matrix columns
# The LF matrix columns are ordered by dimension. Let me check what columns correspond
# to intent by looking at the intent_lambda_matrix.npy
intent_lf = np.load('outputs/m2_lf_matrices/intent_lambda_matrix.npy')
print(f"\nIntent LF matrix shape: {intent_lf.shape}")

# Check which columns are non-abstain (not -1) for each domain
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

# For intent, let me check the weak label JSONL for intent to understand which LFs fire
# Let me look at the intent JSONL more carefully
intent_jsonl = {}
with open('outputs/m2_weak_labels/intent_weak_labels.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line: continue
        rec = json.loads(line)
        intent_jsonl[rec['record_id']] = rec

# Check: do all medical records get medical_information_request?
print("\n--- INTENT: Checking domain shortcut hypothesis ---")
for domain in ['medical', 'financial', 'general_qa', 'multi_hop_qa']:
    domain_records = [r for r in selected if r['domain'] == domain]
    intent_labels = []
    for r in domain_records:
        rid = r['record_id']
        if rid in intent_jsonl:
            label_names, probs = intent_jsonl[rid]['label_names'], intent_jsonl[rid]['probabilities']
            top = label_names[int(np.argmax(probs))]
            intent_labels.append(top)
    counts = Counter(intent_labels)
    print(f"  {domain}: {dict(counts)}")

# Check if the domain shortcut LFs are the ONLY ones firing for medical/financial
# by looking at which LFs fire for a sample medical record that has NO medical terms
print("\n--- INTENT: Checking if semantic LFs alone can cover ---")
# Pick a medical record that might not have medical terms
for r in selected:
    if r['domain'] == 'medical':
        text = r.get('normalized_text', '').lower()
        has_medical = any(t in text for t in ['diagnosis', 'diagnosed', 'patient', 'doctor', 'treatment', 'medication', 'symptom', 'disease', 'prescribed', 'hospital', 'blood', 'fever', 'pain'])
        has_question = '?' in r.get('query_text', '')
        print(f"  {r['record_id']}: has_medical_terms={has_medical}, has_question={has_question}, query={r.get('query_text','')[:80]}")
        break

# Check multi_hop_qa records
print("\n--- INTENT: multi_hop_qa LF vote patterns ---")
for r in selected:
    if r['domain'] == 'multi_hop_qa':
        rid = r['record_id']
        if rid in intent_jsonl:
            label_names, probs = intent_jsonl[rid]['label_names'], intent_jsonl[rid]['probabilities']
            probs = np.array(probs)
            top = label_names[int(np.argmax(probs))]
            # Check which LFs fire (non-abstain)
            text = r.get('normalized_text', '').lower()
            has_medical = any(t in text for t in ['diagnosis', 'diagnosed', 'patient', 'doctor', 'treatment', 'medication', 'symptom', 'disease', 'prescribed', 'hospital', 'blood', 'fever', 'pain'])
            has_financial = any(t in text for t in ['account', 'balance', 'invest', 'mortgage', 'salary', 'credit', 'debit', 'bank', 'loan', 'routing', 'income', 'tax', 'fund'])
            has_personal = any(t in text for t in ['my name', 'i am', 'my address', 'i live at', 'my age', 'my id'])
            has_identity = any(t in text for t in ['who is', 'who am i', 'identify', 'real name', 'identity'])
            starts_wh = text.startswith(("what", "when", "where", "who", "how", "which"))
            print(f"  {rid[:30]}: top={top} ({probs.max():.3f}) | med={has_medical} fin={has_financial} wh={starts_wh} query={r.get('query_text','')[:60]}")

# ====== HAS_ORGANIZATION DIAGNOSIS ======
print("\n")
print("=" * 80)
print("HAS_ORGANIZATION DIAGNOSIS: 25 general_qa records")
print("=" * 80)

# For each general_qa record, check entities and LF votes
general_qa_records = [r for r in selected if r['domain'] == 'general_qa']
for r in general_qa_records[:5]:
    rid = r['record_id']
    print(f"\n  Record: {rid}")
    print(f"  Query: {r.get('query_text','')[:120]}")
    print(f"  Normalized text: {r.get('normalized_text','')[:200]}")
    
    # Check entities
    evidence = r.get('evidence')
    if evidence and isinstance(evidence, dict):
        entities = evidence.get('entities', [])
        print(f"  Entities detected: {len(entities)}")
        for e in entities[:10]:
            print(f"    - label={e.get('label','?')}, text='{e.get('text','')[:50]}'")
    
    # Check weak label
    if rid in weak_data:
        et = weak_data[rid].get('entity_tags')
        if et:
            names, probs = et
            org_idx = names.index('has_organization') if 'has_organization' in names else -1
            if org_idx >= 0:
                print(f"  has_organization posterior: {probs[org_idx]:.6f}")
    
    # Check LF matrix
    if rid in [rec['record_id'] for rec in records]:
        ridx = [rec['record_id'] for rec in records].index(rid)
        org_pos_col = 2  # has_org_pos
        org_neg_col = 3  # has_org_neg
        print(f"  LF matrix has_org_pos vote: {et_lf[ridx, org_pos_col]}, has_org_neg vote: {et_lf[ridx, org_neg_col]}")
        print(f"  LF matrix org_pos_col={org_pos_col}, org_neg_col={org_neg_col}")

# Count how many general_qa records have ORG entities
print("\n--- ORG entity detection rate in general_qa ---")
org_count = 0
entity_count = 0
for r in general_qa_records:
    evidence = r.get('evidence')
    if evidence and isinstance(evidence, dict):
        entities = evidence.get('entities', [])
        entity_count += 1
        org_labels = [e.get('label','') for e in entities]
        if 'ORG' in org_labels:
            org_count += 1
print(f"  {org_count}/{entity_count} general_qa records have ORG entities detected")
print(f"  Organization positive rate: {100*org_count/entity_count:.1f}%")

# Check what ORG entities are found
print("\n--- Sample ORG entity texts in general_qa ---")
shown = 0
for r in general_qa_records:
    evidence = r.get('evidence')
    if evidence and isinstance(evidence, dict):
        entities = evidence.get('entities', [])
        org_entities = [e for e in entities if e.get('label') == 'ORG']
        if org_entities and shown < 5:
            print(f"  {r['record_id']}: query={r.get('query_text','')[:80]}")
            for e in org_entities[:3]:
                print(f"    ORG entity: '{e.get('text','')}'")
            shown += 1
