import json
import numpy as np
from pathlib import Path

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

print(f'Loaded weak labels for {len(weak_data)} records')
sample_dims = list(weak_data[list(weak_data.keys())[0]].keys())
print(f'Dimensions: {sample_dims}')
print()

records = []
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
print(f'Loaded {len(records)} M1 train records')

rec_by_id = {r['record_id']: r for r in records}

domains_seen = set()
selected = []
for r in records:
    d = r['domain']
    if d not in domains_seen and r['record_id'] in weak_data:
        selected.append(r)
        domains_seen.add(d)
    if len(selected) == 5:
        break

if len(selected) < 5:
    for r in records:
        if r['record_id'] not in [s['record_id'] for s in selected] and r['record_id'] in weak_data:
            selected.append(r)
        if len(selected) == 5:
            break

print(f'Selected {len(selected)} records:')
for s in selected:
    print(f'  {s["record_id"]} (domain={s["domain"]}, split={s.get("split","?")})')
print()

DIM_LABELS = {
    'sensitivity': None,
    'intent': None,
    'disclosure_scope': None,
    'entity_tags': None,
    'threat_content': None,
}

for s in selected:
    rid = s['record_id']
    wd = weak_data.get(rid, {})
    print('=' * 80)
    print(f'RECORD: {rid}')
    print(f'Domain: {s["domain"]} | Split: {s.get("split","?")}')
    print(f'Source dataset: {s["source_dataset"]}')
    print()
    print(f'QUERY TEXT:')
    print(s.get('query_text', '')[:500])
    print()
    print(f'CONTEXT TEXT:')
    print(s.get('context_text', '')[:500])
    print()

    for dim in ['sensitivity', 'intent', 'disclosure_scope', 'entity_tags', 'threat_content']:
        if dim not in wd:
            print(f'  {dim}: NO WEAK LABELS')
            continue
        label_names, probs = wd[dim]
        probs = np.array(probs)
        print(f'  {dim.upper()} (shape={probs.shape}):')
        for name, p in zip(label_names, probs):
            print(f'    {name}: {p:.6f}')
        top_idx = int(np.argmax(probs))
        print(f'    -> PREDICTED: {label_names[top_idx]} ({probs[top_idx]:.6f})')
    print()
    print()
