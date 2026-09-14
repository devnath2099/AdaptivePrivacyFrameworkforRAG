import json, numpy as np
from pathlib import Path

weak_dir = Path('outputs/m2_weak_labels')
for f in sorted(weak_dir.glob('*_weak_labels.jsonl')):
    dim = f.name.replace('_weak_labels.jsonl', '')
    with open(f, 'r', encoding='utf-8') as fh:
        first = json.loads(fh.readline())
    print(f'{dim}: labels={first["label_names"]}, shape={len(first["label_names"])}, sample_probs={[round(p,4) for p in first["probabilities"]]}')

print()
for f in sorted(weak_dir.glob('*_weak_labels.npy')):
    arr = np.load(str(f))
    print(f'{f.name}: shape={arr.shape}')

# Check alignment between M1 train split and M2 JSONL for each dimension
print("\nAlignment check:")
m1_train_ids = set()
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        d = json.loads(line)
        m1_train_ids.add(d['record_id'])
for f in sorted(weak_dir.glob('*_weak_labels.jsonl')):
    dim = f.name.replace('_weak_labels.jsonl', '')
    m2_ids = set()
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            m2_ids.add(json.loads(line)['record_id'])
    overlap = len(m1_train_ids & m2_ids)
    missing = len(m1_train_ids - m2_ids)
    print(f'  {dim}: overlap={overlap}, missing_from_m2={missing}, total_m1_train={len(m1_train_ids)}')

# Check that train/val/test records are correctly separated in M2 JSONL
print("\nM2 JSONL record split distribution:")
train_ids = set()
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        train_ids.add(json.loads(line)['record_id'])
val_ids = set()
with open('outputs/m1_validation_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        val_ids.add(json.loads(line)['record_id'])
test_ids = set()
with open('outputs/m1_test_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        test_ids.add(json.loads(line)['record_id'])

dim_jsonl = 'outputs/m2_weak_labels/intent_weak_labels.jsonl'
split_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
with open(dim_jsonl, 'r', encoding='utf-8') as fh:
    for line in fh:
        rid = json.loads(line)['record_id']
        if rid in train_ids: split_counts['train'] += 1
        elif rid in val_ids: split_counts['val'] += 1
        elif rid in test_ids: split_counts['test'] += 1
        else: split_counts['unknown'] += 1
print(f'  {split_counts}')
print(f'  Total: {sum(split_counts.values())}')
print(f'  Train:Val:Test = {split_counts["train"]}:{split_counts["val"]}:{split_counts["test"]}')
