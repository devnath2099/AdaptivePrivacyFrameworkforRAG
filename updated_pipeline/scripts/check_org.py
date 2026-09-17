import json
import numpy as np
import sys
sys.path.insert(0, 'src')

from m2_label_generation.labeling_functions import ENTITY_TAG_LFS
from m2_label_generation.lf_engine import build_lf_matrix
from m2_label_generation.taxonomy import DimensionSpec
from m2_label_generation.generative_model import fit_and_infer

records = []
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))

et = np.load('outputs/m2_lf_matrices/entity_tags_lambda_matrix.npy')

target = 'nq_014509_faa90344'
idx = next(i for i, r in enumerate(records) if r['record_id'] == target)
print(f'Record {target} at row {idx}')
print(f'  et_lf row: {et[idx]}')
print(f'  has_org_pos(col2)={et[idx,2]}, has_org_neg(col3)={et[idx,3]}')

# Check JSONL
with open('outputs/m2_weak_labels/entity_tags_weak_labels.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec['record_id'] == target:
            probs = rec['probabilities']
            names = rec['label_names']
            org_idx = names.index('has_organization')
            print(f'  JSONL has_organization posterior: {probs[org_idx]}')
            print(f'  JSONL all probabilities: {dict(zip(names, [round(p,6) for p in probs]))}')
            break

# Build the exact 2-col matrix
r = records[idx]
lfs = ENTITY_TAG_LFS['has_organization']
binary_spec = DimensionSpec(name='test', labels=['absent', 'present'], is_multi_label=False)
lf_result = build_lf_matrix([r], lfs, binary_spec)
print(f'\n  2-col matrix: {lf_result.matrix}')
print(f'  LF names: {lf_result.lf_names}')
result = fit_and_infer(lf_result.matrix, 2, seed=42)
print(f'  fit_and_infer: method={result.method}')
print(f'  probs: {result.probs}')
print(f'  class_priors: {result.class_priors}')

# Now check: what about nq_037692 (no entities)?
target2 = 'nq_037692_983a4dc6'
idx2 = next(i for i, r in enumerate(records) if r['record_id'] == target2)
print(f'\nRecord {target2} at row {idx2}')
print(f'  et_lf row: {et[idx2]}')
print(f'  has_org_pos(col2)={et[idx2,2]}, has_org_neg(col3)={et[idx2,3]}')
print(f'  NER entities: {r.get("evidence",{}).get("entities",[]) if r.get("evidence") else "none"}')

r2 = records[idx2]
lf_result2 = build_lf_matrix([r2], lfs, binary_spec)
print(f'  2-col matrix: {lf_result2.matrix}')
result2 = fit_and_infer(lf_result2.matrix, 2, seed=42)
print(f'  fit_and_infer: method={result2.method}')
print(f'  probs: {result2.probs}')

# Check the JSONL for this record
with open('outputs/m2_weak_labels/entity_tags_weak_labels.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec['record_id'] == target2:
            probs = rec['probabilities']
            names = rec['label_names']
            org_idx = names.index('has_organization')
            print(f'  JSONL has_organization posterior: {probs[org_idx]}')
            break
