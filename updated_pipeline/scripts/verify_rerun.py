import json
import numpy as np
import sys
sys.path.insert(0, 'src')

from m1_data_integration.config import load_config
from m1_data_integration.pipeline import merge_partitions
from m2_label_generation.pipeline import run_m2, save_m2_outputs
from m2_label_generation.weak_labels import synthesize_weak_labels_for_dimension
from m2_label_generation.taxonomy import build_taxonomy

cfg = load_config('configs/review1.yaml')

# Load records exactly like rerun_m2.py
records = []
with open('outputs/m1_train_dataset.jsonl', 'r', encoding='utf-8') as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        from m1_data_integration.schemas import EvidenceBundle, UnifiedRecord
        ev = d.get('evidence')
        if isinstance(ev, dict):
            ev = EvidenceBundle(
                entities=ev.get('entities', []),
                dependency_relations=ev.get('dependency_relations', []),
                regex_matches=ev.get('regex_matches', {}),
                embedding=ev.get('embedding'),
            )
        rec = UnifiedRecord(
            record_id=d['record_id'],
            domain=d['domain'],
            source_dataset=d['source_dataset'],
            query_text=d.get('query_text', '') or '',
            context_text=d.get('context_text', '') or '',
            normalized_text=d.get('normalized_text', '') or '',
            metadata=d.get('metadata', {}),
            evidence=ev,
            is_duplicate=d.get('is_duplicate', False),
            split=d.get('split'),
        )
        records.append(rec)

print(f'Loaded {len(records)} records')

taxonomy = build_taxonomy(cfg.label_taxonomy)

# Test entity_tags has_organization dimension
spec = taxonomy['entity_tags']
print(f'Entity tags labels: {spec.labels}')

# Build has_organization result directly
from m2_label_generation.labeling_functions import ENTITY_TAG_LFS
from m2_label_generation.taxonomy import DimensionSpec
from m2_label_generation.lf_engine import build_lf_matrix
from m2_label_generation.generative_model import fit_and_infer

lfs = ENTITY_TAG_LFS['has_organization']
binary_spec = DimensionSpec(name='entity_tags_has_organization', labels=['absent', 'present'], is_multi_label=False)
lf_result = build_lf_matrix(records, lfs, binary_spec, balance_domains=False, seed=42)
print(f'LF matrix shape: {lf_result.matrix.shape}')
print(f'LF names: {lf_result.lf_names}')
print(f'Row 81844: {lf_result.matrix[81844]}')

gen_result = fit_and_infer(lf_result.matrix, num_classes=2, seed=42)
print(f'Backend: {gen_result.method}')
print(f'Probs for row 81844: {gen_result.probs[81844]}')
print(f'has_organization posterior (col 1): {gen_result.probs[81844, 1]}')

# Compare with .npy
et_npy = np.load('outputs/m2_lf_matrices/entity_tags_lambda_matrix.npy')
print(f'\n.npy row 81844: {et_npy[81844]}')

# Compare with JSONL
with open('outputs/m2_weak_labels/entity_tags_weak_labels.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec['record_id'] == records[81844].record_id:
            print(f'JSONL has_organization: {rec["probabilities"][1]}')
            break

# Now find nq_014509 in the full records
target = 'nq_014509_faa90344'
tidx = next(i for i, r in enumerate(records) if r.record_id == target)
print(f'\n{target} at row {tidx}')
print(f'LF matrix row {tidx}: {lf_result.matrix[tidx]}')
print(f'fit_and_infer probs row {tidx}: {gen_result.probs[tidx]}')
