"""Run M2 on merged M1 outputs + audit."""
import sys, json
sys.path.insert(0, "src")

from m1_data_integration.config import load_config
from m1_data_integration.pipeline import merge_partitions
from m2_label_generation.pipeline import run_m2, save_m2_outputs

cfg = load_config("configs/review1.yaml")

print("=== Step 1: Merge partitions ===")
result = merge_partitions(cfg, num_partitions=4)
print(f"Merged: {result.statistics['final_record_count']} records")

print("\n=== Step 2: Run M2 ===")
m2_result = run_m2(result.records, cfg)
record_ids = [r.record_id for r in result.records]
save_m2_outputs(m2_result, cfg, record_ids)
print("M2 complete!")

print("\n=== Step 3: Audit M2 ===")
with open(cfg.resolve_output("m2_diagnostics")) as f:
    d = json.load(f)
print(f"Dimensions: {len(d)}")
for dim, diag in d.items():
    print(f"  {dim}: n={diag.get('n_records',0)}, entropy={diag.get('mean_entropy',0):.3f}, conf={diag.get('mean_max_confidence',0):.3f}, uncertain={diag.get('n_uncertain_records',0)}")
print("M2 AUDIT DONE")
