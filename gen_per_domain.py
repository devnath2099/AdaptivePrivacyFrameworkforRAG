#!/usr/bin/env python
"""Generate per-domain diagnostics JSON."""
import json
import sys
sys.path.insert(0, "src")

from m1_data_integration.config import load_config
from m1_data_integration.pipeline import run_m1
from m2_label_generation.pipeline import run_m2
from m2_label_generation.diagnostics import full_diagnostics
from m2_label_generation.taxonomy import build_taxonomy

# Load config and run M1 to get records
cfg = load_config("configs/review1.yaml")
result = run_m1(cfg)

# Run M2
m2_result = run_m2(result.records, cfg)

# Get per-domain diagnostics
diags = full_diagnostics(m2_result.dimension_results, records=result.records)

# Save to file
with open("outputs01/per-domain-diagnostics.json", "w") as f:
    json.dump(diags, f, indent=2)

print("Saved per-domain-diagnostics.json")
print("Dimensions:", list(diags.keys()))
for dim, diag in diags.items():
    has_per = "per_domain" in diag
    print(f"  {dim}: has_per_domain={has_per}, n_records={diag['n_records']}")
    if has_per:
        doms = list(diag["per_domain"].keys())
        print(f"    domains: {doms}")
        for d in doms:
            pd = diag["per_domain"][d]
            print(f"      {d}: n={pd['n_records']}, H={pd['mean_entropy']:.3f}, C={pd['mean_max_confidence']:.3f}")