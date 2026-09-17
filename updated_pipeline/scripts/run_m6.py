#!/usr/bin/env python3
"""
M6 Decision Engine Runner
Runs the risk-to-policy decision engine on M5 outputs
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent)))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from m6_decision import RiskPolicyDecisionEngine

def main():
    # Load M5 outputs with decisions
    m5_dir = PROJECT_ROOT / "outputs" / "m5" / "full_m4"
    input_file = m5_dir / "uncertainty_predictions_with_decision.jsonl"
    
    if not input_file.exists():
        # Fallback: generate from separate files
        print("Generating combined M5+M6 output...")
        with open(m5_dir / "uncertainty_predictions.jsonl") as f:
            m5_records = [json.loads(line) for line in f]
        with open(PROJECT_ROOT / "outputs" / "m6" / "decision_results.jsonl") as f:
            m6_records = [json.loads(line) for line in f]
        
        combined = []
        for m5, m6 in zip(m5_records, m6_records):
            combined.append({**m5, **m6})
        
        with open(m5_dir / "uncertainty_predictions_with_decision.jsonl", "w") as f:
            for r in combined:
                f.write(json.dumps(r) + "\n")
        print(f"Created {len(combined)} combined records")
    else:
        print(f"Using existing combined file: {input_file}")
        with open(input_file) as f:
            combined = [json.loads(line) for line in f]
        print(f"Loaded {len(combined)} records")
    
    # Summary
    from collections import Counter
    risk_dist = Counter(r["risk_level"] for r in combined)
    policy_dist = Counter(r["selected_policy"] for r in combined)
    fallback_count = sum(1 for r in combined if r.get("fallback_applied", False))
    
    print(f"\nDecision Summary:")
    print(f"  Risk: {dict(risk_dist)}")
    print(f"  Policy: {dict(policy_dist)}")
    print(f"  Fallback: {fallback_count}/{len(combined)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())