"""Re-run ONLY M2 (LF matrix + generative model + weak labels + diagnostics) with
the current labeling functions, on the already-restored full-scale M1 records.

Does NOT re-run M1 dataset loading/cleaning/dedup/evidence/embeddings and does
NOT regenerate embeddings. It reads the unified records from the M1 split files
already restored under outputs/, re-runs the LF/generative-model/diagnostics
computation, and overwrites the M2 artifacts (weak labels + diagnostics).
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from m1_data_integration.config import load_config
from m1_data_integration.schemas import EvidenceBundle, UnifiedRecord
from m2_label_generation.pipeline import run_m2, save_m2_outputs


def load_unified_records(cfg) -> list:
    dataset_path = cfg.resolve_output("m1_dataset")
    out_dir = dataset_path.parent
    split_files = [out_dir / f"m1_{s}_dataset.jsonl" for s in ("train", "validation", "test")]
    files = split_files if all(p.exists() for p in split_files) else ([dataset_path] if dataset_path.exists() else split_files)

    records = []
    for path in files:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ev = d.get("evidence")
                if isinstance(ev, dict):
                    ev = EvidenceBundle(
                        entities=ev.get("entities", []),
                        dependency_relations=ev.get("dependency_relations", []),
                        regex_matches=ev.get("regex_matches", {}),
                        embedding=ev.get("embedding"),
                    )
                rec = UnifiedRecord(
                    record_id=d["record_id"],
                    domain=d["domain"],
                    source_dataset=d["source_dataset"],
                    query_text=d.get("query_text", "") or "",
                    context_text=d.get("context_text", "") or "",
                    normalized_text=d.get("normalized_text", "") or "",
                    metadata=d.get("metadata", {}),
                    evidence=ev,
                    is_duplicate=d.get("is_duplicate", False),
                    split=d.get("split"),
                )
                records.append(rec)
    return records


def main() -> None:
    cfg = load_config("configs/review1.yaml")
    print("Loading unified M1 records from restored split files...")
    records = load_unified_records(cfg)
    print(f"  {len(records)} records loaded")

    print("Re-running M2 (LFs + generative model + diagnostics)...")
    m2_result = run_m2(records, cfg)
    save_m2_outputs(m2_result, cfg, [r.record_id for r in records])

    print("M2 re-run complete.")
    for dim, res in m2_result.dimension_results.items():
        print(f"  {dim}: weak_labels shape={res.weak_labels.shape}, "
              f"method={res.generative_result.method}")


if __name__ == "__main__":
    main()