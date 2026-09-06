"""Audit M1 outputs to verify quality before M2."""
import sys, json
from pathlib import Path
sys.path.insert(0, "src")

from m1_data_integration.config import load_config, ReviewConfig

def audit_m1(cfg: ReviewConfig):
    print("=" * 60)
    print("M1 OUTPUT AUDIT")
    print("=" * 60)

    # Load unified dataset
    dataset_path = cfg.resolve_output("m1_dataset")
    records = []
    with open(dataset_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"\n1. Total records: {len(records)}")

    # Split distribution
    splits = {}
    for r in records:
        s = r.get("split", "unknown")
        splits[s] = splits.get(s, 0) + 1
    print(f"\n2. Split distribution:")
    for s, c in sorted(splits.items()):
        print(f"   {s}: {c} ({c/len(records)*100:.1f}%)")

    # Domain distribution
    domains = {}
    for r in records:
        d = r.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    print(f"\n3. Domain distribution:")
    for d, c in sorted(domains.items()):
        print(f"   {d}: {c}")

    # Split by domain
    split_domain = {}
    for r in records:
        s = r.get("split", "unknown")
        d = r.get("domain", "unknown")
        split_domain.setdefault(s, {})
        split_domain[s][d] = split_domain[s].get(d, 0) + 1
    print(f"\n4. Domain distribution per split:")
    for s in sorted(split_domain.keys()):
        print(f"   {s}:")
        for d, c in sorted(split_domain[s].items()):
            print(f"     {d}: {c}")

    # Evidence quality
    has_evidence = sum(1 for r in records if r.get("evidence"))
    has_entities = sum(1 for r in records if r.get("evidence") and r["evidence"].get("entities"))
    has_embedding = sum(1 for r in records if r.get("evidence") and r["evidence"].get("embedding"))
    total_entities = sum(len(r["evidence"]["entities"]) for r in records if r.get("evidence") and r["evidence"].get("entities"))
    total_regex = sum(len(v) for r in records if r.get("evidence") and r["evidence"].get("regex_matches") for v in r["evidence"]["regex_matches"].values())

    print(f"\n5. Evidence quality:")
    print(f"   Records with evidence: {has_evidence}/{len(records)} ({has_evidence/len(records)*100:.1f}%)")
    print(f"   Records with entities: {has_entities}/{len(records)} ({has_entities/len(records)*100:.1f}%)")
    print(f"   Records with embedding: {has_embedding}/{len(records)} ({has_embedding/len(records)*100:.1f}%)")
    print(f"   Total entities: {total_entities}")
    print(f"   Total regex hits: {total_regex}")

    # Check for duplicates in normalized_text
    texts = [r.get("normalized_text", "") for r in records]
    unique_texts = len(set(texts))
    print(f"\n6. Dedup check: {unique_texts} unique texts out of {len(records)} ({unique_texts/len(records)*100:.1f}% unique)")

    # Check embedding dimensions
    embedding_dims = set()
    for r in records:
        if r.get("evidence") and r["evidence"].get("embedding"):
            embedding_dims.add(len(r["evidence"]["embedding"]))
    print(f"\n7. Embedding dimensions: {embedding_dims}")

    # Leakage from stats
    stats_path = cfg.resolve_output("m1_stats")
    with open(stats_path, "r") as fh:
        stats = json.load(fh)

    print(f"\n8. Split leakage check:")
    leakage = stats.get("split_leakage", {})
    nt = leakage.get("normalized_text_overlap", {})
    for pair, count in nt.items():
        if isinstance(count, int):
            print(f"   normalized_text {pair}: {count} overlaps")
        elif isinstance(count, dict):
            print(f"   normalized_text {pair}: {count.get('train_validation', 0)} / {count.get('train_test', 0)} / {count.get('validation_test', 0)}")

    sk = leakage.get("source_key_overlap", {})
    for pair, count in sk.items():
        if isinstance(count, int):
            print(f"   source_key {pair}: {count} overlaps")

    print(f"\n9. Reconciliation: {stats.get('reconciliation', {}).get('checks_passed', 'N/A')}")

    print(f"\n{'=' * 60}")
    if has_evidence / len(records) > 0.9 and has_embedding / len(records) > 0.9 and stats.get("reconciliation", {}).get("checks_passed"):
        print("✅ M1 OUTPUTS LOOK GOOD FOR M2")
    else:
        print("⚠️  M1 OUTPUTS MAY HAVE ISSUES — REVIEW MANUALLY")
    print(f"{'=' * 60}")

    return records


if __name__ == "__main__":
    cfg = load_config("configs/review1.yaml")
    audit_m1(cfg)
