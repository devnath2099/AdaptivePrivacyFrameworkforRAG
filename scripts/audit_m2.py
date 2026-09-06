"""Audit M2 outputs to verify quality."""
import sys, json, os
from pathlib import Path
sys.path.insert(0, "src")

from m1_data_integration.config import load_config, ReviewConfig

def audit_m2(cfg: ReviewConfig):
    print("=" * 60)
    print("M2 OUTPUT AUDIT")
    print("=" * 60)

    diag_path = cfg.resolve_output("m2_diagnostics")
    with open(diag_path, "r") as fh:
        diagnostics = json.load(fh)

    print(f"\n1. Dimensions analyzed: {len(diagnostics)}")

    for dim, diag in diagnostics.items():
        n = diag.get("n_records", 0)
        entropy = diag.get("mean_entropy", 0)
        confidence = diag.get("mean_max_confidence", 0)
        uncertain = diag.get("n_uncertain_records", 0)
        method = diag.get("generative_model_method", "N/A")
        coverage = diag.get("lf_coverage", "N/A")
        label_dist = diag.get("label_distribution", {})

        print(f"\n2. Dimension: {dim}")
        print(f"   Records: {n}")
        print(f"   Method: {method}")
        print(f"   LF Coverage: {coverage}")
        print(f"   Mean Entropy: {entropy:.4f}")
        print(f"   Mean Max Confidence: {confidence:.4f}")
        print(f"   Uncertain Records (<0.6): {uncertain}")
        print(f"   Label Distribution: {label_dist}")

        # Per-domain breakdown
        per_domain = diag.get("per_domain", {})
        if per_domain:
            print(f"   Per-domain:")
            for d, dd in sorted(per_domain.items()):
                d_n = dd.get("n_records", 0)
                d_entropy = dd.get("mean_entropy", 0)
                d_conf = dd.get("mean_max_confidence", 0)
                d_uncertain = dd.get("n_uncertain_records", 0)
                print(f"     {d}: n={d_n}, entropy={d_entropy:.4f}, confidence={d_conf:.4f}, uncertain={d_uncertain}")

    # Check weak labels files
    weak_dir = cfg.resolve_output("m2_weak_labels_dir")
    print(f"\n3. Weak label files:")
    if weak_dir.exists():
        for f in sorted(weak_dir.glob("*.npy")):
            print(f"   {f.name}: exists")
        for f in sorted(weak_dir.glob("*.jsonl")):
            size = os.path.getsize(f) / 1024
            print(f"   {f.name}: {size:.1f} KB")

    # Check LF matrices
    lf_dir = cfg.resolve_output("m2_lf_matrix_dir")
    print(f"\n4. LF matrix files:")
    if lf_dir.exists():
        for f in sorted(lf_dir.glob("*.npy")):
            print(f"   {f.name}: exists")

    # Quality summary
    print(f"\n{'=' * 60}")
    all_good = True
    for dim, diag in diagnostics.items():
        entropy = diag.get("mean_entropy", 1)
        confidence = diag.get("mean_max_confidence", 0)
        if entropy > 1.0 or confidence < 0.5:
            all_good = False
            print(f"⚠️  {dim}: high entropy ({entropy:.3f}) or low confidence ({confidence:.3f})")

    if all_good:
        print("✅ M2 OUTPUTS LOOK GOOD")
    else:
        print("⚠️  M2 OUTPUTS HAVE ISSUES — REVIEW MANUALLY")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    cfg = load_config("configs/review1.yaml")
    audit_m2(cfg)
