"""Audit M2 outputs to verify quality."""
import sys, json, os
from pathlib import Path
sys.path.insert(0, "src")

from m1_data_integration.config import load_config, ReviewConfig


def _print_label_summary(dim: str, label_dist: dict):
    """Render the label distribution; multi-label dims get per-category stats."""
    if "per_category_statistics" in label_dist:
        print(f"   Multi-label categories (independent):")
        for cat, st in label_dist["per_category_statistics"].items():
            q = st.get("quantiles", {})
            ath = st.get("above_threshold", {})
            print(f"     {cat}:")
            print(f"       mean={st.get('mean', 0):.4f} median={st.get('median', 0):.4f} "
                  f"std={st.get('std', 0):.4f} min={st.get('min', 0):.4f} max={st.get('max', 0):.4f}")
            print(f"       quantiles: " + " ".join(f"{k}={v:.4f}" for k, v in q.items()))
            for t, info in ath.items():
                print(f"       {t}: {info['count']} ({info['proportion']*100:.2f}%)")
        co = label_dist.get("co_occurrence", {})
        print(f"     co-occurrence: zero={co.get('zero_positive_records')} "
              f"one={co.get('exactly_one_positive_record')} "
              f"two={co.get('two_positive_records')} "
              f">=3={co.get('three_or_more_positive_records')}")
    else:
        print(f"   Label Distribution (argmax): {label_dist}")


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

        _print_label_summary(dim, label_dist)

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
                d_dist = dd.get("label_distribution", {})
                if "per_category_statistics" in d_dist:
                    for cat, st in d_dist["per_category_statistics"].items():
                        print(f"        {cat}: mean={st.get('mean', 0):.3f} median={st.get('median', 0):.3f} "
                              f"gt_0.5={st['above_threshold']['gt_0.5']['count']} "
                              f"gt_0.8={st['above_threshold']['gt_0.8']['count']}")

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
