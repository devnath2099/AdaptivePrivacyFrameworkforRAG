"""Audit M3 outputs: verify alignment, shapes, and metrics."""
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from m1_data_integration.config import load_config


def main() -> None:
    cfg = load_config("configs/review1.yaml")
    m3_dir = cfg.resolve_output("m3_dir")

    print("=== M3 Audit ===")

    # Check artifacts exist
    artifacts = ["best_model.pt", "training_history.json", "validation_metrics.json", "m3_config.json"]
    for art in artifacts:
        path = m3_dir / art
        print(f"  {art}: {'EXISTS' if path.exists() else 'MISSING'} ({path.stat().st_size if path.exists() else 0} bytes)")

    # Load training history
    history_path = m3_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        print(f"\nTraining epochs: {len(history)}")
        if history:
            last = history[-1]
            print(f"Last epoch: train_loss={last.get('train_loss', 'N/A'):.4f}, val_loss={last.get('val_loss', 'N/A'):.4f}")

    # Load validation metrics
    metrics_path = m3_dir / "validation_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"\nValidation metrics:")
        for dim, m in metrics.items():
            if isinstance(m, dict):
                acc = m.get("accuracy", "N/A")
                f1 = m.get("macro_f1", "N/A")
                print(f"  {dim}: acc={acc}, macro_f1={f1}")

    print("\nM3 audit complete.")


if __name__ == "__main__":
    main()
