from datasets import load_from_disk
from pathlib import Path

root = Path(r"C:\Users\Devnath\Desktop\datasets\data\raw")

datasets = {
    "Healthcare": ("healthcaremagic", "train"),
    "FiQA": ("fiqa", "corpus"),
    "HotpotQA": ("hotpotqa", "validation"),
    "Natural Questions": ("natural_questions", "train"),
}

print("=" * 70)
print("LOCAL SAVED DATASET SIZES")
print("=" * 70)

total = 0

for name, (folder, split) in datasets.items():
    path = root / folder

    try:
        ds = load_from_disk(str(path))

        # Handle DatasetDict vs Dataset
        if hasattr(ds, "keys"):
            dataset = ds[split]
        else:
            dataset = ds

        count = len(dataset)
        total += count

        print(f"{name:<22} : {count:,} records")
        print(f"{'  Path':<22} : {path}")
        print(f"{'  Split':<22} : {split}")
        print()

    except Exception as e:
        print(f"{name:<22} : ERROR")
        print(f"  {e}")
        print()

print("-" * 70)
print(f"{'TOTAL':<22} : {total:,} records")
print("=" * 70)