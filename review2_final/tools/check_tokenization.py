"""Check full development splits without training or opening test annotations."""
from pathlib import Path
from review2_final.common import read, write
from review2_final.m1 import load_split
from review2_final.neural import tokenizer_for
from review2_final.experiments import make_batches


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = read(root/"config.json")
    labels = read(root/"data/prepared/labels.json")
    tokenizer = tokenizer_for()
    report = {}
    for name in ("validation", "train", "calibration"):
        records = load_split(root/"data/prepared", name)
        batches, coverage = make_batches(records, tokenizer, labels, cfg)
        seen = [[] for _ in records]
        for window in batches.dataset:
            assert len(window["input_ids"]) <= cfg["max_length"]
            seen[window["doc"]].extend(w for w in window["words"] if w >= 0)
        assert all(indices == list(range(len(row["tokens"]))) for indices, row in zip(seen, records))
        report[name] = {**coverage, "all_native_tokens_scored_once": True}
        write(root/"reports/tokenization_preflight.json", report)
        print(f"{name}: {coverage['documents']} documents, {coverage['windows']} windows, "
              f"{coverage['oversized_native_tokens']} oversized native tokens; coverage passed", flush=True)
        del records, batches, seen


if __name__ == "__main__":
    main()
