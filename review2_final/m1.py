"""Verified PIILO v2 ingestion and deterministic grouped multilabel splitting."""
from collections import Counter, defaultdict
from pathlib import Path
import random
import re

from .common import read, write, digest, fingerprint, fresh_dir


def spans(labels, strict=True):
    result, start, kind = [], None, None
    for i, tag in enumerate(list(labels) + ["O"]):
        if tag != "O" and not re.fullmatch(r"[BI]-.+", tag):
            raise ValueError(f"Malformed BIO tag {tag!r}")
        prefix, current = ("O", None) if tag == "O" else tag.split("-", 1)
        continuation = prefix == "I" and kind == current
        if prefix == "I" and not continuation and strict:
            raise ValueError(f"Orphan I tag at token {i}: {tag}")
        if kind is not None and not continuation:
            result.append((start, i, kind))
            start, kind = None, None
        if prefix == "B" or (prefix == "I" and not continuation):
            start, kind = i, current
    return result


def canonical(row, provenance):
    required = {"document", "full_text", "tokens", "trailing_whitespace", "labels"}
    if not required <= row.keys():
        raise ValueError(f"Missing columns: {required - row.keys()}")
    tokens, labels, spaces = row["tokens"], row["labels"], row["trailing_whitespace"]
    if not all(isinstance(x, list) for x in (tokens, labels, spaces)):
        raise ValueError("Token columns must be lists")
    if not tokens or not len(tokens) == len(labels) == len(spaces):
        raise ValueError("Empty/misaligned annotation arrays")
    if any(not isinstance(t, str) or not t for t in tokens) or any(type(w) is not bool for w in spaces):
        raise ValueError("Invalid tokens or whitespace flags")
    if not isinstance(row["full_text"], str) or any(not isinstance(t, str) for t in labels):
        raise ValueError("Text/labels must be strings")
    rebuilt = "".join(t + (" " if w else "") for t, w in zip(tokens, spaces))
    if rebuilt != row["full_text"]:
        raise ValueError(f"Native token reconstruction differs: document {row['document']}")
    offsets, position = [], 0
    for token, space in zip(tokens, spaces):
        offsets.append([position, position + len(token)])
        position += len(token) + int(space)
    entities = [{"token_start": a, "token_end": b, "start": offsets[a][0],
                 "end": offsets[b-1][1], "type": k} for a, b, k in spans(labels)]
    return {"record_id": str(row["document"]), "document_id": row["document"],
            "original": row, "text": row["full_text"], "tokens": tokens, "labels": labels,
            "offsets": offsets, "entities": entities, "source": provenance,
            "group": fingerprint(" ".join(row["full_text"].casefold().split()))}


def statistics(records):
    entity = Counter(e["type"] for r in records for e in r["entities"])
    docs = Counter(k for r in records for k in {e["type"] for e in r["entities"]})
    return {"documents": len(records), "tokens": sum(len(r["tokens"]) for r in records),
            "pii_positive_documents": sum(bool(r["entities"]) for r in records),
            "pii_negative_documents": sum(not r["entities"] for r in records),
            "entity_counts": dict(sorted(entity.items())), "category_documents": dict(sorted(docs.items())),
            "label_counts": dict(sorted(Counter(t for r in records for t in r["labels"]).items()))}


def features(records):
    return {e["type"] for r in records for e in r["entities"]} | {
        "__positive__" if any(r["entities"] for r in records) else "__negative__"}


def allocate(records, proportions, seed):
    """Rarest-label-first iterative allocation of indivisible duplicate groups.

    Quotas count groups (not tokens); document size is the secondary criterion.
    Minimum coverage targets are aspirations, not guarantees for co-occurring labels.
    """
    if any(v <= 0 for v in proportions.values()) or abs(sum(proportions.values())-1) > 1e-8:
        raise ValueError("Split proportions must be positive and sum to one")
    names = list(proportions)
    groups = defaultdict(list)
    for row in records:
        groups[row["group"]].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    rank = {key: i for i, key in enumerate(keys)}
    tags = {key: features(rows) for key, rows in groups.items()}
    counts = Counter(t for ts in tags.values() for t in ts)
    # Preserve rare training exposure and independent test evidence where possible.
    priority = [x for x in ("train", "test", "validation", "calibration") if x in names]
    priority += [x for x in names if x not in priority]
    targets = {}
    for tag, n in counts.items():
        q = {name: int(n * proportions[name]) for name in names}
        for name in sorted(names, key=lambda s: (-(n*proportions[s]-q[s]), priority.index(s)))[:n-sum(q.values())]:
            q[name] += 1
        for name in priority[:min(n, len(names))]:
            if q[name] == 0:
                donor = max(names, key=lambda s: q[s])
                if q[donor] > 1:
                    q[donor] -= 1
                    q[name] += 1
        targets[tag] = q
    remaining, assigned = set(keys), {name: [] for name in names}
    actual = {name: Counter() for name in names}
    sizes = Counter()
    while remaining:
        remaining_counts = Counter(t for key in remaining for t in tags[key])
        rare = min(remaining_counts, key=lambda t: (remaining_counts[t], t))
        batch = sorted((key for key in remaining if rare in tags[key]), key=lambda k: rank[k])
        for key in batch:
            def score(name):
                capacity = len(records)*proportions[name] - sizes[name]
                return (targets[rare][name]-actual[name][rare], capacity, -priority.index(name))
            chosen = max(names, key=score)
            assigned[chosen].extend(groups[key])
            sizes[chosen] += len(groups[key])
            actual[chosen].update(tags[key])
            remaining.remove(key)
    return assigned


def audit(raw, metadata):
    rows = read(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected nonempty JSON array")
    source = {"dataset": metadata["title"], "release": metadata["currentVersionNumber"],
              "ref": metadata["ref"], "license": metadata["licenseName"],
              "source_sha256": digest(raw)}
    records = [canonical(r, source) for r in rows]
    ids = [r["record_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated document IDs; resolve provenance before splitting")
    by_text = defaultdict(list)
    for r in records:
        by_text[r["text"]].append(r)
    conflicting = [rs[0]["record_id"] for rs in by_text.values()
                   if len({fingerprint((r["tokens"], r["labels"])) for r in rs}) > 1]
    if conflicting:
        raise ValueError(f"Conflicting duplicate annotations: {conflicting}")
    report = {"source": source, "columns": sorted(rows[0]), "statistics": statistics(records),
              "exact_duplicate_excess": len(records)-len(by_text),
              "normalized_duplicate_excess": len(records)-len({r["group"] for r in records}),
              "group_information": "Document ID and normalized text only; author identity unavailable. Semantic near duplicates not established.",
              "official_labelled_splits": ["train.json"],
              "validation": "All native BIO sequences, lengths and character offsets validated without annotation repair."}
    return records, report


def build(raw, metadata_path, output, proportions, seed=20260922):
    records, report = audit(raw, read(metadata_path))
    root = fresh_dir(output)
    # Exact duplicates retained in audit but only one exemplar used in experiments.
    unique = list({r["text"]: r for r in reversed(records)}.values())
    splits = allocate(unique, proportions, seed)
    observed = sorted({tag for r in records for tag in r["labels"]})
    categories = sorted({e["type"] for r in records for e in r["entities"]})
    # BIO closure supports valid continuation predictions even if a tag is absent in gold.
    labels = ["O"] + [f"{p}-{k}" for k in categories for p in ("B", "I")]
    report["observed_labels"] = observed
    report["model_labels"] = labels
    report["bio_closure_added"] = sorted(set(labels)-set(observed))
    manifest = {"source_sha256": digest(raw), "seed": seed, "proportions": proportions,
                "strategy": "rarest-label-first grouped multilabel allocation; deduplication first",
                "splits": {}, "files": {}, "coverage_warnings": []}
    for name, rows in splits.items():
        rows.sort(key=lambda r: r["record_id"])
        for row in rows:
            row["split"] = name
        path = root / f"{name}.json"
        write(path, rows)
        manifest["files"][name] = digest(path)
        manifest["splits"][name] = statistics(rows)
        for kind in categories:
            if manifest["splits"][name]["entity_counts"].get(kind, 0) == 0:
                manifest["coverage_warnings"].append(f"{name}: no {kind} gold entities; recall is undefined")
    sets = [{r["group"] for r in rows} for rows in splits.values()]
    if any(a & b for i, a in enumerate(sets) for b in sets[i+1:]):
        raise AssertionError("Duplicate leakage")
    manifest["leakage_check"] = "No document or normalized duplicate group crosses splits"
    write(root / "audit.json", report)
    write(root / "labels.json", labels)
    manifest["labels_sha256"] = digest(root / "labels.json")
    write(root / "manifest.json", manifest)
    # Nested training fractions: same held-out splits, rare-aware quarter allocation.
    quarters = allocate(splits["train"], {f"q{i}": .25 for i in range(1,5)}, seed)
    selected, subsets = [], {}
    for i in range(1, 5):
        selected.extend(quarters[f"q{i}"])
        subsets[str(i*25)] = {"record_ids": sorted(r["record_id"] for r in selected),
                              "statistics": statistics(selected)}
    write(root / "subsets.json", subsets)
    return manifest


def load_split(root, split):
    root = Path(root)
    manifest = read(root / "manifest.json")
    if digest(root / f"{split}.json") != manifest["files"][split]:
        raise ValueError("Split fingerprint changed")
    if digest(root / "labels.json") != manifest["labels_sha256"]:
        raise ValueError("Label map changed")
    return read(root / f"{split}.json")
