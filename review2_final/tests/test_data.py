import copy
import pytest
from review2_final.m1 import canonical, spans, allocate, build, load_split
from review2_final.common import write, development_guard


def native(doc, labels=("B-NAME", "I-NAME", "O")):
    return {"document": doc, "full_text": "Ada Lovelace wrote", "tokens": ["Ada", "Lovelace", "wrote"],
            "trailing_whitespace": [True, True, False], "labels": list(labels)}


def test_native_roundtrip_and_offsets():
    row = native(1)
    result = canonical(row, {})
    assert result["original"] == row
    entity = result["entities"][0]
    assert result["text"][entity["start"]:entity["end"]] == "Ada Lovelace"


@pytest.mark.parametrize("change", ["labels", "spaces", "bio"])
def test_malformed_annotations_fail(change):
    row = native(1)
    if change == "labels":
        row["labels"].pop()
    elif change == "spaces":
        row["trailing_whitespace"][0] = False
    else:
        row["labels"][0] = "I-NAME"
    with pytest.raises(ValueError):
        canonical(row, {})


def test_group_isolation_reproducibility_and_rare_coverage():
    records = []
    for i in range(40):
        r = canonical(native(i), {})
        r["group"] = str(i//2)
        r["entities"] = [{"type": "RARE" if i < 4 else "COMMON"}]
        records.append(r)
    proportions = {"train": .7, "calibration": .1, "validation": .1, "test": .1}
    a, b = allocate(records, proportions, 1), allocate(records, proportions, 1)
    assert a == b
    assert sum(map(len, a.values())) == 40
    groups = [{r["group"] for r in rs} for rs in a.values()]
    assert all(not x & y for i, x in enumerate(groups) for y in groups[i+1:])
    assert any(r["entities"][0]["type"] == "RARE" for r in a["train"])
    assert any(r["entities"][0]["type"] == "RARE" for r in a["test"])


def test_test_lock(tmp_path):
    development_guard(tmp_path)
    write(tmp_path/"TEST_OPENED.json", {})
    with pytest.raises(RuntimeError):
        development_guard(tmp_path)


def test_tampered_split_rejected(tmp_path):
    write(tmp_path/"train.json", [])
    write(tmp_path/"manifest.json", {"files": {"train": "not-the-hash"}})
    with pytest.raises(ValueError, match="fingerprint"):
        load_split(tmp_path, "train")
