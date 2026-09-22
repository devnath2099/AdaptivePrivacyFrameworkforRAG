import pytest
from review2_final.experiments import final_test


def test_smoke_cannot_open_research_test(monkeypatch, tmp_path):
    import review2_final.experiments as experiments
    monkeypatch.setattr(experiments, "validate_parent", lambda *args: {"smoke": True})
    with pytest.raises(RuntimeError, match="Smoke checkpoints"):
        final_test(tmp_path, "m2", "m3", "m4", tmp_path/"output")
    assert not (tmp_path/"TEST_OPENED.json").exists()
    assert not (tmp_path/"output").exists()
