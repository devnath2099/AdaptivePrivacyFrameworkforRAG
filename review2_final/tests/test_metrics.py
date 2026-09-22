import numpy as np
import pytest
from review2_final.metrics import entity_metrics, calibration_metrics
from review2_final.m4 import fit_temperature, scale


def test_strict_type_and_boundary_errors():
    records = [{"labels": ["B-NAME", "I-NAME", "O"]}]
    for pred in (["B-NAME", "O", "O"], ["B-ID", "I-ID", "O"]):
        m = entity_metrics(records, [pred], ["NAME", "ID"])
        assert m["micro"]["f1"] == 0
    assert entity_metrics(records, [["B-NAME", "I-NAME", "O"]], ["NAME"])["micro"]["f1"] == 1


def test_absent_class_is_not_perfect_recall():
    m = entity_metrics([{"labels": ["O"]}], [["O"]], ["NAME"])
    assert m["per_class"]["NAME"]["recall"] is None
    assert m["macro_f1_supported_classes"] is None


def test_calibration_known_answer():
    m = calibration_metrics(np.array([[.8, .2], [.2, .8]]), np.array([0, 1]))
    assert m["nll"] == pytest.approx(-np.log(.8))
    assert m["brier"] == pytest.approx(.08)
    assert m["ece"] == pytest.approx(.2)
    assert m["aurc"] == 0
    assert m["error_auroc"] is None


def test_temperature_improves_overconfident_errors():
    p = np.array([[.99, .01]]*10, dtype=np.float32)
    y = np.array([0]*7+[1]*3)
    t = fit_temperature(p, y)
    assert t > 1
    assert calibration_metrics(scale(p, t), y)["nll"] < calibration_metrics(p, y)["nll"]
