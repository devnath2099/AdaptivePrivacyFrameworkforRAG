"""Tests for M6/M7 RiskPolicyDecisionEngine.

Tests the deterministic rule engine with YAML-driven configuration.
"""
import sys
from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent.parent)))
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pytest
import numpy as np

from m6_decision import RiskPolicyDecisionEngine


def make_record(sensitivity_mean, intent_mean, disclosure_mean, entity_mean, threat_mean, entropy=0.3):
    """Helper to create a test record dict."""
    return {
        "sensitivity_mean": sensitivity_mean,
        "intent_mean": intent_mean,
        "disclosure_scope_mean": disclosure_mean,
        "entity_tags_mean": entity_mean,
        "threat_content_mean": threat_mean,
        "sensitivity_entropy": entropy,
        "intent_entropy": entropy,
        "disclosure_scope_entropy": entropy,
        "entity_tags_entropy": entropy,
        "threat_content_entropy": entropy,
        "composite_entropy": entropy,
    }


class TestRiskPolicyDecisionEngine:
    """Test the deterministic decision engine."""

    def test_low_risk(self):
        """Low sensitivity, no threats → P0_low."""
        engine = RiskPolicyDecisionEngine(config={
            "uncertainty_threshold": 0.5,
            "conservative_fallback": False,
            "risk_score_thresholds": {"critical": 0.70, "high": 0.45, "medium": 0.25},
            "task_weights": {"sensitivity": 0.30, "threat_content": 0.30, "intent": 0.15, "disclosure_scope": 0.10, "entity_tags": 0.10},
        })
        record = make_record(
            sensitivity_mean=[0.8, 0.15, 0.05],  # low
            intent_mean=[0.7, 0.1, 0.1, 0.05, 0.05],  # general_information
            disclosure_mean=[0.9, 0.1],  # single_document
            entity_mean=[0.1, 0.1, 0.1, 0.05],  # no contact
            threat_mean=[0.1, 0.1, 0.8],  # membership_inference (not critical)
            entropy=0.1,
        )
        result = engine.predict(record)
        assert result["risk_level"] in ["low", "medium"]
        assert result["selected_policy"] in ["P0_low", "P1_moderate"]
        assert result["fallback_applied"] is False

    def test_high_risk(self):
        """High sensitivity + critical threat → P2_high or P3_restrictive_fallback."""
        engine = RiskPolicyDecisionEngine(config={
            "uncertainty_threshold": 0.5,
            "conservative_fallback": False,
            "risk_score_thresholds": {"critical": 0.70, "high": 0.45, "medium": 0.25},
            "task_weights": {"sensitivity": 0.30, "threat_content": 0.30, "intent": 0.15, "disclosure_scope": 0.10, "entity_tags": 0.10},
        })
        record = make_record(
            sensitivity_mean=[0.1, 0.1, 0.8],  # high
            intent_mean=[0.1, 0.1, 0.1, 0.1, 0.6],  # identity_related
            disclosure_mean=[0.1, 0.9],  # multi_document
            entity_mean=[0.1, 0.1, 0.1, 0.7],  # contact_identifier
            threat_mean=[0.7, 0.1, 0.2],  # re_identification (critical)
            entropy=0.3,
        )
        result = engine.predict(record)
        assert result["risk_level"] in ["high", "critical"]
        assert result["selected_policy"] in ["P2_high", "P3_restrictive_fallback"]
        assert len(result["triggered_rules"]) > 0

    def test_fallback_on_high_uncertainty(self):
        """High composite entropy → fallback_applied=True."""
        engine = RiskPolicyDecisionEngine(config={
            "uncertainty_threshold": 0.5,
            "conservative_fallback": True,
            "risk_score_thresholds": {"critical": 0.70, "high": 0.45, "medium": 0.25},
            "task_weights": {"sensitivity": 0.30, "threat_content": 0.30, "intent": 0.15, "disclosure_scope": 0.10, "entity_tags": 0.10},
        })
        record = make_record(
            sensitivity_mean=[0.33, 0.33, 0.34],  # uncertain
            intent_mean=[0.2, 0.2, 0.2, 0.2, 0.2],  # uncertain
            disclosure_mean=[0.5, 0.5],  # uncertain
            entity_mean=[0.25, 0.25, 0.25, 0.25],  # uncertain
            threat_mean=[0.33, 0.33, 0.34],  # uncertain
            entropy=0.8,  # Very high uncertainty
        )
        result = engine.predict(record)
        assert result["fallback_applied"] is True
        assert result["risk_level"] in ["high", "critical"]  # Upgraded by one tier
        assert "uncertainty_fallback" in result["triggered_rules"]

    def test_yaml_config_loading(self, tmp_path):
        """Test loading configuration from YAML."""
        import yaml
        config = {
            "m6_decision": {
                "uncertainty_threshold": 0.3,
                "conservative_fallback": False,
                "risk_score_thresholds": {"critical": 0.8, "high": 0.6, "medium": 0.4},
                "task_weights": {"sensitivity": 0.4, "threat_content": 0.3, "intent": 0.15, "disclosure_scope": 0.1, "entity_tags": 0.05},
                "critical_threats": ["re_identification"],
            }
        }
        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        engine = RiskPolicyDecisionEngine(config_path=str(config_path))
        assert engine.uncertainty_threshold == 0.3
        assert engine.conservative_fallback is False
        assert engine.thresholds["critical"] == 0.8

    def test_policy_mapping(self):
        """Test risk level → policy code mapping."""
        engine = RiskPolicyDecisionEngine()
        assert engine._risk_to_policy("low") == "P0_low"
        assert engine._risk_to_policy("medium") == "P1_moderate"
        assert engine._risk_to_policy("high") == "P2_high"
        assert engine._risk_to_policy("critical") == "P3_restrictive_fallback"

    def test_summarize(self):
        """Test batch summarization."""
        engine = RiskPolicyDecisionEngine(config={"conservative_fallback": False})
        results = [
            {"risk_level": "low", "selected_policy": "P0_low", "fallback_applied": False},
            {"risk_level": "high", "selected_policy": "P2_high", "fallback_applied": False},
            {"risk_level": "high", "selected_policy": "P2_high", "fallback_applied": True},
        ]
        summary = engine.summarize(results)
        assert summary["total_records"] == 3
        assert summary["risk_distribution"]["low"] == 1
        assert summary["risk_distribution"]["high"] == 2
        assert summary["policy_distribution"]["P0_low"] == 1
        assert summary["policy_distribution"]["P2_high"] == 2
        assert summary["fallback_count"] == 1

    def test_task_probabilities_in_output(self):
        """Test that output includes task_probabilities and task_uncertainties."""
        engine = RiskPolicyDecisionEngine(config={"conservative_fallback": False})
        record = make_record(
            sensitivity_mean=[0.8, 0.15, 0.05],
            intent_mean=[0.7, 0.1, 0.1, 0.05, 0.05],
            disclosure_mean=[0.9, 0.1],
            entity_mean=[0.1, 0.1, 0.1, 0.05],
            threat_mean=[0.1, 0.1, 0.8],
            entropy=0.1,
        )
        result = engine.predict(record)
        assert "task_probabilities" in result
        assert "task_uncertainties" in result
        assert "risk_score" in result
        assert "triggered_rules" in result
        assert "composite_entropy" in result

    def test_policy_description(self):
        """Test human-readable policy descriptions."""
        engine = RiskPolicyDecisionEngine()
        desc = engine.get_policy_description("P0_low")
        assert "No restrictions" in desc
        desc = engine.get_policy_description("P3_restrictive_fallback")
        assert "Blocked" in desc or "escalation" in desc.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])