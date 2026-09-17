"""M6/M7 — Deterministic Risk-to-Policy Decision Engine.

Transparent rule engine (not a trained classifier). All thresholds and
mappings are loaded from YAML configuration. Outputs:
  - risk_level: low | medium | high | critical
  - selected_policy: P0_low | P1_moderate | P2_high | P3_restrictive_fallback
  - fallback_applied: bool
  - triggered_rules: list[str]
  - task_probabilities: dict of task → predicted class
  - task_uncertainties: dict of task → mean_entropy

Review-2 decision prototype only. RAG enforcement and policy mapping are deferred.
"""
from __future__ import annotations

import yaml
import numpy as np
from pathlib import Path


def load_decision_config(config_path: str | Path = None) -> dict:
    """Load decision engine configuration from YAML."""
    if config_path is None:
        config_path = Path("configs/review1.yaml")
    config_path = Path(config_path)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("m6_decision", {})


class RiskPolicyDecisionEngine:
    """Transparent deterministic rule engine for risk-to-policy decisions.

    All thresholds, weights, and mappings come from YAML config.
    No learned parameters — purely rule-based.

    Policy codes (Review-2 temporary, enforcement deferred):
      P0_low         — No restrictions, standard processing
      P1_moderate    — Normal processing with logging
      P2_high        — Restricted processing, human review recommended
      P3_restrictive_fallback — Block/escalate, uncertainty fallback applied
    """

    RISK_LEVELS = ["low", "medium", "high", "critical"]
    POLICY_CODES = ["P0_low", "P1_moderate", "P2_high", "P3_restrictive_fallback"]
    POLICY_DESCRIPTIONS = {
        "P0_low": "No restrictions. Standard query processing allowed.",
        "P1_moderate": "Normal processing with standard logging and monitoring.",
        "P2_high": "Restricted processing: warnings, enhanced logging, human review recommended.",
        "P3_restrictive_fallback": "Blocked or heavily filtered. Escalation required. (Review-2 prototype — enforcement deferred)",
    }

    def __init__(self, config: dict = None, config_path: str | Path = None):
        """Initialize with YAML-driven configuration.

        Parameters
        ----------
        config : dict | None
            Pre-loaded configuration dict.
        config_path : str | Path | None
            Path to review1.yaml. Used if config is None.
        """
        if config is None:
            if config_path is None:
                config_path = Path("configs/review1.yaml")
            with open(config_path) as f:
                full_cfg = yaml.safe_load(f)
            config = full_cfg.get("m6_decision", {})

        self.thresholds = config.get("risk_score_thresholds", {
            "critical": 0.70,
            "high": 0.45,
            "medium": 0.25,
        })
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.5)
        self.high_risk_sensitivity = config.get("high_risk_sensitivity", "high")
        self.critical_threats = config.get("critical_threats", ["re_identification"])
        self.conservative_fallback = config.get("conservative_fallback", True)

        # Load task weights from config
        self.task_weights = config.get("task_weights", {
            "sensitivity": 0.30,
            "threat_content": 0.30,
            "intent": 0.15,
            "disclosure_scope": 0.10,
            "entity_tags": 0.10,
        })

        # Load risk triggers from config
        self.risk_triggers = config.get("risk_triggers", {})

    def predict(self, record: dict) -> dict:
        """Classify a single record into risk level and policy.

        Parameters
        ----------
        record : dict
            Must contain:
            - sensitivity_mean, intent_mean, disclosure_scope_mean,
              entity_tags_mean, threat_content_mean (probability lists)
            - sensitivity_entropy, intent_entropy, disclosure_scope_entropy,
              entity_tags_entropy, threat_content_entropy (per-dimension)
            - composite_entropy (optional, computed if absent)

        Returns
        -------
        result : dict
            {
                "risk_level": str,
                "selected_policy": str (P0/P1/P2/P3),
                "fallback_applied": bool,
                "triggered_rules": list[str],
                "task_probabilities": dict,
                "task_uncertainties": dict,
                "risk_score": float,
            }
        """
        sensitivity_mean = np.array(record["sensitivity_mean"])
        intent_mean = np.array(record["intent_mean"])
        disclosure_mean = np.array(record["disclosure_scope_mean"])
        entity_tags_mean = np.array(record["entity_tags_mean"])
        threat_mean = np.array(record["threat_content_mean"])

        composite_entropy = record.get("composite_entropy", 0.0)
        if composite_entropy == 0.0:
            entropies = [record.get(f"{dim}_entropy", 0) for dim in
                ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]]
            composite_entropy = sum(entropies) / len(entropies)

        # Determine if uncertainty makes decision unreliable
        uncertainty_reliable = composite_entropy < self.uncertainty_threshold
        fallback_applied = False
        triggered_rules = []

        # Compute risk score using YAML-configured weights
        risk_score = 0.0

        # Sensitivity
        sensitivity_pred = int(np.argmax(sensitivity_mean))
        sensitivity_conf = float(sensitivity_mean[sensitivity_pred])
        triggered_rules.append(f"sensitivity={sensitivity_pred}")
        if sensitivity_pred == 2:
            risk_score += self.task_weights["sensitivity"]
            triggered_rules.append("high_sensitivity")
        elif sensitivity_pred == 1:
            risk_score += self.task_weights["sensitivity"] * 0.5
            triggered_rules.append("medium_sensitivity")

        # Threat content
        threat_pred = int(np.argmax(threat_mean))
        threat_conf = float(threat_mean[threat_pred])
        triggered_rules.append(f"threat={threat_pred}")
        if threat_pred in self.critical_threats or threat_conf > 0.7:
            risk_score += self.task_weights["threat_content"]
            triggered_rules.append("critical_threat")
        elif threat_pred == 1:
            risk_score += self.task_weights["threat_content"] * 0.67
            triggered_rules.append("attribute_inference_risk")

        # Intent
        intent_pred = int(np.argmax(intent_mean))
        triggered_rules.append(f"intent={intent_pred}")
        if intent_pred == 4:
            risk_score += self.task_weights["intent"]
            triggered_rules.append("identity_related_intent")
        elif intent_pred in (2, 3):
            risk_score += self.task_weights["intent"] * 0.67
            triggered_rules.append("sensitive_intent_request")

        # Disclosure scope
        disclosure_pred = int(np.argmax(disclosure_mean))
        triggered_rules.append(f"disclosure={disclosure_pred}")
        if disclosure_pred == 1:
            risk_score += self.task_weights["disclosure_scope"]
            triggered_rules.append("multi_document_disclosure")

        # Entity tags
        entity_pred = int(np.argmax(entity_tags_mean))
        triggered_rules.append(f"entity={entity_pred}")
        if entity_pred == 3:
            risk_score += self.task_weights["entity_tags"]
            triggered_rules.append("contact_identifier_present")
        elif entity_pred == 0:
            risk_score += self.task_weights["entity_tags"] * 0.5
            triggered_rules.append("person_entity_present")

        risk_score = min(risk_score, 1.0)

        # Determine risk level from thresholds
        risk_level = self._score_to_level(risk_score)

        # Uncertainty fallback
        if not uncertainty_reliable and self.conservative_fallback:
            fallback_applied = True
            risk_idx = self.RISK_LEVELS.index(risk_level)
            risk_level = self.RISK_LEVELS[min(risk_idx + 1, 3)]
            triggered_rules.append("uncertainty_fallback")

        # Map to policy code
        selected_policy = self._risk_to_policy(risk_level)

        # Build task probabilities and uncertainties
        task_probabilities = {
            "sensitivity": {"predicted": sensitivity_pred, "confidence": sensitivity_conf},
            "intent": {"predicted": intent_pred, "confidence": float(intent_mean[intent_pred])},
            "disclosure_scope": {"predicted": disclosure_pred, "confidence": float(disclosure_mean[disclosure_pred])},
            "entity_tags": {"predicted": entity_pred, "confidence": float(entity_tags_mean[entity_pred])},
            "threat_content": {"predicted": threat_pred, "confidence": float(threat_mean[threat_pred])},
        }
        task_uncertainties = {
            dim: record.get(f"{dim}_entropy", 0.0)
            for dim in ["sensitivity", "intent", "disclosure_scope", "entity_tags", "threat_content"]
        }

        return {
            "risk_level": risk_level,
            "selected_policy": selected_policy,
            "fallback_applied": fallback_applied,
            "triggered_rules": triggered_rules,
            "task_probabilities": task_probabilities,
            "task_uncertainties": task_uncertainties,
            "risk_score": round(risk_score, 4),
            "composite_entropy": round(composite_entropy, 4),
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Classify a batch of records."""
        return [self.predict(r) for r in records]

    def _score_to_level(self, score: float) -> str:
        """Convert risk score to risk level using configured thresholds."""
        if score >= self.thresholds["critical"]:
            return "critical"
        elif score >= self.thresholds["high"]:
            return "high"
        elif score >= self.thresholds["medium"]:
            return "medium"
        else:
            return "low"

    def _risk_to_policy(self, risk_level: str) -> str:
        """Map risk level to Review-2 policy code."""
        mapping = {
            "low": "P0_low",
            "medium": "P1_moderate",
            "high": "P2_high",
            "critical": "P3_restrictive_fallback",
        }
        return mapping[risk_level]

    def get_policy_description(self, policy_code: str) -> str:
        """Return human-readable description of a policy code."""
        return self.POLICY_DESCRIPTIONS.get(policy_code, "Unknown policy code")

    def summarize(self, results: list[dict]) -> dict:
        """Summarize batch classification results."""
        summary = {
            "total_records": len(results),
            "risk_distribution": {level: 0 for level in self.RISK_LEVELS},
            "policy_distribution": {code: 0 for code in self.POLICY_CODES},
            "fallback_count": 0,
        }
        for r in results:
            summary["risk_distribution"][r["risk_level"]] += 1
            summary["policy_distribution"][r["selected_policy"]] += 1
            if r["fallback_applied"]:
                summary["fallback_count"] += 1
        return summary