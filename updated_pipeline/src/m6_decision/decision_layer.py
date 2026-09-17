"""M6/M7 — Deterministic Risk-to-Policy Decision Layer.

Merges the formal DS (M6) and conformal (M7) approaches into a single
deterministic decision layer that maps model predictions + uncertainty
signals to explicit risk tiers and privacy policies.
"""
from __future__ import annotations

import torch
import numpy as np


class RiskPolicyClassifier:
    """Deterministic risk-to-policy decision module.

    Maps multi-task predictions, uncertainty metrics, and M2 weak labels
    to an explicit risk tier and predefined privacy policy.

    Risk Tiers:
      - low: no restrictions needed
      - medium: standard processing with logging
      - high: restricted processing, human review required
      - critical: block or escalate immediately

    Policy Actions:
      - restrict: block query or apply heavy filtering
      - cautious: allow with warnings, additional logging
      - standard: normal processing
      - open: no restrictions

    Conservative fallback: when uncertainty is too high to make a reliable
    decision, default to the most restrictive policy.
    """

    RISK_LEVELS = ["low", "medium", "high", "critical"]
    POLICY_ACTIONS = ["open", "standard", "cautious", "restrict"]

    def __init__(
        self,
        uncertainty_threshold: float = 0.5,
        high_risk_sensitivity: str = "high",
        critical_threats: list[str] = None,
        conservative_fallback: bool = True,
        config: dict | None = None,
    ):
        """Initialize the decision layer.

        Parameters
        ----------
        uncertainty_threshold : float
            Composite entropy threshold above which conservative fallback activates.
        high_risk_sensitivity : str
            Sensitivity label that triggers at least "high" risk.
        critical_threats : list[str]
            Threat content labels that trigger "critical" risk.
        conservative_fallback : bool
            If True, high uncertainty defaults to restrictive policy.
        config : dict | None
            Optional custom configuration dict for thresholds and rules.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.high_risk_sensitivity = high_risk_sensitivity
        self.critical_threats = critical_threats or ["re_identification"]
        self.conservative_fallback = conservative_fallback

        if config:
            self.uncertainty_threshold = config.get("uncertainty_threshold", self.uncertainty_threshold)
            self.high_risk_sensitivity = config.get("high_risk_sensitivity", self.high_risk_sensitivity)
            self.critical_threats = config.get("critical_threats", self.critical_threats)
            self.conservative_fallback = config.get("conservative_fallback", self.conservative_fallback)

    def predict(self, record: dict) -> dict:
        """Classify a single record into a risk tier and policy action.

        Parameters
        ----------
        record : dict
            Must contain:
            - sensitivity_mean: list of 3 probabilities (low, medium, high)
            - intent_mean: list of 5 probabilities
            - disclosure_scope_mean: list of 2 probabilities
            - entity_tags_mean: list of 4 probabilities
            - threat_content_mean: list of 3 probabilities
            - composite_entropy: float (sum of all dimensional entropies / 5)
            - sensitivity_entropy, intent_entropy, etc.: float per-dimension entropy
            - m2_labels (optional): dict of M2 weak labels for cross-validation

        Returns
        -------
        result : dict
            {
                "risk_level": str ("low"|"medium"|"high"|"critical"),
                "policy_action": str ("open"|"standard"|"cautious"|"restrict"),
                "risk_score": float (0.0-1.0),
                "reasons": list[str],
                "uncertainty_reliable": bool,
                "fallback_triggered": bool,
            }
        """
        sensitivity_mean = np.array(record["sensitivity_mean"])
        intent_mean = np.array(record["intent_mean"])
        disclosure_mean = np.array(record["disclosure_scope_mean"])
        entity_tags_mean = np.array(record["entity_tags_mean"])
        threat_mean = np.array(record["threat_content_mean"])
        composite_entropy = record.get("composite_entropy", 0.0)

        uncertainty_reliable = composite_entropy < self.uncertainty_threshold
        fallback_triggered = False

        risk_score = 0.0
        reasons = []

        # Factor 1: Sensitivity level (weight: 0.30)
        sensitivity_pred = int(np.argmax(sensitivity_mean))
        if sensitivity_pred == 2:  # high sensitivity
            risk_score += 0.30
            reasons.append("high_sensitivity")
        elif sensitivity_pred == 1:  # medium sensitivity
            risk_score += 0.15
            reasons.append("medium_sensitivity")
        else:
            reasons.append("low_sensitivity")

        # Factor 2: Threat content (weight: 0.30)
        threat_pred = int(np.argmax(threat_mean))
        threat_conf = float(threat_mean[threat_pred])
        if threat_pred in self.critical_threats or threat_conf > 0.7:
            risk_score += 0.30
            reasons.append(f"threat_confidence_{threat_conf:.2f}")
        elif threat_pred == 1:  # attribute_inference
            risk_score += 0.20
            reasons.append("attribute_inference_risk")
        else:
            reasons.append("no_threat")

        # Factor 3: Intent (weight: 0.15)
        intent_pred = int(np.argmax(intent_mean))
        if intent_pred == 4:  # identity_related_request
            risk_score += 0.15
            reasons.append("identity_related_intent")
        elif intent_pred in (2, 3):  # medical/financial request
            risk_score += 0.10
            reasons.append("sensitive_intent_request")
        else:
            reasons.append("general_intent")

        # Factor 4: Disclosure scope (weight: 0.10)
        disclosure_pred = int(np.argmax(disclosure_mean))
        if disclosure_pred == 1:  # multi_document
            risk_score += 0.10
            reasons.append("multi_document_disclosure")
        else:
            reasons.append("single_document_disclosure")

        # Factor 5: Entity tags (weight: 0.10)
        entity_pred = int(np.argmax(entity_tags_mean))
        if entity_pred == 3:  # has_contact_identifier
            risk_score += 0.10
            reasons.append("contact_identifier_present")
        elif entity_pred == 0:  # has_person
            risk_score += 0.05
            reasons.append("person_entity_present")
        else:
            reasons.append("entity_risk_low")

        risk_score = min(risk_score, 1.0)

        if risk_score >= 0.70:
            risk_level = "critical"
        elif risk_score >= 0.45:
            risk_level = "high"
        elif risk_score >= 0.25:
            risk_level = "medium"
        else:
            risk_level = "low"

        if not uncertainty_reliable and self.conservative_fallback:
            fallback_triggered = True
            risk_idx = self.RISK_LEVELS.index(risk_level)
            risk_level = self.RISK_LEVELS[min(risk_idx + 1, 3)]
            reasons.append("uncertainty_fallback")

        risk_to_policy = {
            "low": "open",
            "medium": "standard",
            "high": "cautious",
            "critical": "restrict",
        }
        policy_action = risk_to_policy[risk_level]

        return {
            "risk_level": risk_level,
            "policy_action": policy_action,
            "risk_score": round(risk_score, 4),
            "reasons": reasons,
            "uncertainty_reliable": uncertainty_reliable,
            "fallback_triggered": fallback_triggered,
            "sensitivity_pred": str(sensitivity_pred),
            "threat_pred": threat_pred,
            "intent_pred": intent_pred,
            "disclosure_pred": disclosure_pred,
            "entity_pred": entity_pred,
        }

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Classify a batch of records."""
        return [self.predict(r) for r in records]

    def get_policy_description(self, policy_action: str) -> str:
        """Return a human-readable description of a policy action."""
        descriptions = {
            "open": "No restrictions. Standard query processing allowed.",
            "standard": "Normal processing with standard logging and monitoring.",
            "cautious": "Restricted processing: additional warnings, enhanced logging, human review recommended.",
            "restrict": "Blocked or heavily filtered. Escalation required before any processing.",
        }
        return descriptions.get(policy_action, "Unknown policy action.")

    def summarize(self, results: list[dict]) -> dict:
        """Summarize batch classification results."""
        summary = {
            "total_records": len(results),
            "risk_distribution": {level: 0 for level in self.RISK_LEVELS},
            "policy_distribution": {action: 0 for action in self.POLICY_ACTIONS},
            "fallback_count": 0,
        }
        for r in results:
            summary["risk_distribution"][r["risk_level"]] += 1
            summary["policy_distribution"][r["policy_action"]] += 1
            if r["fallback_triggered"]:
                summary["fallback_count"] += 1
        return summary