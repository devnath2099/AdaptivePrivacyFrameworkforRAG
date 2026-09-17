"""M6/M7 — Deterministic Risk-to-Policy Decision Engine.

Transparent rule engine (not a trained classifier). All thresholds and
mappings are loaded from YAML configuration. Review-2 decision prototype only.
"""
from __future__ import annotations

from m6_decision.decision_engine import RiskPolicyDecisionEngine, load_decision_config