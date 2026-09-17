"""M5 -- MC-Dropout Uncertainty Quantification & Calibration."""
from __future__ import annotations

from m5_uncertainty.mc_dropout import (
    set_dropout_training,
    mc_dropout_inference,
    compute_predictive_mean,
    compute_predictive_variance,
    compute_categorical_entropy,
    compute_binary_entropy,
    compute_confidence,
    compute_uncertainty_summary,
    run_m5_inference,
)
from m5_uncertainty.calibration import (
    compute_ece,
    compute_categorical_ece,
    compute_multilabel_ece,
    compute_correct_vs_incorrect_uncertainty,
    save_high_uncertainty_examples,
)
from m5_uncertainty.temperature_scaling import (
    fit_temperature_scaling,
)