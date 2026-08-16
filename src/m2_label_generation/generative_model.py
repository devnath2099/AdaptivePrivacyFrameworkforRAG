"""Fits a Snorkel-style generative model over the LF matrix and infers
posterior probabilities for the latent true label (steps 3-4 of the
SynthesizeWeakLabels algorithm).

Two backends:
  1. `snorkel_label_model` -- uses the real Snorkel `LabelModel` when the
     `snorkel` package is installed.
  2. `fallback_generative` -- a small, documented EM-style approximation
     implemented from scratch when Snorkel is unavailable. This fallback
     is a project-specific numerical approximation of the same posterior
     formula given in the spec; it is NOT a re-implementation claiming
     to be Snorkel's exact algorithm, and it is always logged as such.

Method foundation: Ratner et al. (2016) Data Programming; Bach et al.
(2019) Snorkel DryBell -- for the general weak-supervision idea only.
The exact fallback math below is this project's own design.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .taxonomy import ABSTAIN

logger = logging.getLogger(__name__)


@dataclass
class GenerativeModelResult:
    method: str
    class_priors: np.ndarray       # mu_k, shape (K,)
    lf_accuracy: np.ndarray        # theta_j, shape (m,) - P(LF correct | not abstaining)
    probs: np.ndarray              # L_w, shape (n, K)


def fit_and_infer(matrix: np.ndarray, num_classes: int, seed: int = 42,
                   n_em_iterations: int = 8) -> GenerativeModelResult:
    """Fit a generative model on `matrix` (n x m, values in {ABSTAIN, 0..K-1})
    and return posterior probabilities L_w of shape (n, num_classes).
    """
    try:
        return _fit_snorkel(matrix, num_classes, seed)
    except Exception as exc:  # noqa: BLE001
        logger.info("Snorkel LabelModel unavailable/failed (%s); using fallback_generative.", exc)
        return _fit_fallback(matrix, num_classes, n_em_iterations)


def _fit_snorkel(matrix: np.ndarray, num_classes: int, seed: int) -> GenerativeModelResult:
    from snorkel.labeling.model import LabelModel

    # Snorkel expects ABSTAIN = -1, which matches our convention.
    label_model = LabelModel(cardinality=num_classes, verbose=False)
    label_model.fit(L_train=matrix, seed=seed, n_epochs=200, log_freq=100)
    probs = label_model.predict_proba(L=matrix)
    probs = _safeguard_probs(probs, num_classes)

    class_priors = probs.mean(axis=0)
    lf_accuracy = _empirical_lf_accuracy(matrix, probs)
    return GenerativeModelResult(
        method="snorkel_label_model",
        class_priors=class_priors,
        lf_accuracy=lf_accuracy,
        probs=probs,
    )


def _fit_fallback(matrix: np.ndarray, num_classes: int, n_iterations: int) -> GenerativeModelResult:
    """Documented from-scratch approximation of the generative model.

    E-step / M-step sketch:
      - mu_k: current estimate of the class prior P(y=k).
      - theta_j: current estimate of P(lf_j correct | lf_j != ABSTAIN).
      - E-step: nu_k(i) = mu_k * prod_{j active} P(Lambda(i,j) | y_i=k; theta_j),
        where P(Lambda(i,j)=k | y_i=k) = theta_j and, for a wrong vote,
        the remaining mass (1 - theta_j) is spread uniformly over the
        other (K-1) classes. Normalize -> posterior L_w(i, k).
      - M-step: refit mu_k as the mean posterior; refit theta_j as the
        expected fraction of times LF j's vote matches the current
        posterior-weighted true class, among non-abstaining rows.
    A uniform-distribution fallback is used whenever the normalizing
    denominator would be zero (the "numerical safeguard" required by
    the spec); this has no relation to Confident Learning.
    """
    n, m = matrix.shape
    if n == 0:
        return GenerativeModelResult(
            method="fallback_generative",
            class_priors=np.full(num_classes, 1.0 / num_classes),
            lf_accuracy=np.full(m, 0.5),
            probs=np.zeros((0, num_classes)),
        )

    mu = np.full(num_classes, 1.0 / num_classes)
    theta = np.full(m, 0.7)  # initial optimistic-but-modest LF accuracy prior

    probs = np.full((n, num_classes), 1.0 / num_classes)

    for _ in range(max(1, n_iterations)):
        probs = _e_step(matrix, mu, theta, num_classes)
        mu, theta = _m_step(matrix, probs, num_classes)

    lf_accuracy = theta
    return GenerativeModelResult(
        method="fallback_generative",
        class_priors=mu,
        lf_accuracy=lf_accuracy,
        probs=probs,
    )


def _e_step(matrix: np.ndarray, mu: np.ndarray, theta: np.ndarray, num_classes: int) -> np.ndarray:
    n, m = matrix.shape
    nu = np.tile(mu, (n, 1))  # start with class priors, shape (n, K)

    for j in range(m):
        votes = matrix[:, j]
        active_mask = votes != ABSTAIN
        if not np.any(active_mask):
            continue
        correct_prob = theta[j]
        wrong_prob_each = (1.0 - theta[j]) / max(num_classes - 1, 1)
        for k in range(num_classes):
            like = np.where(votes == k, correct_prob, wrong_prob_each)
            like = np.where(active_mask, like, 1.0)  # abstaining LFs contribute no evidence
            nu[:, k] *= like

    return _safeguard_probs(nu, num_classes)


def _m_step(matrix: np.ndarray, probs: np.ndarray, num_classes: int):
    n, m = matrix.shape
    mu = probs.mean(axis=0)
    mu = mu / mu.sum() if mu.sum() > 0 else np.full(num_classes, 1.0 / num_classes)

    theta = np.full(m, 0.5)
    for j in range(m):
        votes = matrix[:, j]
        active_mask = votes != ABSTAIN
        n_active = active_mask.sum()
        if n_active == 0:
            theta[j] = 0.5
            continue
        # expected correctness = sum over active rows of posterior mass on the voted class
        idx = np.where(active_mask)[0]
        matched_mass = probs[idx, votes[idx]]
        theta[j] = float(np.clip(matched_mass.mean(), 0.05, 0.99))
    return mu, theta


def _safeguard_probs(raw: np.ndarray, num_classes: int) -> np.ndarray:
    """Normalize rows; fall back to a uniform distribution when the
    denominator is (numerically) zero. This is the numerical safeguard
    required by the spec -- not an application of Confident Learning.
    """
    row_sums = raw.sum(axis=1, keepdims=True)
    uniform_row = np.full(num_classes, 1.0 / num_classes)
    safe = np.divide(raw, row_sums, out=np.tile(uniform_row, (raw.shape[0], 1)), where=row_sums > 1e-12)
    safe = np.nan_to_num(safe, nan=1.0 / num_classes, posinf=1.0 / num_classes, neginf=1.0 / num_classes)
    # re-normalize defensively in case of residual floating point drift
    row_sums2 = safe.sum(axis=1, keepdims=True)
    safe = np.divide(safe, row_sums2, out=np.tile(uniform_row, (safe.shape[0], 1)), where=row_sums2 > 1e-12)
    return safe


def _empirical_lf_accuracy(matrix: np.ndarray, probs: np.ndarray) -> np.ndarray:
    n, m = matrix.shape
    num_classes = probs.shape[1]
    acc = np.full(m, 0.5)
    predicted = probs.argmax(axis=1)
    for j in range(m):
        votes = matrix[:, j]
        active = votes != ABSTAIN
        if not np.any(active):
            continue
        acc[j] = float(np.mean(votes[active] == predicted[active]))
    return acc
