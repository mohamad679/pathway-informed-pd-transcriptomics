"""Bootstrap confidence intervals for development-only Phase 5 validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from eval.metrics import (
    _validate_binary_inputs,
    compute_binary_metrics,
    expected_calibration_error,
)


BOOTSTRAP_METRICS = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")
BOOTSTRAP_CI_COLUMNS = ("metric", "estimate", "ci_lower", "ci_upper", "n_bootstrap")


def _single_class_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    present_class = int(y_true[0])
    return float(np.mean(y_pred == present_class))


def bootstrap_metric_cis(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    n_bootstrap: int = 2000,
    seed: int = 20260710,
) -> pd.DataFrame:
    """Return percentile bootstrap 95% CIs for pooled binary OOF metrics."""
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1")
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)
    estimates = compute_binary_metrics(labels, probabilities)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[sampled_indices]
        sampled_probabilities = probabilities[sampled_indices]
        if np.unique(sampled_labels).size == 2:
            metric_values = compute_binary_metrics(sampled_labels, sampled_probabilities)
            for metric in BOOTSTRAP_METRICS:
                samples[metric].append(float(metric_values[metric]))
            continue

        predictions = (sampled_probabilities >= 0.5).astype(int)
        samples["balanced_accuracy"].append(
            _single_class_balanced_accuracy(sampled_labels, predictions)
        )
        samples["brier"].append(float(brier_score_loss(sampled_labels, sampled_probabilities)))
        samples["ece"].append(
            float(expected_calibration_error(sampled_labels, sampled_probabilities))
        )

    rows: list[dict[str, float | int | str]] = []
    for metric in BOOTSTRAP_METRICS:
        metric_samples = np.asarray(samples[metric], dtype=float)
        if metric_samples.size == 0:
            lower = upper = float("nan")
        else:
            lower, upper = np.quantile(metric_samples, (0.025, 0.975))
        rows.append(
            {
                "metric": metric,
                "estimate": float(estimates[metric]),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "n_bootstrap": int(n_bootstrap),
            }
        )
    return pd.DataFrame(rows, columns=BOOTSTRAP_CI_COLUMNS)
