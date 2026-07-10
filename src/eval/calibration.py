"""Calibration helpers for development-only Phase 5 validation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from eval.metrics import _validate_binary_inputs, expected_calibration_error


CALIBRATION_BIN_COLUMNS = (
    "bin_index",
    "bin_left",
    "bin_right",
    "n_samples",
    "mean_predicted_probability",
    "observed_fraction",
)


def reliability_curve(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    n_bins: int = 15,
) -> pd.DataFrame:
    """Return an equal-width reliability table for binary probabilities."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)

    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        n_samples = int(np.sum(in_bin))
        rows.append(
            {
                "bin_index": int(bin_index),
                "bin_left": float(bin_edges[bin_index]),
                "bin_right": float(bin_edges[bin_index + 1]),
                "n_samples": n_samples,
                "mean_predicted_probability": (
                    float(np.mean(probabilities[in_bin])) if n_samples else float("nan")
                ),
                "observed_fraction": float(np.mean(labels[in_bin])) if n_samples else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=CALIBRATION_BIN_COLUMNS)


def calibration_summary(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    n_bins: int = 15,
) -> dict[str, float | int]:
    """Return Brier score and equal-width ECE for binary probabilities."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)
    return {
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": float(expected_calibration_error(labels, probabilities, n_bins=n_bins)),
        "n_bins": int(n_bins),
        "n_samples": int(labels.size),
    }
