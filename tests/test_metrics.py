from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.metrics import (
    bootstrap_metric_ci,
    compute_binary_metrics,
    expected_calibration_error,
)


def test_compute_binary_metrics_has_required_keys() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = compute_binary_metrics(y_true, y_prob)

    assert set(metrics) == {"auroc", "auprc", "balanced_accuracy", "brier", "ece"}


def test_expected_calibration_error_is_bounded() -> None:
    ece = expected_calibration_error(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
    )

    assert 0.0 <= ece <= 1.0


def test_bootstrap_metric_ci_contains_estimate() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3])

    interval = bootstrap_metric_ci(
        y_true,
        y_prob,
        metric_fn=lambda labels, probabilities: float(np.mean((probabilities >= 0.5) == labels)),
        n_resamples=200,
    )

    assert interval["lower"] <= interval["estimate"] <= interval["upper"]
