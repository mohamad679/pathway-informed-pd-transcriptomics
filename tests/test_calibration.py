from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.calibration import calibration_summary, reliability_curve


def test_reliability_curve_has_requested_shape_and_columns() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    bins = reliability_curve(y_true, y_prob, n_bins=4)

    assert bins.shape == (4, 6)
    assert list(bins.columns) == [
        "bin_index",
        "bin_left",
        "bin_right",
        "n_samples",
        "mean_predicted_probability",
        "observed_fraction",
    ]


def test_calibration_summary_is_finite() -> None:
    summary = calibration_summary(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_bins=4
    )

    assert np.isfinite(summary["brier"])
    assert np.isfinite(summary["ece"])
    assert summary["n_bins"] == 4
    assert summary["n_samples"] == 4


def test_invalid_probabilities_fail() -> None:
    with pytest.raises(ValueError, match="finite probabilities"):
        reliability_curve(np.array([0, 1]), np.array([0.2, 1.2]))
