from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.baseline_summary import (
    METRIC_NAMES,
    bootstrap_model_metrics_from_predictions,
    compute_model_summary_from_predictions,
)
from models.logistic_baseline import evaluate_logistic_regression_cv_with_predictions


def synthetic_prediction_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    probabilities = {
        "strong_model": [0.05, 0.15, 0.80, 0.95],
        "weak_model": [0.40, 0.70, 0.45, 0.55],
    }
    for model, values in probabilities.items():
        for seed in (11, 23):
            for sample_index, (y_true, y_prob) in enumerate(zip([0, 0, 1, 1], values, strict=True)):
                rows.append({"model": model, "seed": seed, "fold": sample_index % 2 + 1, "sample_index": sample_index, "y_true": y_true, "y_prob": y_prob})
    return rows


def test_bootstrap_ci_has_ordered_metric_keys() -> None:
    result = bootstrap_model_metrics_from_predictions(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_resamples=25
    )

    assert tuple(result) == METRIC_NAMES
    for metric in ("auroc", "auprc", "brier", "ece"):
        assert tuple(result[metric]) == ("ci_lower", "ci_upper")


def test_summary_chooses_best_model_by_mean_auroc() -> None:
    summary = compute_model_summary_from_predictions(synthetic_prediction_rows(), n_resamples=25)

    assert summary.loc[summary["auroc_mean"].idxmax(), "model"] == "strong_model"


def test_prediction_rows_contain_required_columns() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    folds = [
        {"train_indices": [1, 3], "validation_indices": [0, 2]},
        {"train_indices": [0, 2], "validation_indices": [1, 3]},
    ]

    _, prediction_rows = evaluate_logistic_regression_cv_with_predictions(X, y, folds, seeds=[11])

    assert len(prediction_rows) == len(y)
    required = {"model", "seed", "fold", "sample_index", "y_true", "y_prob"}
    assert all(required <= set(row) for row in prediction_rows)
