"""Development-only cross-validation baseline using Random Forest."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from eval.cv import iter_folds
from eval.metrics import compute_binary_metrics


MODEL_NAME = "random_forest"
DEFAULT_SEEDS = (11, 23, 37)
DEFAULT_N_ESTIMATORS = 500


def build_random_forest(seed: int, n_estimators: int = DEFAULT_N_ESTIMATORS) -> RandomForestClassifier:
    """Build the baseline classifier; fitting occurs separately within each fold."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def evaluate_random_forest_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> list[dict[str, float | int | str]]:
    """Evaluate Random Forest on predefined development folds only."""
    metric_rows, _ = evaluate_random_forest_cv_with_predictions(
        X, y, folds, seeds=seeds, n_estimators=n_estimators
    )
    return metric_rows


def evaluate_random_forest_cv_with_predictions(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Evaluate development folds and return metrics plus validation predictions."""
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for fold_index, (X_train, X_validation, y_train, y_validation) in enumerate(
            iter_folds(X, y, folds), start=1
        ):
            classifier = build_random_forest(int(seed), n_estimators=n_estimators)
            classifier.fit(X_train, y_train)
            probabilities = classifier.predict_proba(X_validation)[:, 1]
            metrics = compute_binary_metrics(y_validation, probabilities)
            metric_rows.append(
                {
                    "model": MODEL_NAME,
                    "seed": int(seed),
                    "fold": fold_index,
                    **metrics,
                }
            )
            validation_indices = np.asarray(folds[fold_index - 1]["validation_indices"], dtype=int)
            prediction_rows.extend(
                {
                    "model": MODEL_NAME,
                    "seed": int(seed),
                    "fold": fold_index,
                    "sample_index": int(sample_index),
                    "y_true": int(y_true),
                    "y_prob": float(y_prob),
                }
                for sample_index, y_true, y_prob in zip(
                    validation_indices, y_validation, probabilities, strict=True
                )
            )
    return metric_rows, prediction_rows
