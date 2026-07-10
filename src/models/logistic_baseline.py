"""Development-only cross-validation baseline using logistic regression."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval.cv import iter_folds
from eval.metrics import compute_binary_metrics


MODEL_NAME = "logistic_regression"
DEFAULT_SEEDS = (11, 23, 37)


def build_logistic_pipeline(seed: int) -> Pipeline:
    """Build the baseline pipeline; fitting occurs separately within each fold."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_logistic_regression_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[dict[str, float | int | str]]:
    """Evaluate Logistic Regression on predefined development folds only.

    The pipeline is newly constructed and fit for every seed/fold pair, ensuring
    ``StandardScaler`` learns its parameters from that fold's training samples only.
    """
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for fold_index, (X_train, X_validation, y_train, y_validation) in enumerate(
            iter_folds(X, y, folds), start=1
        ):
            pipeline = build_logistic_pipeline(int(seed))
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_validation)[:, 1]
            metrics = compute_binary_metrics(y_validation, probabilities)
            rows.append(
                {
                    "model": MODEL_NAME,
                    "seed": int(seed),
                    "fold": fold_index,
                    **metrics,
                }
            )
    return rows
