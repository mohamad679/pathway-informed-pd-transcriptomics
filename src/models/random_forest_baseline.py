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
    rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for fold_index, (X_train, X_validation, y_train, y_validation) in enumerate(
            iter_folds(X, y, folds), start=1
        ):
            classifier = build_random_forest(int(seed), n_estimators=n_estimators)
            classifier.fit(X_train, y_train)
            probabilities = classifier.predict_proba(X_validation)[:, 1]
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
