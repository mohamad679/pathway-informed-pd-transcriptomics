"""Development-only cross-validation baseline using an unconstrained MLP."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval.cv import iter_folds
from eval.metrics import compute_binary_metrics


MODEL_NAME = "unconstrained_mlp"
DEFAULT_SEEDS = (11, 23, 37)
DEFAULT_HIDDEN_LAYER_SIZES = (128,)
DEFAULT_MAX_ITER = 300


def build_mlp_pipeline(
    seed: int,
    *,
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES,
    max_iter: int = DEFAULT_MAX_ITER,
) -> Pipeline:
    """Build the MLP pipeline; it is fit separately within each training fold."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=max_iter,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=seed,
                ),
            ),
        ]
    )


def evaluate_mlp_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES,
    max_iter: int = DEFAULT_MAX_ITER,
) -> list[dict[str, float | int | str]]:
    """Evaluate the unconstrained MLP on predefined development folds only.

    A new pipeline is fit for every seed/fold pair, so ``StandardScaler`` is
    fit using that fold's training data only.
    """
    metric_rows, _ = evaluate_mlp_cv_with_predictions(
        X,
        y,
        folds,
        seeds=seeds,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
    )
    return metric_rows


def evaluate_mlp_cv_with_predictions(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES,
    max_iter: int = DEFAULT_MAX_ITER,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    """Evaluate development folds and return metrics plus validation predictions."""
    metric_rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | int | str]] = []
    for seed in seeds:
        for fold_index, (X_train, X_validation, y_train, y_validation) in enumerate(
            iter_folds(X, y, folds), start=1
        ):
            pipeline = build_mlp_pipeline(
                int(seed),
                hidden_layer_sizes=hidden_layer_sizes,
                max_iter=max_iter,
            )
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_validation)[:, 1]
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
