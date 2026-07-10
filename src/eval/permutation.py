"""Development-only label permutation helpers for Phase 5 validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from eval.metrics import compute_binary_metrics
from models.binn_training import run_binn_cv


PERMUTATION_NULL_COLUMNS = ("permutation_index", "null_auroc")


def empirical_p_value(observed_score: float, null_scores: Sequence[float] | np.ndarray) -> float:
    """Return the one-sided empirical p-value for scores where larger is better."""
    null_array = np.asarray(null_scores, dtype=float).reshape(-1)
    if null_array.size < 1:
        raise ValueError("null_scores must contain at least one permutation score")
    if not np.isfinite(observed_score) or not np.isfinite(null_array).all():
        raise ValueError("observed_score and null_scores must be finite")
    return float((1 + np.sum(null_array >= observed_score)) / (1 + null_array.size))


def run_label_permutation_binn_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    pathway_mask: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 20260710,
    training_kwargs: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Run development-only BINN CV after permuting labels within development data."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    labels = np.asarray(y, dtype=int).reshape(-1)
    if labels.size == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isin(np.unique(labels), (0, 1))):
        raise ValueError("y must contain only binary labels 0 and 1")
    if np.unique(labels).size != 2:
        raise ValueError("label permutation requires both binary classes")

    kwargs = dict(training_kwargs or {})
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for permutation_index in range(1, n_permutations + 1):
        permuted_y = rng.permutation(labels)
        _, oof_df = run_binn_cv(X, permuted_y, folds, pathway_mask, **kwargs)
        metrics = compute_binary_metrics(
            oof_df["y_true"].to_numpy(dtype=int),
            oof_df["y_prob"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "permutation_index": int(permutation_index),
                "null_auroc": float(metrics["auroc"]),
            }
        )
    return pd.DataFrame(rows, columns=PERMUTATION_NULL_COLUMNS)
