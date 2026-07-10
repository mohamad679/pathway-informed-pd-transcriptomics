"""Development-only cross-validation input and fold iteration helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np


def load_dev_arrays(processed_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load only the development feature matrix and labels from processed data."""
    processed_path = Path(processed_dir)
    X = np.load(processed_path / "dev_X.npy", allow_pickle=False)
    y = np.load(processed_path / "dev_y.npy", allow_pickle=False)
    if X.ndim != 2:
        raise ValueError(f"dev_X must be 2D, found shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"dev_y must be 1D, found shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("dev_X and dev_y have different sample counts")
    return X, y


def load_dev_folds(processed_dir: str | Path) -> list[dict[str, object]]:
    """Load the pre-defined development-only folds."""
    path = Path(processed_dir) / "dev_folds.json"
    with path.open(encoding="utf-8") as handle:
        folds = json.load(handle)
    if not isinstance(folds, list):
        raise ValueError("dev_folds.json must contain a JSON list")
    if not all(isinstance(fold, dict) for fold in folds):
        raise ValueError("Each development fold must be a JSON object")
    return folds


def validate_fold_indices(folds: Sequence[dict[str, object]], n_samples: int) -> None:
    """Ensure folds are disjoint, in bounds, and cover each validation sample once."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if not folds:
        raise ValueError("folds must not be empty")

    seen_validation: set[int] = set()
    for position, fold in enumerate(folds, start=1):
        if "train_indices" not in fold or "validation_indices" not in fold:
            raise ValueError(f"Fold {position} must contain train_indices and validation_indices")
        train_indices = np.asarray(fold["train_indices"])
        validation_indices = np.asarray(fold["validation_indices"])
        if train_indices.ndim != 1 or validation_indices.ndim != 1:
            raise ValueError(f"Fold {position} indices must be one-dimensional")
        if train_indices.size == 0 or validation_indices.size == 0:
            raise ValueError(f"Fold {position} train and validation indices must not be empty")
        if not np.issubdtype(train_indices.dtype, np.integer) or not np.issubdtype(validation_indices.dtype, np.integer):
            raise ValueError(f"Fold {position} indices must be integers")
        train_set = set(train_indices.tolist())
        validation_set = set(validation_indices.tolist())
        if len(train_set) != train_indices.size or len(validation_set) != validation_indices.size:
            raise ValueError(f"Fold {position} contains duplicate indices")
        if any(index < 0 or index >= n_samples for index in train_set | validation_set):
            raise ValueError(f"Fold {position} contains an out-of-range index")
        if train_set & validation_set:
            raise ValueError(f"Fold {position} has train/validation overlap")
        if seen_validation & validation_set:
            raise ValueError("Validation indices appear in more than one fold")
        seen_validation.update(validation_set)
    if seen_validation != set(range(n_samples)):
        raise ValueError("Validation indices must cover every sample exactly once")


def iter_folds(
    X: np.ndarray, y: np.ndarray, folds: Sequence[dict[str, object]]
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Yield ``X_train, X_validation, y_train, y_validation`` for each validated fold."""
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    if X_array.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if y_array.ndim != 1:
        raise ValueError("y must be a one-dimensional array")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y have different sample counts")
    validate_fold_indices(folds, X_array.shape[0])
    for fold in folds:
        train_indices = np.asarray(fold["train_indices"], dtype=int)
        validation_indices = np.asarray(fold["validation_indices"], dtype=int)
        yield (
            X_array[train_indices],
            X_array[validation_indices],
            y_array[train_indices],
            y_array[validation_indices],
        )
