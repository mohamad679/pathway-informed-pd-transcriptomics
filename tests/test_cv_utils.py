from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.cv import iter_folds


def test_fold_iterator_has_no_train_validation_overlap() -> None:
    X = np.arange(24).reshape(6, 4)
    y = np.array([0, 1, 0, 1, 0, 1])
    folds = [
        {"train_indices": [2, 3, 4, 5], "validation_indices": [0, 1]},
        {"train_indices": [0, 1, 4, 5], "validation_indices": [2, 3]},
        {"train_indices": [0, 1, 2, 3], "validation_indices": [4, 5]},
    ]

    for X_train, X_validation, _, _ in iter_folds(X, y, folds):
        assert set(X_train[:, 0]).isdisjoint(set(X_validation[:, 0]))


def test_fold_iterator_covers_validation_samples_exactly_once() -> None:
    X = np.arange(24).reshape(6, 4)
    y = np.array([0, 1, 0, 1, 0, 1])
    folds = [
        {"train_indices": [2, 3, 4, 5], "validation_indices": [0, 1]},
        {"train_indices": [0, 1, 4, 5], "validation_indices": [2, 3]},
        {"train_indices": [0, 1, 2, 3], "validation_indices": [4, 5]},
    ]

    validation_markers = []
    for _, X_validation, _, _ in iter_folds(X, y, folds):
        validation_markers.extend(X_validation[:, 0].tolist())

    assert sorted(validation_markers) == [0, 4, 8, 12, 16, 20]
