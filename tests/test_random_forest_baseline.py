from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import models.random_forest_baseline as random_forest_baseline


def synthetic_inputs() -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    X = np.array(
        [
            [-2.0, -1.0], [-1.5, -0.5], [-1.0, -1.5],
            [1.0, 1.5], [1.5, 0.5], [2.0, 1.0],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    folds = [
        {"train_indices": [1, 2, 4, 5], "validation_indices": [0, 3]},
        {"train_indices": [0, 2, 3, 5], "validation_indices": [1, 4]},
        {"train_indices": [0, 1, 3, 4], "validation_indices": [2, 5]},
    ]
    return X, y, folds


def test_returns_one_metric_row_per_seed_and_fold() -> None:
    X, y, folds = synthetic_inputs()
    seeds = [11, 23]

    rows = random_forest_baseline.evaluate_random_forest_cv(
        X, y, folds, seeds=seeds, n_estimators=5
    )

    assert len(rows) == len(seeds) * len(folds)
    assert {row["model"] for row in rows} == {"random_forest"}


def test_metric_rows_have_required_metrics() -> None:
    X, y, folds = synthetic_inputs()
    rows = random_forest_baseline.evaluate_random_forest_cv(X, y, folds, seeds=[11], n_estimators=5)

    required = {"model", "seed", "fold", "auroc", "auprc", "balanced_accuracy", "brier", "ece"}
    assert all(required <= set(row) for row in rows)


def test_classifier_predicts_valid_probabilities() -> None:
    X, y, folds = synthetic_inputs()
    classifier = random_forest_baseline.build_random_forest(11, n_estimators=5)
    classifier.fit(X[folds[0]["train_indices"]], y[folds[0]["train_indices"]])

    probabilities = classifier.predict_proba(X[folds[0]["validation_indices"]])[:, 1]

    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_model_module_does_not_load_external_or_ndd_files() -> None:
    source = inspect.getsource(random_forest_baseline).lower()

    assert "np.load" not in source
    assert ".npy" not in source
    assert "external" not in source
    assert "ndd" not in source
