from __future__ import annotations

import inspect
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import models.mlp_baseline as mlp_baseline


def synthetic_inputs() -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    negative = np.column_stack((np.linspace(-3.0, -1.0, 15), np.linspace(-2.0, -0.5, 15)))
    positive = np.column_stack((np.linspace(1.0, 3.0, 15), np.linspace(0.5, 2.0, 15)))
    X = np.vstack((negative, positive))
    y = np.array([0] * 15 + [1] * 15)
    folds = [
        {"train_indices": list(range(5, 15)) + list(range(20, 30)), "validation_indices": list(range(0, 5)) + list(range(15, 20))},
        {"train_indices": list(range(0, 5)) + list(range(10, 15)) + list(range(15, 20)) + list(range(25, 30)), "validation_indices": list(range(5, 10)) + list(range(20, 25))},
        {"train_indices": list(range(0, 10)) + list(range(15, 25)), "validation_indices": list(range(10, 15)) + list(range(25, 30))},
    ]
    return X, y, folds


def test_returns_one_metric_row_per_seed_and_fold() -> None:
    X, y, folds = synthetic_inputs()
    seeds = [11, 23]

    rows = mlp_baseline.evaluate_mlp_cv(
        X, y, folds, seeds=seeds, hidden_layer_sizes=(8,), max_iter=20
    )

    assert len(rows) == len(seeds) * len(folds)
    assert {row["model"] for row in rows} == {"unconstrained_mlp"}


def test_metric_rows_have_required_metrics() -> None:
    X, y, folds = synthetic_inputs()
    rows = mlp_baseline.evaluate_mlp_cv(
        X, y, folds, seeds=[11], hidden_layer_sizes=(8,), max_iter=20
    )

    required = {"model", "seed", "fold", "auroc", "auprc", "balanced_accuracy", "brier", "ece"}
    assert all(required <= set(row) for row in rows)


def test_pipeline_predicts_valid_probabilities() -> None:
    X, y, folds = synthetic_inputs()
    pipeline = mlp_baseline.build_mlp_pipeline(11, hidden_layer_sizes=(8,), max_iter=20)
    pipeline.fit(X[folds[0]["train_indices"]], y[folds[0]["train_indices"]])

    probabilities = pipeline.predict_proba(X[folds[0]["validation_indices"]])[:, 1]

    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_model_module_does_not_load_external_or_ndd_files() -> None:
    source = inspect.getsource(mlp_baseline).lower()

    assert "np.load" not in source
    assert ".npy" not in source
    assert "external" not in source
    assert "ndd" not in source
