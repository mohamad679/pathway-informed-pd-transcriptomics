from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from models.binn_training import (
    run_binn_cv,
    standardize_train_only,
    train_one_binn_fold,
    train_one_binn_fold_return_model,
)


def synthetic_inputs() -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    rng = np.random.default_rng(4)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.array([0, 1] * 10)
    folds = [
        {"train_indices": list(range(10, 20)), "validation_indices": list(range(10))},
        {"train_indices": list(range(10)), "validation_indices": list(range(10, 20))},
    ]
    return X, y, folds, np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.float32)


def test_standardization_is_fit_on_training_data_only() -> None:
    train = np.array([[0.0, 2.0], [2.0, 4.0]])
    validation = np.array([[100.0, 200.0]])
    train_scaled, validation_scaled = standardize_train_only(train, validation)
    assert np.allclose(train_scaled.mean(axis=0), 0.0, atol=1e-7)
    assert np.all(validation_scaled > 50.0)


def test_train_one_fold_returns_metrics_metadata_and_zero_off_mask_weights() -> None:
    X, y, folds, mask = synthetic_inputs()
    result = train_one_binn_fold(X, y, folds[0]["train_indices"], folds[0]["validation_indices"], mask, 11, 1, hidden_dim=4, dropout=0.0, max_epochs=3, patience=2, batch_size=4)
    assert {"metrics", "y_prob", "metadata"} <= set(result)
    assert {"seed", "fold", "best_epoch", "n_epochs_run", "best_validation_loss", "max_abs_masked_weight_after_training", "n_masked_weights", "n_unmasked_weights"} <= set(result["metadata"])
    assert result["metadata"]["max_abs_masked_weight_after_training"] == 0.0


def test_train_one_fold_accepts_explicit_cpu_device() -> None:
    X, y, folds, mask = synthetic_inputs()
    result = train_one_binn_fold(
        X,
        y,
        folds[0]["train_indices"],
        folds[0]["validation_indices"],
        mask,
        11,
        1,
        hidden_dim=4,
        dropout=0.0,
        max_epochs=2,
        patience=1,
        batch_size=4,
        device="cpu",
    )
    assert isinstance(result["y_prob"], np.ndarray)
    assert result["metadata"]["max_abs_masked_weight_after_training"] == 0.0


def test_returned_model_remains_on_selected_device() -> None:
    X, y, folds, mask = synthetic_inputs()
    result = train_one_binn_fold_return_model(
        X,
        y,
        folds[0]["train_indices"],
        folds[0]["validation_indices"],
        mask,
        11,
        1,
        hidden_dim=4,
        dropout=0.0,
        max_epochs=2,
        patience=1,
        batch_size=4,
        device="cpu",
    )
    model = result["model"]
    assert next(model.parameters()).device.type == "cpu"


def test_run_cv_has_fixed_seed_fold_rows_and_valid_oof_probabilities() -> None:
    X, y, folds, mask = synthetic_inputs()
    metrics_df, oof_df = run_binn_cv(X, y, folds, mask, hidden_dim=4, dropout=0.0, max_epochs=2, patience=1, batch_size=4)
    assert len(metrics_df) == 3 * len(folds)
    assert len(oof_df) == 3 * len(X)
    assert {"sample_index", "y_true", "y_prob"} <= set(oof_df.columns)
    assert oof_df["y_prob"].between(0.0, 1.0).all()


def test_run_cv_accepts_explicit_cpu_device() -> None:
    X, y, folds, mask = synthetic_inputs()
    metrics_df, oof_df = run_binn_cv(
        X,
        y,
        folds,
        mask,
        device="cpu",
        hidden_dim=4,
        dropout=0.0,
        max_epochs=2,
        patience=1,
        batch_size=4,
    )
    assert len(metrics_df) == 3 * len(folds)
    assert len(oof_df) == 3 * len(X)
