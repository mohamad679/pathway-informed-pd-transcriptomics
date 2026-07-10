from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.binn_training import MODEL_NAME
from models.final_training import fit_full_development_binn, select_final_epoch_count, serialize_scaler


def _cv_df(best_epochs: list[int] | None = None) -> pd.DataFrame:
    values = best_epochs or [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    rows = []
    index = 0
    for seed in (11, 23, 37):
        for fold in (1, 2, 3, 4, 5):
            rows.append(
                {
                    "model": MODEL_NAME,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": values[index],
                }
            )
            index += 1
    return pd.DataFrame(rows)


def _training_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(6)
    X = rng.normal(size=(16, 4)).astype(np.float32)
    y = np.array([0, 1] * 8)
    mask = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]], dtype=np.float32)
    return X, y, mask


def test_final_epoch_selection_validates_exact_15_row_seed_fold_coverage() -> None:
    assert select_final_epoch_count(_cv_df()) == 8
    with pytest.raises(ValueError, match="exactly 15 rows"):
        select_final_epoch_count(_cv_df().iloc[:-1])
    duplicate = _cv_df()
    duplicate.loc[0, "fold"] = 2
    with pytest.raises(ValueError, match="seed/fold pair"):
        select_final_epoch_count(duplicate)
    wrong_seed = _cv_df()
    wrong_seed.loc[0, "seed"] = 99
    with pytest.raises(ValueError, match="seeds must be exactly"):
        select_final_epoch_count(wrong_seed)


def test_integer_median_rule_is_deterministic() -> None:
    df = _cv_df([18, 26, 1, 74, 23, 26, 10, 3, 17, 11, 14, 7, 16, 17, 14])
    assert select_final_epoch_count(df.sample(frac=1.0, random_state=1)) == 16


def test_rejects_invalid_best_epoch_values() -> None:
    df = _cv_df().astype({"best_epoch": object})
    df.loc[0, "best_epoch"] = 1.5
    with pytest.raises(ValueError, match="integers"):
        select_final_epoch_count(df)
    df = _cv_df()
    df.loc[0, "best_epoch"] = 0
    with pytest.raises(ValueError, match="positive"):
        select_final_epoch_count(df)


def test_full_development_training_accepts_explicit_cpu_and_returns_metadata() -> None:
    X, y, mask = _training_inputs()
    model, scaler, metadata = fit_full_development_binn(
        X,
        y,
        mask,
        n_epochs=2,
        hidden_dim=4,
        dropout=0.0,
        batch_size=4,
        device="cpu",
    )
    assert next(model.parameters()).device.type == "cpu"
    assert metadata["seed"] == 11
    assert metadata["n_epochs"] == 2
    assert metadata["parameter_count"] == sum(parameter.numel() for parameter in model.parameters())
    assert serialize_scaler(scaler)["n_features_in_"] == X.shape[1]


def test_masked_weights_remain_zero() -> None:
    X, y, mask = _training_inputs()
    model, _scaler, metadata = fit_full_development_binn(
        X, y, mask, n_epochs=2, hidden_dim=4, dropout=0.0, batch_size=4, device="cpu"
    )
    assert metadata["max_abs_masked_weight_after_training"] == 0.0
    assert model.mask_integrity_summary()["max_abs_masked_weight"] == 0.0


def test_scaler_fitted_on_development_only() -> None:
    X, y, mask = _training_inputs()
    shifted_external_like_rows = np.full((3, X.shape[1]), 1000.0, dtype=np.float32)
    _model, scaler, metadata = fit_full_development_binn(
        X, y, mask, n_epochs=1, hidden_dim=4, dropout=0.0, batch_size=4, device="cpu"
    )
    assert np.allclose(scaler.mean_, X.mean(axis=0))
    assert np.all(scaler.transform(shifted_external_like_rows) > 500.0)
    assert metadata["scaler_fit_scope"] == "full_development_only"
