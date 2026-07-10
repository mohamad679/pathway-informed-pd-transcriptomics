"""Final development-only BINN retraining utilities for the frozen Phase 6 bundle."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from models.binn import BINNClassifier
from models.binn_training import DEFAULT_SEEDS, MODEL_NAME, set_torch_seed


EXPECTED_FOLDS = (1, 2, 3, 4, 5)


def _validated_positive_integer_column(values: pd.Series, column_name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_array = numeric.to_numpy(dtype=float)
    if not np.isfinite(numeric_array).all():
        raise ValueError(f"{column_name} values must be finite.")
    if not np.all(numeric_array > 0):
        raise ValueError(f"{column_name} values must be positive.")
    if not np.all(np.equal(numeric_array, np.floor(numeric_array))):
        raise ValueError(f"{column_name} values must be integers.")
    return numeric_array.astype(int)


def select_final_epoch_count(binn_cv_df: pd.DataFrame) -> int:
    """Select the full-development epoch count from Phase 3 CV metadata only."""
    required_columns = {"model", "seed", "fold", "best_epoch"}
    missing = required_columns - set(binn_cv_df.columns)
    if missing:
        raise ValueError(f"binn_cv_df is missing required columns: {sorted(missing)}")
    if len(binn_cv_df) != 15:
        raise ValueError("binn_cv_df must contain exactly 15 rows.")
    if set(binn_cv_df["model"]) != {MODEL_NAME}:
        raise ValueError(f"model must be exactly {MODEL_NAME}.")
    seeds = _validated_positive_integer_column(binn_cv_df["seed"], "seed")
    folds = _validated_positive_integer_column(binn_cv_df["fold"], "fold")
    if set(seeds) != set(DEFAULT_SEEDS):
        raise ValueError(f"seeds must be exactly {list(DEFAULT_SEEDS)}.")
    if set(folds) != set(EXPECTED_FOLDS):
        raise ValueError(f"folds must be exactly {list(EXPECTED_FOLDS)}.")

    coverage = set(zip(seeds, folds, strict=False))
    expected_coverage = {(seed, fold) for seed in DEFAULT_SEEDS for fold in EXPECTED_FOLDS}
    if coverage != expected_coverage:
        raise ValueError("binn_cv_df must contain exactly one row for each seed/fold pair.")

    best_epoch = _validated_positive_integer_column(binn_cv_df["best_epoch"], "best_epoch")
    median_epoch = Decimal(str(float(np.median(best_epoch))))
    return int(median_epoch.quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def fit_full_development_binn(
    X: np.ndarray,
    y: np.ndarray,
    pathway_mask: np.ndarray,
    *,
    seed: int = 11,
    hidden_dim: int = 64,
    dropout: float = 0.25,
    n_epochs: int,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[BINNClassifier, StandardScaler, dict[str, float | int | str]]:
    """Fit the final BINN on all development samples for a fixed epoch count."""
    if not isinstance(n_epochs, int) or n_epochs <= 0:
        raise ValueError("n_epochs must be a positive integer.")
    X_array = np.asarray(X)
    y_array = np.asarray(y)
    mask_array = np.asarray(pathway_mask)
    if X_array.ndim != 2 or y_array.ndim != 1 or X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must be compatible 2D and 1D development arrays.")
    if mask_array.ndim != 2 or mask_array.shape[1] != X_array.shape[1]:
        raise ValueError("pathway_mask must have one column per development feature.")
    if not np.isin(y_array, [0, 1]).all():
        raise ValueError("y must use binary class mapping HC=0 and PD=1.")

    set_torch_seed(seed)
    resolved_device = torch.device(device)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_array).astype(np.float32, copy=False)
    y_float = y_array.astype(np.float32, copy=False)

    model = BINNClassifier(mask_array.astype(np.float32, copy=False), hidden_dim=hidden_dim, dropout=dropout)
    model = model.to(resolved_device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    X_tensor = torch.as_tensor(X_scaled, dtype=torch.float32, device=resolved_device)
    y_tensor = torch.as_tensor(y_float, dtype=torch.float32, device=resolved_device)

    for _epoch in range(1, n_epochs + 1):
        model.train()
        for indices in torch.randperm(X_tensor.shape[0], device=resolved_device).split(batch_size):
            optimizer.zero_grad()
            loss = criterion(model(X_tensor[indices]), y_tensor[indices])
            loss.backward()
            optimizer.step()
            model.apply_masks_()

    model.apply_masks_()
    model.eval()
    integrity = model.mask_integrity_summary()
    max_abs = float(integrity["max_abs_masked_weight"])
    if max_abs != 0.0:
        raise RuntimeError("Masked BINN weights must remain exactly 0.0 after final training.")

    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    training_metadata: dict[str, float | int | str] = {
        "model": MODEL_NAME,
        "training_scope": "full_development_only",
        "seed": int(seed),
        "n_epochs": int(n_epochs),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "batch_size": int(batch_size),
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "scaler_fit_scope": "full_development_only",
        "validation_set": "none",
        "metric_based_tuning": "none",
        "n_development_samples": int(X_array.shape[0]),
        "n_genes": int(X_array.shape[1]),
        "n_pathways": int(mask_array.shape[0]),
        "parameter_count": parameter_count,
        "max_abs_masked_weight_after_training": max_abs,
        "n_masked_weights": int(integrity["n_masked_weights"]),
        "n_unmasked_weights": int(integrity["n_unmasked_weights"]),
    }
    return model, scaler, training_metadata


def serialize_scaler(scaler: StandardScaler) -> dict[str, object]:
    """Return JSON-safe StandardScaler parameters in feature order."""
    return {
        "mean_": [float(value) for value in scaler.mean_],
        "scale_": [float(value) for value in scaler.scale_],
        "var_": [float(value) for value in scaler.var_],
        "n_features_in_": int(scaler.n_features_in_),
    }
