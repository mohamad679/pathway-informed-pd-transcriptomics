"""Development-only cross-validation training for the pathway-constrained BINN."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from torch import nn

from eval.metrics import compute_binary_metrics
from models.binn import BINNClassifier


MODEL_NAME = "pathway_constrained_binn"
DEFAULT_SEEDS = (11, 23, 37)


def load_binn_inputs(
    dev_X_path: str | Path,
    dev_y_path: str | Path,
    folds_path: str | Path,
    mask_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    """Load the four development-only BINN inputs and validate their dimensions."""
    X = np.load(Path(dev_X_path), allow_pickle=False)
    y = np.load(Path(dev_y_path), allow_pickle=False)
    with Path(folds_path).open(encoding="utf-8") as handle:
        folds = json.load(handle)
    pathway_mask = sparse.load_npz(Path(mask_path)).toarray().astype(np.float32, copy=False)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0]:
        raise ValueError("Development X and y must be compatible 2D and 1D arrays.")
    if pathway_mask.ndim != 2 or pathway_mask.shape[1] != X.shape[1]:
        raise ValueError("pathway_mask must have one column per development feature.")
    if not isinstance(folds, list) or not all(isinstance(fold, dict) for fold in folds):
        raise ValueError("folds_path must contain a JSON list of fold objects.")
    return X.astype(np.float32, copy=False), y.astype(int, copy=False), folds, pathway_mask


def standardize_train_only(X_train: np.ndarray, X_val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit a scaler only on a training fold and transform both fold partitions."""
    scaler = StandardScaler()
    return (
        scaler.fit_transform(np.asarray(X_train)).astype(np.float32, copy=False),
        scaler.transform(np.asarray(X_val)).astype(np.float32, copy=False),
    )


def _train_binn_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    pathway_mask: np.ndarray,
    seed: int,
    fold_id: int,
    hidden_dim: int = 64,
    dropout: float = 0.25,
    max_epochs: int = 300,
    patience: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
) -> dict[str, object]:
    """Train one fold and retain the fitted model and train-only scaler internally."""
    set_torch_seed(seed)
    train_idx_array, val_idx_array = np.asarray(train_idx, dtype=int), np.asarray(val_idx, dtype=int)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.asarray(X)[train_idx_array]).astype(np.float32, copy=False)
    X_val = scaler.transform(np.asarray(X)[val_idx_array]).astype(np.float32, copy=False)
    y_train, y_val = np.asarray(y)[train_idx_array], np.asarray(y)[val_idx_array]
    model = BINNClassifier(pathway_mask, hidden_dim=hidden_dim, dropout=dropout)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32)
    has_both_validation_classes = np.unique(y_val).size == 2
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -float("inf")
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for indices in torch.randperm(len(train_idx_array)).split(batch_size):
            optimizer.zero_grad()
            loss = criterion(model(X_train_tensor[indices]), y_train_tensor[indices])
            loss.backward()
            optimizer.step()
            model.apply_masks_()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor)
            val_loss = float(criterion(val_logits, y_val_tensor).item())
            val_prob = torch.sigmoid(val_logits).cpu().numpy()
        score = float(compute_binary_metrics(y_val, val_prob)["auroc"]) if has_both_validation_classes else -val_loss
        improved = score > best_score or (score == best_score and val_loss < best_loss)
        if improved:
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_score, best_loss, best_epoch, stale_epochs = score, val_loss, epoch, 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.apply_masks_()
    model.eval()
    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
    integrity = model.mask_integrity_summary()
    max_abs = float(integrity["max_abs_masked_weight"])
    assert max_abs == 0.0
    metrics = _validation_metrics(y_val, y_prob)
    metadata: dict[str, float | int] = {
        "seed": int(seed), "fold": int(fold_id), "best_epoch": best_epoch,
        "n_epochs_run": epoch, "best_validation_loss": best_loss,
        "max_abs_masked_weight_after_training": max_abs,
        "n_masked_weights": int(integrity["n_masked_weights"]),
        "n_unmasked_weights": int(integrity["n_unmasked_weights"]),
    }
    return {
        "model": model, "scaler": scaler, "metrics": metrics, "y_prob": y_prob,
        "metadata": metadata,
    }


def set_torch_seed(seed: int) -> None:
    """Set CPU-relevant random seeds for reproducible fold training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validation_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Use the shared metrics helper when a validation fold has both classes."""
    if np.unique(y_true).size == 2:
        return compute_binary_metrics(y_true, y_prob)
    return {name: float("nan") for name in ("auroc", "auprc", "balanced_accuracy", "brier", "ece")}


def train_one_binn_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    pathway_mask: np.ndarray,
    seed: int,
    fold_id: int,
    hidden_dim: int = 64,
    dropout: float = 0.25,
    max_epochs: int = 300,
    patience: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
) -> dict[str, object]:
    """Train one CPU BINN fold with train-only scaling and early stopping."""
    result = _train_binn_fold(
        X, y, train_idx, val_idx, pathway_mask, seed, fold_id, hidden_dim, dropout,
        max_epochs, patience, learning_rate, weight_decay, batch_size,
    )
    return {
        "metrics": result["metrics"], "y_prob": result["y_prob"], "metadata": result["metadata"],
        **result["metrics"], **result["metadata"],
    }


def train_one_binn_fold_return_model(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    pathway_mask: np.ndarray,
    seed: int,
    fold_id: int,
    hidden_dim: int = 64,
    dropout: float = 0.25,
    max_epochs: int = 300,
    patience: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
) -> dict[str, object]:
    """Retrain one development fold and return its model, scaler, and audit metadata."""
    result = _train_binn_fold(
        X, y, train_idx, val_idx, pathway_mask, seed, fold_id, hidden_dim, dropout,
        max_epochs, patience, learning_rate, weight_decay, batch_size,
    )
    return {key: result[key] for key in ("model", "scaler", "metadata")}


def run_binn_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    pathway_mask: np.ndarray,
    **training_kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run BINN CV on the existing development folds with Phase 2's fixed seeds."""
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for seed in DEFAULT_SEEDS:
        for fold_id, fold in enumerate(folds, start=1):
            train_idx = np.asarray(fold["train_indices"], dtype=int)
            val_idx = np.asarray(fold["validation_indices"], dtype=int)
            result = train_one_binn_fold(
                X, y, train_idx, val_idx, pathway_mask, seed, fold_id, **training_kwargs
            )
            metric_rows.append({"model": MODEL_NAME, **result["metrics"], **result["metadata"]})
            prediction_rows.extend(
                {"model": MODEL_NAME, "seed": seed, "fold": fold_id,
                 "sample_index": int(sample_index), "y_true": int(y_true), "y_prob": float(y_prob)}
                for sample_index, y_true, y_prob in zip(val_idx, y[val_idx], result["y_prob"], strict=True)
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(prediction_rows)
