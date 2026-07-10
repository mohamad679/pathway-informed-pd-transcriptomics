"""Immutable frozen-model loading, preprocessing, and Phase 6 scoring helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from scipy import sparse

from eval.metrics import compute_binary_metrics
from models.binn import BINNClassifier
from models.frozen_bundle import verify_hash_manifest


FIXED_THRESHOLD = 0.5


def _read_nonempty_lines(path: Path) -> list[str]:
    values = path.read_text(encoding="utf-8").splitlines()
    if not values or any(not value for value in values):
        raise ValueError(f"Frozen text payload must contain only nonempty lines: {path.name}")
    if len(values) != len(set(values)):
        raise ValueError(f"Frozen text payload contains duplicate entries: {path.name}")
    return values


def _require_metadata_match(
    checkpoint: Mapping[str, Any],
    training_metadata: Mapping[str, Any],
    field: str,
) -> None:
    if field not in checkpoint or field not in training_metadata:
        raise ValueError(f"Missing frozen metadata field: {field}")
    if checkpoint[field] != training_metadata[field]:
        raise ValueError(f"Frozen checkpoint and training metadata disagree on {field}")


def _scaler_arrays(scaler_config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    parameters = scaler_config.get("parameters", scaler_config)
    if not isinstance(parameters, Mapping):
        raise ValueError("Frozen StandardScaler parameters must be a mapping")
    if "mean_" not in parameters or "scale_" not in parameters:
        raise ValueError("Frozen StandardScaler parameters require mean_ and scale_")
    mean = np.asarray(parameters["mean_"], dtype=np.float64)
    scale = np.asarray(parameters["scale_"], dtype=np.float64)
    if mean.ndim != 1 or scale.ndim != 1:
        raise ValueError("Frozen StandardScaler mean_ and scale_ must be one-dimensional")
    return mean, scale


def load_and_verify_frozen_bundle(
    frozen_dir: str | Path,
) -> tuple[BINNClassifier, dict[str, Any], dict[str, Any]]:
    """Verify and reconstruct the immutable BINN bundle for inference only."""
    bundle_dir = Path(frozen_dir)
    hash_before_path = bundle_dir / "HASH_BEFORE.txt"

    # This check must precede every frozen payload read, especially torch.load.
    verify_hash_manifest(bundle_dir, hash_before_path)

    preprocessing_config = json.loads(
        (bundle_dir / "preprocessing_config.json").read_text(encoding="utf-8")
    )
    training_metadata = json.loads(
        (bundle_dir / "training_metadata.json").read_text(encoding="utf-8")
    )
    gene_space = _read_nonempty_lines(bundle_dir / "gene_space.txt")
    pathway_names = _read_nonempty_lines(bundle_dir / "pathway_names.txt")
    pathway_mask = (
        sparse.load_npz(bundle_dir / "pathway_mask.npz")
        .toarray()
        .astype(np.float32, copy=False)
    )
    checkpoint = torch.load(
        bundle_dir / "model_v1.pt",
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Frozen model checkpoint must be a mapping")
    if checkpoint.get("model_class") != "BINNClassifier":
        raise ValueError("Frozen model_class must be BINNClassifier")
    if pathway_mask.ndim != 2:
        raise ValueError("Frozen pathway mask must be two-dimensional")
    expected_shape = (len(pathway_names), len(gene_space))
    if pathway_mask.shape != expected_shape:
        raise ValueError("Frozen pathway mask dimensions do not match pathway/gene files")

    for field in ("hidden_dim", "dropout", "n_genes", "n_pathways", "seed", "n_epochs"):
        _require_metadata_match(checkpoint, training_metadata, field)
    if int(checkpoint["n_genes"]) != len(gene_space):
        raise ValueError("Frozen n_genes does not match gene_space.txt")
    if int(checkpoint["n_pathways"]) != len(pathway_names):
        raise ValueError("Frozen n_pathways does not match pathway_names.txt")

    standard_scaler = preprocessing_config.get("standard_scaler")
    if not isinstance(standard_scaler, Mapping):
        raise ValueError("Missing frozen standard_scaler configuration")
    scaler_parameters = standard_scaler.get("parameters")
    if not isinstance(scaler_parameters, Mapping):
        raise ValueError("Missing frozen StandardScaler parameters")
    mean, scale = _scaler_arrays(scaler_parameters)
    variance = np.asarray(scaler_parameters.get("var_"), dtype=np.float64)
    n_genes = len(gene_space)
    if mean.size != n_genes or scale.size != n_genes or variance.shape != (n_genes,):
        raise ValueError("Frozen StandardScaler parameter lengths must equal n_genes")
    if int(scaler_parameters.get("n_features_in_", -1)) != n_genes:
        raise ValueError("Frozen StandardScaler n_features_in_ must equal n_genes")

    model = BINNClassifier(
        pathway_mask,
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mask_summary = model.mask_integrity_summary()
    if float(mask_summary["max_abs_masked_weight"]) != 0.0:
        raise ValueError("Frozen model has nonzero masked weights")

    frozen_metadata: dict[str, Any] = {
        "preprocessing_config": preprocessing_config,
        "training_metadata": training_metadata,
        "gene_space": gene_space,
        "pathway_names": pathway_names,
        "pathway_mask": pathway_mask,
        "model_metadata": {
            key: checkpoint[key]
            for key in (
                "model_class",
                "hidden_dim",
                "dropout",
                "n_genes",
                "n_pathways",
                "seed",
                "n_epochs",
            )
        },
        "hash_before_manifest": str(hash_before_path),
        "hash_before_verified": True,
        "max_abs_masked_weight": float(mask_summary["max_abs_masked_weight"]),
    }
    return model, dict(scaler_parameters), frozen_metadata


def apply_frozen_preprocessing(
    X: np.ndarray,
    scaler_config: Mapping[str, Any],
    expected_gene_count: int,
) -> np.ndarray:
    """Apply serialized StandardScaler parameters without fitting or imputation."""
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("X must be a finite two-dimensional array")
    if values.shape[1] != expected_gene_count:
        raise ValueError(
            f"X feature count {values.shape[1]} does not match expected {expected_gene_count}"
        )
    if not np.isfinite(values).all():
        raise ValueError("X must be finite; frozen preprocessing does not impute")

    mean, scale = _scaler_arrays(scaler_config)
    if mean.size != expected_gene_count or scale.size != expected_gene_count:
        raise ValueError("Frozen StandardScaler parameter lengths must equal expected_gene_count")
    if not np.isfinite(mean).all():
        raise ValueError("Frozen StandardScaler mean_ must be finite")
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("Frozen StandardScaler scale_ must be finite and strictly positive")

    scaled = (values - mean) / scale
    if not np.isfinite(scaled).all():
        raise ValueError("Frozen preprocessing produced nonfinite values")
    return scaled.astype(np.float32, copy=False)


def score_frozen_model(
    model: torch.nn.Module,
    X_scaled: np.ndarray,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Run ordered batched inference and return logits and PD probabilities."""
    values = np.asarray(X_scaled, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("X_scaled must be a finite two-dimensional array")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    target_device = torch.device(device)
    model = model.to(target_device)
    model.eval()
    logits_batches: list[np.ndarray] = []
    probability_batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, values.shape[0], batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(target_device)
            batch_logits = model(batch).reshape(-1)
            logits_batches.append(batch_logits.cpu().numpy())
            probability_batches.append(torch.sigmoid(batch_logits).cpu().numpy())

    if not logits_batches:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty.copy()
    return np.concatenate(logits_batches), np.concatenate(probability_batches)


def compute_external_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Compute the existing binary metrics at the immutable 0.5 threshold."""
    return compute_binary_metrics(y_true, y_prob, threshold=FIXED_THRESHOLD)


def summarize_ndd_specificity(
    y_prob: np.ndarray,
    threshold: float = FIXED_THRESHOLD,
) -> dict[str, float | int]:
    """Summarize unlabeled NDD predictions as a specificity stress test only."""
    if threshold != FIXED_THRESHOLD:
        raise ValueError("NDD threshold must be exactly 0.5")
    probabilities = np.asarray(y_prob, dtype=np.float64).reshape(-1)
    if probabilities.size == 0:
        raise ValueError("y_prob must not be empty")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0.0) | (probabilities > 1.0)
    ):
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")

    predicted_pd = probabilities >= FIXED_THRESHOLD
    return {
        "n_samples": int(probabilities.size),
        "mean_pd_probability": float(np.mean(probabilities)),
        "median_pd_probability": float(np.median(probabilities)),
        "std_pd_probability": float(np.std(probabilities)),
        "min_pd_probability": float(np.min(probabilities)),
        "max_pd_probability": float(np.max(probabilities)),
        "fraction_predicted_pd_at_0_5": float(np.mean(predicted_pd)),
        "fraction_predicted_hc_at_0_5": float(np.mean(~predicted_pd)),
    }
