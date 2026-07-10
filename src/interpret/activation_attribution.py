"""Development-only out-of-fold pathway activation attribution."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from interpret.pathway_attribution import (
    aggregate_pathway_signal,
    classify_rna_processing_pathway,
    rank_pathways,
)
from models.binn import BINNClassifier
from models.binn_training import train_one_binn_fold_return_model


def compute_validation_pathway_activations(
    model: BINNClassifier, X_val_scaled: np.ndarray,
) -> np.ndarray:
    """Return post-ReLU, pre-dropout pathway activations for validation samples."""
    values = np.asarray(X_val_scaled, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("X_val_scaled must have shape (n_validation_samples, n_features).")
    model.eval()
    with torch.no_grad():
        _, activations = model.forward_with_pathway_activations(
            torch.as_tensor(values, dtype=torch.float32)
        )
    return activations.cpu().numpy()


def run_activation_attribution_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    pathway_mask: np.ndarray,
    pathway_names: Sequence[str],
    seeds: Sequence[int] = (11, 23, 37),
    rna_processing_keywords: Sequence[str] = (),
    **training_kwargs: object,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | float | bool]]:
    """Retrain development folds solely to calculate out-of-fold pathway activations."""
    X_values, y_values, mask = np.asarray(X), np.asarray(y), np.asarray(pathway_mask)
    if X_values.ndim != 2 or y_values.ndim != 1 or X_values.shape[0] != y_values.shape[0]:
        raise ValueError("X and y must be compatible 2D and 1D development arrays.")
    if mask.ndim != 2 or mask.shape[1] != X_values.shape[1]:
        raise ValueError("pathway_mask must have one column per X feature.")
    if len(pathway_names) != mask.shape[0]:
        raise ValueError("pathway_names must have one entry per pathway-mask row.")
    if not folds or not seeds:
        raise ValueError("folds and seeds must be non-empty.")

    score_frames: list[pd.DataFrame] = []
    masked_weight_values: list[float] = []
    for seed in seeds:
        for fold_id, fold in enumerate(folds, start=1):
            train_idx = np.asarray(fold["train_indices"], dtype=int)
            val_idx = np.asarray(fold["validation_indices"], dtype=int)
            trained = train_one_binn_fold_return_model(
                X_values, y_values, train_idx, val_idx, mask, int(seed), fold_id, **training_kwargs
            )
            X_val_scaled = trained["scaler"].transform(X_values[val_idx]).astype(np.float32, copy=False)
            activations = compute_validation_pathway_activations(trained["model"], X_val_scaled)
            ranked = rank_pathways(
                aggregate_pathway_signal(activations, pathway_names, fold_id, int(seed), "activation")
            )
            ranked["is_rna_processing"] = ranked["pathway_name"].map(
                lambda name: classify_rna_processing_pathway(name, rna_processing_keywords)
            )
            score_frames.append(ranked)
            masked_weight_values.append(float(trained["metadata"]["max_abs_masked_weight_after_training"]))

    pathway_scores_df = pd.concat(score_frames, ignore_index=True).sort_values(
        ["seed", "fold", "rank", "pathway_name"], kind="mergesort"
    ).reset_index(drop=True)
    fold_stability_df = (
        pathway_scores_df.groupby(["pathway_name", "method", "is_rna_processing"], as_index=False, sort=True)
        .agg(
            mean_rank=("rank", "mean"),
            rank_variance=("rank", "var"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["method", "mean_rank", "pathway_name"], kind="mergesort")
        .reset_index(drop=True)
    )
    fold_stability_df = fold_stability_df[
        [
            "pathway_name", "method", "mean_rank", "rank_variance", "min_rank", "max_rank",
            "n_folds", "is_rna_processing",
        ]
    ]
    audit_summary: dict[str, int | float | bool] = {
        "n_pathways": int(mask.shape[0]),
        "n_seeds": len(seeds),
        "n_folds": len(folds),
        "max_masked_weight_after_training": max(masked_weight_values),
        "n_rna_processing_pathways": int(
            pathway_scores_df.drop_duplicates("pathway_name")["is_rna_processing"].sum()
        ),
        "confirmation_no_external_ndd": True,
    }
    return pathway_scores_df, fold_stability_df, audit_summary
