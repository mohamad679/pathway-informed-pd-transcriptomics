"""Development-only label permutation helpers for Phase 5 validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from eval.metrics import compute_binary_metrics
from models.binn_training import run_binn_cv


PERMUTATION_NULL_COLUMNS = ("permutation_index", "null_auroc")


def empirical_p_value(observed_score: float, null_scores: Sequence[float] | np.ndarray) -> float:
    """Return the one-sided empirical p-value for scores where larger is better."""
    null_array = np.asarray(null_scores, dtype=float).reshape(-1)
    if null_array.size < 1:
        raise ValueError("null_scores must contain at least one permutation score")
    if not np.isfinite(observed_score) or not np.isfinite(null_array).all():
        raise ValueError("observed_score and null_scores must be finite")
    return float((1 + np.sum(null_array >= observed_score)) / (1 + null_array.size))


def _validate_binary_labels(y: np.ndarray) -> np.ndarray:
    labels = np.asarray(y, dtype=int).reshape(-1)
    if labels.size == 0:
        raise ValueError("y must not be empty")
    if not np.all(np.isin(np.unique(labels), (0, 1))):
        raise ValueError("y must contain only binary labels 0 and 1")
    if np.unique(labels).size != 2:
        raise ValueError("label permutation requires both binary classes")
    return labels


def generate_permuted_labels(y: np.ndarray, permutation_index: int, seed: int) -> np.ndarray:
    """Return deterministic permuted binary labels for a 1-based permutation index."""
    if permutation_index < 1:
        raise ValueError("permutation_index must be at least 1")
    labels = _validate_binary_labels(y)
    seed_sequence = np.random.SeedSequence([int(seed), int(permutation_index)])
    rng = np.random.default_rng(seed_sequence)
    return rng.permutation(labels)


def _sorted_unique_permutation_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=PERMUTATION_NULL_COLUMNS)
    sorted_df = df.loc[:, PERMUTATION_NULL_COLUMNS].copy()
    sorted_df["permutation_index"] = sorted_df["permutation_index"].astype(int)
    duplicate_mask = sorted_df["permutation_index"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_indices = sorted(sorted_df.loc[duplicate_mask, "permutation_index"].unique())
        raise ValueError(f"duplicate permutation_index rows found: {duplicate_indices}")
    return sorted_df.sort_values("permutation_index").reset_index(drop=True)


def _combine_permutation_rows(
    existing_df: pd.DataFrame,
    rows: Sequence[dict[str, float | int]],
) -> pd.DataFrame:
    new_df = pd.DataFrame(rows, columns=PERMUTATION_NULL_COLUMNS)
    if existing_df.empty and new_df.empty:
        return pd.DataFrame(columns=PERMUTATION_NULL_COLUMNS)
    if existing_df.empty:
        combined_df = new_df
    elif new_df.empty:
        combined_df = existing_df
    else:
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    return _sorted_unique_permutation_df(combined_df)


def _write_permutation_checkpoint(df: pd.DataFrame, checkpoint_path: str | Path) -> None:
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    _sorted_unique_permutation_df(df).to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def run_label_permutation_binn_cv(
    X: np.ndarray,
    y: np.ndarray,
    folds: Sequence[dict[str, object]],
    pathway_mask: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 20260710,
    training_kwargs: Mapping[str, object] | None = None,
    start_permutation_index: int = 1,
    append_existing: pd.DataFrame | None = None,
    progress: bool = False,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 1,
) -> pd.DataFrame:
    """Run development-only BINN CV after permuting labels within development data."""
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    if start_permutation_index < 1:
        raise ValueError("start_permutation_index must be at least 1")
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    _validate_binary_labels(y)

    kwargs = dict(training_kwargs or {})
    if append_existing is None:
        existing_df = pd.DataFrame(columns=PERMUTATION_NULL_COLUMNS)
    else:
        missing_columns = set(PERMUTATION_NULL_COLUMNS) - set(append_existing.columns)
        if missing_columns:
            raise ValueError(
                f"append_existing must contain columns {sorted(PERMUTATION_NULL_COLUMNS)}"
            )
        existing_df = append_existing.loc[:, PERMUTATION_NULL_COLUMNS].copy()
        existing_df["permutation_index"] = existing_df["permutation_index"].astype(int)
        existing_df = _sorted_unique_permutation_df(existing_df)

    existing_indices = set(existing_df["permutation_index"].tolist())
    existing_null_scores = dict(
        zip(
            existing_df["permutation_index"].astype(int),
            existing_df["null_auroc"].astype(float),
            strict=True,
        )
    )
    rows: list[dict[str, float | int]] = []
    end_permutation_index = start_permutation_index + n_permutations - 1
    requested_indices = range(start_permutation_index, end_permutation_index + 1)
    iterator = tqdm(requested_indices, desc="Phase 5 permutations") if progress else requested_indices
    newly_completed_since_checkpoint = 0
    for permutation_index in iterator:
        if permutation_index in existing_indices:
            if progress:
                iterator.set_postfix(
                    permutation_index=int(permutation_index),
                    null_auroc=existing_null_scores.get(int(permutation_index)),
                )
            continue
        permuted_y = generate_permuted_labels(y, permutation_index, seed)
        _, oof_df = run_binn_cv(X, permuted_y, folds, pathway_mask, **kwargs)
        metrics = compute_binary_metrics(
            oof_df["y_true"].to_numpy(dtype=int),
            oof_df["y_prob"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "permutation_index": int(permutation_index),
                "null_auroc": float(metrics["auroc"]),
            }
        )
        newly_completed_since_checkpoint += 1
        if progress:
            iterator.set_postfix(
                permutation_index=int(permutation_index),
                null_auroc=f"{float(metrics['auroc']):.6f}",
            )
        if checkpoint_path is not None and newly_completed_since_checkpoint >= checkpoint_every:
            _write_permutation_checkpoint(
                _combine_permutation_rows(existing_df, rows),
                checkpoint_path,
            )
            newly_completed_since_checkpoint = 0

    combined_df = _combine_permutation_rows(existing_df, rows)
    if checkpoint_path is not None:
        _write_permutation_checkpoint(combined_df, checkpoint_path)
    return combined_df
