"""Metrics and aggregation helpers for development-only binary evaluation."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)


METRIC_NAMES = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")


def sigmoid(logits: np.ndarray | Sequence[float] | float) -> np.ndarray | float:
    """Convert logits to probabilities using a numerically stable sigmoid."""
    values = np.asarray(logits, dtype=float)
    probabilities = np.empty_like(values, dtype=float)
    positive = values >= 0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    probabilities[~positive] = exp_values / (1.0 + exp_values)
    if values.ndim == 0:
        return float(probabilities)
    return probabilities


def _validate_binary_inputs(
    y_true: np.ndarray | Sequence[int], y_prob: np.ndarray | Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true).reshape(-1)
    probabilities = np.asarray(y_prob, dtype=float).reshape(-1)
    if labels.size == 0:
        raise ValueError("y_true and y_prob must not be empty")
    if labels.shape != probabilities.shape:
        raise ValueError("y_true and y_prob must have the same length")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, (0, 1))):
        raise ValueError("y_true must contain only binary labels 0 and 1")
    return labels.astype(int, copy=False), probabilities


def expected_calibration_error(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    n_bins: int = 15,
) -> float:
    """Compute equal-width expected calibration error for binary probabilities."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)
    bin_indices = np.minimum((probabilities * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for bin_index in range(n_bins):
        in_bin = bin_indices == bin_index
        if not np.any(in_bin):
            continue
        confidence = float(np.mean(probabilities[in_bin]))
        accuracy = float(np.mean(labels[in_bin]))
        ece += float(np.mean(in_bin)) * abs(accuracy - confidence)
    return float(ece)


def compute_binary_metrics(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute discrimination, thresholded accuracy, and calibration metrics."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)
    if np.unique(labels).size != 2:
        raise ValueError("AUROC and AUPRC require both binary classes")
    predictions = (probabilities >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
    }


def bootstrap_metric_ci(
    y_true: np.ndarray | Sequence[int],
    y_prob: np.ndarray | Sequence[float],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 2000,
    random_state: int = 20260710,
) -> dict[str, float]:
    """Return a percentile bootstrap 95% CI for a metric on paired observations."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be at least 1")
    labels, probabilities = _validate_binary_inputs(y_true, y_prob)
    estimate = float(metric_fn(labels, probabilities))
    if not np.isfinite(estimate):
        raise ValueError("metric_fn must return a finite value")

    rng = np.random.default_rng(random_state)
    samples = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled_indices = rng.integers(0, labels.size, size=labels.size)
        samples[index] = float(metric_fn(labels[sampled_indices], probabilities[sampled_indices]))
    if not np.isfinite(samples).all():
        raise ValueError("metric_fn returned a non-finite bootstrap value")
    lower, upper = np.quantile(samples, (0.025, 0.975))
    return {"lower": float(lower), "estimate": estimate, "upper": float(upper)}


def summarize_seed_fold_results(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-seed, per-fold metric rows by their non-run identifiers."""
    if not rows:
        return []

    run_fields = {"seed", "fold", "random_state"}
    groups: dict[tuple[tuple[str, object], ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        missing_metrics = set(METRIC_NAMES) - set(row)
        if missing_metrics:
            raise ValueError(f"Metric row is missing keys: {sorted(missing_metrics)}")
        group_key = tuple(
            sorted((key, value) for key, value in row.items() if key not in METRIC_NAMES and key not in run_fields)
        )
        groups[group_key].append(row)

    summaries: list[dict[str, object]] = []
    for group_key, group_rows in groups.items():
        summary = dict(group_key)
        summary["n_rows"] = len(group_rows)
        for metric_name in METRIC_NAMES:
            values = np.asarray([float(row[metric_name]) for row in group_rows], dtype=float)
            summary[f"{metric_name}_mean"] = float(np.mean(values))
            summary[f"{metric_name}_std"] = float(np.std(values, ddof=0))
        summaries.append(summary)
    return summaries


def save_metrics_csv(rows: Sequence[Mapping[str, object]], output_path: str | Path) -> None:
    """Write metric rows to CSV without creating result directories implicitly."""
    path = Path(output_path)
    if not rows:
        raise ValueError("rows must not be empty")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
