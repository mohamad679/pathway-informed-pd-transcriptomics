"""Development-only summaries from out-of-fold baseline predictions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from eval.metrics import compute_binary_metrics


METRIC_NAMES = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")
PREDICTION_COLUMNS = ("model", "seed", "fold", "sample_index", "y_true", "y_prob")


def _as_prediction_frame(prediction_rows: Sequence[dict[str, object]] | pd.DataFrame) -> pd.DataFrame:
    frame = prediction_rows.copy() if isinstance(prediction_rows, pd.DataFrame) else pd.DataFrame(prediction_rows)
    missing = set(PREDICTION_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction rows are missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Prediction rows must not be empty")
    return frame.loc[:, PREDICTION_COLUMNS].copy()


def load_baseline_prediction_csv(path: str | Path) -> pd.DataFrame:
    """Load and validate baseline out-of-fold prediction rows."""
    return _as_prediction_frame(pd.read_csv(path))


def bootstrap_model_metrics_from_predictions(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    n_resamples: int = 2000,
    random_state: int = 20260710,
) -> dict[str, dict[str, float]]:
    """Return deterministic 95% paired, class-stratified bootstrap metric CIs.

    Labels and probabilities are always sampled together. Sampling within each class
    preserves both classes in every replicate, which is required by AUROC/AUPRC.
    """
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    labels = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    if labels.ndim != 1 or probabilities.ndim != 1 or labels.size != probabilities.size:
        raise ValueError("y_true and y_prob must be one-dimensional arrays of equal length")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Bootstrap requires both binary classes")

    rng = np.random.default_rng(random_state)
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    samples = {metric: np.empty(n_resamples, dtype=float) for metric in METRIC_NAMES}
    for resample_index in range(n_resamples):
        indices = np.concatenate([rng.choice(indices, size=indices.size, replace=True) for indices in class_indices])
        metrics = compute_binary_metrics(labels[indices], probabilities[indices])
        for metric in METRIC_NAMES:
            samples[metric][resample_index] = metrics[metric]
    return {
        metric: {
            "ci_lower": float(np.quantile(samples[metric], 0.025)),
            "ci_upper": float(np.quantile(samples[metric], 0.975)),
        }
        for metric in METRIC_NAMES
    }


def compute_model_summary_from_predictions(
    prediction_rows: Sequence[dict[str, object]] | pd.DataFrame,
    *,
    n_resamples: int = 2000,
    random_state: int = 20260710,
) -> pd.DataFrame:
    """Compute pooled-OOF seed means and CIs for each model.

    A metric is calculated once per model/seed after pooling that seed's five
    validation folds. CIs use all pooled seed-level OOF rows for a model.
    """
    frame = _as_prediction_frame(prediction_rows)
    rows: list[dict[str, float | int | str]] = []
    for model, model_frame in frame.groupby("model", sort=True):
        seed_metrics = [
            compute_binary_metrics(seed_frame["y_true"].to_numpy(), seed_frame["y_prob"].to_numpy())
            for _, seed_frame in model_frame.groupby("seed", sort=True)
        ]
        ci = bootstrap_model_metrics_from_predictions(
            model_frame["y_true"].to_numpy(),
            model_frame["y_prob"].to_numpy(),
            n_resamples=n_resamples,
            random_state=random_state,
        )
        row: dict[str, float | int | str] = {"model": str(model), "n_seeds": len(seed_metrics), "n_oof_rows": len(model_frame)}
        for metric in METRIC_NAMES:
            row[f"{metric}_mean"] = float(np.mean([values[metric] for values in seed_metrics]))
            row[f"{metric}_ci_lower"] = ci[metric]["ci_lower"]
            row[f"{metric}_ci_upper"] = ci[metric]["ci_upper"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_baseline_summary_csv(summary: pd.DataFrame, path: str | Path) -> None:
    """Write the model-level development-only summary CSV."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def write_baseline_summary_report(summary: pd.DataFrame, path: str | Path) -> None:
    """Write a scope-limited Markdown report for the baseline comparison."""
    lines = [
        "# Phase 2 development-only baseline summary",
        "",
        "This is a development-only baseline comparison, not an external-validation result or final performance claim.",
        "",
        "## Method",
        "",
        "- Inputs: `data/processed/dev_X.npy`, `data/processed/dev_y.npy`, and `data/processed/dev_folds.json` only.",
        "- For each model and seed, out-of-fold probabilities were pooled across all five validation folds before computing its metric.",
        "- The reported mean is the mean of these seed-level pooled-OOF metrics.",
        "- 95% bootstrap CIs use 2,000 deterministic paired, class-stratified resamples of all seed-level OOF prediction rows within each model (`random_state=20260710`). Repeated rows across seeds are retained; each resample keeps its paired `y_true` and `y_prob` values together.",
        "- No external cohort or held-out NDD data was loaded or used.",
        "- No BINN, pathway masks, or MSigDB logic was implemented or used.",
        "",
        "## Pooled out-of-fold metrics",
        "",
        "| Model | AUROC mean [95% CI] | AUPRC mean [95% CI] | Balanced accuracy mean [95% CI] | Brier mean [95% CI] | ECE mean [95% CI] |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.sort_values("model").iterrows():
        cells = [str(row["model"])]
        for metric in METRIC_NAMES:
            cells.append(f"{row[f'{metric}_mean']:.6f} [{row[f'{metric}_ci_lower']:.6f}, {row[f'{metric}_ci_upper']:.6f}]")
        lines.append("| " + " | ".join(cells) + " |")
    best_model = summary.loc[summary["auroc_mean"].idxmax(), "model"]
    lines.extend(["", "## Development-only comparison", "", f"- Highest mean pooled-OOF AUROC: `{best_model}`.", "- This is a baseline-comparison observation only; it is not a final performance claim.", ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")
