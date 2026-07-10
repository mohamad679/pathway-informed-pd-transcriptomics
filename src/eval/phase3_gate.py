from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


MODEL_NAME = "pathway_constrained_binn"
EXPECTED_SEEDS = frozenset({11, 23, 37})
EXPECTED_FOLDS = frozenset({1, 2, 3, 4, 5})
EXPECTED_METRIC_ROWS = 15
EXPECTED_OOF_ROWS = 1_314
EXPECTED_SAMPLES_PER_SEED = 438
METRIC_COLUMNS = (
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
    "ece",
)
METADATA_COLUMNS = (
    "best_epoch",
    "n_epochs_run",
    "best_validation_loss",
    "max_abs_masked_weight_after_training",
    "n_masked_weights",
    "n_unmasked_weights",
)
REQUIRED_CV_COLUMNS = frozenset({"model", "seed", "fold"} | set(METRIC_COLUMNS) | set(METADATA_COLUMNS))
REQUIRED_OOF_COLUMNS = frozenset({"model", "seed", "fold", "sample_index", "y_true", "y_prob"})


def load_phase3_outputs(
    processed_dir: str | Path, results_dir: str | Path
) -> dict[str, object]:
    """Load the existing Phase 3 development artifacts without running training."""
    processed_path = Path(processed_dir)
    results_path = Path(results_dir)
    required_paths = {
        "mask_path": processed_path / "pathway_mask.npz",
        "pathway_names_path": processed_path / "pathway_names.txt",
        "gene_space_path": processed_path / "gene_space.txt",
        "metrics_path": results_path / "binn_cv.csv",
        "oof_path": results_path / "binn_oof_predictions.csv",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 3 output: {path}")

    mask = sparse.load_npz(required_paths["mask_path"])

    pathway_names = _load_nonempty_lines(required_paths["pathway_names_path"])
    gene_space = _load_nonempty_lines(required_paths["gene_space_path"])
    outputs: dict[str, object] = {
        "mask": mask,
        "pathway_names": pathway_names,
        "gene_space": gene_space,
        "metrics": pd.read_csv(required_paths["metrics_path"]),
        "oof": pd.read_csv(required_paths["oof_path"]),
    }
    baseline_summary_path = results_path / "baseline_summary.csv"
    if baseline_summary_path.is_file():
        outputs["baseline_summary"] = pd.read_csv(baseline_summary_path)
    return outputs


def _load_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_pathway_mask(
    mask: np.ndarray | sparse.spmatrix, pathway_names: list[str], gene_space: list[str]
) -> None:
    """Validate pathway-to-gene mask dimensions and nontrivial sparsity."""
    if mask.ndim != 2:
        raise ValueError(f"pathway mask must be two-dimensional, found shape {mask.shape}")
    expected_shape = (len(pathway_names), len(gene_space))
    if mask.shape != expected_shape:
        raise ValueError(
            f"pathway mask shape {mask.shape} must match pathway/gene dimensions {expected_shape}"
        )
    nnz = _mask_nnz(mask)
    if nnz <= 0:
        raise ValueError("pathway mask must have nnz > 0")
    size = int(np.prod(mask.shape))
    density = nnz / size if size else 0.0
    if not 0.0 < density < 1.0:
        raise ValueError("pathway mask density must be in (0, 1)")


def validate_binn_cv(metrics_df: pd.DataFrame) -> None:
    """Validate exact BINN CV coverage, metrics, and development sanity threshold."""
    missing_columns = REQUIRED_CV_COLUMNS.difference(metrics_df.columns)
    if missing_columns:
        raise ValueError(f"binn_cv.csv is missing required columns: {sorted(missing_columns)}")
    if len(metrics_df) != EXPECTED_METRIC_ROWS:
        raise ValueError(f"binn_cv.csv must contain exactly {EXPECTED_METRIC_ROWS} metric rows")
    if set(metrics_df["model"].astype(str)) != {MODEL_NAME}:
        raise ValueError(f"binn_cv.csv model must be exactly {MODEL_NAME}")

    seeds = _integer_column(metrics_df, "seed")
    folds = _integer_column(metrics_df, "fold")
    if set(seeds) != EXPECTED_SEEDS:
        raise ValueError(f"binn_cv.csv seeds must be exactly {sorted(EXPECTED_SEEDS)}")
    if set(folds) != EXPECTED_FOLDS:
        raise ValueError(f"binn_cv.csv folds must be exactly {sorted(EXPECTED_FOLDS)}")
    combinations = set(zip(seeds, folds, strict=True))
    expected_combinations = {(seed, fold) for seed in EXPECTED_SEEDS for fold in EXPECTED_FOLDS}
    if combinations != expected_combinations or len(combinations) != EXPECTED_METRIC_ROWS:
        raise ValueError("binn_cv.csv must contain each required seed/fold combination exactly once")

    for metric in METRIC_COLUMNS:
        values = pd.to_numeric(metrics_df[metric], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"binn_cv.csv {metric} values must be finite")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"binn_cv.csv {metric} values must be in [0, 1]")
    mean_auroc = float(pd.to_numeric(metrics_df["auroc"], errors="coerce").mean())
    if mean_auroc <= 0.6:
        raise ValueError("BINN mean AUROC must be > 0.6 for the development sanity gate")


def validate_binn_oof(oof_df: pd.DataFrame) -> None:
    """Validate development OOF row count, probability bounds, and seed coverage."""
    missing_columns = REQUIRED_OOF_COLUMNS.difference(oof_df.columns)
    if missing_columns:
        raise ValueError(f"binn_oof_predictions.csv is missing required columns: {sorted(missing_columns)}")
    if len(oof_df) != EXPECTED_OOF_ROWS:
        raise ValueError(f"binn_oof_predictions.csv must contain exactly {EXPECTED_OOF_ROWS} rows")
    if set(oof_df["model"].astype(str)) != {MODEL_NAME}:
        raise ValueError(f"binn_oof_predictions.csv model must be exactly {MODEL_NAME}")
    seeds = _integer_column(oof_df, "seed")
    if set(seeds) != EXPECTED_SEEDS:
        raise ValueError(f"OOF seeds must be exactly {sorted(EXPECTED_SEEDS)}")
    y_prob = pd.to_numeric(oof_df["y_prob"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_prob).all() or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("OOF y_prob values must be finite and in [0, 1]")
    for seed in sorted(EXPECTED_SEEDS):
        seed_df = oof_df.loc[np.asarray(seeds) == seed]
        unique_samples = seed_df["sample_index"].nunique()
        if len(seed_df) != EXPECTED_SAMPLES_PER_SEED or unique_samples != EXPECTED_SAMPLES_PER_SEED:
            raise ValueError(
                f"OOF seed {seed} must cover {EXPECTED_SAMPLES_PER_SEED} unique sample_index values exactly once"
            )


def _integer_column(dataframe: pd.DataFrame, column: str) -> list[int]:
    values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{column} values must be finite integers")
    return values.astype(int).tolist()


def _mask_nnz(mask: np.ndarray | sparse.spmatrix) -> int:
    return int(mask.nnz) if sparse.issparse(mask) else int(np.count_nonzero(mask))


def validate_mask_integrity(metrics_df: pd.DataFrame, mask: np.ndarray | sparse.spmatrix) -> None:
    """Verify that all off-mask weights stayed exactly zero after training."""
    missing_columns = set(METADATA_COLUMNS).difference(metrics_df.columns)
    if missing_columns:
        raise ValueError(f"binn_cv.csv is missing required columns: {sorted(missing_columns)}")
    masked_weights = pd.to_numeric(
        metrics_df["max_abs_masked_weight_after_training"], errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(masked_weights).all() or not np.equal(masked_weights, 0.0).all():
        raise ValueError("max_abs_masked_weight_after_training must be exactly 0.0 for all rows")
    expected_unmasked = _mask_nnz(mask)
    expected_masked = int(np.prod(mask.shape)) - expected_unmasked
    for column, expected in (("n_unmasked_weights", expected_unmasked), ("n_masked_weights", expected_masked)):
        values = _integer_column(metrics_df, column)
        if any(value != expected for value in values):
            raise ValueError(f"{column} must equal {expected} for all rows")


def summarize_phase3_gate(metrics_df: pd.DataFrame, mask: np.ndarray | sparse.spmatrix, oof_df: pd.DataFrame) -> dict[str, float | int | tuple[int, int]]:
    """Return concise development-only audit facts after validation."""
    return {
        "mask_shape": tuple(int(value) for value in mask.shape),
        "mask_nnz": _mask_nnz(mask),
        "mean_auroc": float(pd.to_numeric(metrics_df["auroc"]).mean()),
        "mean_auprc": float(pd.to_numeric(metrics_df["auprc"]).mean()),
        "mean_balanced_accuracy": float(pd.to_numeric(metrics_df["balanced_accuracy"]).mean()),
        "mean_brier": float(pd.to_numeric(metrics_df["brier"]).mean()),
        "mean_ece": float(pd.to_numeric(metrics_df["ece"]).mean()),
        "max_masked_weight": float(pd.to_numeric(metrics_df["max_abs_masked_weight_after_training"]).max()),
        "oof_rows": int(len(oof_df)),
    }


def write_phase3_gate_report(output_path: str | Path, summary: dict[str, float | int | tuple[int, int]]) -> None:
    """Write the Phase 3 final gate report with explicit scope boundaries."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Phase 3 Final Gate Audit",
                "",
                "This is a development-only audit of existing BINN outputs, not final validation or a final performance claim.",
                "",
                "## Gate Status",
                "",
                "- Status: `PASS`",
                "- The Phase 3 gate concerns stable training and pathway-mask integrity, not model superiority.",
                "",
                "## Validated Development Outputs",
                "",
                f"- Mask shape: `{summary['mask_shape']}`; nonzero entries: `{summary['mask_nnz']}`.",
                "- Exact model: `pathway_constrained_binn`; seeds: `11`, `23`, `37`; folds: `1`–`5`.",
                "- All 15 metric rows and 1,314 OOF rows passed integrity checks.",
                "- Each seed covers 438 unique development sample indices exactly once.",
                "- Off-mask weights are exactly zero; reported masked/unmasked weight counts match the pathway mask.",
                "- Mean development metrics across fold/seed rows:",
                "",
                "| AUROC | AUPRC | Balanced accuracy | Brier | ECE | Max masked weight |",
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
                f"| {summary['mean_auroc']:.6f} | {summary['mean_auprc']:.6f} | {summary['mean_balanced_accuracy']:.6f} | {summary['mean_brier']:.6f} | {summary['mean_ece']:.6f} | {summary['max_masked_weight']:.1f} |",
                "",
                "## Boundary Confirmation",
                "",
                "- No external cohort or held-out NDD result was used.",
                "- No model was trained and BINN CV was not rerun inside this gate.",
                "- This is development-only and not final validation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
