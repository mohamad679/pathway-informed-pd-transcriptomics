from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_MODELS = frozenset(
    {"logistic_regression", "random_forest", "unconstrained_mlp"}
)
METRIC_NAMES = (
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
    "ece",
)
REQUIRED_SUMMARY_COLUMNS = frozenset(
    {"model", "n_seeds", "n_oof_rows"}
    | {
        f"{metric}_{stat}"
        for metric in METRIC_NAMES
        for stat in ("mean", "ci_lower", "ci_upper")
    }
)
REQUIRED_OOF_COLUMNS = frozenset(
    {"model", "seed", "fold", "sample_index", "y_true", "y_prob"}
)
EXPECTED_SEEDS_PER_MODEL = 3
EXPECTED_OOF_ROWS_PER_MODEL = 1314
EXPECTED_SAMPLES_PER_SEED = 438


def load_phase2_outputs(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load the development-only Phase 2 baseline output CSVs."""
    results_path = Path(results_dir)
    filenames = {
        "logistic_cv": "logistic_baseline_cv.csv",
        "random_forest_cv": "random_forest_baseline_cv.csv",
        "mlp_cv": "mlp_baseline_cv.csv",
        "oof": "baseline_oof_predictions.csv",
        "summary": "baseline_summary.csv",
    }
    outputs: dict[str, pd.DataFrame] = {}
    for name, filename in filenames.items():
        path = results_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 2 output: {path}")
        outputs[name] = pd.read_csv(path)
    return outputs


def validate_required_models(summary_df: pd.DataFrame) -> None:
    """Validate the exact baseline set and required summary metric columns."""
    missing_columns = REQUIRED_SUMMARY_COLUMNS.difference(summary_df.columns)
    if missing_columns:
        raise ValueError(
            "baseline_summary.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    observed_models = set(summary_df["model"].astype(str))
    if observed_models != REQUIRED_MODELS:
        raise ValueError(
            "Phase 2 models must be exactly "
            f"{sorted(REQUIRED_MODELS)}, found {sorted(observed_models)}"
        )
    if summary_df["model"].duplicated().any():
        raise ValueError("baseline_summary.csv contains duplicate model rows")
    n_seeds = pd.to_numeric(summary_df["n_seeds"], errors="coerce")
    n_oof_rows = pd.to_numeric(summary_df["n_oof_rows"], errors="coerce")
    if not (n_seeds == EXPECTED_SEEDS_PER_MODEL).all():
        raise ValueError(
            f"Each summary model must report {EXPECTED_SEEDS_PER_MODEL} seeds"
        )
    if not (n_oof_rows == EXPECTED_OOF_ROWS_PER_MODEL).all():
        raise ValueError(
            f"Each summary model must report {EXPECTED_OOF_ROWS_PER_MODEL} OOF rows"
        )


def validate_oof_prediction_integrity(oof_df: pd.DataFrame) -> None:
    """Validate development-only OOF row counts, seed coverage, and probabilities."""
    missing_columns = REQUIRED_OOF_COLUMNS.difference(oof_df.columns)
    if missing_columns:
        raise ValueError(
            "baseline_oof_predictions.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    observed_models = set(oof_df["model"].astype(str))
    if observed_models != REQUIRED_MODELS:
        raise ValueError(
            "OOF models must be exactly "
            f"{sorted(REQUIRED_MODELS)}, found {sorted(observed_models)}"
        )

    y_prob = pd.to_numeric(oof_df["y_prob"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_prob).all() or np.any((y_prob < 0.0) | (y_prob > 1.0)):
        raise ValueError("OOF y_prob values must be finite and in [0, 1]")

    for model, model_df in oof_df.groupby("model", sort=True):
        if len(model_df) != EXPECTED_OOF_ROWS_PER_MODEL:
            raise ValueError(
                f"{model} must have {EXPECTED_OOF_ROWS_PER_MODEL} OOF rows, "
                f"found {len(model_df)}"
            )
        seeds = model_df["seed"].unique()
        if len(seeds) != EXPECTED_SEEDS_PER_MODEL:
            raise ValueError(
                f"{model} must have {EXPECTED_SEEDS_PER_MODEL} seeds, found {len(seeds)}"
            )
        for seed, seed_df in model_df.groupby("seed", sort=True):
            unique_samples = seed_df["sample_index"].nunique()
            if len(seed_df) != EXPECTED_SAMPLES_PER_SEED or unique_samples != EXPECTED_SAMPLES_PER_SEED:
                raise ValueError(
                    f"{model} seed {seed} must cover {EXPECTED_SAMPLES_PER_SEED} "
                    f"unique development samples exactly once; found {unique_samples} unique "
                    f"samples across {len(seed_df)} rows"
                )


def validate_sanity_gate(summary_df: pd.DataFrame, min_auroc: float = 0.6) -> str:
    """Require every required model's mean AUROC to exceed the sanity threshold."""
    validate_required_models(summary_df)
    auroc_means = pd.to_numeric(summary_df["auroc_mean"], errors="coerce")
    if not np.isfinite(auroc_means.to_numpy(dtype=float)).all():
        raise ValueError("AUROC means must be finite")
    failing_models = summary_df.loc[auroc_means <= min_auroc, "model"].astype(str).tolist()
    if failing_models:
        raise ValueError(
            f"AUROC sanity gate requires all means > {min_auroc}; "
            f"failing models: {failing_models}"
        )
    return str(summary_df.loc[auroc_means.idxmax(), "model"])


def write_phase2_gate_report(
    output_path: str | Path,
    *,
    summary_df: pd.DataFrame,
    best_model: str,
    min_auroc: float = 0.6,
) -> None:
    """Write a development-only Phase 2 gate report without performance claims."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_summary = summary_df.sort_values("model")
    lines = [
        "# Phase 2 Final Gate Audit",
        "",
        "This is a gate/audit of existing development-only baseline outputs. It is not an external-validation result or a final performance claim.",
        "",
        "## Gate Status",
        "",
        "- Status: `PASS`",
        "",
        "## Validated Baselines",
        "",
        "- Exact model set: `logistic_regression`, `random_forest`, `unconstrained_mlp`.",
        "- Required mean and 95% CI columns are present for AUROC, AUPRC, balanced accuracy, Brier, and ECE.",
        "- Each model has 3 seeds and 1,314 OOF rows.",
        "- Each model/seed covers 438 unique development samples exactly once.",
        "- All OOF probabilities are finite and within [0, 1].",
        "",
        "## AUROC Sanity Gate",
        "",
    ]
    lines.extend(
        f"- `{row.model}` mean AUROC: `{float(row.auroc_mean):.6f}`"
        for row in ordered_summary.itertuples(index=False)
    )
    lines.extend(
        [
            f"- All model AUROC means are greater than `{min_auroc:.1f}`.",
            f"- Best model by mean AUROC: `{best_model}`.",
            "",
            "## Boundary Confirmation",
            "",
            "- Only existing Phase 2 development baseline outputs were loaded.",
            "- No external cohort or held-out NDD data was loaded or used.",
            "- No new modeling was performed.",
            "- No BINN, pathway masks, or MSigDB logic was implemented or used.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
