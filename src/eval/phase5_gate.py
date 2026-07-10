from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PERMUTATION_COLUMNS = ("permutation_index", "null_auroc")
BOOTSTRAP_COLUMNS = ("metric", "estimate", "ci_lower", "ci_upper", "n_bootstrap")
CALIBRATION_COLUMNS = (
    "bin_index",
    "bin_left",
    "bin_right",
    "n_samples",
    "mean_predicted_probability",
    "observed_fraction",
)
BOOTSTRAP_METRICS = frozenset({"auroc", "auprc", "balanced_accuracy", "brier", "ece"})
EXPECTED_PERMUTATIONS = 50
EXPECTED_BOOTSTRAPS = 2000
EXPECTED_CALIBRATION_BINS = 15
MINIMUM_ATTAINABLE_P_VALUE = 1.0 / (EXPECTED_PERMUTATIONS + 1)
FLOAT_TOLERANCE = 1e-12

FALSE_P_LT_001_FLAGS = frozenset(
    {
        "p_less_than_0_01",
        "p_lt_0_01",
        "p_below_0_01",
        "p_value_less_than_0_01",
        "p_value_lt_0_01",
        "supports_p_less_than_0_01",
        "supports_p_lt_0_01",
        "claims_p_less_than_0_01",
        "claims_p_lt_0_01",
        "significant_at_0_01",
    }
)


def load_phase5_outputs(results_dir: str | Path) -> dict[str, Any]:
    """Load existing Phase 5 artifacts without running training, permutations, or bootstrap."""
    base = Path(results_dir)
    paths = {
        "permutation_df": base / "statistical_validation_permutation_null.csv",
        "summary": base / "statistical_validation_summary.json",
        "bootstrap_df": base / "statistical_validation_bootstrap_ci.csv",
        "calibration_df": base / "statistical_validation_calibration_bins.csv",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 5 output: {path}")

    root = base.parents[1] if len(base.parents) > 1 else base.parent
    source_report_path = root / "docs" / "phase5_statistical_validation.md"
    source_report_text = (
        source_report_path.read_text(encoding="utf-8") if source_report_path.is_file() else ""
    )
    return {
        "permutation_df": pd.read_csv(paths["permutation_df"]),
        "summary": json.loads(paths["summary"].read_text(encoding="utf-8")),
        "bootstrap_df": pd.read_csv(paths["bootstrap_df"]),
        "calibration_df": pd.read_csv(paths["calibration_df"]),
        "source_report_text": source_report_text,
    }


def validate_permutation_output(permutation_df: pd.DataFrame) -> None:
    """Validate the fixed 50-row development-only permutation null artifact."""
    _validate_no_external_ndd_columns(permutation_df, "permutation null")
    _validate_exact_columns(permutation_df, PERMUTATION_COLUMNS, "permutation null")
    if len(permutation_df) != EXPECTED_PERMUTATIONS:
        raise ValueError(f"permutation null must contain exactly {EXPECTED_PERMUTATIONS} rows")

    indices = _integer_values(permutation_df, "permutation_index", "permutation null")
    duplicate_indices = sorted(permutation_df.loc[permutation_df["permutation_index"].duplicated(), "permutation_index"].unique())
    if duplicate_indices:
        raise ValueError(f"permutation null contains duplicate permutation_index values: {duplicate_indices}")
    expected_indices = list(range(1, EXPECTED_PERMUTATIONS + 1))
    if indices != expected_indices:
        raise ValueError("permutation null permutation_index values must be exactly 1..50")
    _validate_probability_column(permutation_df, "null_auroc", "permutation null")


def validate_summary(
    summary: dict[str, Any],
    permutation_df: pd.DataFrame | None = None,
    source_report_text: str | None = None,
) -> None:
    """Validate Phase 5 summary metadata and optional cross-checks against the null artifact."""
    expected_values: dict[str, object] = {
        "phase": 5,
        "scope": "development-only",
        "fast_smoke": False,
        "n_permutations": EXPECTED_PERMUTATIONS,
        "requested_n_permutations": EXPECTED_PERMUTATIONS,
        "completed_n_permutations": EXPECTED_PERMUTATIONS,
        "final_unique_permutation_count": EXPECTED_PERMUTATIONS,
        "final_index_coverage_complete": True,
        "start_permutation_index": 1,
        "end_permutation_index": EXPECTED_PERMUTATIONS,
        "external_or_ndd_used": False,
        "final_validation": False,
        "model_frozen": False,
        "n_bootstrap": EXPECTED_BOOTSTRAPS,
        "device_resolved": "cuda",
    }
    for key, expected in expected_values.items():
        _require_summary_value(summary, key, expected)

    observed_auroc = _summary_probability(summary, "observed_auroc")
    _summary_probability(summary, "null_auroc_mean")
    _summary_nonnegative_finite(summary, "null_auroc_std")
    empirical_p = _summary_probability(summary, "empirical_p_value")
    _summary_probability(summary, "brier")
    _summary_probability(summary, "ece")

    if permutation_df is not None:
        expected_p = _empirical_p_value(
            observed_auroc,
            pd.to_numeric(permutation_df["null_auroc"], errors="coerce").to_numpy(dtype=float),
        )
        if not math.isclose(empirical_p, expected_p, rel_tol=0.0, abs_tol=FLOAT_TOLERANCE):
            raise ValueError(
                "summary empirical_p_value must equal empirical_p_value(observed_auroc, permutation null scores)"
            )
    if not math.isclose(
        empirical_p,
        MINIMUM_ATTAINABLE_P_VALUE,
        rel_tol=0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError("summary empirical_p_value must equal 1/51 for this production result")

    _validate_no_false_p_lt_001_metadata(summary)
    if source_report_text is not None:
        _validate_limitation_acknowledgement(source_report_text)


def validate_bootstrap_output(
    bootstrap_df: pd.DataFrame,
    summary: dict[str, Any] | None = None,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    """Validate the existing bootstrap confidence interval artifact."""
    _validate_no_external_ndd_columns(bootstrap_df, "bootstrap CI")
    _validate_exact_columns(bootstrap_df, BOOTSTRAP_COLUMNS, "bootstrap CI")
    if len(bootstrap_df) != len(BOOTSTRAP_METRICS):
        raise ValueError("bootstrap CI must contain exactly five metric rows")

    metrics = bootstrap_df["metric"].astype(str).tolist()
    if set(metrics) != BOOTSTRAP_METRICS:
        raise ValueError(f"bootstrap CI metrics must be exactly {sorted(BOOTSTRAP_METRICS)}")
    duplicate_metrics = sorted(bootstrap_df.loc[bootstrap_df["metric"].duplicated(), "metric"].astype(str).unique())
    if duplicate_metrics:
        raise ValueError(f"bootstrap CI must contain exactly one row per metric: {duplicate_metrics}")

    n_bootstrap = _integer_values(bootstrap_df, "n_bootstrap", "bootstrap CI")
    if any(value != EXPECTED_BOOTSTRAPS for value in n_bootstrap):
        raise ValueError(f"bootstrap CI n_bootstrap must be exactly {EXPECTED_BOOTSTRAPS}")
    for column in ("estimate", "ci_lower", "ci_upper"):
        _validate_finite_column(bootstrap_df, column, "bootstrap CI")

    estimates = pd.to_numeric(bootstrap_df["estimate"], errors="coerce").to_numpy(dtype=float)
    lower = pd.to_numeric(bootstrap_df["ci_lower"], errors="coerce").to_numpy(dtype=float)
    upper = pd.to_numeric(bootstrap_df["ci_upper"], errors="coerce").to_numpy(dtype=float)
    if np.any(lower > estimates) or np.any(estimates > upper):
        raise ValueError("bootstrap CI rows must satisfy ci_lower <= estimate <= ci_upper")

    if summary is not None:
        metric_estimates = bootstrap_df.set_index("metric")["estimate"].astype(float).to_dict()
        _validate_estimate_matches_summary(metric_estimates, "auroc", summary, "observed_auroc", tolerance)
        _validate_estimate_matches_summary(metric_estimates, "brier", summary, "brier", tolerance)
        _validate_estimate_matches_summary(metric_estimates, "ece", summary, "ece", tolerance)


def validate_calibration_output(
    calibration_df: pd.DataFrame,
    summary: dict[str, Any] | None = None,
) -> None:
    """Validate the existing 15-bin development-only calibration artifact."""
    _validate_no_external_ndd_columns(calibration_df, "calibration bins")
    _validate_exact_columns(calibration_df, CALIBRATION_COLUMNS, "calibration bins")
    if len(calibration_df) != EXPECTED_CALIBRATION_BINS:
        raise ValueError(f"calibration bins must contain exactly {EXPECTED_CALIBRATION_BINS} rows")

    bin_indices = _integer_values(calibration_df, "bin_index", "calibration bins")
    if bin_indices != list(range(EXPECTED_CALIBRATION_BINS)):
        raise ValueError("calibration bin_index values must be exactly 0..14")
    n_samples = _integer_values(calibration_df, "n_samples", "calibration bins")
    if any(value < 0 for value in n_samples):
        raise ValueError("calibration n_samples values must be nonnegative integers")
    if summary is not None and sum(n_samples) != _summary_integer(summary, "n_samples"):
        raise ValueError("calibration n_samples total must equal summary n_samples")

    left = _finite_array(calibration_df, "bin_left", "calibration bins")
    right = _finite_array(calibration_df, "bin_right", "calibration bins")
    if np.any(left >= right) or np.any(np.diff(left) < -FLOAT_TOLERANCE) or np.any(np.diff(right) < -FLOAT_TOLERANCE):
        raise ValueError("calibration bin edges must be finite and ordered")

    mean_pred = pd.to_numeric(
        calibration_df["mean_predicted_probability"], errors="coerce"
    ).to_numpy(dtype=float)
    observed = pd.to_numeric(calibration_df["observed_fraction"], errors="coerce").to_numpy(dtype=float)
    for row_index, count in enumerate(n_samples):
        values = (mean_pred[row_index], observed[row_index])
        if count > 0:
            if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
                raise ValueError("nonempty calibration means/fractions must be finite and in [0, 1]")
            continue
        if not all(np.isnan(value) or (np.isfinite(value) and 0.0 <= value <= 1.0) for value in values):
            raise ValueError("empty calibration means/fractions must be NaN or in [0, 1]")


def summarize_phase5_gate(
    permutation_df: pd.DataFrame,
    summary: dict[str, Any],
    bootstrap_df: pd.DataFrame,
    calibration_df: pd.DataFrame,
) -> dict[str, float | int | bool]:
    """Return the Phase 5 gate summary without rerunning statistical validation."""
    n_bootstrap_values = _integer_values(bootstrap_df, "n_bootstrap", "bootstrap CI")
    return {
        "status_pass": True,
        "permutation_rows": int(len(permutation_df)),
        "exact_permutation_coverage": True,
        "observed_auroc": float(summary["observed_auroc"]),
        "null_auroc_mean": float(summary["null_auroc_mean"]),
        "null_auroc_std": float(summary["null_auroc_std"]),
        "empirical_p_value": float(summary["empirical_p_value"]),
        "bootstrap_resamples": int(n_bootstrap_values[0]),
        "brier": float(summary["brier"]),
        "ece": float(summary["ece"]),
        "calibration_bins": int(len(calibration_df)),
        "n_samples": int(summary["n_samples"]),
        "external_or_ndd_used": False,
        "training_permutation_bootstrap_rerun_inside_gate": False,
        "final_validation": False,
        "model_frozen": False,
        "minimum_attainable_p_value": MINIMUM_ATTAINABLE_P_VALUE,
        "supports_p_less_than_0_01": False,
        "supports_biological_claims": False,
    }


def write_phase5_gate_report(output_path: str | Path, gate_summary: dict[str, float | int | bool]) -> None:
    """Write the Phase 5 final statistical-validation gate report."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Phase 5 Final Statistical-Validation Gate",
                "",
                "This gate audits integrity of the limited-resolution development-only Phase 5 statistical artifacts. It does not train or freeze a model, rerun permutation testing, rerun bootstrap, use external or held-out NDD data, perform final validation, or support biological claims.",
                "",
                "## Gate Status",
                "",
                "- Status: `PASS`",
                "- PASS means the existing limited-resolution development-only artifacts are internally consistent.",
                "- PASS does not mean final validation.",
                "- PASS does not mean `p < 0.01`.",
                "- PASS does not remove the computational limitation.",
                "- PASS does not support biological claims.",
                "",
                "## Validated Development Outputs",
                "",
                f"- Permutation rows: `{gate_summary['permutation_rows']}`.",
                "- Exact permutation coverage: `1` through `50`, with unique indices.",
                f"- Observed AUROC: `{gate_summary['observed_auroc']:.6f}`.",
                f"- Null AUROC mean: `{gate_summary['null_auroc_mean']:.6f}`.",
                f"- Null AUROC std: `{gate_summary['null_auroc_std']:.6f}`.",
                f"- Empirical p-value: `{gate_summary['empirical_p_value']:.6f}`.",
                f"- Bootstrap resamples per metric: `{gate_summary['bootstrap_resamples']}`.",
                f"- Brier: `{gate_summary['brier']:.6f}`.",
                f"- ECE: `{gate_summary['ece']:.6f}`.",
                f"- Calibration bins: `{gate_summary['calibration_bins']}`.",
                f"- Calibration sample total: `{gate_summary['n_samples']}`.",
                "",
                "## Production Limitation",
                "",
                "- 50 permutations only.",
                "- The minimum attainable p-value is `1/51 = 0.019608`.",
                "- This result cannot support `p < 0.01`.",
                "",
                "## Boundary Confirmation",
                "",
                "- No external cohort or held-out NDD data was used.",
                "- No training, retraining, permutation rerun, or bootstrap rerun happened inside the gate.",
                "- No model was frozen.",
                "- No final external validation claim is made.",
                "- No biological claim is made.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _validate_exact_columns(df: pd.DataFrame, expected_columns: tuple[str, ...], name: str) -> None:
    if tuple(df.columns) != expected_columns:
        raise ValueError(f"{name} columns must be exactly {list(expected_columns)}")


def _validate_no_external_ndd_columns(df: pd.DataFrame, name: str) -> None:
    forbidden = [
        column
        for column in df.columns
        if "external" in str(column).lower() or "ndd" in str(column).lower()
    ]
    if forbidden:
        raise ValueError(f"{name} must not contain external/NDD columns: {sorted(forbidden)}")


def _integer_values(df: pd.DataFrame, column: str, name: str) -> list[int]:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} {column} values must be finite integers")
    return values.astype(int).tolist()


def _validate_probability_column(df: pd.DataFrame, column: str, name: str) -> None:
    values = _finite_array(df, column, name)
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(f"{name} {column} values must be in [0, 1]")


def _validate_finite_column(df: pd.DataFrame, column: str, name: str) -> None:
    _finite_array(df, column, name)


def _finite_array(df: pd.DataFrame, column: str, name: str) -> np.ndarray:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} {column} values must be finite")
    return values


def _require_summary_value(summary: dict[str, Any], key: str, expected: object) -> None:
    if key not in summary:
        raise ValueError(f"summary is missing required key: {key}")
    value = summary[key]
    if isinstance(expected, bool):
        if not isinstance(value, bool) or value is not expected:
            raise ValueError(f"summary {key} must be exactly {expected}")
        return
    if value != expected:
        raise ValueError(f"summary {key} must be exactly {expected}")


def _summary_probability(summary: dict[str, Any], key: str) -> float:
    value = _summary_float(summary, key)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"summary {key} must be in [0, 1]")
    return value


def _summary_nonnegative_finite(summary: dict[str, Any], key: str) -> float:
    value = _summary_float(summary, key)
    if value < 0.0:
        raise ValueError(f"summary {key} must be nonnegative")
    return value


def _summary_float(summary: dict[str, Any], key: str) -> float:
    if key not in summary:
        raise ValueError(f"summary is missing required key: {key}")
    try:
        value = float(summary[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f"summary {key} must be finite") from error
    if not math.isfinite(value):
        raise ValueError(f"summary {key} must be finite")
    return value


def _summary_integer(summary: dict[str, Any], key: str) -> int:
    value = _summary_float(summary, key)
    if not value.is_integer():
        raise ValueError(f"summary {key} must be an integer")
    return int(value)


def _empirical_p_value(observed_score: float, null_scores: np.ndarray) -> float:
    null_array = np.asarray(null_scores, dtype=float).reshape(-1)
    if null_array.size < 1:
        raise ValueError("permutation null scores must not be empty")
    if not np.isfinite(observed_score) or not np.isfinite(null_array).all():
        raise ValueError("observed AUROC and null AUROC scores must be finite")
    return float((1 + np.sum(null_array >= observed_score)) / (1 + null_array.size))


def _validate_estimate_matches_summary(
    metric_estimates: dict[str, float],
    metric: str,
    summary: dict[str, Any],
    summary_key: str,
    tolerance: float,
) -> None:
    if not math.isclose(
        float(metric_estimates[metric]),
        _summary_float(summary, summary_key),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError(f"bootstrap {metric} estimate must match summary {summary_key}")


def _validate_no_false_p_lt_001_metadata(summary: dict[str, Any]) -> None:
    for key, value in summary.items():
        normalized_key = str(key).strip().lower()
        if normalized_key in FALSE_P_LT_001_FLAGS and _truthy_metadata_value(value):
            raise ValueError("summary must not represent this 50-permutation result as supporting p < 0.01")
        if isinstance(value, str):
            normalized_value = value.strip().lower()
            mentions_p_lt_001 = "p < 0.01" in normalized_value or "p<0.01" in normalized_value
            negates_support = (
                "does not support" in normalized_value
                or "cannot support" in normalized_value
                or "not evidence for" in normalized_value
            )
            if mentions_p_lt_001 and "support" in normalized_value and not negates_support:
                raise ValueError("summary text must not claim support for p < 0.01")


def _truthy_metadata_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "supported", "support", "significant"}
    return bool(value)


def _validate_limitation_acknowledgement(source_report_text: str) -> None:
    text = source_report_text.lower()
    has_50_permutation_limit = "50 permutations" in text
    has_minimum_p = (
        "0.019608" in text
        and ("1 / (50 + 1)" in text or "1 / 51" in text or "1/51" in text)
        and ("smallest attainable" in text or "minimum attainable" in text)
    )
    rejects_p_lt_001 = ("p < 0.01" in text or "p<0.01" in text) and (
        "does not support" in text or "cannot support" in text or "not evidence for" in text
    )
    if not (has_50_permutation_limit and has_minimum_p and rejects_p_lt_001):
        raise ValueError(
            "Phase 5 source report must acknowledge the 50-permutation, 1/51 = 0.019608, no p < 0.01 limitation"
        )
