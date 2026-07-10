from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase5_gate import (  # noqa: E402
    validate_bootstrap_output,
    validate_calibration_output,
    validate_permutation_output,
    validate_summary,
)


def _permutation_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "permutation_index": list(range(1, 51)),
            "null_auroc": [0.40 + index * 0.001 for index in range(50)],
        }
    )


def _summary() -> dict[str, object]:
    return {
        "phase": 5,
        "scope": "development-only",
        "fast_smoke": False,
        "n_permutations": 50,
        "requested_n_permutations": 50,
        "completed_n_permutations": 50,
        "final_unique_permutation_count": 50,
        "final_index_coverage_complete": True,
        "start_permutation_index": 1,
        "end_permutation_index": 50,
        "external_or_ndd_used": False,
        "final_validation": False,
        "model_frozen": False,
        "n_bootstrap": 2000,
        "observed_auroc": 0.80,
        "null_auroc_mean": 0.4245,
        "null_auroc_std": 0.0144,
        "empirical_p_value": 1 / 51,
        "brier": 0.20,
        "ece": 0.10,
        "device_resolved": "cuda",
        "n_samples": 100,
    }


def _bootstrap_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "auroc", "estimate": 0.80, "ci_lower": 0.75, "ci_upper": 0.85, "n_bootstrap": 2000},
            {"metric": "auprc", "estimate": 0.70, "ci_lower": 0.65, "ci_upper": 0.75, "n_bootstrap": 2000},
            {
                "metric": "balanced_accuracy",
                "estimate": 0.66,
                "ci_lower": 0.61,
                "ci_upper": 0.71,
                "n_bootstrap": 2000,
            },
            {"metric": "brier", "estimate": 0.20, "ci_lower": 0.15, "ci_upper": 0.25, "n_bootstrap": 2000},
            {"metric": "ece", "estimate": 0.10, "ci_lower": 0.05, "ci_upper": 0.15, "n_bootstrap": 2000},
        ]
    )


def _calibration_df() -> pd.DataFrame:
    rows = []
    counts = [10] * 10 + [0] * 5
    for bin_index, count in enumerate(counts):
        rows.append(
            {
                "bin_index": bin_index,
                "bin_left": bin_index / 15,
                "bin_right": (bin_index + 1) / 15,
                "n_samples": count,
                "mean_predicted_probability": 0.50 if count else float("nan"),
                "observed_fraction": 0.40 if count else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _source_report_text() -> str:
    return (
        "This report documents 50 permutations only. "
        "The smallest attainable empirical p-value is 1 / (50 + 1) = 0.019608. "
        "Therefore, this result does not support p < 0.01."
    )


def _validate_all() -> None:
    permutation_df = _permutation_df()
    summary = _summary()
    bootstrap_df = _bootstrap_df()
    calibration_df = _calibration_df()

    validate_permutation_output(permutation_df)
    validate_summary(summary, permutation_df, _source_report_text())
    validate_bootstrap_output(bootstrap_df, summary)
    validate_calibration_output(calibration_df, summary)


def test_complete_valid_50_permutation_fixture_passes() -> None:
    _validate_all()


def test_catches_wrong_permutation_coverage() -> None:
    permutation_df = _permutation_df()
    permutation_df.loc[49, "permutation_index"] = 51

    with pytest.raises(ValueError, match="exactly 1..50"):
        validate_permutation_output(permutation_df)


def test_catches_duplicate_permutation_index() -> None:
    permutation_df = _permutation_df()
    permutation_df.loc[49, "permutation_index"] = 1

    with pytest.raises(ValueError, match="duplicate permutation_index"):
        validate_permutation_output(permutation_df)


def test_catches_mismatched_empirical_p_value() -> None:
    summary = _summary()
    summary["empirical_p_value"] = 2 / 51

    with pytest.raises(ValueError, match="empirical_p_value must equal"):
        validate_summary(summary, _permutation_df(), _source_report_text())


def test_catches_invalid_bootstrap_ci_ordering() -> None:
    bootstrap_df = _bootstrap_df()
    bootstrap_df.loc[0, "ci_lower"] = 0.81

    with pytest.raises(ValueError, match="ci_lower <= estimate <= ci_upper"):
        validate_bootstrap_output(bootstrap_df, _summary())


def test_catches_wrong_bootstrap_count() -> None:
    bootstrap_df = _bootstrap_df()
    bootstrap_df.loc[0, "n_bootstrap"] = 1999

    with pytest.raises(ValueError, match="n_bootstrap must be exactly 2000"):
        validate_bootstrap_output(bootstrap_df, _summary())


def test_catches_wrong_calibration_sample_total() -> None:
    calibration_df = _calibration_df()
    calibration_df.loc[0, "n_samples"] = 9

    with pytest.raises(ValueError, match="must equal summary n_samples"):
        validate_calibration_output(calibration_df, _summary())


def test_catches_forbidden_external_ndd_column() -> None:
    bootstrap_df = _bootstrap_df()
    bootstrap_df["held_out_ndd_score"] = 0.0

    with pytest.raises(ValueError, match="external/NDD columns"):
        validate_bootstrap_output(bootstrap_df, _summary())


def test_catches_false_p_less_than_0_01_interpretation_metadata_if_represented() -> None:
    summary = _summary()
    summary["supports_p_lt_0_01"] = True

    with pytest.raises(ValueError, match="supporting p < 0.01"):
        validate_summary(summary, _permutation_df(), _source_report_text())
