"""Final read-only integrity gate for existing Phase 6 validation artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from eval.metrics import compute_binary_metrics
from models.frozen_bundle import PAYLOAD_FILES, verify_hash_manifest


FIXED_THRESHOLD = 0.5
EXPECTED_FROZEN_COMMIT = "6b150f8f64935e3bbd4b7bb64a5ec59e665a7f22"
EXPECTED_FROZEN_TAG = "frozen-v1"
EXPECTED_EXTERNAL_ROWS = 72
EXPECTED_EXTERNAL_HC = 22
EXPECTED_EXTERNAL_PD = 50
EXPECTED_NDD_ROWS = 48
FLOAT_TOLERANCE = 1e-8
CALIBRATION_APPROX_TOLERANCE = 1e-6

IMMUTABLE_PAYLOAD_FILES = (
    "model_v1.pt",
    "preprocessing_config.json",
    "gene_space.txt",
    "pathway_names.txt",
    "pathway_mask.npz",
    "training_metadata.json",
)
EXTERNAL_PREDICTION_COLUMNS = (
    "sample_id",
    "y_true",
    "logit",
    "y_prob",
    "y_pred",
)
EXTERNAL_METRIC_FIELDS = {
    "auroc",
    "auprc",
    "balanced_accuracy",
    "brier",
    "ece",
    "threshold",
    "n_samples",
    "n_hc",
    "n_pd",
    "frozen_commit",
    "frozen_tag",
    "hash_before_verified",
    "model_frozen",
    "external_scored_once",
}
METRIC_NAMES = ("auroc", "auprc", "balanced_accuracy", "brier", "ece")
NDD_PREDICTION_COLUMNS = ("sample_id", "logit", "y_prob", "predicted_class")
NDD_SUMMARY_FIELDS = {
    "n_samples",
    "mean_pd_probability",
    "median_pd_probability",
    "std_pd_probability",
    "min_pd_probability",
    "max_pd_probability",
    "fraction_predicted_pd_at_0_5",
    "fraction_predicted_hc_at_0_5",
}
SCORING_AUDIT_FIELDS = {
    "execution_scope",
    "frozen_commit",
    "frozen_tag",
    "hash_before_manifest",
    "hash_before_verified",
    "external_input_paths",
    "ndd_input_paths",
    "frozen_payload_modified",
    "threshold",
    "scaler_refit",
    "model_retrained",
    "external_metric_used_for_selection",
    "external_scored_once",
    "ndd_scored_once",
}


def _load_json_mapping(path: Path, artifact_name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{artifact_name} must contain a JSON object")
    return value


def _require_fields(mapping: Mapping[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")


def _validate_exact_columns(
    dataframe: pd.DataFrame,
    expected: tuple[str, ...],
    name: str,
) -> None:
    observed = tuple(str(column) for column in dataframe.columns)
    if observed != expected:
        raise ValueError(f"{name} columns must be exactly {list(expected)}; observed={list(observed)}")


def _numeric_array(dataframe: pd.DataFrame, column: str, name: str) -> np.ndarray:
    try:
        values = pd.to_numeric(dataframe[column], errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} {column} must be numeric") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{name} {column} must be finite")
    return values


def _binary_integer_array(dataframe: pd.DataFrame, column: str, name: str) -> np.ndarray:
    values = _numeric_array(dataframe, column, name)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError(f"{name} {column} must be binary")
    return values.astype(np.int64)


def _validate_sample_ids(dataframe: pd.DataFrame, expected_rows: int, name: str) -> None:
    if len(dataframe) != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} rows")
    sample_ids = dataframe["sample_id"]
    if sample_ids.isna().any() or sample_ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} sample_id values must be nonempty")
    if sample_ids.duplicated().any():
        raise ValueError(f"{name} contains duplicate rows by sample_id")
    if sample_ids.nunique(dropna=False) != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} unique sample IDs")


def _require_boolean(mapping: Mapping[str, Any], key: str, expected: bool, name: str) -> None:
    value = mapping[key]
    if not isinstance(value, bool) or value is not expected:
        raise ValueError(f"{name} {key} must be {str(expected).lower()}")


def _require_exact_integer(mapping: Mapping[str, Any], key: str, expected: int, name: str) -> None:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) != expected:
        raise ValueError(f"{name} {key} must be exactly {expected}")


def _finite_probability(mapping: Mapping[str, Any], key: str, name: str) -> float:
    value = mapping[key]
    if isinstance(value, bool):
        raise ValueError(f"{name} {key} must be a finite value in [0, 1]")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} {key} must be a finite value in [0, 1]") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} {key} must be a finite value in [0, 1]")
    return number


def _metadata_value_is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized not in {"", "false", "no", "none", "not supported", "not applicable"}
    return value is not None


def _validate_no_forbidden_claim_metadata(mapping: Mapping[str, Any], name: str) -> None:
    """Reject affirmative clinical, diagnostic, deployment, or biological claim flags."""

    domains = ("clinical", "diagnostic", "deployment", "biological")
    assertions = ("valid", "ready", "readiness", "claim", "support")

    def visit(value: object, path: tuple[str, ...]) -> None:
        if not isinstance(value, Mapping):
            return
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_").replace(" ", "_")
            child_path = (*path, key)
            joined_path = "_".join(child_path)
            is_claim_key = any(domain in joined_path for domain in domains) and any(
                assertion in joined_path for assertion in assertions
            )
            is_negative_key = joined_path.startswith("not_") or "does_not_" in joined_path
            if is_claim_key and not is_negative_key and _metadata_value_is_true(child):
                raise ValueError(
                    f"{name} contains forbidden clinical/diagnostic/deployment/biological "
                    f"claim metadata: {'.'.join(child_path)}"
                )
            visit(child, child_path)

    visit(mapping, ())


def load_phase6_outputs(
    frozen_dir: str | Path,
    external_results_dir: str | Path,
) -> dict[str, Any]:
    """Load existing Phase 6 artifacts without training, fitting, inference, or scoring."""
    frozen_base = Path(frozen_dir)
    results_base = Path(external_results_dir)
    required_paths = {
        "hash_before_path": frozen_base / "HASH_BEFORE.txt",
        "hash_after_path": frozen_base / "HASH_AFTER.txt",
        "external_predictions": results_base / "external_predictions.csv",
        "external_metrics": results_base / "external_metrics.json",
        "ndd_predictions": results_base / "ndd_predictions.csv",
        "ndd_summary": results_base / "ndd_specificity_summary.json",
        "scoring_audit": results_base / "scoring_audit.json",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 6 artifact: {path}")

    return {
        "hash_before_path": required_paths["hash_before_path"],
        "hash_after_path": required_paths["hash_after_path"],
        "external_predictions": pd.read_csv(required_paths["external_predictions"]),
        "external_metrics": _load_json_mapping(
            required_paths["external_metrics"], "external metrics"
        ),
        "ndd_predictions": pd.read_csv(required_paths["ndd_predictions"]),
        "ndd_summary": _load_json_mapping(required_paths["ndd_summary"], "NDD summary"),
        "scoring_audit": _load_json_mapping(
            required_paths["scoring_audit"], "scoring audit"
        ),
    }


def validate_frozen_hash_chain(frozen_dir: str | Path) -> dict[str, Any]:
    """Validate both frozen manifests and their exact chain-of-custody equality."""
    base = Path(frozen_dir)
    before_path = base / "HASH_BEFORE.txt"
    after_path = base / "HASH_AFTER.txt"
    if not before_path.is_file():
        raise FileNotFoundError(f"Missing frozen hash manifest: {before_path}")
    if not after_path.is_file():
        raise FileNotFoundError(f"Missing frozen hash manifest: {after_path}")
    if tuple(PAYLOAD_FILES) != IMMUTABLE_PAYLOAD_FILES:
        raise ValueError("Immutable frozen payload file contract is not the exact Phase 6 set")

    before_text = before_path.read_text(encoding="utf-8")
    after_text = after_path.read_text(encoding="utf-8")
    if before_text != after_text:
        raise ValueError("HASH_BEFORE.txt and HASH_AFTER.txt must match exactly")
    verify_hash_manifest(base, before_path)
    verify_hash_manifest(base, after_path)
    return {
        "hash_before_after_equal": True,
        "hash_before_verified": True,
        "hash_after_verified": True,
        "immutable_payload_files": IMMUTABLE_PAYLOAD_FILES,
    }


def validate_external_predictions(predictions_df: pd.DataFrame) -> None:
    """Validate the saved 72-row external prediction artifact without inference."""
    _validate_exact_columns(
        predictions_df, EXTERNAL_PREDICTION_COLUMNS, "external predictions"
    )
    _validate_sample_ids(predictions_df, EXPECTED_EXTERNAL_ROWS, "external predictions")
    labels = _binary_integer_array(predictions_df, "y_true", "external predictions")
    if int(np.sum(labels == 0)) != EXPECTED_EXTERNAL_HC:
        raise ValueError("external predictions must contain exactly 22 HC labels")
    if int(np.sum(labels == 1)) != EXPECTED_EXTERNAL_PD:
        raise ValueError("external predictions must contain exactly 50 PD labels")
    _numeric_array(predictions_df, "logit", "external predictions")
    probabilities = _numeric_array(predictions_df, "y_prob", "external predictions")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("external predictions y_prob must be in [0, 1]")
    predictions = _binary_integer_array(predictions_df, "y_pred", "external predictions")
    expected_predictions = (probabilities >= FIXED_THRESHOLD).astype(np.int64)
    if not np.array_equal(predictions, expected_predictions):
        raise ValueError("external predictions y_pred must exactly match y_prob >= 0.5")


def validate_external_metrics(
    metrics: Mapping[str, Any],
    predictions_df: pd.DataFrame | None = None,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    """Validate stored metadata and recompute metrics from saved predictions only."""
    _require_fields(metrics, EXTERNAL_METRIC_FIELDS, "external metrics")
    _validate_no_forbidden_claim_metadata(metrics, "external metrics")
    threshold = _finite_probability(metrics, "threshold", "external metrics")
    if threshold != FIXED_THRESHOLD:
        raise ValueError("external metrics threshold must be exactly 0.5")
    _require_exact_integer(metrics, "n_samples", EXPECTED_EXTERNAL_ROWS, "external metrics")
    _require_exact_integer(metrics, "n_hc", EXPECTED_EXTERNAL_HC, "external metrics")
    _require_exact_integer(metrics, "n_pd", EXPECTED_EXTERNAL_PD, "external metrics")
    if metrics["frozen_tag"] != EXPECTED_FROZEN_TAG:
        raise ValueError(f"external metrics frozen_tag must be exactly {EXPECTED_FROZEN_TAG}")
    if metrics["frozen_commit"] != EXPECTED_FROZEN_COMMIT:
        raise ValueError(
            f"external metrics frozen_commit must be exactly {EXPECTED_FROZEN_COMMIT}"
        )
    _require_boolean(metrics, "hash_before_verified", True, "external metrics")
    _require_boolean(metrics, "model_frozen", True, "external metrics")
    _require_boolean(metrics, "external_scored_once", True, "external metrics")

    stored_metrics = {
        metric_name: _finite_probability(metrics, metric_name, "external metrics")
        for metric_name in METRIC_NAMES
    }
    if predictions_df is not None:
        recomputed = compute_binary_metrics(
            predictions_df["y_true"].to_numpy(),
            predictions_df["y_prob"].to_numpy(),
            threshold=FIXED_THRESHOLD,
        )
        for metric_name in METRIC_NAMES:
            if not math.isclose(
                stored_metrics[metric_name],
                recomputed[metric_name],
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise ValueError(
                    f"external metric mismatch for {metric_name}: "
                    f"stored={stored_metrics[metric_name]}, recomputed={recomputed[metric_name]}"
                )
    if not math.isclose(
        stored_metrics["balanced_accuracy"], 0.5, rel_tol=0.0, abs_tol=tolerance
    ):
        raise ValueError("external balanced_accuracy must be 0.5 for the frozen result")
    if not math.isclose(
        stored_metrics["brier"],
        0.694444,
        rel_tol=0.0,
        abs_tol=CALIBRATION_APPROX_TOLERANCE,
    ):
        raise ValueError("external Brier must remain approximately 0.694444")
    if not math.isclose(
        stored_metrics["ece"],
        0.694441,
        rel_tol=0.0,
        abs_tol=CALIBRATION_APPROX_TOLERANCE,
    ):
        raise ValueError("external ECE must remain approximately 0.694441")


def validate_ndd_predictions(predictions_df: pd.DataFrame) -> None:
    """Validate the saved 48-row unlabeled NDD stress-test predictions."""
    _validate_exact_columns(predictions_df, NDD_PREDICTION_COLUMNS, "NDD predictions")
    _validate_sample_ids(predictions_df, EXPECTED_NDD_ROWS, "NDD predictions")
    _numeric_array(predictions_df, "logit", "NDD predictions")
    probabilities = _numeric_array(predictions_df, "y_prob", "NDD predictions")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("NDD predictions y_prob must be in [0, 1]")
    predicted_classes = predictions_df["predicted_class"]
    if predicted_classes.isna().any() or not predicted_classes.isin(("HC", "PD")).all():
        raise ValueError("NDD predictions predicted_class must contain only HC/PD")
    expected_classes = np.where(probabilities >= FIXED_THRESHOLD, "PD", "HC")
    if not np.array_equal(predicted_classes.to_numpy(dtype=str), expected_classes):
        raise ValueError(
            "NDD predictions predicted_class must exactly match the 0.5 threshold"
        )


def validate_ndd_summary(
    summary: Mapping[str, Any],
    predictions_df: pd.DataFrame | None = None,
    tolerance: float = FLOAT_TOLERANCE,
) -> None:
    """Recompute the specificity/stress-test summary from saved NDD predictions."""
    _require_fields(summary, NDD_SUMMARY_FIELDS, "NDD summary")
    _validate_no_forbidden_claim_metadata(summary, "NDD summary")
    _require_exact_integer(summary, "n_samples", EXPECTED_NDD_ROWS, "NDD summary")
    observed_values = {
        field: _finite_probability(summary, field, "NDD summary")
        for field in NDD_SUMMARY_FIELDS - {"n_samples"}
    }
    if predictions_df is not None:
        probabilities = predictions_df["y_prob"].to_numpy(dtype=np.float64)
        predicted_pd = probabilities >= FIXED_THRESHOLD
        recomputed = {
            "mean_pd_probability": float(np.mean(probabilities)),
            "median_pd_probability": float(np.median(probabilities)),
            "std_pd_probability": float(np.std(probabilities)),
            "min_pd_probability": float(np.min(probabilities)),
            "max_pd_probability": float(np.max(probabilities)),
            "fraction_predicted_pd_at_0_5": float(np.mean(predicted_pd)),
            "fraction_predicted_hc_at_0_5": float(np.mean(~predicted_pd)),
        }
        for field, expected in recomputed.items():
            observed = observed_values[field]
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
                raise ValueError(
                    f"NDD summary mismatch for {field}: stored={observed}, recomputed={expected}"
                )
    fraction_sum = (
        float(summary["fraction_predicted_pd_at_0_5"])
        + float(summary["fraction_predicted_hc_at_0_5"])
    )
    if not math.isclose(fraction_sum, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError("NDD summary predicted fractions must sum to 1")


def validate_scoring_audit(
    audit: Mapping[str, Any],
    external_metrics: Mapping[str, Any] | None = None,
) -> None:
    """Validate the one-time scoring boundary and chain-of-custody audit."""
    _require_fields(audit, SCORING_AUDIT_FIELDS, "scoring audit")
    _validate_no_forbidden_claim_metadata(audit, "scoring audit")
    if audit["frozen_commit"] != EXPECTED_FROZEN_COMMIT:
        raise ValueError(f"scoring audit frozen_commit must be {EXPECTED_FROZEN_COMMIT}")
    if audit["frozen_tag"] != EXPECTED_FROZEN_TAG:
        raise ValueError(f"scoring audit frozen_tag must be {EXPECTED_FROZEN_TAG}")
    if external_metrics is not None and (
        audit["frozen_commit"] != external_metrics.get("frozen_commit")
        or audit["frozen_tag"] != external_metrics.get("frozen_tag")
    ):
        raise ValueError("scoring audit frozen commit/tag must match external metrics")
    if not isinstance(audit["execution_scope"], str) or not audit["execution_scope"].strip():
        raise ValueError("scoring audit execution_scope must be a nonempty string")
    if (
        not isinstance(audit["hash_before_manifest"], str)
        or Path(audit["hash_before_manifest"]).name != "HASH_BEFORE.txt"
    ):
        raise ValueError("scoring audit hash_before_manifest must identify HASH_BEFORE.txt")
    for key in ("external_input_paths", "ndd_input_paths"):
        paths = audit[key]
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(path, str) or not path.strip() for path in paths)
        ):
            raise ValueError(f"scoring audit {key} must be a nonempty list of paths")
    threshold = _finite_probability(audit, "threshold", "scoring audit")
    if threshold != FIXED_THRESHOLD:
        raise ValueError("scoring audit threshold must be exactly 0.5")
    _require_boolean(audit, "hash_before_verified", True, "scoring audit")
    _require_boolean(audit, "frozen_payload_modified", False, "scoring audit")
    _require_boolean(audit, "scaler_refit", False, "scoring audit")
    _require_boolean(audit, "model_retrained", False, "scoring audit")
    _require_boolean(audit, "external_metric_used_for_selection", False, "scoring audit")
    _require_boolean(audit, "external_scored_once", True, "scoring audit")
    _require_boolean(audit, "ndd_scored_once", True, "scoring audit")


def summarize_phase6_gate(
    hash_chain: Mapping[str, Any],
    external_predictions: pd.DataFrame,
    external_metrics: Mapping[str, Any],
    ndd_predictions: pd.DataFrame,
    ndd_summary: Mapping[str, Any],
    scoring_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize validated saved artifacts without performing any inference or scoring."""
    labels = external_predictions["y_true"].to_numpy(dtype=np.int64)
    return {
        "status": "PASS",
        "hash_before_after_equal": bool(hash_chain["hash_before_after_equal"]),
        "hash_before_verified": bool(hash_chain["hash_before_verified"]),
        "hash_after_verified": bool(hash_chain["hash_after_verified"]),
        "frozen_payload_modified": bool(scoring_audit["frozen_payload_modified"]),
        "frozen_commit": str(external_metrics["frozen_commit"]),
        "frozen_tag": str(external_metrics["frozen_tag"]),
        "external_rows": int(len(external_predictions)),
        "external_hc": int(np.sum(labels == 0)),
        "external_pd": int(np.sum(labels == 1)),
        **{metric: float(external_metrics[metric]) for metric in METRIC_NAMES},
        "threshold": float(external_metrics["threshold"]),
        "ndd_rows": int(len(ndd_predictions)),
        "ndd_fraction_predicted_pd": float(ndd_summary["fraction_predicted_pd_at_0_5"]),
        "no_retraining_refit_rescoring_inside_gate": True,
        "chain_of_custody_and_artifact_integrity_only": True,
        "clinical_validation": False,
        "deployment_ready": False,
        "calibration_acceptable": False,
        "ndd_stress_test_only": True,
    }


def write_phase6_gate_report(output_path: str | Path, gate_summary: Mapping[str, Any]) -> None:
    """Write the final Phase 6 frozen external-validation integrity gate report."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Phase 6 Final Frozen External-Validation Gate",
                "",
                "This read-only gate validates chain of custody and internal consistency of the existing frozen bundle, one-time external outputs, NDD stress-test outputs, and scoring audit. No training, preprocessing fitting, inference, rescoring, threshold tuning, or model selection occurs inside the gate.",
                "",
                "## Gate Status",
                "",
                "- Status: `PASS`.",
                "- PASS means chain-of-custody and artifact integrity only.",
                "- PASS does not mean clinical validation.",
                "- PASS does not mean model deployment readiness.",
                "- PASS does not mean calibration is acceptable.",
                "- No clinical, diagnostic, deployment, or biological claim is made.",
                "",
                "## Frozen Chain of Custody",
                "",
                "- `HASH_BEFORE.txt` and `HASH_AFTER.txt` are exactly identical.",
                "- Both manifests independently verify the exact six-file immutable payload.",
                f"- Frozen commit: `{gate_summary['frozen_commit']}`.",
                f"- Frozen tag: `{gate_summary['frozen_tag']}`.",
                "- Frozen payload modified: `false`.",
                "",
                "## External PD/HC Artifact",
                "",
                f"- Rows: `{gate_summary['external_rows']}` (`{gate_summary['external_hc']}` HC, `{gate_summary['external_pd']}` PD).",
                f"- Fixed threshold: `{gate_summary['threshold']:.1f}`.",
                f"- AUROC: `{gate_summary['auroc']:.6f}`; this indicates ranking discrimination only and is not clinical validity.",
                f"- AUPRC: `{gate_summary['auprc']:.6f}`.",
                f"- Fixed-threshold balanced accuracy: `{gate_summary['balanced_accuracy']:.6f}`.",
                f"- Brier: `{gate_summary['brier']:.6f}`.",
                f"- ECE: `{gate_summary['ece']:.6f}`.",
                "- The fixed-threshold balanced accuracy and poor calibration remain explicit limitations.",
                "",
                "## Held-Out NDD Artifact",
                "",
                f"- Rows: `{gate_summary['ndd_rows']}`.",
                f"- Fraction predicted PD at 0.5: `{gate_summary['ndd_fraction_predicted_pd']:.6f}`.",
                "- NDD remains an unlabeled specificity/stress test only, not diagnostic validation.",
                "",
                "## Boundary Confirmation",
                "",
                "- No model retraining or weight update occurred inside the gate.",
                "- No preprocessing or scaler refit occurred inside the gate.",
                "- No external or NDD inference/rescoring occurred inside the gate.",
                "- No external metric was used for model selection.",
                "- No Phase 7 work was performed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
