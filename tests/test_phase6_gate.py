from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.metrics import compute_binary_metrics  # noqa: E402
from eval.phase6_gate import (  # noqa: E402
    EXPECTED_FROZEN_COMMIT,
    EXPECTED_FROZEN_TAG,
    load_phase6_outputs,
    summarize_phase6_gate,
    validate_external_metrics,
    validate_external_predictions,
    validate_frozen_hash_chain,
    validate_ndd_predictions,
    validate_ndd_summary,
    validate_scoring_audit,
    write_phase6_gate_report,
)
from models.frozen_bundle import (  # noqa: E402
    PAYLOAD_FILES,
    compute_bundle_hashes,
    write_hash_manifest,
)


def _external_predictions() -> pd.DataFrame:
    labels = np.array([0] * 22 + [1] * 50, dtype=np.int64)
    target_ece = 0.6944412781278375
    pd_probability = 1e-12
    total_probability = 72 * ((50 / 72) - target_ece)
    hc_probability = (total_probability - 50 * pd_probability) / 22
    probabilities = np.where(labels == 0, hc_probability, pd_probability)
    logits = np.log(probabilities / (1.0 - probabilities))
    return pd.DataFrame(
        {
            "sample_id": [f"synthetic-external-{index:02d}" for index in range(72)],
            "y_true": labels,
            "logit": logits,
            "y_prob": probabilities,
            "y_pred": (probabilities >= 0.5).astype(np.int64),
        }
    )


def _external_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = compute_binary_metrics(
        predictions["y_true"].to_numpy(),
        predictions["y_prob"].to_numpy(),
        threshold=0.5,
    )
    metrics.update(
        {
            "threshold": 0.5,
            "n_samples": 72,
            "n_hc": 22,
            "n_pd": 50,
            "frozen_commit": EXPECTED_FROZEN_COMMIT,
            "frozen_tag": EXPECTED_FROZEN_TAG,
            "hash_before_verified": True,
            "model_frozen": True,
            "external_scored_once": True,
        }
    )
    return metrics


def _ndd_predictions() -> pd.DataFrame:
    probabilities = np.linspace(0.01, 0.99, 48, dtype=np.float64)
    logits = np.log(probabilities / (1.0 - probabilities))
    return pd.DataFrame(
        {
            "sample_id": [f"synthetic-ndd-{index:02d}" for index in range(48)],
            "logit": logits,
            "y_prob": probabilities,
            "predicted_class": np.where(probabilities >= 0.5, "PD", "HC"),
        }
    )


def _ndd_summary(predictions: pd.DataFrame) -> dict[str, float | int]:
    probabilities = predictions["y_prob"].to_numpy(dtype=np.float64)
    predicted_pd = probabilities >= 0.5
    return {
        "n_samples": 48,
        "mean_pd_probability": float(np.mean(probabilities)),
        "median_pd_probability": float(np.median(probabilities)),
        "std_pd_probability": float(np.std(probabilities)),
        "min_pd_probability": float(np.min(probabilities)),
        "max_pd_probability": float(np.max(probabilities)),
        "fraction_predicted_pd_at_0_5": float(np.mean(predicted_pd)),
        "fraction_predicted_hc_at_0_5": float(np.mean(~predicted_pd)),
    }


def _scoring_audit() -> dict[str, object]:
    return {
        "execution_scope": "synthetic one-time frozen scoring audit fixture",
        "frozen_commit": EXPECTED_FROZEN_COMMIT,
        "frozen_tag": EXPECTED_FROZEN_TAG,
        "hash_before_manifest": "frozen/HASH_BEFORE.txt",
        "hash_before_verified": True,
        "external_input_paths": ["synthetic/external.npy"],
        "ndd_input_paths": ["synthetic/ndd.npy"],
        "frozen_payload_modified": False,
        "threshold": 0.5,
        "scaler_refit": False,
        "model_retrained": False,
        "external_metric_used_for_selection": False,
        "external_scored_once": True,
        "ndd_scored_once": True,
    }


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    frozen_dir = tmp_path / "frozen"
    results_dir = tmp_path / "results" / "external"
    frozen_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    for filename in PAYLOAD_FILES:
        (frozen_dir / filename).write_bytes(f"synthetic payload: {filename}\n".encode())
    write_hash_manifest(compute_bundle_hashes(frozen_dir), frozen_dir / "HASH_BEFORE.txt")
    (frozen_dir / "HASH_AFTER.txt").write_text(
        (frozen_dir / "HASH_BEFORE.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    external_predictions = _external_predictions()
    external_predictions.to_csv(results_dir / "external_predictions.csv", index=False)
    (results_dir / "external_metrics.json").write_text(
        json.dumps(_external_metrics(external_predictions)), encoding="utf-8"
    )
    ndd_predictions = _ndd_predictions()
    ndd_predictions.to_csv(results_dir / "ndd_predictions.csv", index=False)
    (results_dir / "ndd_specificity_summary.json").write_text(
        json.dumps(_ndd_summary(ndd_predictions)), encoding="utf-8"
    )
    (results_dir / "scoring_audit.json").write_text(
        json.dumps(_scoring_audit()), encoding="utf-8"
    )
    return frozen_dir, results_dir


def _validate_all(tmp_path: Path) -> dict[str, object]:
    frozen_dir, results_dir = _write_fixture(tmp_path)
    outputs = load_phase6_outputs(frozen_dir, results_dir)
    hash_chain = validate_frozen_hash_chain(frozen_dir)
    validate_external_predictions(outputs["external_predictions"])
    validate_external_metrics(outputs["external_metrics"], outputs["external_predictions"])
    validate_ndd_predictions(outputs["ndd_predictions"])
    validate_ndd_summary(outputs["ndd_summary"], outputs["ndd_predictions"])
    validate_scoring_audit(outputs["scoring_audit"], outputs["external_metrics"])
    summary = summarize_phase6_gate(
        hash_chain,
        outputs["external_predictions"],
        outputs["external_metrics"],
        outputs["ndd_predictions"],
        outputs["ndd_summary"],
        outputs["scoring_audit"],
    )
    write_phase6_gate_report(tmp_path / "phase6_gate.md", summary)
    return summary


def test_complete_valid_fixture_passes(tmp_path: Path) -> None:
    summary = _validate_all(tmp_path)
    assert summary["status"] == "PASS"
    assert summary["clinical_validation"] is False
    assert summary["deployment_ready"] is False


def test_detects_hash_before_after_mismatch(tmp_path: Path) -> None:
    frozen_dir, _ = _write_fixture(tmp_path)
    (frozen_dir / "HASH_AFTER.txt").write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must match exactly"):
        validate_frozen_hash_chain(frozen_dir)


def test_detects_altered_frozen_payload(tmp_path: Path) -> None:
    frozen_dir, _ = _write_fixture(tmp_path)
    (frozen_dir / "gene_space.txt").write_text("altered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen payload hash mismatch"):
        validate_frozen_hash_chain(frozen_dir)


def test_detects_wrong_external_row_count() -> None:
    predictions = _external_predictions().iloc[:-1].copy()
    with pytest.raises(ValueError, match="exactly 72 rows"):
        validate_external_predictions(predictions)


def test_detects_wrong_threshold_derived_y_pred() -> None:
    predictions = _external_predictions()
    predictions.loc[0, "y_pred"] = 1
    with pytest.raises(ValueError, match="y_prob >= 0.5"):
        validate_external_predictions(predictions)


def test_detects_metric_mismatch() -> None:
    predictions = _external_predictions()
    metrics = _external_metrics(predictions)
    metrics["auroc"] = 0.01
    with pytest.raises(ValueError, match="external metric mismatch for auroc"):
        validate_external_metrics(metrics, predictions)


def test_detects_wrong_ndd_predicted_class() -> None:
    predictions = _ndd_predictions()
    predictions.loc[0, "predicted_class"] = "PD"
    with pytest.raises(ValueError, match="exactly match the 0.5 threshold"):
        validate_ndd_predictions(predictions)


def test_detects_ndd_summary_mismatch() -> None:
    predictions = _ndd_predictions()
    summary = _ndd_summary(predictions)
    summary["mean_pd_probability"] += 0.01
    with pytest.raises(ValueError, match="NDD summary mismatch"):
        validate_ndd_summary(summary, predictions)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("scaler_refit", "scaler_refit must be false"),
        ("model_retrained", "model_retrained must be false"),
        (
            "external_metric_used_for_selection",
            "external_metric_used_for_selection must be false",
        ),
    ],
)
def test_detects_forbidden_scoring_action(field: str, message: str) -> None:
    audit = _scoring_audit()
    audit[field] = True
    with pytest.raises(ValueError, match=message):
        validate_scoring_audit(audit, _external_metrics(_external_predictions()))


@pytest.mark.parametrize("field", ["clinical_validation", "deployment_ready"])
def test_detects_forbidden_clinical_or_deployment_claim_metadata(field: str) -> None:
    metrics = _external_metrics(_external_predictions())
    metrics[field] = True
    with pytest.raises(ValueError, match="forbidden clinical/diagnostic/deployment/biological"):
        validate_external_metrics(metrics, _external_predictions())
