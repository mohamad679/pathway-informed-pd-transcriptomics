from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.frozen_scoring import (
    apply_frozen_preprocessing,
    compute_external_metrics,
    load_and_verify_frozen_bundle,
    score_frozen_model,
    summarize_ndd_specificity,
)
from models.binn import BINNClassifier
from models.frozen_bundle import compute_bundle_hashes, save_frozen_bundle, write_hash_manifest


class _FirstFeatureModel(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs[:, 0]


def test_load_and_verify_reconstructs_synthetic_frozen_bundle(tmp_path: Path) -> None:
    mask = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    model = BINNClassifier(mask, hidden_dim=3, dropout=0.0)
    scaler = StandardScaler().fit(
        np.array([[0.0, 2.0], [2.0, 6.0], [4.0, 10.0]], dtype=np.float32)
    )
    training_metadata = {
        "hidden_dim": 3,
        "dropout": 0.0,
        "n_genes": 2,
        "n_pathways": 2,
        "seed": 11,
        "n_epochs": 4,
    }
    save_frozen_bundle(
        frozen_dir=tmp_path,
        model=model,
        scaler=scaler,
        gene_space=["g1", "g2"],
        pathway_names=["p1", "p2"],
        pathway_mask=mask,
        training_metadata=training_metadata,
        hidden_dim=3,
        dropout=0.0,
        seed=11,
        n_epochs=4,
    )
    write_hash_manifest(compute_bundle_hashes(tmp_path), tmp_path / "HASH_BEFORE.txt")

    loaded_model, scaler_parameters, metadata = load_and_verify_frozen_bundle(tmp_path)

    assert loaded_model.training is False
    assert scaler_parameters["n_features_in_"] == 2
    assert metadata["gene_space"] == ["g1", "g2"]
    assert metadata["pathway_names"] == ["p1", "p2"]
    assert metadata["hash_before_verified"] is True
    assert metadata["max_abs_masked_weight"] == 0.0


def test_frozen_preprocessing_reproduces_manual_scaling() -> None:
    X = np.array([[3.0, 10.0], [5.0, 14.0]], dtype=np.float64)
    scaler = {"mean_": [1.0, 6.0], "scale_": [2.0, 4.0]}

    transformed = apply_frozen_preprocessing(X, scaler, expected_gene_count=2)

    expected = ((X - np.array([1.0, 6.0])) / np.array([2.0, 4.0])).astype(np.float32)
    np.testing.assert_allclose(transformed, expected)
    assert transformed.dtype == np.float32


def test_frozen_preprocessing_wrong_feature_count_fails() -> None:
    with pytest.raises(ValueError, match="feature count"):
        apply_frozen_preprocessing(
            np.ones((3, 2), dtype=np.float32),
            {"mean_": [0.0, 0.0], "scale_": [1.0, 1.0]},
            expected_gene_count=3,
        )


@pytest.mark.parametrize("bad_scale", [0.0, np.inf, np.nan])
def test_frozen_preprocessing_zero_or_nonfinite_scale_fails(bad_scale: float) -> None:
    with pytest.raises(ValueError, match="scale_"):
        apply_frozen_preprocessing(
            np.ones((2, 2), dtype=np.float32),
            {"mean_": [0.0, 0.0], "scale_": [1.0, bad_scale]},
            expected_gene_count=2,
        )


def test_model_scoring_preserves_row_count_and_order() -> None:
    X_scaled = np.array([[2.0, 0.0], [-1.0, 5.0], [0.5, 8.0]], dtype=np.float32)

    logits, probabilities = score_frozen_model(
        _FirstFeatureModel(), X_scaled, batch_size=2, device="cpu"
    )

    np.testing.assert_allclose(logits, X_scaled[:, 0])
    np.testing.assert_allclose(
        probabilities,
        1.0 / (1.0 + np.exp(-X_scaled[:, 0])),
        rtol=1e-6,
    )
    assert logits.shape == probabilities.shape == (X_scaled.shape[0],)


def test_external_metrics_return_required_fields() -> None:
    metrics = compute_external_metrics(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
    )

    assert set(metrics) == {"auroc", "auprc", "balanced_accuracy", "brier", "ece"}
    assert all(np.isfinite(value) for value in metrics.values())


def test_ndd_summary_returns_required_fields() -> None:
    summary = summarize_ndd_specificity(np.array([0.1, 0.5, 0.8]))

    assert set(summary) == {
        "n_samples",
        "mean_pd_probability",
        "median_pd_probability",
        "std_pd_probability",
        "min_pd_probability",
        "max_pd_probability",
        "fraction_predicted_pd_at_0_5",
        "fraction_predicted_hc_at_0_5",
    }
    assert summary["n_samples"] == 3


def test_threshold_is_exactly_point_five() -> None:
    metrics = compute_external_metrics(np.array([0, 1]), np.array([0.5, 0.49]))
    summary = summarize_ndd_specificity(np.array([0.5]))

    assert metrics["balanced_accuracy"] == 0.0
    assert summary["fraction_predicted_pd_at_0_5"] == 1.0
    with pytest.raises(ValueError, match="exactly 0.5"):
        summarize_ndd_specificity(np.array([0.5]), threshold=0.51)
