from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase3_gate import (
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    MODEL_NAME,
    summarize_phase3_gate,
    validate_binn_cv,
    validate_binn_oof,
    validate_mask_integrity,
    validate_pathway_mask,
)


def _mask() -> np.ndarray:
    return np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)


def _metrics(mask: np.ndarray) -> pd.DataFrame:
    rows = []
    for seed in sorted(EXPECTED_SEEDS):
        for fold in sorted(EXPECTED_FOLDS):
            rows.append(
                {
                    "model": MODEL_NAME,
                    "seed": seed,
                    "fold": fold,
                    "auroc": 0.7,
                    "auprc": 0.7,
                    "balanced_accuracy": 0.7,
                    "brier": 0.2,
                    "ece": 0.1,
                    "best_epoch": 2,
                    "n_epochs_run": 4,
                    "best_validation_loss": 0.5,
                    "max_abs_masked_weight_after_training": 0.0,
                    "n_masked_weights": mask.size - np.count_nonzero(mask),
                    "n_unmasked_weights": np.count_nonzero(mask),
                }
            )
    return pd.DataFrame(rows)


def _oof() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": MODEL_NAME,
                "seed": seed,
                "fold": (sample_index % 5) + 1,
                "sample_index": sample_index,
                "y_true": sample_index % 2,
                "y_prob": 0.7,
            }
            for seed in sorted(EXPECTED_SEEDS)
            for sample_index in range(438)
        ]
    )


def test_validates_correct_synthetic_mask_dimensions() -> None:
    validate_pathway_mask(_mask(), ["pathway_a", "pathway_b"], ["g1", "g2", "g3"])


def test_catches_wrong_pathway_names_count() -> None:
    with pytest.raises(ValueError, match="must match pathway/gene dimensions"):
        validate_pathway_mask(_mask(), ["pathway_a"], ["g1", "g2", "g3"])


def test_catches_missing_binn_cv_rows() -> None:
    with pytest.raises(ValueError, match="exactly 15 metric rows"):
        validate_binn_cv(_metrics(_mask()).iloc[:-1])


def test_catches_nonzero_masked_weight() -> None:
    metrics = _metrics(_mask())
    metrics.loc[0, "max_abs_masked_weight_after_training"] = 0.01
    with pytest.raises(ValueError, match="must be exactly 0.0"):
        validate_mask_integrity(metrics, _mask())


def test_catches_wrong_oof_row_coverage() -> None:
    oof = _oof()
    oof.loc[0, "sample_index"] = 1
    with pytest.raises(ValueError, match="unique sample_index values exactly once"):
        validate_binn_oof(oof)


def test_complete_small_synthetic_gate_fixture_passes() -> None:
    mask = _mask()
    metrics = _metrics(mask)
    oof = _oof()
    validate_pathway_mask(mask, ["pathway_a", "pathway_b"], ["g1", "g2", "g3"])
    validate_binn_cv(metrics)
    validate_binn_oof(oof)
    validate_mask_integrity(metrics, mask)
    summary = summarize_phase3_gate(metrics, mask, oof)
    assert summary["mask_nnz"] == 3
    assert summary["oof_rows"] == 1_314
