from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.phase1_gate import (
    compute_dev_pca,
    validate_folds_cover_development_once,
    validate_processed_shapes,
    validate_zscore_sanity,
)


def test_shape_validation_accepts_expected_phase1_shapes() -> None:
    gene_count = 7
    dev_X = np.zeros((438, gene_count), dtype=np.float32)
    dev_y = np.array([0] * 233 + [1] * 205, dtype=np.int64)
    dev_sample_ids = [f"dev_{index}" for index in range(438)]
    ndd_X = np.zeros((48, gene_count), dtype=np.float32)
    ndd_sample_ids = [f"ndd_{index}" for index in range(48)]
    ext_X = np.zeros((72, gene_count), dtype=np.float32)
    ext_y = np.array([0] * 22 + [1] * 50, dtype=np.int64)
    ext_sample_ids = [f"ext_{index}" for index in range(72)]
    gene_space = [f"gene_{index}" for index in range(gene_count)]

    summary = validate_processed_shapes(
        dev_X,
        dev_y,
        dev_sample_ids,
        ndd_X,
        ndd_sample_ids,
        ext_X,
        ext_y,
        ext_sample_ids,
        gene_space,
    )

    assert summary["gene_count"] == gene_count
    assert summary["dev_shape"] == (438, gene_count)
    assert summary["ext_label_counts"] == {"HC": 22, "PD": 50}


def test_zscore_sanity_accepts_rowwise_standardized_matrix() -> None:
    X = np.array(
        [
            [-1.3416408, -0.4472136, 0.4472136, 1.3416408],
            [-1.3416408, -0.4472136, 0.4472136, 1.3416408],
        ],
        dtype=np.float32,
    )

    summary = validate_zscore_sanity(X, dataset_name="synthetic")

    assert summary["max_abs_mean"] < 1e-6
    assert summary["max_abs_std_delta"] < 1e-6


def test_fold_validation_catches_duplicate_validation_coverage() -> None:
    sample_ids = ["s0", "s1", "s2", "s3"]
    folds = [
        {
            "fold": 1,
            "train_indices": [2, 3],
            "validation_indices": [0, 1],
            "train_sample_ids": ["s2", "s3"],
            "validation_sample_ids": ["s0", "s1"],
        },
        {
            "fold": 2,
            "train_indices": [1, 3],
            "validation_indices": [0, 2],
            "train_sample_ids": ["s1", "s3"],
            "validation_sample_ids": ["s0", "s2"],
        },
    ]

    with pytest.raises(ValueError, match="appear in more than one fold"):
        validate_folds_cover_development_once(folds, sample_ids)


def test_pca_output_shape() -> None:
    dev_X = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )

    summary = compute_dev_pca(dev_X)

    assert summary["coordinates"].shape == (4, 2)
    assert summary["explained_variance_ratio"].shape == (2,)
