from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.permutation import empirical_p_value, run_label_permutation_binn_cv


def synthetic_inputs() -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
    rng = np.random.default_rng(5)
    X = rng.normal(size=(20, 4)).astype(np.float32)
    y = np.array([0, 1] * 10)
    folds = [
        {"train_indices": list(range(10, 20)), "validation_indices": list(range(10))},
        {"train_indices": list(range(10)), "validation_indices": list(range(10, 20))},
    ]
    mask = np.array([[1, 0, 1, 0], [0, 1, 1, 0]], dtype=np.float32)
    return X, y, folds, mask


def test_empirical_p_value_uses_requested_formula() -> None:
    observed = 0.7
    null_scores = np.array([0.4, 0.7, 0.8])

    assert empirical_p_value(observed, null_scores) == 3 / 4


def test_run_label_permutation_binn_cv_returns_requested_null_rows() -> None:
    X, y, folds, mask = synthetic_inputs()

    null_df = run_label_permutation_binn_cv(
        X,
        y,
        folds,
        mask,
        n_permutations=2,
        seed=20260710,
        training_kwargs={
            "hidden_dim": 4,
            "dropout": 0.0,
            "max_epochs": 2,
            "patience": 1,
            "batch_size": 4,
        },
    )

    assert len(null_df) == 2
    assert list(null_df.columns) == ["permutation_index", "null_auroc"]
    assert np.isfinite(null_df["null_auroc"]).all()
    assert not {"external", "ndd", "held_out_ndd"} & set(null_df.columns)
