from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import eval.permutation as permutation
from eval.permutation import (
    empirical_p_value,
    generate_permuted_labels,
    run_label_permutation_binn_cv,
)


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


def test_generate_permuted_labels_is_deterministic_for_same_index() -> None:
    y = np.array([0, 1] * 10)

    first = generate_permuted_labels(y, permutation_index=7, seed=20260710)
    second = generate_permuted_labels(y, permutation_index=7, seed=20260710)

    assert np.array_equal(first, second)
    assert np.array_equal(y, np.array([0, 1] * 10))


def test_generate_permuted_labels_changes_with_different_index() -> None:
    y = np.array([0, 1] * 10)

    first = generate_permuted_labels(y, permutation_index=1, seed=20260710)
    second = generate_permuted_labels(y, permutation_index=2, seed=20260710)

    assert not np.array_equal(first, second)


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


def test_run_label_permutation_binn_cv_uses_start_permutation_index(monkeypatch) -> None:
    X, y, folds, mask = synthetic_inputs()

    def fake_run_binn_cv(
        X: np.ndarray,
        y: np.ndarray,
        folds: list[dict[str, object]],
        pathway_mask: np.ndarray,
        **kwargs: object,
    ) -> tuple[None, pd.DataFrame]:
        return None, pd.DataFrame({"y_true": y, "y_prob": np.linspace(0.1, 0.9, len(y))})

    monkeypatch.setattr(permutation, "run_binn_cv", fake_run_binn_cv)

    null_df = run_label_permutation_binn_cv(
        X,
        y,
        folds,
        mask,
        n_permutations=3,
        seed=20260710,
        start_permutation_index=5,
    )

    assert null_df["permutation_index"].tolist() == [5, 6, 7]


def test_run_label_permutation_binn_cv_skips_append_existing(monkeypatch) -> None:
    X, y, folds, mask = synthetic_inputs()
    call_count = 0

    def fake_run_binn_cv(
        X: np.ndarray,
        y: np.ndarray,
        folds: list[dict[str, object]],
        pathway_mask: np.ndarray,
        **kwargs: object,
    ) -> tuple[None, pd.DataFrame]:
        nonlocal call_count
        call_count += 1
        return None, pd.DataFrame({"y_true": y, "y_prob": np.linspace(0.1, 0.9, len(y))})

    monkeypatch.setattr(permutation, "run_binn_cv", fake_run_binn_cv)
    existing_df = pd.DataFrame({"permutation_index": [3], "null_auroc": [0.42]})

    null_df = run_label_permutation_binn_cv(
        X,
        y,
        folds,
        mask,
        n_permutations=3,
        seed=20260710,
        start_permutation_index=2,
        append_existing=existing_df,
    )

    assert null_df["permutation_index"].tolist() == [2, 3, 4]
    assert null_df.loc[null_df["permutation_index"] == 3, "null_auroc"].item() == 0.42
    assert call_count == 2
