from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from interpret.integrated_gradients_attribution import (
    compute_pathway_integrated_gradients,
    run_integrated_gradients_attribution_cv,
)
from models.binn import BINNClassifier


def synthetic_inputs() -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray, list[str]]:
    rng = np.random.default_rng(41)
    X = rng.normal(size=(12, 4)).astype(np.float32)
    y = np.array([0, 1] * 6)
    folds = [
        {"train_indices": list(range(6, 12)), "validation_indices": list(range(6))},
        {"train_indices": list(range(6)), "validation_indices": list(range(6, 12))},
    ]
    mask = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32)
    return X, y, folds, mask, ["REACTOME_SPLICEOSOME", "REACTOME_CELL_CYCLE"]


def test_compute_pathway_integrated_gradients_returns_expected_shape() -> None:
    _, _, _, mask, _ = synthetic_inputs()
    model = BINNClassifier(mask, hidden_dim=3, dropout=0.0)

    attributions = compute_pathway_integrated_gradients(model, np.ones((3, 2), dtype=np.float32), n_steps=4)

    assert attributions.shape == (3, 2)


def test_integrated_gradients_attribution_cv_has_required_columns_and_audit() -> None:
    X, y, folds, mask, pathway_names = synthetic_inputs()
    scores, stability, audit = run_integrated_gradients_attribution_cv(
        X, y, folds, mask, pathway_names, seeds=(11, 23, 37),
        rna_processing_keywords=("SPLICEOSOME",), n_steps=4,
        hidden_dim=3, dropout=0.0, max_epochs=2, patience=1, batch_size=3,
    )

    assert list(scores.columns) == [
        "pathway_name", "method", "seed", "fold", "mean_score", "abs_mean_score", "rank",
        "is_rna_processing",
    ]
    assert list(stability.columns) == [
        "pathway_name", "method", "mean_rank", "rank_variance", "min_rank", "max_rank",
        "n_folds", "is_rna_processing",
    ]
    assert scores["method"].eq("integrated_gradients").all()
    assert scores.groupby(["seed", "fold"])["rank"].apply(lambda ranks: ranks.tolist() == [1, 2]).all()
    assert stability["n_folds"].eq(len(folds)).all()
    assert scores.loc[scores["pathway_name"] == "REACTOME_SPLICEOSOME", "is_rna_processing"].all()
    assert audit["n_steps"] == 4
    assert audit["max_masked_weight_after_training"] == 0.0
    assert audit["confirmation_no_external_ndd"] is True


def test_integrated_gradients_ranks_are_deterministic_per_seed_and_fold() -> None:
    X, y, folds, mask, pathway_names = synthetic_inputs()
    kwargs = dict(
        seeds=(11,), rna_processing_keywords=("SPLICEOSOME",), n_steps=4,
        hidden_dim=3, dropout=0.0, max_epochs=2, patience=1, batch_size=3,
    )
    first_scores, _, _ = run_integrated_gradients_attribution_cv(X, y, folds, mask, pathway_names, **kwargs)
    second_scores, _, _ = run_integrated_gradients_attribution_cv(X, y, folds, mask, pathway_names, **kwargs)

    assert first_scores[["seed", "fold", "pathway_name", "rank"]].equals(
        second_scores[["seed", "fold", "pathway_name", "rank"]]
    )
