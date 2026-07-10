from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interpret.attribution_agreement import (
    compute_global_stability_agreement,
    compute_seed_fold_agreement,
    summarize_rna_processing_tier,
    validate_attribution_scores,
    validate_attribution_stability,
)


SEEDS = [11, 23, 37]
FOLDS = [1, 2, 3, 4, 5]
PATHWAYS = ["RNA_A", "RNA_B", "OTHER_C", "OTHER_D"]
RNA_FLAGS = [True, True, False, False]


def synthetic_scores(method: str = "activation") -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        for fold in FOLDS:
            for rank, (pathway_name, is_rna_processing) in enumerate(zip(PATHWAYS, RNA_FLAGS), start=1):
                rows.append(
                    {
                        "pathway_name": pathway_name,
                        "method": method,
                        "seed": seed,
                        "fold": fold,
                        "mean_score": float(5 - rank),
                        "abs_mean_score": float(5 - rank),
                        "rank": rank,
                        "is_rna_processing": is_rna_processing,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_stability(method: str = "activation") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pathway_name": PATHWAYS,
            "method": [method] * len(PATHWAYS),
            "mean_rank": [1.0, 2.0, 3.0, 4.0],
            "rank_variance": [0.0, 0.5, 0.25, 0.0],
            "min_rank": [1, 1, 2, 4],
            "max_rank": [1, 3, 4, 4],
            "n_folds": [5, 5, 5, 5],
            "is_rna_processing": RNA_FLAGS,
        }
    )


def test_validation_passes_for_correct_synthetic_scores_and_stability() -> None:
    validate_attribution_scores(synthetic_scores("activation"), "activation")
    validate_attribution_scores(synthetic_scores("integrated_gradients"), "integrated_gradients")
    validate_attribution_stability(synthetic_stability("activation"), "activation")
    validate_attribution_stability(
        synthetic_stability("integrated_gradients"), "integrated_gradients"
    )


def test_validation_catches_wrong_method() -> None:
    with pytest.raises(ValueError, match="Method must be exactly activation"):
        validate_attribution_scores(synthetic_scores("integrated_gradients"), "activation")


def test_seed_fold_agreement_computes_spearman_and_topk_overlap() -> None:
    activation = pd.DataFrame(
        {
            "pathway_name": ["A", "B", "C"],
            "seed": [11, 11, 11],
            "fold": [1, 1, 1],
            "rank": [1, 2, 3],
        }
    )
    ig = pd.DataFrame(
        {
            "pathway_name": ["A", "B", "C"],
            "seed": [11, 11, 11],
            "fold": [1, 1, 1],
            "rank": [3, 2, 1],
        }
    )

    agreement = compute_seed_fold_agreement(activation, ig, k=2)

    assert agreement.shape == (1, 5)
    row = agreement.iloc[0]
    assert row["spearman_rank_correlation"] == -1.0
    assert row["top20_overlap"] == 1
    assert row["n_pathways"] == 3


def test_global_stability_agreement_computes_spearman_and_topk_overlap() -> None:
    activation = synthetic_stability("activation")
    ig = synthetic_stability("integrated_gradients")
    ig["mean_rank"] = [2.0, 1.0, 3.0, 4.0]

    agreement = compute_global_stability_agreement(activation, ig, k=2)

    row = agreement.iloc[0]
    assert row["spearman_rank_correlation"] == pytest.approx(0.8)
    assert row["top20_overlap"] == 2
    assert row["activation_top20_rna_count"] == 2
    assert row["ig_top20_rna_count"] == 2
    assert row["n_pathways"] == 4


def test_rna_processing_tier_summary_includes_only_rna_processing_pathways() -> None:
    activation = synthetic_stability("activation")
    ig = synthetic_stability("integrated_gradients")

    summary = summarize_rna_processing_tier(activation, ig)

    assert summary["pathway_name"].tolist() == ["RNA_A", "RNA_B"]
    assert list(summary.columns) == [
        "pathway_name",
        "activation_mean_rank",
        "ig_mean_rank",
        "activation_min_rank",
        "ig_min_rank",
        "activation_max_rank",
        "ig_max_rank",
        "activation_top20",
        "ig_top20",
    ]


def test_missing_columns_fail() -> None:
    missing_rank = synthetic_scores("activation").drop(columns=["rank"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_attribution_scores(missing_rank, "activation")
