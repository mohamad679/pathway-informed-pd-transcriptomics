from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase4_gate import (
    EXPECTED_FOLDS,
    EXPECTED_PATHWAYS,
    EXPECTED_RNA_PROCESSING_PATHWAYS,
    EXPECTED_SEEDS,
    validate_activation_outputs,
    validate_agreement_outputs,
    validate_ig_outputs,
)


PATHWAYS = [
    *(f"RNA_{index}" for index in range(1, EXPECTED_RNA_PROCESSING_PATHWAYS + 1)),
    *(f"PATHWAY_{index}" for index in range(1, EXPECTED_PATHWAYS - EXPECTED_RNA_PROCESSING_PATHWAYS + 1)),
]
RNA_FLAGS = {
    pathway: pathway.startswith("RNA_")
    for pathway in PATHWAYS
}


def _scores(method: str = "activation") -> pd.DataFrame:
    rows = []
    for seed in sorted(EXPECTED_SEEDS):
        for fold in sorted(EXPECTED_FOLDS):
            for rank, pathway_name in enumerate(PATHWAYS, start=1):
                rows.append(
                    {
                        "pathway_name": pathway_name,
                        "method": method,
                        "seed": seed,
                        "fold": fold,
                        "mean_score": float(rank) / EXPECTED_PATHWAYS,
                        "abs_mean_score": float(rank) / EXPECTED_PATHWAYS,
                        "rank": rank,
                        "is_rna_processing": RNA_FLAGS[pathway_name],
                    }
                )
    return pd.DataFrame(rows)


def _stability(method: str = "activation") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pathway_name": pathway_name,
                "method": method,
                "mean_rank": float(rank),
                "rank_variance": 0.0,
                "min_rank": rank,
                "max_rank": rank,
                "n_folds": 5,
                "is_rna_processing": RNA_FLAGS[pathway_name],
            }
            for rank, pathway_name in enumerate(PATHWAYS, start=1)
        ]
    )


def _seed_fold_agreement() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "fold": fold,
                "spearman_rank_correlation": 0.8,
                "top20_overlap": 12,
                "n_pathways": EXPECTED_PATHWAYS,
            }
            for seed in sorted(EXPECTED_SEEDS)
            for fold in sorted(EXPECTED_FOLDS)
        ]
    )


def _global_agreement() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "spearman_rank_correlation": 0.85,
                "top20_overlap": 14,
                "activation_top20_rna_count": 2,
                "ig_top20_rna_count": 2,
                "n_pathways": EXPECTED_PATHWAYS,
            }
        ]
    )


def _rna_processing_tier() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pathway_name": f"RNA_{index}",
                "activation_mean_rank": float(index),
                "ig_mean_rank": float(index + 1),
                "activation_min_rank": index,
                "ig_min_rank": index,
                "activation_max_rank": index + 10,
                "ig_max_rank": index + 11,
                "activation_top20": True,
                "ig_top20": True,
            }
            for index in range(1, EXPECTED_RNA_PROCESSING_PATHWAYS + 1)
        ]
    )


def test_complete_synthetic_phase4_fixture_passes() -> None:
    validate_activation_outputs(_scores("activation"), _stability("activation"))
    validate_ig_outputs(_scores("integrated_gradients"), _stability("integrated_gradients"))
    validate_agreement_outputs(_seed_fold_agreement(), _global_agreement(), _rna_processing_tier())


def test_catches_missing_required_score_columns() -> None:
    scores = _scores("activation").drop(columns=["rank"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_activation_outputs(scores, _stability("activation"))


def test_catches_wrong_method() -> None:
    with pytest.raises(ValueError, match="method must be exactly activation"):
        validate_activation_outputs(_scores("integrated_gradients"), _stability("activation"))


def test_catches_wrong_rank_coverage() -> None:
    scores = _scores("activation")
    scores.loc[(scores["seed"] == 11) & (scores["fold"] == 1) & (scores["rank"] == 1), "rank"] = 2
    with pytest.raises(ValueError, match="ranks for seed 11 fold 1 must be exactly"):
        validate_activation_outputs(scores, _stability("activation"))


def test_catches_invalid_spearman_range() -> None:
    seed_fold_agreement = _seed_fold_agreement()
    seed_fold_agreement.loc[0, "spearman_rank_correlation"] = 1.2
    with pytest.raises(ValueError, match="Spearman values must be finite and in"):
        validate_agreement_outputs(seed_fold_agreement, _global_agreement(), _rna_processing_tier())


def test_catches_wrong_rna_processing_tier_row_count() -> None:
    with pytest.raises(ValueError, match="RNA-processing tier must contain exactly"):
        validate_agreement_outputs(_seed_fold_agreement(), _global_agreement(), _rna_processing_tier().iloc[:-1])


def test_catches_forbidden_external_ndd_column() -> None:
    scores = _scores("activation")
    scores["external_cohort_score"] = 0.0
    with pytest.raises(ValueError, match="external/NDD columns"):
        validate_activation_outputs(scores, _stability("activation"))
