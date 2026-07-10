from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from interpret.pathway_attribution import (
    aggregate_pathway_signal,
    classify_rna_processing_pathway,
    compute_spearman_rank_correlation,
    compute_topk_overlap,
    load_rna_processing_keywords,
    rank_pathways,
    summarize_fold_stability,
)


def test_rank_pathways_is_deterministic_and_breaks_ties_by_name() -> None:
    ranked = rank_pathways(
        pd.DataFrame({"pathway_name": ["Z_PATH", "A_PATH", "B_PATH"], "abs_mean_score": [1.0, 1.0, 2.0]})
    )

    assert ranked["pathway_name"].tolist() == ["B_PATH", "A_PATH", "Z_PATH"]
    assert ranked["rank"].tolist() == [1, 2, 3]


def test_compute_topk_overlap() -> None:
    ranking_a = pd.DataFrame({"pathway_name": ["A", "B", "C"], "rank": [1, 2, 3]})
    ranking_b = pd.DataFrame({"pathway_name": ["B", "D", "A"], "rank": [1, 2, 3]})

    assert compute_topk_overlap(ranking_a, ranking_b, k=2) == 1


def test_compute_spearman_rank_correlation() -> None:
    ranking_a = pd.DataFrame({"pathway_name": ["A", "B", "C"], "rank": [1, 2, 3]})
    ranking_b = pd.DataFrame({"pathway_name": ["A", "B", "C"], "rank": [3, 2, 1]})

    assert compute_spearman_rank_correlation(ranking_a, ranking_b) == -1.0


def test_summarize_fold_stability() -> None:
    summary = summarize_fold_stability(
        pd.DataFrame(
            {
                "pathway_name": ["A", "A", "B", "B"],
                "method": ["activation"] * 4,
                "fold": [1, 2, 1, 2],
                "rank": [1, 3, 2, 1],
            }
        )
    )

    a_row = summary.loc[summary["pathway_name"] == "A"].iloc[0]
    assert list(summary.columns) == ["pathway_name", "method", "mean_rank", "rank_variance", "min_rank", "max_rank", "n_folds"]
    assert a_row[["mean_rank", "rank_variance", "min_rank", "max_rank", "n_folds"]].tolist() == [2.0, 2.0, 1, 3, 2]


def test_rna_processing_classification_uses_configured_keywords(tmp_path: Path) -> None:
    config_path = tmp_path / "pathways.yaml"
    config_path.write_text("msigdb:\n  rna_processing_keywords:\n    - SPLICEOSOME\n", encoding="utf-8")

    keywords = load_rna_processing_keywords(config_path)

    assert keywords == ["SPLICEOSOME"]
    assert classify_rna_processing_pathway("REACTOME_SPLICEOSOME", keywords)
    assert not classify_rna_processing_pathway("REACTOME_CELL_CYCLE", keywords)


def test_aggregate_pathway_signal_returns_one_row_per_pathway() -> None:
    aggregated = aggregate_pathway_signal(
        np.array([[1.0, -2.0], [3.0, 2.0]]), ["A", "B"], fold=1, seed=42, method="activation"
    )

    assert list(aggregated.columns) == ["pathway_name", "method", "seed", "fold", "mean_score", "abs_mean_score"]
    assert aggregated.shape == (2, 6)
    assert aggregated["mean_score"].tolist() == [2.0, 0.0]
    assert aggregated["abs_mean_score"].tolist() == [2.0, 0.0]
