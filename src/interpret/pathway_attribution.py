"""Development-only utilities for Phase 4 pathway-attribution summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


def _load_nonempty_lines(path: str | Path) -> list[str]:
    """Load an ordered newline-delimited name file."""
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_pathway_names(path: str | Path) -> list[str]:
    """Load ordered pathway names from the fixed pathway-name file."""
    return _load_nonempty_lines(path)


def load_gene_names(path: str | Path) -> list[str]:
    """Load ordered gene names from the fixed gene-space file."""
    return _load_nonempty_lines(path)


def load_rna_processing_keywords(config_path: str | Path) -> list[str]:
    """Load configured RNA-processing pathway keywords."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    try:
        keywords = config["msigdb"]["rna_processing_keywords"]
    except (KeyError, TypeError) as error:
        raise ValueError("Config must define msigdb.rna_processing_keywords.") from error
    if not isinstance(keywords, list) or not all(isinstance(keyword, str) for keyword in keywords):
        raise ValueError("msigdb.rna_processing_keywords must be a list of strings.")
    return [keyword.upper() for keyword in keywords]


def classify_rna_processing_pathway(pathway_name: str, keywords: Sequence[str]) -> bool:
    """Classify a pathway by case-insensitive configured keyword matching."""
    normalized_name = pathway_name.upper()
    return any(str(keyword).upper() in normalized_name for keyword in keywords)


def aggregate_pathway_signal(
    values_by_sample: np.ndarray | Sequence[Sequence[float]],
    pathway_names: Sequence[str],
    fold: int,
    seed: int,
    method: str,
) -> pd.DataFrame:
    """Aggregate per-sample pathway values into one row per pathway."""
    values = np.asarray(values_by_sample, dtype=float)
    if values.ndim != 2:
        raise ValueError("values_by_sample must have shape (n_samples, n_pathways).")
    if values.shape[1] != len(pathway_names):
        raise ValueError("pathway_names length must match values_by_sample n_pathways.")
    if values.shape[0] == 0:
        raise ValueError("values_by_sample must contain at least one sample.")

    mean_scores = values.mean(axis=0)
    return pd.DataFrame(
        {
            "pathway_name": list(pathway_names),
            "method": method,
            "seed": seed,
            "fold": fold,
            "mean_score": mean_scores,
            "abs_mean_score": np.abs(mean_scores),
        }
    )


def rank_pathways(df: pd.DataFrame, score_column: str = "abs_mean_score") -> pd.DataFrame:
    """Rank pathways by descending score, breaking ties by pathway name."""
    required_columns = {"pathway_name", score_column}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}.")
    ranked = df.sort_values(
        [score_column, "pathway_name"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked


def _topk_pathway_names(ranking: pd.DataFrame, k: int) -> set[str]:
    if k <= 0:
        raise ValueError("k must be positive.")
    if "pathway_name" not in ranking.columns:
        raise ValueError("Ranking must contain a pathway_name column.")
    ordered = ranking.sort_values("rank", kind="mergesort") if "rank" in ranking else ranking
    return set(ordered["pathway_name"].head(k))


def compute_topk_overlap(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame, k: int = 20) -> int:
    """Return the number of pathways shared by the two top-k rankings."""
    return len(_topk_pathway_names(ranking_a, k).intersection(_topk_pathway_names(ranking_b, k)))


def compute_spearman_rank_correlation(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame) -> float:
    """Compute Spearman agreement between pathway rankings on shared pathways."""
    for ranking in (ranking_a, ranking_b):
        if not {"pathway_name", "rank"}.issubset(ranking.columns):
            raise ValueError("Each ranking must contain pathway_name and rank columns.")
    merged = ranking_a[["pathway_name", "rank"]].merge(
        ranking_b[["pathway_name", "rank"]], on="pathway_name", suffixes=("_a", "_b"), validate="one_to_one"
    )
    if len(merged) < 2:
        raise ValueError("At least two shared pathways are required for Spearman correlation.")
    return float(spearmanr(merged["rank_a"], merged["rank_b"]).statistic)


def summarize_fold_stability(ranked_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize pathway rank stability across development folds by method."""
    required_columns = {"pathway_name", "method", "fold", "rank"}
    missing_columns = required_columns.difference(ranked_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}.")
    return (
        ranked_df.groupby(["pathway_name", "method"], as_index=False, sort=True)
        .agg(
            mean_rank=("rank", "mean"),
            rank_variance=("rank", "var"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["method", "pathway_name"], kind="mergesort")
        .reset_index(drop=True)
    )
