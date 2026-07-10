"""Development-only agreement summaries for Phase 4 attribution outputs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from interpret.pathway_attribution import compute_spearman_rank_correlation, compute_topk_overlap


EXPECTED_SEEDS = {11, 23, 37}
EXPECTED_FOLDS = {1, 2, 3, 4, 5}
SCORE_COLUMNS = {
    "pathway_name",
    "method",
    "seed",
    "fold",
    "mean_score",
    "abs_mean_score",
    "rank",
    "is_rna_processing",
}
STABILITY_COLUMNS = {
    "pathway_name",
    "method",
    "mean_rank",
    "rank_variance",
    "min_rank",
    "max_rank",
    "n_folds",
    "is_rna_processing",
}
ALLOWED_METHODS = {"activation", "integrated_gradients"}


@dataclass(frozen=True)
class AttributionOutputs:
    """Container for the four existing development attribution CSVs."""

    activation_scores: pd.DataFrame
    activation_stability: pd.DataFrame
    ig_scores: pd.DataFrame
    ig_stability: pd.DataFrame

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.activation_scores
        yield self.activation_stability
        yield self.ig_scores
        yield self.ig_stability

    def __getitem__(self, key: str) -> pd.DataFrame:
        return getattr(self, key)


def load_attribution_outputs(results_dir: str | Path) -> AttributionOutputs:
    """Load existing development activation and Integrated Gradients attribution outputs."""
    base = Path(results_dir)
    return AttributionOutputs(
        activation_scores=pd.read_csv(base / "pathway_activation_scores.csv"),
        activation_stability=pd.read_csv(base / "pathway_activation_stability.csv"),
        ig_scores=pd.read_csv(base / "pathway_ig_scores.csv"),
        ig_stability=pd.read_csv(base / "pathway_ig_stability.csv"),
    )


def validate_attribution_scores(scores_df: pd.DataFrame, expected_method: str) -> None:
    """Validate a development-only per-seed/fold attribution score table."""
    _validate_expected_method(expected_method)
    _validate_required_columns(scores_df, SCORE_COLUMNS)
    _validate_no_external_ndd_columns(scores_df)
    _validate_method(scores_df, expected_method)
    _validate_integer_column(scores_df, "seed")
    _validate_integer_column(scores_df, "fold")
    _validate_integer_column(scores_df, "rank")

    seeds = set(pd.to_numeric(scores_df["seed"]).astype(int))
    folds = set(pd.to_numeric(scores_df["fold"]).astype(int))
    if seeds != EXPECTED_SEEDS:
        raise ValueError(f"Seeds must be exactly {sorted(EXPECTED_SEEDS)}; found {sorted(seeds)}.")
    if folds != EXPECTED_FOLDS:
        raise ValueError(f"Folds must be exactly {sorted(EXPECTED_FOLDS)}; found {sorted(folds)}.")

    observed_pairs = set(
        map(tuple, scores_df[["seed", "fold"]].drop_duplicates().astype(int).to_numpy())
    )
    expected_pairs = {(seed, fold) for seed in EXPECTED_SEEDS for fold in EXPECTED_FOLDS}
    if observed_pairs != expected_pairs:
        raise ValueError("Scores must contain every predefined seed/fold pair exactly.")
    if scores_df.duplicated(["seed", "fold", "pathway_name"]).any():
        raise ValueError("Scores must contain at most one row per seed/fold/pathway.")


def validate_attribution_stability(stability_df: pd.DataFrame, expected_method: str) -> None:
    """Validate a development-only attribution stability table."""
    _validate_expected_method(expected_method)
    _validate_required_columns(stability_df, STABILITY_COLUMNS)
    _validate_no_external_ndd_columns(stability_df)
    _validate_method(stability_df, expected_method)
    _validate_positive_finite_column(stability_df, "mean_rank")
    _validate_nonnegative_finite_column(stability_df, "rank_variance")
    _validate_integer_column(stability_df, "min_rank")
    _validate_integer_column(stability_df, "max_rank")
    _validate_integer_column(stability_df, "n_folds")
    n_folds = set(pd.to_numeric(stability_df["n_folds"]).astype(int))
    if n_folds != {len(EXPECTED_FOLDS)}:
        raise ValueError(f"n_folds must be exactly {len(EXPECTED_FOLDS)} for all pathways; found {sorted(n_folds)}.")
    if stability_df.duplicated(["pathway_name"]).any():
        raise ValueError("Stability table must contain at most one row per pathway.")


def compute_seed_fold_agreement(
    activation_scores: pd.DataFrame, ig_scores: pd.DataFrame, k: int = 20
) -> pd.DataFrame:
    """Compare activation and Integrated Gradients ranks within each seed/fold."""
    _validate_k(k)
    _validate_seed_fold_pathway_counts(activation_scores, ig_scores)
    rows: list[dict[str, float | int]] = []
    pairs = activation_scores[["seed", "fold"]].drop_duplicates().sort_values(["seed", "fold"])
    for pair in pairs.itertuples(index=False):
        activation_fold = activation_scores[
            (activation_scores["seed"] == pair.seed) & (activation_scores["fold"] == pair.fold)
        ]
        ig_fold = ig_scores[(ig_scores["seed"] == pair.seed) & (ig_scores["fold"] == pair.fold)]
        merged = activation_fold[["pathway_name", "rank"]].merge(
            ig_fold[["pathway_name", "rank"]],
            on="pathway_name",
            suffixes=("_activation", "_ig"),
            validate="one_to_one",
        )
        if len(merged) < 2:
            raise ValueError("At least two shared pathways are required for seed/fold agreement.")
        rows.append(
            {
                "seed": int(pair.seed),
                "fold": int(pair.fold),
                "spearman_rank_correlation": compute_spearman_rank_correlation(activation_fold, ig_fold),
                "top20_overlap": compute_topk_overlap(activation_fold, ig_fold, k=k),
                "n_pathways": int(len(merged)),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["seed", "fold", "spearman_rank_correlation", "top20_overlap", "n_pathways"],
    )


def compute_global_stability_agreement(
    activation_stability: pd.DataFrame, ig_stability: pd.DataFrame, k: int = 20
) -> pd.DataFrame:
    """Compare global mean-rank stability summaries between activation and IG."""
    _validate_k(k)
    activation_ranking = _mean_rank_as_rank(activation_stability)
    ig_ranking = _mean_rank_as_rank(ig_stability)
    merged = activation_ranking[["pathway_name", "rank"]].merge(
        ig_ranking[["pathway_name", "rank"]],
        on="pathway_name",
        suffixes=("_activation", "_ig"),
        validate="one_to_one",
    )
    if len(merged) < 2:
        raise ValueError("At least two shared pathways are required for global agreement.")

    activation_top = _topk_names_by_mean_rank(activation_stability, k)
    ig_top = _topk_names_by_mean_rank(ig_stability, k)
    return pd.DataFrame(
        [
            {
                "spearman_rank_correlation": compute_spearman_rank_correlation(
                    activation_ranking, ig_ranking
                ),
                "top20_overlap": len(activation_top.intersection(ig_top)),
                "activation_top20_rna_count": _topk_rna_count(activation_stability, k),
                "ig_top20_rna_count": _topk_rna_count(ig_stability, k),
                "n_pathways": int(len(merged)),
            }
        ]
    )


def summarize_rna_processing_tier(
    activation_stability: pd.DataFrame, ig_stability: pd.DataFrame
) -> pd.DataFrame:
    """Summarize activation and IG rank ranges for RNA-processing-flagged pathways."""
    activation_top20 = _topk_names_by_mean_rank(activation_stability, 20)
    ig_top20 = _topk_names_by_mean_rank(ig_stability, 20)
    merged = activation_stability[
        ["pathway_name", "mean_rank", "min_rank", "max_rank", "is_rna_processing"]
    ].merge(
        ig_stability[["pathway_name", "mean_rank", "min_rank", "max_rank", "is_rna_processing"]],
        on="pathway_name",
        suffixes=("_activation", "_ig"),
        validate="one_to_one",
    )
    rna_mask = merged["is_rna_processing_activation"].map(_to_bool) | merged[
        "is_rna_processing_ig"
    ].map(_to_bool)
    summary = merged.loc[rna_mask].copy()
    summary["activation_top20"] = summary["pathway_name"].isin(activation_top20)
    summary["ig_top20"] = summary["pathway_name"].isin(ig_top20)
    return (
        summary.rename(
            columns={
                "mean_rank_activation": "activation_mean_rank",
                "mean_rank_ig": "ig_mean_rank",
                "min_rank_activation": "activation_min_rank",
                "min_rank_ig": "ig_min_rank",
                "max_rank_activation": "activation_max_rank",
                "max_rank_ig": "ig_max_rank",
            }
        )[
            [
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
        ]
        .sort_values(["activation_mean_rank", "ig_mean_rank", "pathway_name"], kind="mergesort")
        .reset_index(drop=True)
    )


def write_agreement_report(
    seed_fold_agreement: pd.DataFrame,
    global_agreement: pd.DataFrame,
    rna_processing_tier: pd.DataFrame,
    report_path: str | Path,
) -> Path:
    """Write the Phase 4 development-only attribution agreement report."""
    global_row = global_agreement.iloc[0]
    seed_fold_rows = "\n".join(
        (
            f"| {int(row.seed)} | {int(row.fold)} | "
            f"{row.spearman_rank_correlation:.6f} | {int(row.top20_overlap)} | "
            f"{int(row.n_pathways)} |"
        )
        for row in seed_fold_agreement.itertuples(index=False)
    )
    rna_rows = _format_rna_rows(rna_processing_tier)
    output_path = Path(report_path)
    output_path.write_text(
        f"""# Phase 4 activation-vs-Integrated-Gradients agreement

Generated at: {datetime.now(timezone.utc).isoformat()}

This is a development-only interpretation summary comparing existing activation and Integrated Gradients pathway attribution outputs. No retraining was performed in this step. This is not final validation and is not a biological claim.

- Inputs: `results/development/pathway_activation_scores.csv`, `results/development/pathway_activation_stability.csv`, `results/development/pathway_ig_scores.csv`, and `results/development/pathway_ig_stability.csv` only.
- No external cohort or held-out NDD data was loaded or used.
- Attention, permutation testing, final validation, and later-phase logic are not included.

## Agreement summary

| Metric | Value |
| --- | ---: |
| Mean seed/fold Spearman | {seed_fold_agreement["spearman_rank_correlation"].mean():.6f} |
| Minimum seed/fold Spearman | {seed_fold_agreement["spearman_rank_correlation"].min():.6f} |
| Maximum seed/fold Spearman | {seed_fold_agreement["spearman_rank_correlation"].max():.6f} |
| Mean top-20 overlap | {seed_fold_agreement["top20_overlap"].mean():.6f} |
| Global Spearman | {global_row.spearman_rank_correlation:.6f} |
| Global top-20 overlap | {int(global_row.top20_overlap)} |
| Activation top-20 RNA-processing count | {int(global_row.activation_top20_rna_count)} |
| IG top-20 RNA-processing count | {int(global_row.ig_top20_rna_count)} |
| No external/NDD data used | yes |

## Seed/fold agreement

| Seed | Fold | Spearman | Top-20 overlap | Pathways compared |
| ---: | ---: | ---: | ---: | ---: |
{seed_fold_rows}

## RNA-processing-flagged tier

This table reports ranking agreement metadata only; it is not a biological claim.

| Pathway | Activation mean rank | IG mean rank | Activation rank range | IG rank range | Activation top 20 | IG top 20 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
{rna_rows}
""",
        encoding="utf-8",
    )
    return output_path


def _validate_required_columns(df: pd.DataFrame, required_columns: set[str]) -> None:
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}.")


def _validate_no_external_ndd_columns(df: pd.DataFrame) -> None:
    forbidden = [
        column
        for column in df.columns
        if "external" in str(column).lower() or "ndd" in str(column).lower()
    ]
    if forbidden:
        raise ValueError(f"External/NDD columns are not allowed: {sorted(forbidden)}.")


def _validate_expected_method(expected_method: str) -> None:
    if expected_method not in ALLOWED_METHODS:
        raise ValueError(f"expected_method must be one of {sorted(ALLOWED_METHODS)}.")


def _validate_method(df: pd.DataFrame, expected_method: str) -> None:
    observed_methods = set(df["method"].dropna().astype(str).unique())
    if observed_methods != {expected_method}:
        raise ValueError(f"Method must be exactly {expected_method}; found {sorted(observed_methods)}.")


def _validate_integer_column(df: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        raise ValueError(f"{column} must contain finite positive integers.")
    if (values <= 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{column} must contain finite positive integers.")


def _validate_positive_finite_column(df: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{column} must contain positive finite values.")


def _validate_nonnegative_finite_column(df: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{column} must contain nonnegative finite values.")


def _validate_seed_fold_pathway_counts(activation_scores: pd.DataFrame, ig_scores: pd.DataFrame) -> None:
    _validate_required_columns(activation_scores, {"seed", "fold", "pathway_name", "rank"})
    _validate_required_columns(ig_scores, {"seed", "fold", "pathway_name", "rank"})
    activation_counts = (
        activation_scores.groupby(["seed", "fold"], sort=True)["pathway_name"].nunique().rename("activation")
    )
    ig_counts = ig_scores.groupby(["seed", "fold"], sort=True)["pathway_name"].nunique().rename("ig")
    counts = pd.concat([activation_counts, ig_counts], axis=1)
    if counts.isna().any().any() or not (counts["activation"] == counts["ig"]).all():
        raise ValueError("Each seed/fold must contain the same number of pathways for activation and IG.")


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive.")


def _mean_rank_as_rank(stability_df: pd.DataFrame) -> pd.DataFrame:
    return stability_df[["pathway_name", "mean_rank"]].rename(columns={"mean_rank": "rank"})


def _topk_names_by_mean_rank(stability_df: pd.DataFrame, k: int) -> set[str]:
    return set(
        stability_df.sort_values(["mean_rank", "pathway_name"], kind="mergesort")["pathway_name"].head(k)
    )


def _topk_rna_count(stability_df: pd.DataFrame, k: int) -> int:
    topk = stability_df.sort_values(["mean_rank", "pathway_name"], kind="mergesort").head(k)
    return int(topk["is_rna_processing"].map(_to_bool).sum())


def _to_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _format_rna_rows(rna_processing_tier: pd.DataFrame) -> str:
    if rna_processing_tier.empty:
        return "| None |  |  |  |  | false | false |"
    return "\n".join(
        (
            f"| {row.pathway_name} | {row.activation_mean_rank:.3f} | {row.ig_mean_rank:.3f} | "
            f"{int(row.activation_min_rank)}-{int(row.activation_max_rank)} | "
            f"{int(row.ig_min_rank)}-{int(row.ig_max_rank)} | "
            f"{str(bool(row.activation_top20)).lower()} | {str(bool(row.ig_top20)).lower()} |"
        )
        for row in rna_processing_tier.itertuples(index=False)
    )
