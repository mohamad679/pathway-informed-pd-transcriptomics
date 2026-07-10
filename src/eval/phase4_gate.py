from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_SEEDS = frozenset({11, 23, 37})
EXPECTED_FOLDS = frozenset({1, 2, 3, 4, 5})
EXPECTED_PATHWAYS = 1_297
EXPECTED_SCORE_ROWS = 19_455
EXPECTED_STABILITY_ROWS = 1_297
EXPECTED_AGREEMENT_ROWS = 15
EXPECTED_RNA_PROCESSING_PATHWAYS = 4
ACTIVATION_METHOD = "activation"
IG_METHOD = "integrated_gradients"
SCORE_COLUMNS = frozenset(
    {
        "pathway_name",
        "method",
        "seed",
        "fold",
        "mean_score",
        "abs_mean_score",
        "rank",
        "is_rna_processing",
    }
)
STABILITY_COLUMNS = frozenset(
    {
        "pathway_name",
        "method",
        "mean_rank",
        "rank_variance",
        "min_rank",
        "max_rank",
        "n_folds",
        "is_rna_processing",
    }
)
SEED_FOLD_AGREEMENT_COLUMNS = frozenset(
    {"seed", "fold", "spearman_rank_correlation", "top20_overlap", "n_pathways"}
)
GLOBAL_AGREEMENT_COLUMNS = frozenset(
    {
        "spearman_rank_correlation",
        "top20_overlap",
        "activation_top20_rna_count",
        "ig_top20_rna_count",
        "n_pathways",
    }
)
RNA_PROCESSING_TIER_COLUMNS = frozenset(
    {
        "pathway_name",
        "activation_mean_rank",
        "ig_mean_rank",
        "activation_min_rank",
        "ig_min_rank",
        "activation_max_rank",
        "ig_max_rank",
        "activation_top20",
        "ig_top20",
    }
)


def load_phase4_outputs(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load existing Phase 4 development attribution artifacts without rerunning attribution."""
    base = Path(results_dir)
    required_paths = {
        "activation_scores": base / "pathway_activation_scores.csv",
        "activation_stability": base / "pathway_activation_stability.csv",
        "ig_scores": base / "pathway_ig_scores.csv",
        "ig_stability": base / "pathway_ig_stability.csv",
        "seed_fold_agreement": base / "pathway_attribution_seed_fold_agreement.csv",
        "global_agreement": base / "pathway_attribution_global_agreement.csv",
        "rna_processing_tier": base / "pathway_rna_processing_tier.csv",
    }
    for path in required_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 4 output: {path}")
    return {name: pd.read_csv(path) for name, path in required_paths.items()}


def validate_activation_outputs(
    activation_scores: pd.DataFrame, activation_stability: pd.DataFrame
) -> None:
    """Validate existing activation attribution score and stability artifacts."""
    _validate_score_outputs(activation_scores, ACTIVATION_METHOD, "activation scores")
    _validate_stability_outputs(
        activation_stability, activation_scores, ACTIVATION_METHOD, "activation stability"
    )


def validate_ig_outputs(ig_scores: pd.DataFrame, ig_stability: pd.DataFrame) -> None:
    """Validate existing Integrated Gradients score and stability artifacts."""
    _validate_score_outputs(ig_scores, IG_METHOD, "IG scores")
    _validate_stability_outputs(ig_stability, ig_scores, IG_METHOD, "IG stability")


def validate_agreement_outputs(
    seed_fold_agreement: pd.DataFrame,
    global_agreement: pd.DataFrame,
    rna_processing_tier: pd.DataFrame,
) -> None:
    """Validate existing activation-vs-IG agreement artifacts."""
    _validate_required_columns(seed_fold_agreement, SEED_FOLD_AGREEMENT_COLUMNS, "seed/fold agreement")
    _validate_required_columns(global_agreement, GLOBAL_AGREEMENT_COLUMNS, "global agreement")
    _validate_required_columns(rna_processing_tier, RNA_PROCESSING_TIER_COLUMNS, "RNA-processing tier")
    for name, dataframe in (
        ("seed/fold agreement", seed_fold_agreement),
        ("global agreement", global_agreement),
        ("RNA-processing tier", rna_processing_tier),
    ):
        _validate_no_external_ndd_columns(dataframe, name)

    if len(seed_fold_agreement) != EXPECTED_AGREEMENT_ROWS:
        raise ValueError(f"seed/fold agreement must contain exactly {EXPECTED_AGREEMENT_ROWS} rows")
    seeds = _integer_values(seed_fold_agreement, "seed", positive=True)
    folds = _integer_values(seed_fold_agreement, "fold", positive=True)
    if set(seeds) != EXPECTED_SEEDS:
        raise ValueError(f"seed/fold agreement seeds must be exactly {sorted(EXPECTED_SEEDS)}")
    if set(folds) != EXPECTED_FOLDS:
        raise ValueError(f"seed/fold agreement folds must be exactly {sorted(EXPECTED_FOLDS)}")
    observed_pairs = set(zip(seeds, folds, strict=True))
    expected_pairs = {(seed, fold) for seed in EXPECTED_SEEDS for fold in EXPECTED_FOLDS}
    if observed_pairs != expected_pairs or len(observed_pairs) != EXPECTED_AGREEMENT_ROWS:
        raise ValueError("seed/fold agreement must contain each required seed/fold exactly once")
    if any(value != EXPECTED_PATHWAYS for value in _integer_values(seed_fold_agreement, "n_pathways", positive=True)):
        raise ValueError(f"seed/fold agreement n_pathways must be exactly {EXPECTED_PATHWAYS}")
    _validate_spearman(seed_fold_agreement, "seed/fold agreement")
    _validate_top20_column(seed_fold_agreement, "top20_overlap", "seed/fold agreement")

    if len(global_agreement) != 1:
        raise ValueError("global agreement must contain exactly 1 row")
    global_row = global_agreement.iloc[0]
    if int(_integer_values(global_agreement, "n_pathways", positive=True)[0]) != EXPECTED_PATHWAYS:
        raise ValueError(f"global agreement n_pathways must be exactly {EXPECTED_PATHWAYS}")
    _validate_spearman(global_agreement, "global agreement")
    for column in ("top20_overlap", "activation_top20_rna_count", "ig_top20_rna_count"):
        _validate_top20_column(global_agreement, column, "global agreement")

    if len(rna_processing_tier) != EXPECTED_RNA_PROCESSING_PATHWAYS:
        raise ValueError(
            f"RNA-processing tier must contain exactly {EXPECTED_RNA_PROCESSING_PATHWAYS} rows"
        )
    rank_columns = [
        "activation_mean_rank",
        "ig_mean_rank",
        "activation_min_rank",
        "ig_min_rank",
        "activation_max_rank",
        "ig_max_rank",
    ]
    for column in rank_columns:
        _validate_positive_finite(rna_processing_tier, column, "RNA-processing tier")
    # Keep a direct row access so schema mistakes surface in static checks and tests.
    _ = global_row


def summarize_phase4_gate(
    activation_scores: pd.DataFrame,
    activation_stability: pd.DataFrame,
    ig_scores: pd.DataFrame,
    ig_stability: pd.DataFrame,
    seed_fold_agreement: pd.DataFrame,
    global_agreement: pd.DataFrame,
    rna_processing_tier: pd.DataFrame,
) -> dict[str, float | int]:
    """Return development-only Phase 4 artifact-integrity summary facts."""
    return {
        "activation_score_rows": int(len(activation_scores)),
        "activation_stability_rows": int(len(activation_stability)),
        "ig_score_rows": int(len(ig_scores)),
        "ig_stability_rows": int(len(ig_stability)),
        "pathway_count": int(activation_scores["pathway_name"].nunique()),
        "mean_seed_fold_spearman": float(
            pd.to_numeric(seed_fold_agreement["spearman_rank_correlation"]).mean()
        ),
        "global_spearman": float(global_agreement.iloc[0]["spearman_rank_correlation"]),
        "global_top20_overlap": int(global_agreement.iloc[0]["top20_overlap"]),
        "activation_top20_rna_count": int(global_agreement.iloc[0]["activation_top20_rna_count"]),
        "ig_top20_rna_count": int(global_agreement.iloc[0]["ig_top20_rna_count"]),
        "rna_processing_tier_rows": int(len(rna_processing_tier)),
    }


def write_phase4_gate_report(output_path: str | Path, summary: dict[str, float | int]) -> None:
    """Write the Phase 4 final gate report with explicit scope boundaries."""
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Phase 4 Final Gate Audit",
                "",
                "This is a development-only attribution artifact integrity audit, not final validation, not a final performance claim, and not a biological claim.",
                "",
                "## Gate Status",
                "",
                "- Status: `PASS`",
                "- The Phase 4 gate concerns existing activation and Integrated Gradients artifact integrity only.",
                "- Activation and IG show agreement/stability summaries only.",
                "",
                "## Validated Development Outputs",
                "",
                f"- Activation score rows: `{summary['activation_score_rows']}`.",
                f"- Integrated Gradients score rows: `{summary['ig_score_rows']}`.",
                f"- Pathway count: `{summary['pathway_count']}`.",
                "- Exact seeds: `11`, `23`, `37`; folds: `1`-`5`; pathways per seed/fold: `1,297`.",
                f"- Mean seed/fold Spearman: `{summary['mean_seed_fold_spearman']:.6f}`.",
                f"- Global Spearman: `{summary['global_spearman']:.6f}`.",
                f"- Global top-20 overlap: `{summary['global_top20_overlap']}`.",
                f"- Activation top-20 RNA-processing count: `{summary['activation_top20_rna_count']}`.",
                f"- IG top-20 RNA-processing count: `{summary['ig_top20_rna_count']}`.",
                f"- RNA-processing tier rows: `{summary['rna_processing_tier_rows']}`.",
                "",
                "## Boundary Confirmation",
                "",
                "- No training or retraining was run inside the gate.",
                "- Activation attribution was not run inside the gate.",
                "- Integrated Gradients attribution was not run inside the gate.",
                "- No external cohort or held-out NDD data was used.",
                "- This is not final validation.",
                "- This is not a biological claim.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _validate_score_outputs(scores: pd.DataFrame, expected_method: str, name: str) -> None:
    _validate_required_columns(scores, SCORE_COLUMNS, name)
    _validate_no_external_ndd_columns(scores, name)
    if len(scores) != EXPECTED_SCORE_ROWS:
        raise ValueError(f"{name} must contain exactly {EXPECTED_SCORE_ROWS} rows")
    _validate_method(scores, expected_method, name)
    seeds = _integer_values(scores, "seed", positive=True)
    folds = _integer_values(scores, "fold", positive=True)
    ranks = _integer_values(scores, "rank", positive=True)
    if set(seeds) != EXPECTED_SEEDS:
        raise ValueError(f"{name} seeds must be exactly {sorted(EXPECTED_SEEDS)}")
    if set(folds) != EXPECTED_FOLDS:
        raise ValueError(f"{name} folds must be exactly {sorted(EXPECTED_FOLDS)}")
    if scores["pathway_name"].nunique() != EXPECTED_PATHWAYS:
        raise ValueError(f"{name} pathway count must be exactly {EXPECTED_PATHWAYS}")
    if scores.duplicated(["seed", "fold", "pathway_name"]).any():
        raise ValueError(f"{name} must not contain duplicate seed/fold/pathway rows")
    _validate_finite(scores, "mean_score", name)
    _validate_finite(scores, "abs_mean_score", name)

    checked = scores.assign(
        seed=np.asarray(seeds, dtype=int),
        fold=np.asarray(folds, dtype=int),
        rank=np.asarray(ranks, dtype=int),
    )
    expected_ranks = set(range(1, EXPECTED_PATHWAYS + 1))
    for (seed, fold), group in checked.groupby(["seed", "fold"], sort=True):
        if len(group) != EXPECTED_PATHWAYS or group["pathway_name"].nunique() != EXPECTED_PATHWAYS:
            raise ValueError(
                f"{name} seed {seed} fold {fold} must contain exactly {EXPECTED_PATHWAYS} pathways"
            )
        if set(group["rank"].astype(int)) != expected_ranks:
            raise ValueError(f"{name} ranks for seed {seed} fold {fold} must be exactly 1..{EXPECTED_PATHWAYS}")
    _validate_rna_flags(scores, name)


def _validate_stability_outputs(
    stability: pd.DataFrame, scores: pd.DataFrame, expected_method: str, name: str
) -> None:
    _validate_required_columns(stability, STABILITY_COLUMNS, name)
    _validate_no_external_ndd_columns(stability, name)
    if len(stability) != EXPECTED_STABILITY_ROWS:
        raise ValueError(f"{name} must contain exactly {EXPECTED_STABILITY_ROWS} rows")
    _validate_method(stability, expected_method, name)
    if stability["pathway_name"].nunique() != EXPECTED_PATHWAYS:
        raise ValueError(f"{name} pathway count must be exactly {EXPECTED_PATHWAYS}")
    if stability.duplicated(["pathway_name"]).any():
        raise ValueError(f"{name} must contain at most one row per pathway")
    if set(stability["pathway_name"].astype(str)) != set(scores["pathway_name"].astype(str)):
        raise ValueError(f"{name} pathway set must match score file")
    if any(value != len(EXPECTED_FOLDS) for value in _integer_values(stability, "n_folds", positive=True)):
        raise ValueError(f"{name} n_folds must be exactly {len(EXPECTED_FOLDS)} for all pathways")
    _validate_positive_finite(stability, "mean_rank", name)
    _validate_nonnegative_finite(stability, "rank_variance", name)
    _integer_values(stability, "min_rank", positive=True)
    _integer_values(stability, "max_rank", positive=True)
    if _pathway_flag_map(stability) != _pathway_flag_map(scores):
        raise ValueError(f"{name} is_rna_processing flags must match score file")


def _validate_required_columns(df: pd.DataFrame, required_columns: frozenset[str], name: str) -> None:
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{name} is missing required columns: {sorted(missing_columns)}")


def _validate_no_external_ndd_columns(df: pd.DataFrame, name: str) -> None:
    forbidden = [
        column
        for column in df.columns
        if "external" in str(column).lower() or "ndd" in str(column).lower()
    ]
    if forbidden:
        raise ValueError(f"{name} must not contain external/NDD columns: {sorted(forbidden)}")


def _validate_method(df: pd.DataFrame, expected_method: str, name: str) -> None:
    observed_methods = set(df["method"].dropna().astype(str))
    if observed_methods != {expected_method}:
        raise ValueError(f"{name} method must be exactly {expected_method}")


def _integer_values(df: pd.DataFrame, column: str, *, positive: bool) -> list[int]:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{column} values must be finite integers")
    if positive and np.any(values <= 0):
        raise ValueError(f"{column} values must be positive integers")
    return values.astype(int).tolist()


def _validate_finite(df: pd.DataFrame, column: str, name: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} {column} values must be finite")


def _validate_positive_finite(df: pd.DataFrame, column: str, name: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError(f"{name} {column} values must be finite and positive")


def _validate_nonnegative_finite(df: pd.DataFrame, column: str, name: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} {column} values must be finite and nonnegative")


def _validate_spearman(df: pd.DataFrame, name: str) -> None:
    values = pd.to_numeric(df["spearman_rank_correlation"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < -1.0) | (values > 1.0)):
        raise ValueError(f"{name} Spearman values must be finite and in [-1, 1]")


def _validate_top20_column(df: pd.DataFrame, column: str, name: str) -> None:
    values = _integer_values(df, column, positive=False)
    if any(value < 0 or value > 20 for value in values):
        raise ValueError(f"{name} {column} must be an integer in [0, 20]")


def _validate_rna_flags(df: pd.DataFrame, name: str) -> None:
    flag_map = _pathway_flag_map(df)
    if len(flag_map) != EXPECTED_PATHWAYS:
        raise ValueError(f"{name} must have one RNA-processing flag per pathway")
    flagged_count = sum(flag_map.values())
    if flagged_count != EXPECTED_RNA_PROCESSING_PATHWAYS:
        raise ValueError(
            f"{name} must flag exactly {EXPECTED_RNA_PROCESSING_PATHWAYS} RNA-processing pathways"
        )


def _pathway_flag_map(df: pd.DataFrame) -> dict[str, bool]:
    frame = df[["pathway_name", "is_rna_processing"]].copy()
    frame["pathway_name"] = frame["pathway_name"].astype(str)
    frame["is_rna_processing"] = frame["is_rna_processing"].map(_to_bool)
    counts = frame.groupby("pathway_name")["is_rna_processing"].nunique()
    inconsistent = counts[counts > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"is_rna_processing flags must be consistent per pathway: {sorted(inconsistent.index)}"
        )
    return frame.drop_duplicates("pathway_name").set_index("pathway_name")["is_rna_processing"].to_dict()


def _to_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return bool(value)
