from __future__ import annotations

import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault(
    "MPLCONFIGDIR",
    "/private/tmp/pathway-informed-pd-transcriptomics-mpl",
)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DIR = ROOT / "results" / "development"
EXTERNAL_DIR = ROOT / "results" / "external"
FIGURE_DIR = ROOT / "results" / "figures"

FIGURE_FILES = (
    "fig02_development_model_comparison.png",
    "fig03_permutation_validation.png",
    "fig04_top_pathways.png",
    "fig05_attribution_agreement.png",
    "fig06_external_validation.png",
)


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _shorten_pathway_name(name: str, width: int = 42) -> str:
    label = name.removeprefix("REACTOME_").replace("_", " ").title()
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_final_figures(root: Path = ROOT) -> list[Path]:
    development_dir = root / "results" / "development"
    external_dir = root / "results" / "external"
    figure_dir = root / "results" / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary = pd.read_csv(development_dir / "baseline_summary.csv")
    binn_cv = pd.read_csv(development_dir / "binn_cv.csv")
    bootstrap_ci = pd.read_csv(development_dir / "statistical_validation_bootstrap_ci.csv")
    permutation_null = pd.read_csv(development_dir / "statistical_validation_permutation_null.csv")
    statistical_summary = _load_json(development_dir / "statistical_validation_summary.json")
    activation_stability = pd.read_csv(development_dir / "pathway_activation_stability.csv")
    ig_stability = pd.read_csv(development_dir / "pathway_ig_stability.csv")
    global_agreement = pd.read_csv(development_dir / "pathway_attribution_global_agreement.csv")
    external_metrics = _load_json(external_dir / "external_metrics.json")

    outputs: list[Path] = []

    outputs.append(
        _figure_development_comparison(
            baseline_summary,
            bootstrap_ci,
            statistical_summary,
            figure_dir / "fig02_development_model_comparison.png",
        )
    )
    outputs.append(
        _figure_permutation_validation(
            permutation_null,
            statistical_summary,
            figure_dir / "fig03_permutation_validation.png",
        )
    )
    outputs.append(
        _figure_top_pathways(
            activation_stability,
            development_dir / "final_top20_pathways.csv",
            figure_dir / "fig04_top_pathways.png",
        )
    )
    outputs.append(
        _figure_attribution_agreement(
            activation_stability,
            ig_stability,
            global_agreement,
            figure_dir / "fig05_attribution_agreement.png",
        )
    )
    outputs.append(
        _figure_external_validation(
            external_metrics,
            figure_dir / "fig06_external_validation.png",
        )
    )
    return outputs


def _figure_development_comparison(
    baseline_summary: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    statistical_summary: dict[str, object],
    output_path: Path,
) -> Path:
    model_order = ["logistic_regression", "random_forest", "unconstrained_mlp"]
    display = {
        "logistic_regression": "Logistic\nregression",
        "random_forest": "Random\nforest",
        "unconstrained_mlp": "Unconstrained\nMLP",
        "pathway_constrained_binn": "Pathway-\nconstrained\nBINN",
    }
    rows = []
    for model in model_order:
        row = baseline_summary.loc[baseline_summary["model"] == model].iloc[0]
        rows.append(
            {
                "model": model,
                "auroc": float(row["auroc_mean"]),
                "ci_lower": float(row["auroc_ci_lower"]),
                "ci_upper": float(row["auroc_ci_upper"]),
            }
        )
    auroc_ci = bootstrap_ci.loc[bootstrap_ci["metric"] == "auroc"].iloc[0]
    rows.append(
        {
            "model": "pathway_constrained_binn",
            "auroc": float(statistical_summary["observed_auroc"]),
            "ci_lower": float(auroc_ci["ci_lower"]),
            "ci_upper": float(auroc_ci["ci_upper"]),
        }
    )
    frame = pd.DataFrame(rows)
    lower = frame["auroc"] - frame["ci_lower"]
    upper = frame["ci_upper"] - frame["auroc"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        range(len(frame)),
        frame["auroc"],
        yerr=[lower, upper],
        capsize=4,
        color=["#7d8f9b", "#8fae91", "#b59b70", "#7b7f8f"],
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.set_xticks(range(len(frame)), [display[name] for name in frame["model"]])
    ax.set_ylim(0.55, 0.76)
    ax.set_ylabel("Pooled development AUROC")
    ax.set_title("Development-only model comparison")
    ax.text(
        0.01,
        0.98,
        "Uncertainty bars use committed confidence intervals where available",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    _save(fig, output_path)
    return output_path


def _figure_permutation_validation(
    permutation_null: pd.DataFrame,
    statistical_summary: dict[str, object],
    output_path: Path,
) -> Path:
    observed = float(statistical_summary["observed_auroc"])
    empirical_p = float(statistical_summary["empirical_p_value"])
    n_permutations = int(statistical_summary["n_permutations"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.hist(
        permutation_null["null_auroc"].astype(float),
        bins=12,
        color="#9aa7b8",
        edgecolor="#333333",
        linewidth=0.7,
    )
    ax.axvline(observed, color="#8f4f4f", linewidth=2.0, label=f"Observed AUROC {observed:.6f}")
    ax.set_xlabel("Null AUROC")
    ax.set_ylabel("Permutation count")
    ax.set_title(
        f"Development-only permutation validation; {n_permutations} permutations; "
        f"empirical p={empirical_p:.6f}; limited resolution"
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    _save(fig, output_path)
    return output_path


def _figure_top_pathways(
    activation_stability: pd.DataFrame,
    companion_csv: Path,
    output_path: Path,
) -> Path:
    top20 = (
        activation_stability.sort_values("mean_rank", ascending=True)
        .head(20)
        .reset_index(drop=True)
        .copy()
    )
    top20.insert(0, "display_rank", range(1, len(top20) + 1))
    top20["short_label"] = top20["pathway_name"].map(_shorten_pathway_name)
    companion_columns = [
        "display_rank",
        "pathway_name",
        "short_label",
        "mean_rank",
        "rank_variance",
        "min_rank",
        "max_rank",
        "n_folds",
        "is_rna_processing",
    ]
    top20[companion_columns].to_csv(companion_csv, index=False)

    plot_frame = top20.iloc[::-1]
    fig_height = 8.2
    fig, ax = plt.subplots(figsize=(8.0, fig_height))
    ax.barh(
        plot_frame["short_label"],
        plot_frame["mean_rank"].astype(float),
        color="#8fae91",
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.invert_xaxis()
    ax.set_xlabel("Activation mean rank (lower is higher ranking)")
    ax.set_title("Top 20 development pathways by activation mean rank")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    _save(fig, output_path)
    return output_path


def _figure_attribution_agreement(
    activation_stability: pd.DataFrame,
    ig_stability: pd.DataFrame,
    global_agreement: pd.DataFrame,
    output_path: Path,
) -> Path:
    merged = activation_stability[["pathway_name", "mean_rank"]].merge(
        ig_stability[["pathway_name", "mean_rank"]],
        on="pathway_name",
        suffixes=("_activation", "_ig"),
        validate="one_to_one",
    )
    spearman = float(global_agreement.iloc[0]["spearman_rank_correlation"])
    overlap = int(global_agreement.iloc[0]["top20_overlap"])

    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    ax.scatter(
        merged["mean_rank_activation"].astype(float),
        merged["mean_rank_ig"].astype(float),
        s=12,
        alpha=0.65,
        color="#6f7f95",
        linewidths=0,
    )
    max_rank = max(merged["mean_rank_activation"].max(), merged["mean_rank_ig"].max())
    ax.plot([1, max_rank], [1, max_rank], color="#555555", linewidth=1.0)
    ax.set_xlabel("Activation mean rank (lower is better)")
    ax.set_ylabel("Integrated Gradients mean rank (lower is better)")
    ax.set_title("Development model attribution agreement")
    ax.text(
        0.04,
        0.96,
        f"Global Spearman = {spearman:.6f}\nTop-20 overlap = {overlap}\nModel attribution, not causation",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )
    ax.grid(color="#dddddd", linewidth=0.6)
    _save(fig, output_path)
    return output_path


def _figure_external_validation(
    external_metrics: dict[str, object],
    output_path: Path,
) -> Path:
    metric_names = ["auroc", "auprc", "balanced_accuracy", "brier", "ece"]
    labels = ["AUROC", "AUPRC", "Balanced\naccuracy", "Brier", "ECE"]
    values = [float(external_metrics[name]) for name in metric_names]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(
        range(len(values)),
        values,
        color=["#8795a8", "#8795a8", "#b59b70", "#b07a7a", "#b07a7a"],
        edgecolor="#333333",
        linewidth=0.8,
    )
    ax.set_xticks(range(len(values)), labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Metric value")
    ax.set_title("Frozen external result: ranking only; poor calibration; threshold 0.5")
    ax.text(
        0.02,
        0.96,
        "Balanced accuracy = 0.500000\nBrier = 0.694444; ECE = 0.694441\nNo external tuning or recalibration",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb"},
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    _save(fig, output_path)
    return output_path


def main() -> int:
    outputs = generate_final_figures(ROOT)
    print("PASS")
    for output in outputs:
        print(output.relative_to(ROOT))
    print((DEVELOPMENT_DIR / "final_top20_pathways.csv").relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
