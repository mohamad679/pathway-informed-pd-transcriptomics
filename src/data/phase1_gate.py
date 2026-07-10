from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from data.expression import load_gene_space
from data.splits import load_sample_ids, validate_fold_disjointness


LABEL_NAME_MAP = {0: "HC", 1: "PD"}


def _count_labels(values: np.ndarray) -> dict[str, int]:
    unique_values, counts = np.unique(values, return_counts=True)
    summary: dict[str, int] = {}
    for value, count in zip(unique_values.tolist(), counts.tolist(), strict=True):
        label_name = LABEL_NAME_MAP.get(int(value), str(value))
        summary[label_name] = int(count)
    return summary


def load_required_phase1_artifacts(processed_dir: str | Path) -> dict[str, object]:
    processed_path = Path(processed_dir)
    if not processed_path.is_dir():
        raise FileNotFoundError(f"Processed directory not found: {processed_path}")

    artifacts = {
        "dev_X": np.load(processed_path / "dev_X.npy", allow_pickle=False),
        "dev_y": np.load(processed_path / "dev_y.npy", allow_pickle=False),
        "dev_sample_ids": load_sample_ids(processed_path / "dev_sample_ids.txt"),
        "ndd_X": np.load(processed_path / "ndd_X.npy", allow_pickle=False),
        "ndd_sample_ids": load_sample_ids(processed_path / "ndd_sample_ids.txt"),
        "ext_X": np.load(processed_path / "ext_X.npy", allow_pickle=False),
        "ext_y": np.load(processed_path / "ext_y.npy", allow_pickle=False),
        "ext_sample_ids": load_sample_ids(processed_path / "ext_sample_ids.txt"),
        "gene_space": load_gene_space(processed_path / "gene_space.txt"),
    }

    folds_path = processed_path / "dev_folds.json"
    if not folds_path.is_file():
        raise FileNotFoundError(f"Missing fold file: {folds_path}")
    artifacts["dev_folds"] = json.loads(folds_path.read_text(encoding="utf-8"))
    return artifacts


def validate_processed_shapes(
    dev_X: np.ndarray,
    dev_y: np.ndarray,
    dev_sample_ids: list[str],
    ndd_X: np.ndarray,
    ndd_sample_ids: list[str],
    ext_X: np.ndarray,
    ext_y: np.ndarray,
    ext_sample_ids: list[str],
    gene_space: list[str],
) -> dict[str, object]:
    gene_count = len(gene_space)
    if dev_X.shape != (438, gene_count):
        raise ValueError(f"Expected dev_X shape (438, {gene_count}), found {dev_X.shape}")
    if dev_y.shape != (438,):
        raise ValueError(f"Expected dev_y shape (438,), found {dev_y.shape}")
    if len(dev_sample_ids) != 438:
        raise ValueError(f"Expected 438 development sample IDs, found {len(dev_sample_ids)}")

    dev_counts = _count_labels(dev_y)
    if dev_counts != {"HC": 233, "PD": 205}:
        raise ValueError(f"Expected development label counts HC=233, PD=205, found {dev_counts}")

    if ndd_X.shape != (48, gene_count):
        raise ValueError(f"Expected ndd_X shape (48, {gene_count}), found {ndd_X.shape}")
    if len(ndd_sample_ids) != 48:
        raise ValueError(f"Expected 48 held-out NDD sample IDs, found {len(ndd_sample_ids)}")

    if ext_X.shape != (72, gene_count):
        raise ValueError(f"Expected ext_X shape (72, {gene_count}), found {ext_X.shape}")
    if ext_y.shape != (72,):
        raise ValueError(f"Expected ext_y shape (72,), found {ext_y.shape}")
    if len(ext_sample_ids) != 72:
        raise ValueError(f"Expected 72 external sample IDs, found {len(ext_sample_ids)}")

    ext_counts = _count_labels(ext_y)
    if ext_counts != {"HC": 22, "PD": 50}:
        raise ValueError(f"Expected external label counts HC=22, PD=50, found {ext_counts}")

    return {
        "gene_count": gene_count,
        "dev_shape": tuple(int(value) for value in dev_X.shape),
        "ndd_shape": tuple(int(value) for value in ndd_X.shape),
        "ext_shape": tuple(int(value) for value in ext_X.shape),
        "dev_label_counts": dev_counts,
        "ext_label_counts": ext_counts,
    }


def validate_zscore_sanity(
    X: np.ndarray,
    *,
    dataset_name: str,
    mean_atol: float = 1e-5,
    std_atol: float = 1e-4,
) -> dict[str, float]:
    if X.ndim != 2:
        raise ValueError(f"{dataset_name} must be a 2D matrix")

    sample_means = X.mean(axis=1)
    sample_stds = X.std(axis=1, ddof=0)
    max_abs_mean = float(np.max(np.abs(sample_means)))
    max_abs_std_delta = float(np.max(np.abs(sample_stds - 1.0)))

    if max_abs_mean > mean_atol:
        raise ValueError(
            f"{dataset_name} per-sample z-score mean sanity failed: "
            f"max_abs_mean={max_abs_mean:.6g} exceeds {mean_atol:.6g}"
        )
    if max_abs_std_delta > std_atol:
        raise ValueError(
            f"{dataset_name} per-sample z-score std sanity failed: "
            f"max_abs_std_delta={max_abs_std_delta:.6g} exceeds {std_atol:.6g}"
        )

    return {
        "max_abs_mean": max_abs_mean,
        "max_abs_std_delta": max_abs_std_delta,
    }


def validate_folds_cover_development_once(
    folds: list[dict[str, object]],
    dev_sample_ids: list[str],
) -> dict[str, int]:
    if not folds:
        raise ValueError("Development fold list is empty")
    return validate_fold_disjointness(folds, dev_sample_ids)


def compute_dev_pca(dev_X: np.ndarray, random_state: int = 20260710) -> dict[str, np.ndarray]:
    if dev_X.ndim != 2:
        raise ValueError("dev_X must be a 2D matrix")
    if dev_X.shape[0] < 2 or dev_X.shape[1] < 2:
        raise ValueError("dev_X must have at least 2 samples and 2 genes for PCA")

    pca = PCA(n_components=2, svd_solver="randomized", random_state=random_state)
    coordinates = pca.fit_transform(dev_X)
    return {
        "coordinates": coordinates,
        "explained_variance_ratio": pca.explained_variance_ratio_.copy(),
    }


def write_phase1_gate_report(
    output_path: str | Path,
    *,
    shape_summary: dict[str, object],
    dev_zscore_summary: dict[str, float],
    ext_zscore_summary: dict[str, float],
    fold_summary: dict[str, int],
    pca_summary: dict[str, np.ndarray],
    pc1_warning: str | None = None,
) -> None:
    report_path = Path(output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    pca_evr = pca_summary["explained_variance_ratio"]
    lines = [
        "# Phase 1 Final Gate Audit",
        "",
        "This report covers only the Phase 1 final gate audit and cohort overview figure.",
        "Development PCA was computed on the development matrix only and is used here as a sanity visualization only.",
        "No modeling, baseline implementation, training, pathway masking, MSigDB logic, or external-validation model selection was performed.",
        "",
        "## Gate Status",
        "",
        "- Status: `PASS`",
        "",
        "## Processed Artifact Checks",
        "",
        f"- Development matrix shape: `{shape_summary['dev_shape']}`",
        f"- Development label counts: `HC={shape_summary['dev_label_counts']['HC']}`, `PD={shape_summary['dev_label_counts']['PD']}`",
        f"- Held-out NDD matrix shape: `{shape_summary['ndd_shape']}`",
        f"- External matrix shape: `{shape_summary['ext_shape']}`",
        f"- External label counts: `HC={shape_summary['ext_label_counts']['HC']}`, `PD={shape_summary['ext_label_counts']['PD']}`",
        f"- Ordered gene count: `{shape_summary['gene_count']}`",
        "",
        "## Z-Score Sanity",
        "",
        f"- Development max absolute per-sample mean: `{dev_zscore_summary['max_abs_mean']:.6g}`",
        f"- Development max absolute per-sample std delta from 1: `{dev_zscore_summary['max_abs_std_delta']:.6g}`",
        f"- External max absolute per-sample mean: `{ext_zscore_summary['max_abs_mean']:.6g}`",
        f"- External max absolute per-sample std delta from 1: `{ext_zscore_summary['max_abs_std_delta']:.6g}`",
        "",
        "## Fold Integrity",
        "",
        f"- Fold count: `{fold_summary['fold_count']}`",
        f"- Development samples covered once in validation: `{fold_summary['validation_coverage_count']}` of `{fold_summary['sample_count']}`",
        "- No train/validation overlap was detected in any fold.",
        "",
        "## PCA Sanity Visualization",
        "",
        f"- PC1 explained variance ratio: `{float(pca_evr[0]):.6f}`",
        f"- PC2 explained variance ratio: `{float(pca_evr[1]):.6f}`",
        "- PCA used development samples only.",
        "- No class-separation claim is made from this visualization.",
        "",
        "## Boundary Confirmation",
        "",
        "- Modeling performed: `no`",
        "- Baselines implemented: `no`",
        "- Training performed: `no`",
        "- Pathway masks implemented: `no`",
        "- MSigDB logic implemented: `no`",
        "- External validation used for model selection: `no`",
    ]

    if pc1_warning is not None:
        lines.extend(
            [
                "",
                "## Warning",
                "",
                f"- {pc1_warning}",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cohort_overview(
    output_path: str | Path,
    *,
    dev_y: np.ndarray,
    ndd_X: np.ndarray,
    ext_y: np.ndarray,
    pca_summary: dict[str, np.ndarray],
) -> None:
    figure_path = Path(output_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    dev_counts = _count_labels(dev_y)
    ext_counts = _count_labels(ext_y)
    coordinates = pca_summary["coordinates"]
    explained_variance_ratio = pca_summary["explained_variance_ratio"]

    figure, axes = plt.subplots(ncols=2, figsize=(12, 5))

    bar_labels = ["Dev HC", "Dev PD", "NDD", "Ext HC", "Ext PD"]
    bar_values = [
        dev_counts["HC"],
        dev_counts["PD"],
        int(ndd_X.shape[0]),
        ext_counts["HC"],
        ext_counts["PD"],
    ]
    axes[0].bar(bar_labels, bar_values)
    axes[0].set_title("Cohort Counts")
    axes[0].set_ylabel("Samples")
    axes[0].tick_params(axis="x", rotation=20)

    for class_value, label_name in sorted(LABEL_NAME_MAP.items()):
        class_mask = dev_y == class_value
        axes[1].scatter(
            coordinates[class_mask, 0],
            coordinates[class_mask, 1],
            label=label_name,
            s=20,
            alpha=0.8,
        )
    axes[1].set_title("Development PCA")
    axes[1].set_xlabel(f"PC1 ({float(explained_variance_ratio[0]) * 100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({float(explained_variance_ratio[1]) * 100:.1f}%)")
    axes[1].legend(title="Label")

    figure.tight_layout()
    figure.savefig(figure_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
