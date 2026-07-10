from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.phase1_gate import (
    compute_dev_pca,
    load_required_phase1_artifacts,
    plot_cohort_overview,
    validate_folds_cover_development_once,
    validate_processed_shapes,
    validate_zscore_sanity,
    write_phase1_gate_report,
)


PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"
RESULTS_DIR = ROOT / "results" / "figures"
REPORT_PATH = DOCS_DIR / "phase1_gate.md"
FIGURE_PATH = RESULTS_DIR / "fig01_cohort_overview.png"
DECISION_LOG_PATH = DOCS_DIR / "decision_log.md"

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 1 final gate audit and cohort overview figure

- Validated only the existing processed Phase 1 artifacts and development fold file.
- Confirmed development, held-out NDD, and external matrix shapes and Phase 1 label counts against the expected gate values.
- Confirmed per-sample z-score sanity for development and external matrices.
- Confirmed that each development sample appears exactly once in validation across the saved folds with no train/validation overlap.
- Computed a development-only PCA for a sanity visualization and saved a single cohort overview PNG.
- No modeling, baselines, training, pathway masks, MSigDB logic, or external-validation model selection was performed.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def detect_pc1_perfect_class_separation(coordinates, dev_y) -> bool:
    hc_values = coordinates[dev_y == 0, 0]
    pd_values = coordinates[dev_y == 1, 0]
    return bool(hc_values.max() < pd_values.min() or pd_values.max() < hc_values.min())


def main() -> int:
    try:
        artifacts = load_required_phase1_artifacts(PROCESSED_DIR)
        shape_summary = validate_processed_shapes(
            artifacts["dev_X"],
            artifacts["dev_y"],
            artifacts["dev_sample_ids"],
            artifacts["ndd_X"],
            artifacts["ndd_sample_ids"],
            artifacts["ext_X"],
            artifacts["ext_y"],
            artifacts["ext_sample_ids"],
            artifacts["gene_space"],
        )
        dev_zscore_summary = validate_zscore_sanity(artifacts["dev_X"], dataset_name="dev_X")
        ext_zscore_summary = validate_zscore_sanity(artifacts["ext_X"], dataset_name="ext_X")
        fold_summary = validate_folds_cover_development_once(
            artifacts["dev_folds"],
            artifacts["dev_sample_ids"],
        )
        pca_summary = compute_dev_pca(artifacts["dev_X"])

        pc1_warning = None
        if detect_pc1_perfect_class_separation(pca_summary["coordinates"], artifacts["dev_y"]):
            pc1_warning = (
                "PC1 appears perfectly class-separating in the development PCA scatter. "
                "This is reported as a warning only and is not a performance claim or model-selection signal."
            )

        write_phase1_gate_report(
            REPORT_PATH,
            shape_summary=shape_summary,
            dev_zscore_summary=dev_zscore_summary,
            ext_zscore_summary=ext_zscore_summary,
            fold_summary=fold_summary,
            pca_summary=pca_summary,
            pc1_warning=pc1_warning,
        )
        plot_cohort_overview(
            FIGURE_PATH,
            dev_y=artifacts["dev_y"],
            ndd_X=artifacts["ndd_X"],
            ext_y=artifacts["ext_y"],
            pca_summary=pca_summary,
        )
        append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

        print("PASS")
        print(
            f"dev shape: {shape_summary['dev_shape']} label counts: "
            f"HC={shape_summary['dev_label_counts']['HC']} PD={shape_summary['dev_label_counts']['PD']}"
        )
        print(f"ndd shape: {shape_summary['ndd_shape']}")
        print(
            f"external shape: {shape_summary['ext_shape']} label counts: "
            f"HC={shape_summary['ext_label_counts']['HC']} PD={shape_summary['ext_label_counts']['PD']}"
        )
        print(f"gene count: {shape_summary['gene_count']}")
        print(
            "PCA explained variance ratio: "
            f"PC1={float(pca_summary['explained_variance_ratio'][0]):.6f} "
            f"PC2={float(pca_summary['explained_variance_ratio'][1]):.6f}"
        )
        print("No modeling was performed.")
        return 0
    except Exception as exc:
        print("FAIL")
        print(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
