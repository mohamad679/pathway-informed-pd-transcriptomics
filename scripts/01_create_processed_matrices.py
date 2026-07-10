from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.expression import build_processed_matrices


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"

GSE99039_SERIES_PATH = RAW_DIR / "GSE99039" / "GSE99039_series_matrix.txt.gz"
GSE6613_SERIES_PATH = RAW_DIR / "GSE6613" / "GSE6613_series_matrix.txt.gz"
GPL570_PATH = PROCESSED_DIR / "GPL570_probe_to_symbol.tsv"
GPL96_PATH = PROCESSED_DIR / "GPL96_probe_to_symbol.tsv"
GENE_SPACE_INPUT_PATH = PROCESSED_DIR / "gene_space_annotation_intersection.txt"

DEV_X_PATH = PROCESSED_DIR / "dev_X.npy"
DEV_Y_PATH = PROCESSED_DIR / "dev_y.npy"
DEV_GROUPS_PATH = PROCESSED_DIR / "dev_groups.npy"
DEV_SAMPLE_IDS_PATH = PROCESSED_DIR / "dev_sample_ids.txt"
NDD_X_PATH = PROCESSED_DIR / "ndd_X.npy"
NDD_SAMPLE_IDS_PATH = PROCESSED_DIR / "ndd_sample_ids.txt"
EXT_X_PATH = PROCESSED_DIR / "ext_X.npy"
EXT_Y_PATH = PROCESSED_DIR / "ext_y.npy"
EXT_SAMPLE_IDS_PATH = PROCESSED_DIR / "ext_sample_ids.txt"
GENE_SPACE_OUTPUT_PATH = PROCESSED_DIR / "gene_space.txt"
AUDIT_PATH = DOCS_DIR / "processed_matrix_audit.md"
DECISION_LOG_PATH = DOCS_DIR / "decision_log.md"

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 1 processed matrix creation

- Parsed cached GEO series-matrix expression tables only between `!series_matrix_table_begin` and `!series_matrix_table_end`.
- Fixed the shared gene space from the existing annotation-only intersection file before reading external-cohort expression values.
- Mapped probes to genes using cached platform annotation tables only and aggregated multiple probes per gene by median.
- Applied metadata-only labels from `src/data/labels.py`.
- Applied within-sample gene-wise z-scoring only; no train-fitted scaler was used.
- No train/validation/test splits, pathway masks, MSigDB logic, baselines, or modeling artifacts were created.
- External expression values were not used for gene selection, probe selection, fitting, ranking, scaling, or imputation.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def write_lines(path: Path, values: np.ndarray | list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(value) for value in values)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_processed_matrix_audit(result: dict[str, object]) -> None:
    gene_space = result["gene_space"]
    dev_X = result["dev_X"]
    ndd_X = result["ndd_X"]
    ext_X = result["ext_X"]
    dev_counts = result["dev_label_counts"]
    ext_counts = result["ext_label_counts"]

    lines = [
        "# Processed Matrix Audit",
        "",
        "This report covers Phase 1 expression parsing and processed matrix creation.",
        "The shared gene space came from the existing annotation-only intersection file.",
        "Probe-to-gene mapping used cached platform annotation tables only.",
        "External expression values were not used for gene selection.",
        "Within-sample z-scoring was applied independently within each sample across genes.",
        "No train-fitted scaler was used.",
        "",
        "## Inputs",
        "",
        f"- GSE99039 series matrix: `{GSE99039_SERIES_PATH.relative_to(ROOT).as_posix()}`",
        f"- GSE6613 series matrix: `{GSE6613_SERIES_PATH.relative_to(ROOT).as_posix()}`",
        f"- GPL570 probe mapping: `{GPL570_PATH.relative_to(ROOT).as_posix()}`",
        f"- GPL96 probe mapping: `{GPL96_PATH.relative_to(ROOT).as_posix()}`",
        f"- Fixed annotation-only gene space: `{GENE_SPACE_INPUT_PATH.relative_to(ROOT).as_posix()}`",
        "",
        "## Outputs",
        "",
        f"- Development matrix shape: `{dev_X.shape}`",
        f"- Development label counts: `HC={dev_counts['HC']}`, `PD={dev_counts['PD']}`",
        f"- Held-out NDD matrix shape: `{ndd_X.shape}`",
        f"- External matrix shape: `{ext_X.shape}`",
        f"- External label counts: `HC={ext_counts['HC']}`, `PD={ext_counts['PD']}`",
        f"- Ordered gene count: `{len(gene_space)}`",
        f"- Saved gene space file: `{GENE_SPACE_OUTPUT_PATH.relative_to(ROOT).as_posix()}`",
        "",
        "## Boundary Confirmation",
        "",
        "- Train/validation/test splits created: `no`",
        "- Modeling performed: `no`",
        "- Baselines implemented: `no`",
        "- Pathway masks implemented: `no`",
        "- MSigDB logic implemented: `no`",
        "- Train-fitted scaler used: `no`",
        "- External expression used for gene selection: `no`",
    ]
    AUDIT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    result = build_processed_matrices(
        gse99039_series_path=GSE99039_SERIES_PATH,
        gse6613_series_path=GSE6613_SERIES_PATH,
        gpl570_probe_to_symbol_path=GPL570_PATH,
        gpl96_probe_to_symbol_path=GPL96_PATH,
        gene_space_path=GENE_SPACE_INPUT_PATH,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DEV_X_PATH, result["dev_X"])
    np.save(DEV_Y_PATH, result["dev_y"])
    np.save(DEV_GROUPS_PATH, result["dev_groups"])
    np.save(NDD_X_PATH, result["ndd_X"])
    np.save(EXT_X_PATH, result["ext_X"])
    np.save(EXT_Y_PATH, result["ext_y"])
    write_lines(DEV_SAMPLE_IDS_PATH, result["dev_sample_ids"])
    write_lines(NDD_SAMPLE_IDS_PATH, result["ndd_sample_ids"])
    write_lines(EXT_SAMPLE_IDS_PATH, result["ext_sample_ids"])
    write_lines(GENE_SPACE_OUTPUT_PATH, result["gene_space"])
    write_processed_matrix_audit(result)
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

    dev_X = result["dev_X"]
    ndd_X = result["ndd_X"]
    ext_X = result["ext_X"]
    dev_counts = result["dev_label_counts"]
    ext_counts = result["ext_label_counts"]
    print(f"dev shape: {dev_X.shape} label counts: HC={dev_counts['HC']} PD={dev_counts['PD']}")
    print(f"ndd shape: {ndd_X.shape}")
    print(f"external shape: {ext_X.shape} label counts: HC={ext_counts['HC']} PD={ext_counts['PD']}")
    print(f"gene count: {len(result['gene_space'])}")
    print("No train-fitted scaler was used.")
    print("External expression was not used for gene selection.")


if __name__ == "__main__":
    main()
