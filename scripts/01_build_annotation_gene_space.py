from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.gene_space import (
    build_annotation_gene_intersection,
    collect_gene_symbols,
    load_probe_to_symbol_table,
    write_gene_space,
    write_gene_space_audit,
)


PROCESSED_DIR = ROOT / "data" / "processed"
DOCS_DIR = ROOT / "docs"
GPL570_PATH = PROCESSED_DIR / "GPL570_probe_to_symbol.tsv"
GPL96_PATH = PROCESSED_DIR / "GPL96_probe_to_symbol.tsv"
GENE_SPACE_PATH = PROCESSED_DIR / "gene_space_annotation_intersection.txt"
AUDIT_PATH = DOCS_DIR / "gene_space_audit.md"
DECISION_LOG_PATH = DOCS_DIR / "decision_log.md"
MIN_INTERSECTION_SIZE = 8000

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 1 annotation-only cross-platform gene-space construction

- Built the shared gene space from `GPL570` and `GPL96` probe-to-symbol annotation tables only.
- No GEO series expression matrix was read.
- No numeric expression values were parsed.
- No labels, splits, processed expression matrices, or modeling artifacts were created.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def require_annotation_inputs() -> None:
    missing_paths = [path for path in (GPL570_PATH, GPL96_PATH) if not path.is_file()]
    if missing_paths:
        missing_text = ", ".join(path.as_posix() for path in missing_paths)
        raise FileNotFoundError(
            "Missing required platform annotation table(s): "
            f"{missing_text}. Run scripts/01_fetch_platform_annotations.py first."
        )


def main() -> None:
    require_annotation_inputs()

    gpl570_table = load_probe_to_symbol_table(GPL570_PATH)
    gpl96_table = load_probe_to_symbol_table(GPL96_PATH)

    gpl570_symbols = collect_gene_symbols(gpl570_table)
    gpl96_symbols = collect_gene_symbols(gpl96_table)
    intersection_symbols = build_annotation_gene_intersection(gpl570_table, gpl96_table)

    if len(intersection_symbols) < MIN_INTERSECTION_SIZE:
        raise RuntimeError(
            "Annotation-only gene intersection is unexpectedly small: "
            f"{len(intersection_symbols)} < {MIN_INTERSECTION_SIZE}."
        )

    write_gene_space(intersection_symbols, GENE_SPACE_PATH)
    write_gene_space_audit(
        gpl570_path=GPL570_PATH.relative_to(ROOT),
        gpl96_path=GPL96_PATH.relative_to(ROOT),
        gpl570_unique_symbols=len(gpl570_symbols),
        gpl96_unique_symbols=len(gpl96_symbols),
        intersection_symbols=intersection_symbols,
        output_path=AUDIT_PATH,
        gene_space_output_path=GENE_SPACE_PATH.relative_to(ROOT),
    )
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

    print(f"GPL570 unique gene symbols: {len(gpl570_symbols)}")
    print(f"GPL96 unique gene symbols: {len(gpl96_symbols)}")
    print(f"intersection size: {len(intersection_symbols)}")
    print("No expression matrix was read.")


if __name__ == "__main__":
    main()
