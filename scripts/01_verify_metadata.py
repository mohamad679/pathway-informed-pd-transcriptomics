from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.metadata import (
    parse_series_matrix_metadata,
    summarize_platforms,
    summarize_sample_characteristics,
    write_metadata_verification_report,
)


ACCESSION_TO_PATH = {
    "GSE99039": ROOT / "data" / "raw" / "GSE99039" / "GSE99039_series_matrix.txt.gz",
    "GSE6613": ROOT / "data" / "raw" / "GSE6613" / "GSE6613_series_matrix.txt.gz",
}
REPORT_PATH = ROOT / "docs" / "metadata_verification.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
PROVENANCE_PATH = ROOT / "docs" / "data_provenance.md"

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 1 step 2 metadata-only verification

- Verified cached GSE99039 and GSE6613 GEO series-matrix metadata headers only.
- Stopped reading at `!series_matrix_table_begin`; expression tables were not read.
- No labels confirmed.
- No processed matrices, probe mapping, gene intersection, or modeling performed.
"""

PROVENANCE_LINK_SECTION = """## Metadata Verification

- Metadata-only verification report: `docs/metadata_verification.md`
- Verification stops at `!series_matrix_table_begin`; expression tables are not read.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if path == PROVENANCE_PATH and "docs/metadata_verification.md" in existing:
        return
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    accession_to_summary: dict[str, dict[str, object]] = {}

    for accession, raw_path in ACCESSION_TO_PATH.items():
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing required raw series matrix file: {raw_path}")

        parsed = parse_series_matrix_metadata(raw_path)
        accession_to_summary[accession] = {
            "path": raw_path.relative_to(ROOT).as_posix(),
            "sample_count": parsed["sample_count"],
            "metadata_line_count": parsed["metadata_line_count"],
            "platforms": summarize_platforms(parsed),
            "sample_characteristics": summarize_sample_characteristics(parsed),
            "expression_table_read": parsed["expression_table_read"],
        }

    write_metadata_verification_report(accession_to_summary, REPORT_PATH)
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)
    append_if_missing(PROVENANCE_PATH, PROVENANCE_LINK_SECTION)

    for accession in sorted(accession_to_summary):
        summary = accession_to_summary[accession]
        print(
            f"{accession}: sample_count={summary['sample_count']} "
            f"platforms={summary['platforms']}"
        )
    print("Expression table was not read.")


if __name__ == "__main__":
    main()
