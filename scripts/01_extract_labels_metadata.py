from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.labels import (
    GSE6613_FIELDS_USED,
    GSE99039_FIELDS_USED,
    extract_candidate_label_fields,
    infer_gse6613_group,
    infer_gse99039_group,
    summarize_label_counts,
    write_label_audit_report,
)
from data.metadata import parse_series_matrix_metadata


ACCESSION_TO_PATH = {
    "GSE99039": ROOT / "data" / "raw" / "GSE99039" / "GSE99039_series_matrix.txt.gz",
    "GSE6613": ROOT / "data" / "raw" / "GSE6613" / "GSE6613_series_matrix.txt.gz",
}
REPORT_PATH = ROOT / "docs" / "label_audit.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 1 step 3 metadata-only label extraction

- GSE6613 exclusion rule was defined before reading expression values.
- No expression table read.
- No modeling.
- No processed matrices created.
"""

RULE_TEXT = {
    "GSE99039": (
        "Use metadata text only. Primary development task maps `disease label: IPD` to `PD` "
        "and `disease label: CONTROL` to `HC`. Held-out NDD maps `HD`, `HD_HD_BATCH`, "
        "`MSA`, `PSP`, `CBD`, `PD_DEMENTIA`, and `Vascular dementia` to `NDD`. "
        "Exclude `GPD`, `GENETIC_UNAFFECTED`, `DRD`, `DRD-DYT5`, and `ATYPICAL_PD` "
        "from primary development training and held-out NDD scoring."
    ),
    "GSE6613": (
        "Use metadata text only. Explicit exclusion rule: samples marked `neurological disease control` "
        "in metadata are labeled `EXCLUDE` before any expression values are read; "
        "`healthy control` maps to `HC`; `Parkinson's disease` maps to `PD`."
    ),
}


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def build_sample_to_group(accession: str, parsed_metadata: dict[str, object]) -> dict[str, str]:
    samples = parsed_metadata.get("samples", {})
    if not isinstance(samples, dict):
        raise TypeError(f"{accession} metadata missing sample records")

    infer_group = infer_gse99039_group if accession == "GSE99039" else infer_gse6613_group
    sample_to_group: dict[str, str] = {}
    for sample_id, sample_metadata in samples.items():
        if not isinstance(sample_metadata, dict):
            raise TypeError(f"{accession} sample {sample_id} metadata is not a dictionary")
        sample_to_group[str(sample_id)] = infer_group(sample_metadata)
    return sample_to_group


def main() -> None:
    accession_to_summary: dict[str, dict[str, object]] = {}

    for accession, raw_path in ACCESSION_TO_PATH.items():
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing required raw series matrix file: {raw_path}")

        parsed = parse_series_matrix_metadata(raw_path)
        if parsed.get("expression_table_read") is not False:
            raise RuntimeError(f"{accession} expression_table_read must remain False")

        sample_to_group = build_sample_to_group(accession, parsed)
        label_counts = summarize_label_counts(accession, sample_to_group)
        if label_counts.get("UNKNOWN", 0) > 0:
            raise RuntimeError(f"{accession} has UNKNOWN labels: {label_counts['UNKNOWN']}")

        fields_used = GSE99039_FIELDS_USED if accession == "GSE99039" else GSE6613_FIELDS_USED
        accession_to_summary[accession] = {
            "path": raw_path.relative_to(ROOT).as_posix(),
            "sample_count": parsed["sample_count"],
            "metadata_line_count": parsed["metadata_line_count"],
            "candidate_fields": extract_candidate_label_fields(parsed),
            "fields_used": fields_used,
            "rule_text": RULE_TEXT[accession],
            "label_counts": label_counts,
            "expression_table_read": parsed["expression_table_read"],
        }

    write_label_audit_report(accession_to_summary, REPORT_PATH)
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

    counts_99039 = accession_to_summary["GSE99039"]["label_counts"]
    counts_6613 = accession_to_summary["GSE6613"]["label_counts"]
    print(
        "GSE99039: "
        f"PD={counts_99039['PD']} HC={counts_99039['HC']} "
        f"NDD={counts_99039['NDD']} UNKNOWN={counts_99039['UNKNOWN']}"
    )
    print(
        "GSE6613: "
        f"PD={counts_6613['PD']} HC={counts_6613['HC']} "
        f"EXCLUDE={counts_6613['EXCLUDE']} UNKNOWN={counts_6613['UNKNOWN']}"
    )


if __name__ == "__main__":
    main()
