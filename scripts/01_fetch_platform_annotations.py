from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.download import ensure_dir
from data.platforms import (
    build_probe_to_symbol_table,
    download_geo_platform,
    write_platform_annotation_report,
)


RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "docs" / "platform_annotation_audit.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
PLATFORM_IDS = ("GPL570", "GPL96")

DECISION_LOG_ENTRY = """## 2026-07-10 — Phase 1 platform annotation acquisition

- Downloaded or reused cached GEO platform annotation files for `GPL570` and `GPL96`.
- Parsed platform annotation metadata only to build probe-to-symbol tables.
- No GEO series expression matrix was read.
- No labels, splits, processed expression matrices, or modeling artifacts were created.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() in existing:
        return
    separator = "" if not existing or existing.endswith("\n\n") else "\n"
    path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_dir(RAW_DIR)
    ensure_dir(PROCESSED_DIR)

    platform_summaries: list[dict[str, object]] = []

    for platform_id in PLATFORM_IDS:
        platform_path = download_geo_platform(platform_id, RAW_DIR)
        summary = build_probe_to_symbol_table(platform_id, platform_path)
        table = summary["table"]
        if not hasattr(table, "to_csv"):
            raise TypeError(f"Unexpected table object for {platform_id}: {type(table)!r}")
        output_path = PROCESSED_DIR / f"{platform_id}_probe_to_symbol.tsv"
        table.to_csv(output_path, sep="\t", index=False)
        summary["platform_path"] = platform_path.relative_to(ROOT).as_posix()
        platform_summaries.append(summary)

    write_platform_annotation_report(platform_summaries, REPORT_PATH)
    append_if_missing(DECISION_LOG_PATH, DECISION_LOG_ENTRY)

    for summary in sorted(platform_summaries, key=lambda item: str(item["platform_id"])):
        print(
            f"{summary['platform_id']} total probes={summary['total_probes']} "
            f"mapped probes={summary['mapped_probes']} "
            f"unique gene symbols={summary['unique_gene_symbols']}"
        )


if __name__ == "__main__":
    main()
