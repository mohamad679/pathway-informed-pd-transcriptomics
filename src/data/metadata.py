from __future__ import annotations

import csv
import gzip
from collections import Counter
from pathlib import Path


TABLE_BEGIN_MARKER = "!series_matrix_table_begin"


def iter_series_matrix_metadata_lines(path: str | Path):
    file_path = Path(path)
    with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith(TABLE_BEGIN_MARKER):
                return
            yield line


def _parse_metadata_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t", quotechar='"'))


def _normalize_value(value: str) -> str:
    return value.strip()


def _store_series_value(series: dict[str, object], key: str, values: list[str]) -> None:
    cleaned_values = [_normalize_value(value) for value in values if value.strip()]
    if not cleaned_values:
        return
    existing = series.get(key)
    if existing is None:
        series[key] = cleaned_values[0] if len(cleaned_values) == 1 else cleaned_values
        return
    if not isinstance(existing, list):
        existing = [existing]
    existing.extend(cleaned_values)
    series[key] = existing


def _sample_ids_from_series(parsed_rows: dict[str, object]) -> list[str]:
    raw_sample_ids = parsed_rows.get("sample_id")
    if raw_sample_ids is None:
        return []
    if isinstance(raw_sample_ids, list):
        values = raw_sample_ids
    else:
        values = [raw_sample_ids]
    sample_ids: list[str] = []
    for value in values:
        sample_ids.extend(str(value).split())
    return sample_ids


def _ensure_sample_order(
    sample_order: list[str], sample_rows: list[tuple[str, list[str]]], series: dict[str, object]
) -> list[str]:
    if sample_order:
        return sample_order
    for sample_key, values in sample_rows:
        if sample_key == "geo_accession":
            return values
    for sample_key, values in sample_rows:
        if sample_key == "title":
            return [f"sample_{index + 1}" for index in range(len(values))]
    return _sample_ids_from_series(series)


def parse_series_matrix_metadata(path: str | Path) -> dict[str, object]:
    series: dict[str, object] = {}
    sample_rows: list[tuple[str, list[str]]] = []
    sample_order: list[str] = []
    metadata_line_count = 0

    for line in iter_series_matrix_metadata_lines(path):
        if not line:
            continue
        metadata_line_count += 1
        fields = _parse_metadata_line(line)
        if not fields:
            continue

        key = fields[0]
        values = [_normalize_value(value) for value in fields[1:]]
        if key.startswith("!Series_"):
            _store_series_value(series, key[len("!Series_") :].lower(), values)
            continue
        if key.startswith("!Sample_"):
            sample_key = key[len("!Sample_") :].lower()
            sample_rows.append((sample_key, values))
            if sample_key == "geo_accession":
                sample_order = values

    sample_order = _ensure_sample_order(sample_order, sample_rows, series)
    samples: dict[str, dict[str, object]] = {
        sample_id: {"sample_id": sample_id} for sample_id in sample_order
    }

    for sample_key, values in sample_rows:
        if sample_order and len(values) != len(sample_order):
            raise ValueError(
                f"Sample metadata row '{sample_key}' has {len(values)} values for "
                f"{len(sample_order)} samples"
            )
        for index, value in enumerate(values):
            sample_id = sample_order[index] if sample_order else f"sample_{index + 1}"
            sample_record = samples.setdefault(sample_id, {"sample_id": sample_id})
            existing = sample_record.get(sample_key)
            if existing is None:
                sample_record[sample_key] = value
            elif isinstance(existing, list):
                existing.append(value)
            else:
                sample_record[sample_key] = [existing, value]

    accession = series.get("geo_accession")
    if not accession:
        accession = Path(path).name.split("_", 1)[0]

    return {
        "accession": accession,
        "path": str(Path(path)),
        "series": series,
        "samples": samples,
        "sample_order": sample_order,
        "sample_count": len(sample_order),
        "metadata_line_count": metadata_line_count,
        "expression_table_read": False,
    }


def summarize_platforms(parsed: dict[str, object]) -> dict[str, int]:
    sample_count = int(parsed["sample_count"])
    series = parsed["series"]
    if not isinstance(series, dict):
        return {}

    series_platforms = series.get("platform_id")
    if isinstance(series_platforms, list):
        platform_ids = [str(value) for value in series_platforms]
    elif series_platforms is None:
        platform_ids = []
    else:
        platform_ids = [str(series_platforms)]

    samples = parsed["samples"]
    if not isinstance(samples, dict):
        return {}

    sample_platforms = Counter()
    for sample in samples.values():
        if not isinstance(sample, dict):
            continue
        platform_value = sample.get("platform_id")
        if platform_value:
            sample_platforms[str(platform_value)] += 1

    if sample_platforms:
        return dict(sample_platforms)
    if len(platform_ids) == 1:
        return {platform_ids[0]: sample_count}
    return {platform_id: 0 for platform_id in platform_ids}


def summarize_sample_characteristics(parsed: dict[str, object]) -> dict[str, dict[str, object]]:
    samples = parsed["samples"]
    if not isinstance(samples, dict):
        return {}

    summary: dict[str, dict[str, object]] = {}
    platforms = summarize_platforms(parsed)
    fallback_platform = next(iter(platforms), None) if len(platforms) == 1 else None

    for sample_id, sample in samples.items():
        if not isinstance(sample, dict):
            continue
        record: dict[str, object] = {}
        for key in ("title", "geo_accession", "source_name_ch1", "organism_ch1"):
            value = sample.get(key)
            if value:
                record[key] = value

        platform_id = sample.get("platform_id", fallback_platform)
        if platform_id:
            record["platform_id"] = platform_id

        characteristic_fields = {
            key: value
            for key, value in sample.items()
            if key.startswith("characteristics_") and value
        }
        if characteristic_fields:
            record["characteristics"] = characteristic_fields

        summary[sample_id] = record

    return summary


def write_metadata_verification_report(
    accession_to_summary: dict[str, dict[str, object]], output_path: str | Path
) -> None:
    lines = [
        "# Metadata Verification",
        "",
        "This report covers Phase 1 metadata-only verification of cached GEO series matrix files.",
        "The expression tables were not read.",
        "",
    ]

    for accession in sorted(accession_to_summary):
        summary = accession_to_summary[accession]
        platforms = summary.get("platforms", {})
        platform_text = ", ".join(
            f"`{platform}` ({count} samples)" for platform, count in sorted(platforms.items())
        )
        if not platform_text:
            platform_text = "None observed"

        lines.extend(
            [
                f"## {accession}",
                "",
                f"- Raw file: `{summary['path']}`",
                f"- Sample count: `{summary['sample_count']}`",
                f"- Metadata lines read before table marker: `{summary['metadata_line_count']}`",
                f"- Observed platforms: {platform_text}",
                "- Expression table read: `no`",
                "",
            ]
        )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
