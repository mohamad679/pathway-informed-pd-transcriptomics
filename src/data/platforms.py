from __future__ import annotations

import csv
import gzip
import re
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO
from urllib.error import HTTPError, URLError

import pandas as pd
from GEOparse import utils as geoutils

from data.download import ensure_dir


PLATFORM_TABLE_BEGIN_MARKER = "!platform_table_begin"
PLATFORM_TABLE_END_MARKER = "!platform_table_end"
MISSING_SYMBOL_VALUES = {
    "",
    "---",
    "--",
    "na",
    "n/a",
    "nan",
    "null",
    "none",
    "not available",
    "unknown",
}
VALID_SYMBOL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SYMBOL_SPLIT_RE = re.compile(r"\s*///\s*|\s*//\s*|\s*[;,|]\s*")


def _platform_range_subdir(platform_id: str) -> str:
    return re.sub(r"\d{1,3}$", "nnn", platform_id.upper())


def _open_text(path: str | Path) -> TextIO:
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    return file_path.open("rt", encoding="utf-8", errors="replace")


def _iter_platform_table_lines(path: str | Path):
    inside_table = False
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith(PLATFORM_TABLE_BEGIN_MARKER):
                inside_table = True
                continue
            if line.startswith(PLATFORM_TABLE_END_MARKER):
                return
            if inside_table:
                yield line


def _read_bounded_platform_table(path: str | Path) -> pd.DataFrame:
    lines = [line for line in _iter_platform_table_lines(path) if line and not line.startswith("#")]
    if not lines:
        raise ValueError(f"No platform table found in {path}")
    rows = list(csv.reader(lines, delimiter="\t", quotechar='"'))
    if not rows:
        raise ValueError(f"Platform table in {path} is empty")
    header = rows[0]
    data_rows = rows[1:]
    return pd.DataFrame(data_rows, columns=header, dtype=str).fillna("")


def _read_annot_table(path: str | Path) -> pd.DataFrame:
    return _read_bounded_platform_table(path)


def _normalize_column_name(column: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column.lower())


def _split_symbol_cell(value: str) -> list[str]:
    cleaned = value.strip().strip('"').strip("'")
    if not cleaned:
        return []
    return [part for part in SYMBOL_SPLIT_RE.split(cleaned) if part]


def download_geo_platform(platform_id: str, raw_dir: str | Path) -> Path:
    platform_id = platform_id.upper()
    platform_dir = ensure_dir(Path(raw_dir) / platform_id)

    cached_candidates = sorted(platform_dir.glob(f"{platform_id}*"))
    if cached_candidates:
        return cached_candidates[0]

    range_subdir = _platform_range_subdir(platform_id)
    candidates = [
        (
            f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{range_subdir}/"
            f"{platform_id}/annot/{platform_id}.annot.gz",
            platform_dir / f"{platform_id}.annot.gz",
        ),
        (
            f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{range_subdir}/"
            f"{platform_id}/soft/{platform_id}_family.soft.gz",
            platform_dir / f"{platform_id}_family.soft.gz",
        ),
        (
            f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{range_subdir}/"
            f"{platform_id}/soft/{platform_id}.soft.gz",
            platform_dir / f"{platform_id}.soft.gz",
        ),
    ]

    errors: list[str] = []
    for url, destination in candidates:
        try:
            geoutils.download_from_url(url, str(destination), force=False, silent=False)
            if destination.is_file():
                return destination
        except (HTTPError, URLError, OSError, IOError, ValueError) as exc:
            errors.append(f"{url} -> {exc}")
            if destination.exists():
                destination.unlink()

    error_text = "\n".join(errors) if errors else "No candidate URLs attempted."
    raise RuntimeError(
        f"Unable to download GEO platform annotation for {platform_id}. Attempts:\n{error_text}"
    )


def parse_gpl_annotation(platform_path: str | Path) -> pd.DataFrame:
    file_path = Path(platform_path)
    name = file_path.name.lower()
    if ".annot" in name:
        table = _read_annot_table(file_path)
    else:
        table = _read_bounded_platform_table(file_path)
    columns = [str(column).strip() for column in table.columns]
    table.columns = columns
    return table


def detect_probe_id_column(columns: Iterable[str]) -> str:
    probe_candidates = []
    for column in columns:
        normalized = _normalize_column_name(column)
        if normalized == "id":
            return column
        if normalized in {"probeid", "probeids", "idref", "reporterid"}:
            probe_candidates.append((0, column))
        elif "probe" in normalized and "id" in normalized:
            probe_candidates.append((1, column))
        elif normalized.endswith("id"):
            probe_candidates.append((2, column))
    if probe_candidates:
        probe_candidates.sort(key=lambda item: (item[0], item[1]))
        return probe_candidates[0][1]
    raise ValueError(f"Could not detect probe ID column from: {list(columns)}")


def detect_gene_symbol_column(columns: Iterable[str]) -> str:
    symbol_candidates = []
    for column in columns:
        normalized = _normalize_column_name(column)
        if normalized in {"genesymbol", "symbol"}:
            symbol_candidates.append((0, column))
        elif "genesymbol" in normalized:
            symbol_candidates.append((1, column))
        elif normalized.endswith("symbol") or (
            "gene" in normalized and "symbol" in normalized
        ):
            symbol_candidates.append((2, column))
    if symbol_candidates:
        symbol_candidates.sort(key=lambda item: (item[0], item[1]))
        return symbol_candidates[0][1]
    raise ValueError(f"Could not detect gene symbol column from: {list(columns)}")


def normalize_gene_symbol(symbol: str) -> str | None:
    cleaned = symbol.strip().strip('"').strip("'")
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered in MISSING_SYMBOL_VALUES:
        return None
    if "///" in cleaned or ";" in cleaned or "," in cleaned or "|" in cleaned:
        raise ValueError("normalize_gene_symbol expects a single candidate symbol")
    if " " in cleaned:
        return None
    if not VALID_SYMBOL_RE.fullmatch(cleaned):
        return None
    return cleaned


def build_probe_to_symbol_table(
    platform_id: str, platform_path: str | Path
) -> dict[str, object]:
    annotation = parse_gpl_annotation(platform_path)
    probe_column = detect_probe_id_column(annotation.columns)
    gene_symbol_column = detect_gene_symbol_column(annotation.columns)

    records: list[dict[str, str]] = []
    total_probes = 0
    mapped_probes: set[str] = set()

    for row in annotation[[probe_column, gene_symbol_column]].itertuples(index=False):
        probe_id = str(row[0]).strip()
        if not probe_id:
            continue
        total_probes += 1
        raw_symbols = _split_symbol_cell(str(row[1]))
        clean_symbols = []
        for raw_symbol in raw_symbols:
            normalized = normalize_gene_symbol(raw_symbol)
            if normalized is not None:
                clean_symbols.append(normalized)
        unique_symbols = sorted(set(clean_symbols))
        if not unique_symbols:
            continue
        mapped_probes.add(probe_id)
        for gene_symbol in unique_symbols:
            records.append({"probe_id": probe_id, "gene_symbol": gene_symbol})

    table = pd.DataFrame.from_records(records, columns=["probe_id", "gene_symbol"])
    if not table.empty:
        table = table.drop_duplicates().sort_values(["probe_id", "gene_symbol"]).reset_index(
            drop=True
        )

    return {
        "platform_id": platform_id.upper(),
        "platform_path": str(Path(platform_path)),
        "probe_column": probe_column,
        "gene_symbol_column": gene_symbol_column,
        "table": table,
        "total_probes": total_probes,
        "mapped_probes": len(mapped_probes),
        "unique_gene_symbols": int(table["gene_symbol"].nunique()) if not table.empty else 0,
    }


def write_platform_annotation_report(
    platform_summaries: list[dict[str, object]], output_path: str | Path
) -> None:
    lines = [
        "# Platform Annotation Audit",
        "",
        "This report covers Phase 1 platform annotation acquisition and probe-to-symbol parsing only.",
        "GEO platform annotation metadata was parsed without reading any GEO series expression matrix.",
        "",
    ]

    for summary in sorted(platform_summaries, key=lambda item: str(item["platform_id"])):
        lines.extend(
            [
                f"## {summary['platform_id']}",
                "",
                f"- Cached annotation file: `{Path(str(summary['platform_path'])).as_posix()}`",
                f"- Detected probe ID column: `{summary['probe_column']}`",
                f"- Detected gene symbol column: `{summary['gene_symbol_column']}`",
                f"- Total probes observed: `{summary['total_probes']}`",
                f"- Probes with at least one mapped gene symbol: `{summary['mapped_probes']}`",
                f"- Unique mapped gene symbols: `{summary['unique_gene_symbols']}`",
                "- Expression matrices read: `no`",
                "",
            ]
        )

    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
