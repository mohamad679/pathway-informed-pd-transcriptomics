from __future__ import annotations

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from data.gene_space import load_probe_to_symbol_table
from data.labels import infer_gse6613_group, infer_gse99039_group, summarize_label_counts
from data.metadata import parse_series_matrix_metadata


TABLE_BEGIN_MARKER = "!series_matrix_table_begin"
TABLE_END_MARKER = "!series_matrix_table_end"
LABEL_ENCODING = {"HC": 0, "PD": 1}


def _open_series_text(path: str | Path):
    file_path = Path(path)
    if file_path.suffix == ".gz":
        return gzip.open(file_path, "rt", encoding="utf-8", errors="replace")
    return file_path.open("r", encoding="utf-8", errors="replace")


def _parse_table_line(line: str) -> list[str]:
    return next(csv.reader([line], delimiter="\t", quotechar='"'))


def _build_sample_to_group(accession: str, parsed_metadata: dict[str, object]) -> dict[str, str]:
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


def _validate_known_labels(accession: str, sample_to_group: dict[str, str]) -> None:
    label_counts = summarize_label_counts(accession, sample_to_group)
    if label_counts.get("UNKNOWN", 0) > 0:
        raise RuntimeError(f"{accession} has UNKNOWN labels: {label_counts['UNKNOWN']}")


def _align_expression_to_sample_order(
    expression_df: pd.DataFrame, sample_order: list[str], accession: str
) -> pd.DataFrame:
    expression_columns = list(expression_df.columns)
    if expression_columns == sample_order:
        return expression_df

    expression_set = set(expression_columns)
    sample_order_set = set(sample_order)
    missing = [sample_id for sample_id in sample_order if sample_id not in expression_set]
    extra = [sample_id for sample_id in expression_columns if sample_id not in sample_order_set]
    if missing or extra:
        raise ValueError(
            f"{accession} metadata/expression sample mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    return expression_df.loc[:, sample_order]


def _select_samples(
    gene_df: pd.DataFrame,
    sample_to_group: dict[str, str],
    allowed_groups: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    selected_sample_ids = [
        sample_id for sample_id in gene_df.columns if sample_to_group.get(sample_id) in allowed_groups
    ]
    return gene_df.loc[:, selected_sample_ids], [sample_to_group[sample_id] for sample_id in selected_sample_ids]


def _ensure_gene_space_coverage(gene_df: pd.DataFrame, gene_space: list[str], accession: str) -> pd.DataFrame:
    missing_genes = [gene for gene in gene_space if gene not in gene_df.index]
    if missing_genes:
        raise ValueError(
            f"{accession} is missing {len(missing_genes)} annotation-space genes after aggregation"
        )
    return gene_df.loc[gene_space]


def read_series_matrix_expression(path: str | Path) -> pd.DataFrame:
    rows: list[list[str]] = []
    inside_table = False
    saw_table_begin = False
    saw_table_end = False

    with _open_series_text(path) as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.startswith(TABLE_BEGIN_MARKER):
                inside_table = True
                saw_table_begin = True
                continue
            if line.startswith(TABLE_END_MARKER):
                saw_table_end = True
                break
            if inside_table:
                if line:
                    rows.append(_parse_table_line(line))

    if not saw_table_begin:
        raise ValueError(f"Missing {TABLE_BEGIN_MARKER} in {path}")
    if not saw_table_end:
        raise ValueError(f"Missing {TABLE_END_MARKER} in {path}")
    if not rows:
        raise ValueError(f"No expression table rows found in {path}")

    header = rows[0]
    if len(header) < 2:
        raise ValueError(f"Expression header must include ID_REF and at least one sample in {path}")
    if header[0] != "ID_REF":
        raise ValueError(f"Expected first expression column to be ID_REF in {path}, found {header[0]!r}")

    sample_columns = [str(value).strip() for value in header[1:]]
    data_rows = rows[1:]
    for row_index, row in enumerate(data_rows, start=2):
        if len(row) != len(header):
            raise ValueError(
                f"Expression row {row_index} in {path} has {len(row)} columns; expected {len(header)}"
            )

    expression_df = pd.DataFrame(
        [row[1:] for row in data_rows],
        index=[str(row[0]).strip() for row in data_rows],
        columns=sample_columns,
        dtype=str,
    )
    expression_df.index.name = "probe_id"

    try:
        numeric_df = expression_df.apply(lambda column: pd.to_numeric(column, errors="raise"))
    except Exception as exc:  # pragma: no cover - pandas raises parser-specific exceptions
        raise ValueError(f"Non-numeric expression value encountered in {path}") from exc
    return numeric_df


def load_gene_space(path: str | Path) -> list[str]:
    gene_space_path = Path(path)
    if not gene_space_path.is_file():
        raise FileNotFoundError(
            f"Missing gene space file: {gene_space_path}. "
            "Run scripts/01_build_annotation_gene_space.py first."
        )

    genes = [line.strip() for line in gene_space_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = len(genes) - len(set(genes))
    if duplicates:
        raise ValueError(f"Gene space file {gene_space_path} contains duplicate gene symbols")
    if not genes:
        raise ValueError(f"Gene space file {gene_space_path} is empty")
    return genes


def load_probe_to_symbol(path: str | Path) -> pd.DataFrame:
    return load_probe_to_symbol_table(path)


def aggregate_probes_to_genes(
    expression_df: pd.DataFrame, probe_to_symbol_df: pd.DataFrame, gene_space: list[str]
) -> pd.DataFrame:
    if expression_df.index.has_duplicates:
        raise ValueError("Expression dataframe contains duplicate probe identifiers")

    mapping = probe_to_symbol_df.loc[:, ["probe_id", "gene_symbol"]].copy()
    mapping["probe_id"] = mapping["probe_id"].astype(str).str.strip()
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.strip()

    merged = mapping.merge(
        expression_df.reset_index(),
        how="inner",
        on="probe_id",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No probes overlapped between expression data and annotation mapping")

    allowed_genes = set(gene_space)
    merged = merged[merged["gene_symbol"].isin(allowed_genes)].copy()
    if merged.empty:
        raise ValueError("No annotation-mapped genes overlapped the fixed gene space")

    sample_columns = list(expression_df.columns)
    gene_df = merged.groupby("gene_symbol", sort=False)[sample_columns].median()
    return gene_df


def within_sample_zscore(gene_df: pd.DataFrame) -> pd.DataFrame:
    sample_means = gene_df.mean(axis=0)
    sample_stds = gene_df.std(axis=0, ddof=0)
    zero_std_samples = sample_stds.index[sample_stds == 0].tolist()
    if zero_std_samples:
        raise ValueError(f"Cannot z-score samples with zero across-gene standard deviation: {zero_std_samples}")
    return gene_df.sub(sample_means, axis=1).div(sample_stds, axis=1)


def build_processed_matrices(
    *,
    gse99039_series_path: str | Path,
    gse6613_series_path: str | Path,
    gpl570_probe_to_symbol_path: str | Path,
    gpl96_probe_to_symbol_path: str | Path,
    gene_space_path: str | Path,
) -> dict[str, object]:
    gene_space = load_gene_space(gene_space_path)
    gpl570_mapping = load_probe_to_symbol(gpl570_probe_to_symbol_path)
    gpl96_mapping = load_probe_to_symbol(gpl96_probe_to_symbol_path)

    gse99039_metadata = parse_series_matrix_metadata(gse99039_series_path)
    gse99039_sample_to_group = _build_sample_to_group("GSE99039", gse99039_metadata)
    _validate_known_labels("GSE99039", gse99039_sample_to_group)

    gse99039_expression = read_series_matrix_expression(gse99039_series_path)
    gse99039_expression = _align_expression_to_sample_order(
        gse99039_expression,
        list(gse99039_metadata["sample_order"]),
        "GSE99039",
    )
    gse99039_gene_df = aggregate_probes_to_genes(gse99039_expression, gpl570_mapping, gene_space)
    gse99039_gene_df = _ensure_gene_space_coverage(gse99039_gene_df, gene_space, "GSE99039")
    gse99039_gene_df = within_sample_zscore(gse99039_gene_df)

    dev_gene_df, dev_groups = _select_samples(gse99039_gene_df, gse99039_sample_to_group, ("PD", "HC"))
    ndd_gene_df, ndd_groups = _select_samples(gse99039_gene_df, gse99039_sample_to_group, ("NDD",))
    if any(group != "NDD" for group in ndd_groups):
        raise AssertionError("Held-out NDD matrix contains non-NDD labels")

    gse6613_metadata = parse_series_matrix_metadata(gse6613_series_path)
    gse6613_sample_to_group = _build_sample_to_group("GSE6613", gse6613_metadata)
    _validate_known_labels("GSE6613", gse6613_sample_to_group)

    gse6613_expression = read_series_matrix_expression(gse6613_series_path)
    gse6613_expression = _align_expression_to_sample_order(
        gse6613_expression,
        list(gse6613_metadata["sample_order"]),
        "GSE6613",
    )
    gse6613_gene_df = aggregate_probes_to_genes(gse6613_expression, gpl96_mapping, gene_space)
    gse6613_gene_df = _ensure_gene_space_coverage(gse6613_gene_df, gene_space, "GSE6613")
    gse6613_gene_df = within_sample_zscore(gse6613_gene_df)
    ext_gene_df, ext_groups = _select_samples(gse6613_gene_df, gse6613_sample_to_group, ("PD", "HC"))

    dev_y = np.asarray([LABEL_ENCODING[group] for group in dev_groups], dtype=np.int64)
    ext_y = np.asarray([LABEL_ENCODING[group] for group in ext_groups], dtype=np.int64)

    return {
        "gene_space": gene_space,
        "dev_X": dev_gene_df.transpose().to_numpy(dtype=np.float32),
        "dev_y": dev_y,
        "dev_groups": np.asarray(dev_groups),
        "dev_sample_ids": np.asarray(dev_gene_df.columns),
        "ndd_X": ndd_gene_df.transpose().to_numpy(dtype=np.float32),
        "ndd_sample_ids": np.asarray(ndd_gene_df.columns),
        "ext_X": ext_gene_df.transpose().to_numpy(dtype=np.float32),
        "ext_y": ext_y,
        "ext_sample_ids": np.asarray(ext_gene_df.columns),
        "dev_label_counts": {
            "HC": int(np.sum(dev_y == LABEL_ENCODING["HC"])),
            "PD": int(np.sum(dev_y == LABEL_ENCODING["PD"])),
        },
        "ext_label_counts": {
            "HC": int(np.sum(ext_y == LABEL_ENCODING["HC"])),
            "PD": int(np.sum(ext_y == LABEL_ENCODING["PD"])),
        },
    }
