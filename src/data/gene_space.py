from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("probe_id", "gene_symbol")


def load_probe_to_symbol_table(path: str | Path) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.is_file():
        raise FileNotFoundError(
            f"Missing annotation table: {table_path}. "
            "Run scripts/01_fetch_platform_annotations.py first."
        )

    table = pd.read_csv(table_path, sep="\t", dtype=str).fillna("")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing_columns:
        raise ValueError(
            f"Annotation table {table_path} is missing required columns: {missing_columns}"
        )

    normalized = table.loc[:, list(REQUIRED_COLUMNS)].copy()
    normalized["probe_id"] = normalized["probe_id"].astype(str).str.strip()
    normalized["gene_symbol"] = normalized["gene_symbol"].astype(str).str.strip()
    normalized = normalized[
        (normalized["probe_id"] != "") & (normalized["gene_symbol"] != "")
    ].drop_duplicates()
    return normalized.reset_index(drop=True)


def collect_gene_symbols(table: pd.DataFrame) -> set[str]:
    if "gene_symbol" not in table.columns:
        raise ValueError("Expected a 'gene_symbol' column in the annotation table.")
    return {
        str(symbol).strip()
        for symbol in table["gene_symbol"].tolist()
        if str(symbol).strip()
    }


def build_annotation_gene_intersection(
    gpl570_table: pd.DataFrame, gpl96_table: pd.DataFrame
) -> list[str]:
    shared_symbols = collect_gene_symbols(gpl570_table) & collect_gene_symbols(gpl96_table)
    return sorted(shared_symbols)


def write_gene_space(gene_symbols: list[str] | set[str], output_path: str | Path) -> None:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_symbols = sorted({str(symbol).strip() for symbol in gene_symbols if str(symbol).strip()})
    content = "\n".join(ordered_symbols)
    if content:
        content += "\n"
    target_path.write_text(content, encoding="utf-8")


def write_gene_space_audit(
    *,
    gpl570_path: str | Path,
    gpl96_path: str | Path,
    gpl570_unique_symbols: int,
    gpl96_unique_symbols: int,
    intersection_symbols: list[str] | set[str],
    output_path: str | Path,
    gene_space_output_path: str | Path,
) -> None:
    audit_path = Path(output_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    intersection_count = len(set(intersection_symbols))
    content = "\n".join(
        [
            "# Gene Space Audit",
            "",
            "This report covers Phase 1 annotation-only cross-platform gene-space construction.",
            "Only platform annotation probe-to-symbol tables were read.",
            "No GEO series expression matrix was read.",
            "No numeric expression values were parsed.",
            "",
            "## Inputs",
            "",
            f"- GPL570 annotation table: `{Path(gpl570_path).as_posix()}`",
            f"- GPL96 annotation table: `{Path(gpl96_path).as_posix()}`",
            "",
            "## Results",
            "",
            f"- GPL570 unique gene symbols: `{gpl570_unique_symbols}`",
            f"- GPL96 unique gene symbols: `{gpl96_unique_symbols}`",
            f"- Annotation-only intersection size: `{intersection_count}`",
            f"- Gene space output: `{Path(gene_space_output_path).as_posix()}`",
            "",
            "## Boundary Confirmation",
            "",
            "- Expression matrices read: `no`",
            "- Expression-derived feature selection performed: `no`",
            "- Labels used: `no`",
            "- Modeling performed: `no`",
        ]
    )
    audit_path.write_text(content + "\n", encoding="utf-8")
