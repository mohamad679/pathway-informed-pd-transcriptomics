from __future__ import annotations

import gzip
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.platforms import (
    build_probe_to_symbol_table,
    detect_gene_symbol_column,
    detect_probe_id_column,
    normalize_gene_symbol,
    parse_gpl_annotation,
)


def write_annotation_fixture(path: Path) -> None:
    content = "\n".join(
        [
            "# synthetic platform annotation",
            "!platform_table_begin",
            "ID\tGene Symbol\tDescription",
            "1007_s_at\tDDR1 /// MIR4640\tDiscoidin domain receptor tyrosine kinase 1",
            "1053_at\tRFC2;H2BC12\tReplication factor C subunit 2",
            "117_at\t---\tmissing symbol row",
            "!platform_table_end",
        ]
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content)


def test_normalize_gene_symbol() -> None:
    assert normalize_gene_symbol(" SNCA ") == "SNCA"
    assert normalize_gene_symbol("---") is None
    assert normalize_gene_symbol("Gene Symbol") is None


def test_detect_probe_and_gene_symbol_columns() -> None:
    columns = ["Transcript Cluster ID", "ID", "Gene Symbol", "Description"]

    assert detect_probe_id_column(columns) == "ID"
    assert detect_gene_symbol_column(columns) == "Gene Symbol"


def test_parse_output_shape(tmp_path: Path) -> None:
    platform_path = tmp_path / "GPLTEST.annot.gz"
    write_annotation_fixture(platform_path)

    parsed = parse_gpl_annotation(platform_path)
    summary = build_probe_to_symbol_table("GPLTEST", platform_path)
    table = summary["table"]

    assert parsed.shape == (3, 3)
    assert list(table.columns) == ["probe_id", "gene_symbol"]
    assert table.shape == (4, 2)
    assert summary["total_probes"] == 3
    assert summary["mapped_probes"] == 2
    assert summary["unique_gene_symbols"] == 4
