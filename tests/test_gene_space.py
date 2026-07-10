from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.gene_space import (
    build_annotation_gene_intersection,
    collect_gene_symbols,
    load_probe_to_symbol_table,
    write_gene_space,
)


def test_collect_gene_symbols() -> None:
    table = pd.DataFrame(
        {
            "probe_id": ["1007_s_at", "1053_at", "117_at", "121_at"],
            "gene_symbol": ["DDR1", "RFC2", "DDR1", "SNCA"],
        }
    )

    assert collect_gene_symbols(table) == {"DDR1", "RFC2", "SNCA"}


def test_annotation_gene_intersection() -> None:
    gpl570_table = pd.DataFrame(
        {
            "probe_id": ["a", "b", "c", "d"],
            "gene_symbol": ["SNCA", "LRRK2", "MAPT", "GBA1"],
        }
    )
    gpl96_table = pd.DataFrame(
        {
            "probe_id": ["x", "y", "z"],
            "gene_symbol": ["MAPT", "SNCA", "PINK1"],
        }
    )

    assert build_annotation_gene_intersection(gpl570_table, gpl96_table) == ["MAPT", "SNCA"]


def test_write_gene_space_is_sorted_and_deterministic(tmp_path: Path) -> None:
    output_path = tmp_path / "gene_space.txt"

    write_gene_space(["SNCA", "MAPT", "SNCA", "GBA1"], output_path)

    assert output_path.read_text(encoding="utf-8").splitlines() == ["GBA1", "MAPT", "SNCA"]


def test_load_probe_to_symbol_table_normalizes_and_filters(tmp_path: Path) -> None:
    table_path = tmp_path / "probe_to_symbol.tsv"
    table = pd.DataFrame(
        {
            "probe_id": [" a ", "b", "b", ""],
            "gene_symbol": [" SNCA ", "MAPT", "MAPT", "LRRK2"],
            "other_column": ["1", "2", "3", "4"],
        }
    )
    table.to_csv(table_path, sep="\t", index=False)

    loaded = load_probe_to_symbol_table(table_path)

    assert list(loaded.columns) == ["probe_id", "gene_symbol"]
    assert loaded.to_dict(orient="records") == [
        {"probe_id": "a", "gene_symbol": "SNCA"},
        {"probe_id": "b", "gene_symbol": "MAPT"},
    ]
