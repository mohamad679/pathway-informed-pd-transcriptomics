from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from priors.gmt import (
    classify_rna_processing_pathways,
    filter_gene_sets_to_gene_space,
    read_gmt,
)


def test_read_gmt_normalizes_and_parses_members(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.gmt"
    path.write_text("PATH_A\tdescription\t snca \tMAPT\tSNCA\n", encoding="utf-8")

    assert read_gmt(path) == {"PATH_A": {"SNCA", "MAPT"}}


def test_filter_gene_sets_to_gene_space_and_minimum() -> None:
    gene_sets = {"KEEP": {"snca", "MAPT", "GBA1"}, "DROP": {"SNCA", "PINK1"}}

    assert filter_gene_sets_to_gene_space(gene_sets, ["SNCA", "MAPT", "GBA1"], min_genes=3) == {
        "KEEP": {"SNCA", "MAPT", "GBA1"}
    }


def test_rna_processing_classification() -> None:
    assert classify_rna_processing_pathways("REACTOME_MRNA_SPLICING")
    assert classify_rna_processing_pathways("KEGG_SPLICEOSOME")
    assert not classify_rna_processing_pathways("HALLMARK_TNFA_SIGNALING")
