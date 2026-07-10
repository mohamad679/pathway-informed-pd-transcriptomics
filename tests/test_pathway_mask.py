from __future__ import annotations

from pathlib import Path
import sys

from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from priors.build_mask import build_pathway_mask


GENES = ["G3", "G1", "G2", "UNMAPPED"]
SETS = {"Z_PATH": {"G1", "G3"}, "A_PATH": {"G2", "G3"}, "TOO_SMALL": {"G1"}}


def test_mask_shape_order_and_csr_output() -> None:
    mask, pathway_names, gene_names, _ = build_pathway_mask(GENES, SETS, min_genes=2)

    assert sparse.isspmatrix_csr(mask)
    assert mask.shape == (2, 4)
    assert pathway_names == ["A_PATH", "Z_PATH"]
    assert gene_names == GENES


def test_mask_edges_match_memberships_and_counts_unmapped_gene() -> None:
    mask, pathway_names, _, stats = build_pathway_mask(GENES, SETS, min_genes=2)
    memberships = {
        pathway_name: {GENES[column] for column in mask.getrow(row).indices}
        for row, pathway_name in enumerate(pathway_names)
    }

    assert memberships == {"A_PATH": {"G2", "G3"}, "Z_PATH": {"G1", "G3"}}
    assert stats["n_genes_with_no_pathway"] == 1
    assert stats["n_edges"] == 4


def test_normalization_collision_prefers_exact_uppercase_gene_space_symbol() -> None:
    gene_space = ["IGK", "Igk", "SNCA"]
    gene_sets = {"IGK_PATH": {"IGK", "SNCA"}}

    mask, pathway_names, gene_names, stats = build_pathway_mask(gene_space, gene_sets, min_genes=2)

    assert mask.shape == (1, 3)
    assert pathway_names == ["IGK_PATH"]
    assert gene_names == gene_space
    assert mask.getrow(0).indices.tolist() == [0, 2]
    assert stats["n_normalized_gene_collision_groups"] == 1
    assert stats["n_genes_in_normalization_collision_groups"] == 2
