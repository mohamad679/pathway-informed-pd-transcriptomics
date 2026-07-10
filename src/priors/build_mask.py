"""Build the sparse gene-to-pathway membership mask used by future BINN work."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
from scipy import sparse

from priors.gmt import (
    classify_rna_processing_pathways,
    filter_gene_sets_to_gene_space,
    normalize_gene_symbol,
)


def build_pathway_mask(
    gene_space: Iterable[str], gene_sets: Mapping[str, Iterable[str]], min_genes: int = 10
) -> tuple[sparse.csr_matrix, list[str], list[str], dict[str, int | float]]:
    """Return a deterministic CSR pathway-by-gene membership matrix and audit statistics."""
    gene_names = [str(gene).strip() for gene in gene_space]
    normalized_genes = [normalize_gene_symbol(gene) for gene in gene_names]
    if not gene_names or any(gene is None for gene in normalized_genes):
        raise ValueError("gene_space must contain at least one valid gene symbol per line.")
    if len(set(normalized_genes)) != len(normalized_genes):
        raise ValueError("gene_space contains duplicate symbols after normalization.")

    gene_index = {gene: index for index, gene in enumerate(normalized_genes)}
    filtered_sets = filter_gene_sets_to_gene_space(gene_sets, gene_names, min_genes=min_genes)
    pathway_names = sorted(filtered_sets)
    if not pathway_names:
        raise ValueError(f"No pathways remain after filtering for at least {min_genes} genes.")

    rows: list[int] = []
    columns: list[int] = []
    for row, pathway_name in enumerate(pathway_names):
        for gene in sorted(filtered_sets[pathway_name]):
            rows.append(row)
            columns.append(gene_index[gene])

    mask = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, columns)),
        shape=(len(pathway_names), len(gene_names)),
        dtype=np.int8,
    )
    mask.sum_duplicates()
    if mask.nnz != len(rows):
        raise AssertionError("Pathway mask contains duplicate pathway-gene edges.")

    genes_per_pathway = np.asarray(mask.sum(axis=1)).ravel()
    pathways_per_gene = np.asarray(mask.sum(axis=0)).ravel()
    density = mask.nnz / (mask.shape[0] * mask.shape[1])
    assert 0 < density <= 1, "Pathway mask density must be in (0, 1]."
    assert int(genes_per_pathway.min()) >= min_genes, "A pathway violates min_genes."
    assert np.all(pathways_per_gene >= 0), "Pathway counts per gene must be non-negative."

    stats: dict[str, int | float] = {
        "n_pathways": mask.shape[0],
        "n_genes": mask.shape[1],
        "n_edges": int(mask.nnz),
        "density": float(density),
        "min_genes_per_pathway": int(genes_per_pathway.min()),
        "max_genes_per_pathway": int(genes_per_pathway.max()),
        "mean_genes_per_pathway": float(genes_per_pathway.mean()),
        "min_pathways_per_gene": int(pathways_per_gene.min()),
        "max_pathways_per_gene": int(pathways_per_gene.max()),
        "mean_pathways_per_gene": float(pathways_per_gene.mean()),
        "n_genes_with_no_pathway": int((pathways_per_gene == 0).sum()),
        "n_rna_processing_pathways": sum(
            classify_rna_processing_pathways(pathway_name) for pathway_name in pathway_names
        ),
    }
    return mask, pathway_names, gene_names, stats
