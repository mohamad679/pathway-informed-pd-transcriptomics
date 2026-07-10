# Phase 3 pathway mask audit

Generated at: 2026-07-10T09:25:02.972837+00:00

## MSigDB provenance

- Configured MSigDB version: `2024.1.Hs`
- Minimum genes present: `10`
- Maximum genes present: `None`
- `data/raw/msigdb/c2.cp.reactome.v2024.1.Hs.symbols.gmt` — SHA-256 `9ea1b5e656597daf423e41c5ebcaa9892bfedf3292fff768605d7b0d5e5e9703`

## Mask summary

| Metric | Value |
| --- | ---: |
| n_pathways | 1297 |
| n_genes | 13908 |
| n_edges | 75429 |
| density | 0.0041815153174213 |
| min_genes_per_pathway | 10 |
| max_genes_per_pathway | 1111 |
| mean_genes_per_pathway | 58.15651503469545 |
| min_pathways_per_gene | 0 |
| max_pathways_per_gene | 306 |
| mean_pathways_per_gene | 5.423425366695427 |
| n_genes_with_no_pathway | 5499 |
| n_normalized_gene_collision_groups | 1 |
| n_genes_in_normalization_collision_groups | 2 |
| normalization_collision_resolution_policy | prefer_exact_all_uppercase_else_first_gene_space_order |
| n_rna_processing_pathways | 4 |


Outputs: `data/processed/pathway_mask.npz`, `data/processed/pathway_names.txt`, and `data/processed/pathway_gene_edges.tsv`.
