# Gene Space Audit

This report covers Phase 1 annotation-only cross-platform gene-space construction.
Only platform annotation probe-to-symbol tables were read.
No GEO series expression matrix was read.
No numeric expression values were parsed.

## Inputs

- GPL570 annotation table: `data/processed/GPL570_probe_to_symbol.tsv`
- GPL96 annotation table: `data/processed/GPL96_probe_to_symbol.tsv`

## Results

- GPL570 unique gene symbols: `22831`
- GPL96 unique gene symbols: `13908`
- Annotation-only intersection size: `13908`
- Gene space output: `data/processed/gene_space_annotation_intersection.txt`

## Boundary Confirmation

- Expression matrices read: `no`
- Expression-derived feature selection performed: `no`
- Labels used: `no`
- Modeling performed: `no`
