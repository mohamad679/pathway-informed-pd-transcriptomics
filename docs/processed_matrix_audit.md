# Processed Matrix Audit

This report covers Phase 1 expression parsing and processed matrix creation.
The shared gene space came from the existing annotation-only intersection file.
Probe-to-gene mapping used cached platform annotation tables only.
External expression values were not used for gene selection.
Within-sample z-scoring was applied independently within each sample across genes.
No train-fitted scaler was used.

## Inputs

- GSE99039 series matrix: `data/raw/GSE99039/GSE99039_series_matrix.txt.gz`
- GSE6613 series matrix: `data/raw/GSE6613/GSE6613_series_matrix.txt.gz`
- GPL570 probe mapping: `data/processed/GPL570_probe_to_symbol.tsv`
- GPL96 probe mapping: `data/processed/GPL96_probe_to_symbol.tsv`
- Fixed annotation-only gene space: `data/processed/gene_space_annotation_intersection.txt`

## Outputs

- Development matrix shape: `(438, 13908)`
- Development label counts: `HC=233`, `PD=205`
- Held-out NDD matrix shape: `(48, 13908)`
- External matrix shape: `(72, 13908)`
- External label counts: `HC=22`, `PD=50`
- Ordered gene count: `13908`
- Saved gene space file: `data/processed/gene_space.txt`

## Boundary Confirmation

- Train/validation/test splits created: `no`
- Modeling performed: `no`
- Baselines implemented: `no`
- Pathway masks implemented: `no`
- MSigDB logic implemented: `no`
- Train-fitted scaler used: `no`
- External expression used for gene selection: `no`
