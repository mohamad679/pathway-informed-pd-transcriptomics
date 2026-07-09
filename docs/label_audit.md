# Label Audit

This report covers Phase 1 metadata-only label extraction from cached GEO series matrix files.
The expression table was not read.

## GSE6613

- Raw file: `data/raw/GSE6613/GSE6613_series_matrix.txt.gz`
- Sample count: `105`
- Metadata lines read before table marker: `54`
- Exact metadata fields used for label inference: `characteristics_ch1`, `description`, `title`
- Candidate label-bearing fields observed: `characteristics_ch1`, `description`, `title`
- Label rule: Use metadata text only. Explicit exclusion rule: samples marked `neurological disease control` in metadata are labeled `EXCLUDE` before any expression values are read; `healthy control` maps to `HC`; `Parkinson's disease` maps to `PD`.
- Counts per group: `PD`=50, `HC`=22, `EXCLUDE`=33, `UNKNOWN`=0
- Expression table was not read.

## GSE99039

- Raw file: `data/raw/GSE99039/GSE99039_series_matrix.txt.gz`
- Sample count: `558`
- Metadata lines read before table marker: `75`
- Exact metadata fields used for label inference: `characteristics_ch1`, `description`
- Candidate label-bearing fields observed: `characteristics_ch1`, `description`, `supplementary_file`, `title`
- Label rule: Use metadata text only. Primary development task maps `disease label: IPD` to `PD` and `disease label: CONTROL` to `HC`. Held-out NDD maps `HD`, `HD_HD_BATCH`, `MSA`, `PSP`, `CBD`, `PD_DEMENTIA`, and `Vascular dementia` to `NDD`. Exclude `GPD`, `GENETIC_UNAFFECTED`, `DRD`, `DRD-DYT5`, and `ATYPICAL_PD` from primary development training and held-out NDD scoring.
- Counts per group: `PD`=205, `HC`=233, `NDD`=48, `EXCLUDE`=72, `UNKNOWN`=0
- Expression table was not read.
