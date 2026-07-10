# Decision Log

## 2026-07-10

- Project initialized.
- No data inspected.
- No modeling performed.
- External cohort not touched.

## 2026-07-10 — Phase 0 environment pin adjustment

- Pinned torch to 2.2.2 because torch 2.3.1 was not available for the current macOS Intel x64 + Python 3.11 pip environment.
- No data inspected.
- No modeling performed.
- External cohort not touched.

## 2026-07-10 — Phase 0 reproducibility cleanup

- Pinned matplotlib to 3.9.2 to avoid committing a yanked 3.9.1 dependency warning.
- Added MIT LICENSE as part of the roadmap repository structure.
- No data inspected.
- No modeling performed.
- External cohort not touched.

## 2026-07-10 — Phase 1 step 1 download implementation

- Created the Phase 1 step 1 data download script for GEO series matrix caching and provenance logging.
- No labels parsed.
- No expression-driven feature selection performed.
- External cohort downloaded/cache-prepared only; not used for modeling or feature selection.

## 2026-07-10 — Phase 1 step 2 metadata-only verification

- Verified cached GSE99039 and GSE6613 GEO series-matrix metadata headers only.
- Stopped reading at `!series_matrix_table_begin`; expression tables were not read.
- No labels confirmed.
- No processed matrices, probe mapping, gene intersection, or modeling performed.

## 2026-07-10 — Phase 1 step 3 metadata-only label extraction

- GSE6613 exclusion rule was defined before reading expression values.
- No expression table read.
- No modeling.
- No processed matrices created.

## 2026-07-10 — Phase 1 GSE99039 label-rule correction

- Corrected GSE99039 metadata-only label rules to match the roadmap primary task: IPD vs CONTROL only.
- GPD and GENETIC_UNAFFECTED are excluded from primary development training.
- DRD, DRD-DYT5, and ATYPICAL_PD are excluded from the held-out NDD probe.
- Held-out NDD is defined as HD, HD_HD_BATCH, MSA, PSP, CBD, PD_DEMENTIA, and Vascular dementia.
- No expression table read.
- No modeling.
- No processed matrices created.

## 2026-07-10 — Phase 1 platform annotation acquisition

- Downloaded or reused cached GEO platform annotation files for `GPL570` and `GPL96`.
- Parsed platform annotation metadata only to build probe-to-symbol tables.
- No GEO series expression matrix was read.
- No labels, splits, processed expression matrices, or modeling artifacts were created.

## 2026-07-10 — Phase 1 annotation-only cross-platform gene-space construction

- Built the shared gene space from `GPL570` and `GPL96` probe-to-symbol annotation tables only.
- No GEO series expression matrix was read.
- No numeric expression values were parsed.
- No labels, splits, processed expression matrices, or modeling artifacts were created.
