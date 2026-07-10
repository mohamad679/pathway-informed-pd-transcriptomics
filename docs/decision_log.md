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

## 2026-07-10 — Phase 1 processed matrix creation

- Parsed cached GEO series-matrix expression tables only between `!series_matrix_table_begin` and `!series_matrix_table_end`.
- Fixed the shared gene space from the existing annotation-only intersection file before reading external-cohort expression values.
- Mapped probes to genes using cached platform annotation tables only and aggregated multiple probes per gene by median.
- Applied metadata-only labels from `src/data/labels.py`.
- Applied within-sample gene-wise z-scoring only; no train-fitted scaler was used.
- No train/validation/test splits, pathway masks, MSigDB logic, baselines, or modeling artifacts were created.
- External expression values were not used for gene selection, probe selection, fitting, ranking, scaling, or imputation.

## 2026-07-10 — Phase 1 development split creation

- Created stratified 5-fold cross-validation splits from development PD/HC samples only.
- Held-out NDD samples were checked for overlap but were not used for split creation.
- External cohort samples were checked for overlap but were not used for split creation.
- Donor IDs were not available in the processed metadata inputs, so GSM accession was used as the conservative subject identifier for split integrity.
- No modeling, baselines, training, pathway masks, MSigDB logic, or external-validation model selection artifacts were created.

## 2026-07-10 — Phase 1 final gate audit and cohort overview figure

- Validated only the existing processed Phase 1 artifacts and development fold file.
- Confirmed development, held-out NDD, and external matrix shapes and Phase 1 label counts against the expected gate values.
- Confirmed per-sample z-score sanity for development and external matrices.
- Confirmed that each development sample appears exactly once in validation across the saved folds with no train/validation overlap.
- Computed a development-only PCA for a sanity visualization and saved a single cohort overview PNG.
- No modeling, baselines, training, pathway masks, MSigDB logic, or external-validation model selection was performed.

## 2026-07-10 — Phase 2 development-only Logistic Regression baseline

- Evaluated a Logistic Regression baseline using only `dev_X.npy`, `dev_y.npy`, and the predefined development folds.
- Used `StandardScaler` inside a newly fitted sklearn pipeline for every training fold; no validation-fold information was used to fit the scaler.
- Used seeds 11, 23, and 37 with liblinear, L2 regularization, balanced class weights, and max_iter=5000.
- No external cohort or held-out NDD data was loaded or used.
- No Random Forest, MLP, BINN, pathway masks, or MSigDB logic was implemented.
