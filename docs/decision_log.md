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

## 2026-07-10 — Phase 2 development-only Random Forest baseline

- Evaluated a Random Forest baseline using only `dev_X.npy`, `dev_y.npy`, and the predefined development folds.
- Used seeds 11, 23, and 37 with 500 trees, `max_features="sqrt"`, and `class_weight="balanced_subsample"`.
- No external cohort or held-out NDD data was loaded or used.
- No MLP, BINN, pathway masks, or MSigDB logic was implemented.

## 2026-07-10 — Phase 2 development-only unconstrained MLP baseline

- Evaluated an unconstrained MLP baseline using only `dev_X.npy`, `dev_y.npy`, and the predefined development folds.
- Used `StandardScaler` inside a newly fitted sklearn pipeline for every training fold; no validation-fold information was used to fit the scaler.
- Used seeds 11, 23, and 37 with one 128-unit ReLU hidden layer and the predefined Adam/early-stopping configuration.
- No external cohort or held-out NDD data was loaded or used.
- This is an unconstrained baseline, not BINN or pathway-informed; no pathway masks or MSigDB logic was implemented.

## 2026-07-10 — Phase 2 development-only baseline OOF comparison and bootstrap CI

- Re-ran only the completed Logistic Regression, Random Forest, and unconstrained MLP baselines using `dev_X.npy`, `dev_y.npy`, and predefined development folds.
- Exported one out-of-fold probability per validation sample, seed, and fold to support sample-level uncertainty estimation without changing the committed per-fold baseline metric CSVs.
- For each model and seed, pooled OOF predictions across the five validation folds before computing seed-level metrics; reported means are across those seed-level metrics.
- Used 2,000 deterministic paired, class-stratified bootstrap resamples of pooled seed-level OOF prediction rows within model (`random_state=20260710`) for 95% CIs.
- No external cohort or held-out NDD data was loaded or used. No BINN, pathway masks, or MSigDB logic was implemented or used.

## 2026-07-10 — Phase 2 final gate audit

- Audited only the existing Phase 2 development baseline output CSVs.
- Confirmed the exact required model set, summary metric mean/CI columns, three-seed OOF coverage, finite bounded probabilities, and the AUROC sanity gate.
- Recorded `logistic_regression` as the best development-only baseline by mean OOF AUROC.
- No external cohort or held-out NDD data was loaded or used.
- No modeling, BINN, pathway masks, or MSigDB logic was implemented or used.
- This gate result is not an external-validation result or final performance claim.

## 2026-07-10 — Phase 3 pathway-mask construction foundation

- Added local-only MSigDB GMT parsing and deterministic sparse pathway-mask construction.
- A real mask is written only when configured local GMT files and the fixed `gene_space.txt` are available.
- The mask audit records configured MSigDB version, exact input paths, and SHA-256 checksums.
- No BINN model, training, attention, Integrated Gradients, external cohort, or held-out NDD data was used.

## 2026-07-10 — Phase 3 BINN model foundation audit

- Added a fixed gene-to-pathway masked linear layer and BINN classifier foundation.
- The integrity smoke test uses only four development expression rows for one forward pass and a synthetic backward pass.
- Off-mask weights are excluded from the forward computation, receive zero gradients, and are hard-zeroed after mask application.
- No training, cross-validation, attention, Integrated Gradients, external cohort, or held-out NDD data was used.

## 2026-07-10 — Phase 3 development-only BINN CV

- Trained the pathway-constrained BINN only on predefined development folds with seeds 11, 23, and 37.
- Scaling was fit on each training partition only; no external cohort or held-out NDD data was used.
- Off-mask weights were re-zeroed after each optimizer step and verified exactly zero after training.
- This is a development-only training gate, not a final or external-validation performance claim.

## 2026-07-10 — Phase 3 final gate audit

- Audited only existing Phase 3 development BINN outputs and the saved pathway mask.
- Confirmed fixed seed/fold coverage, finite bounded metrics and OOF probabilities, and the development AUROC sanity check.
- Confirmed off-mask weights remained exactly zero and reported weight counts match the saved mask.
- No external cohort or held-out NDD result was used.
- No model was trained and BINN CV was not rerun inside this gate.
- This is development-only and not final validation; the gate concerns stable training and mask integrity, not superiority.
