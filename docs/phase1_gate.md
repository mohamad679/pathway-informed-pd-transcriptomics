# Phase 1 Final Gate Audit

This report covers only the Phase 1 final gate audit and cohort overview figure.
Development PCA was computed on the development matrix only and is used here as a sanity visualization only.
No modeling, baseline implementation, training, pathway masking, MSigDB logic, or external-validation model selection was performed.

## Gate Status

- Status: `PASS`

## Processed Artifact Checks

- Development matrix shape: `(438, 13908)`
- Development label counts: `HC=233`, `PD=205`
- Held-out NDD matrix shape: `(48, 13908)`
- External matrix shape: `(72, 13908)`
- External label counts: `HC=22`, `PD=50`
- Ordered gene count: `13908`

## Z-Score Sanity

- Development max absolute per-sample mean: `1.34641e-07`
- Development max absolute per-sample std delta from 1: `3.8147e-06`
- External max absolute per-sample mean: `1.6178e-07`
- External max absolute per-sample std delta from 1: `4.58956e-06`

## Fold Integrity

- Fold count: `5`
- Development samples covered once in validation: `438` of `438`
- No train/validation overlap was detected in any fold.

## PCA Sanity Visualization

- PC1 explained variance ratio: `0.310215`
- PC2 explained variance ratio: `0.069838`
- PCA used development samples only.
- No class-separation claim is made from this visualization.

## Boundary Confirmation

- Modeling performed: `no`
- Baselines implemented: `no`
- Training performed: `no`
- Pathway masks implemented: `no`
- MSigDB logic implemented: `no`
- External validation used for model selection: `no`
