# Phase 2 Final Gate Audit

This is a gate/audit of existing development-only baseline outputs. It is not an external-validation result or a final performance claim.

## Gate Status

- Status: `PASS`

## Validated Baselines

- Exact model set: `logistic_regression`, `random_forest`, `unconstrained_mlp`.
- Required mean and 95% CI columns are present for AUROC, AUPRC, balanced accuracy, Brier, and ECE.
- Each model has 3 seeds and 1,314 OOF rows.
- Each model/seed covers 438 unique development samples exactly once.
- All OOF probabilities are finite and within [0, 1].

## AUROC Sanity Gate

- `logistic_regression` mean AUROC: `0.687072`
- `random_forest` mean AUROC: `0.664454`
- `unconstrained_mlp` mean AUROC: `0.650682`
- All model AUROC means are greater than `0.6`.
- Best model by mean AUROC: `logistic_regression`.

## Boundary Confirmation

- Only existing Phase 2 development baseline outputs were loaded.
- No external cohort or held-out NDD data was loaded or used.
- No new modeling was performed.
- No BINN, pathway masks, or MSigDB logic was implemented or used.
