# Phase 2 development-only baseline summary

This is a development-only baseline comparison, not an external-validation result or final performance claim.

## Method

- Inputs: `data/processed/dev_X.npy`, `data/processed/dev_y.npy`, and `data/processed/dev_folds.json` only.
- For each model and seed, out-of-fold probabilities were pooled across all five validation folds before computing its metric.
- The reported mean is the mean of these seed-level pooled-OOF metrics.
- 95% bootstrap CIs use 2,000 deterministic paired, class-stratified resamples of all seed-level OOF prediction rows within each model (`random_state=20260710`). Repeated rows across seeds are retained; each resample keeps its paired `y_true` and `y_prob` values together.
- No external cohort or held-out NDD data was loaded or used.
- No BINN, pathway masks, or MSigDB logic was implemented or used.

## Pooled out-of-fold metrics

| Model | AUROC mean [95% CI] | AUPRC mean [95% CI] | Balanced accuracy mean [95% CI] | Brier mean [95% CI] | ECE mean [95% CI] |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.687072 [0.658897, 0.716318] | 0.678433 [0.648576, 0.708569] | 0.615451 [0.588198, 0.643003] | 0.286623 [0.266339, 0.305040] | 0.248903 [0.225495, 0.276305] |
| random_forest | 0.664454 [0.636489, 0.692009] | 0.613488 [0.581933, 0.647102] | 0.625779 [0.600885, 0.650825] | 0.230697 [0.225342, 0.236004] | 0.050093 [0.031577, 0.075626] |
| unconstrained_mlp | 0.650682 [0.620804, 0.679480] | 0.625957 [0.591693, 0.658942] | 0.613266 [0.587804, 0.639648] | 0.342944 [0.320213, 0.365360] | 0.333170 [0.304871, 0.354300] |

## Development-only comparison

- Highest mean pooled-OOF AUROC: `logistic_regression`.
- This is a baseline-comparison observation only; it is not a final performance claim.
