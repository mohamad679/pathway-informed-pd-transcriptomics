# Phase 5 Final Statistical-Validation Gate

This gate audits integrity of the limited-resolution development-only Phase 5 statistical artifacts. It does not train or freeze a model, rerun permutation testing, rerun bootstrap, use external or held-out NDD data, perform final validation, or support biological claims.

## Gate Status

- Status: `PASS`
- PASS means the existing limited-resolution development-only artifacts are internally consistent.
- PASS does not mean final validation.
- PASS does not mean `p < 0.01`.
- PASS does not remove the computational limitation.
- PASS does not support biological claims.

## Validated Development Outputs

- Permutation rows: `50`.
- Exact permutation coverage: `1` through `50`, with unique indices.
- Observed AUROC: `0.702895`.
- Null AUROC mean: `0.536038`.
- Null AUROC std: `0.022579`.
- Empirical p-value: `0.019608`.
- Bootstrap resamples per metric: `2000`.
- Brier: `0.249992`.
- ECE: `0.154340`.
- Calibration bins: `15`.
- Calibration sample total: `1314`.

## Production Limitation

- 50 permutations only.
- The minimum attainable p-value is `1/51 = 0.019608`.
- This result cannot support `p < 0.01`.

## Boundary Confirmation

- No external cohort or held-out NDD data was used.
- No training, retraining, permutation rerun, or bootstrap rerun happened inside the gate.
- No model was frozen.
- No final external validation claim is made.
- No biological claim is made.
