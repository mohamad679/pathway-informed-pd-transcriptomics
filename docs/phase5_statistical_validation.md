# Phase 5 statistical validation foundation

Smoke-only run.

**Smoke-only:** These outputs are runtime checks and are not the final Phase 5 result.

This report is development-only. It evaluates whether the observed development BINN OOF result is distinguishable from label-shuffled development results and documents calibration for the existing development OOF predictions.

- Inputs: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, `pathway_mask.npz`, and `binn_oof_predictions.csv` from development outputs only.
- Permutation labels are shuffled only inside development data.
- No external cohort or held-out NDD data was loaded or used.
- This is not final validation, and it does not freeze a model.
- No biological interpretation or final performance claim is made.

## Summary

- Observed pooled AUROC: 0.702895
- Permutations: 2
- Null AUROC mean: 0.538318
- Null AUROC std: 0.008007
- Empirical p-value: 0.333333
- Brier: 0.249992
- ECE: 0.154340

## Bootstrap confidence intervals

| Metric | Estimate | CI lower | CI upper | n bootstrap |
| --- | ---: | ---: | ---: | ---: |
| auroc | 0.702895 | 0.680828 | 0.729325 | 50 |
| auprc | 0.647955 | 0.605429 | 0.681181 | 50 |
| balanced_accuracy | 0.649360 | 0.626945 | 0.668686 | 50 |
| brier | 0.249992 | 0.234152 | 0.262355 | 50 |
| ece | 0.154340 | 0.144004 | 0.179649 | 50 |
