# Phase 3 BINN development-only training

This is development-only cross-validation, not final performance or external validation.

- Data: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, and `pathway_mask.npz` only.
- Seeds: 11, 23, 37; train-only `StandardScaler`; BCEWithLogitsLoss and Adam.
- No external cohort or held-out NDD data was loaded or used.
- Off-mask weights were hard-zeroed after every optimizer step and after training.

## Mean fold/seed metrics

| AUROC | AUPRC | Balanced accuracy | Brier | ECE | Max masked weight |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.712859 | 0.676449 | 0.649416 | 0.249931 | 0.208059 | 0.0 |
