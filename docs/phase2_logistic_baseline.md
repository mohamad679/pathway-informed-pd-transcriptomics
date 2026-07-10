# Phase 2 Logistic Regression development-only baseline

This report contains development-only cross-validation results. It is not an external-validation result or a final performance claim.

## Configuration

- Model: `logistic_regression`
- Pipeline: `StandardScaler` then `LogisticRegression`
- LogisticRegression: `solver="liblinear"`, `penalty="l2"`, `class_weight="balanced"`, `max_iter=5000`
- Seeds: [11, 23, 37]
- Data loaded: `data/processed/dev_X.npy`, `data/processed/dev_y.npy`, and `data/processed/dev_folds.json` only
- The scaler was fit within each training fold only.
- No external cohort or held-out NDD data was loaded or used.

## Mean metrics across all folds and seeds

| AUROC | AUPRC | Balanced accuracy | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: |
| 0.688836 | 0.682098 | 0.615680 | 0.286516 | 0.273118 |

## Fold AUROC range

- Best: seed 11, fold 2, AUROC 0.749870
- Worst: seed 11, fold 3, AUROC 0.611313
