# Phase 2 Random Forest development-only baseline

This report contains development-only cross-validation results. It is not an external-validation result or a final performance claim.

## Configuration

- Model: `random_forest`
- RandomForestClassifier: `n_estimators=500`, `max_features="sqrt"`, `class_weight="balanced_subsample"`, `n_jobs=-1`
- Seeds: [11, 23, 37]
- Data loaded: `data/processed/dev_X.npy`, `data/processed/dev_y.npy`, and `data/processed/dev_folds.json` only
- No scaling was applied.
- No external cohort or held-out NDD data was loaded or used.

## Mean metrics across all folds and seeds

| AUROC | AUPRC | Balanced accuracy | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: |
| 0.672046 | 0.638951 | 0.626037 | 0.230666 | 0.105912 |

## Fold AUROC range

- Best: seed 11, fold 5, AUROC 0.787646
- Worst: seed 23, fold 3, AUROC 0.594707
