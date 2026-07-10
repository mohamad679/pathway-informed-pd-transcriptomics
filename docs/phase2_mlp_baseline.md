# Phase 2 unconstrained MLP development-only baseline

This report contains development-only cross-validation results. It is not an external-validation result or a final performance claim.

## Configuration

- Model: `unconstrained_mlp`
- Pipeline: `StandardScaler` then `MLPClassifier`
- MLPClassifier: `hidden_layer_sizes=(128,)`, `activation="relu"`, `solver="adam"`, `alpha=0.0001`, `batch_size=64`, `learning_rate_init=0.001`, `max_iter=300`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=20`
- Seeds: [11, 23, 37]
- Data loaded: `data/processed/dev_X.npy`, `data/processed/dev_y.npy`, and `data/processed/dev_folds.json` only
- The scaler was fit within each training fold only.
- No external cohort or held-out NDD data was loaded or used.
- This is an unconstrained baseline, not BINN or pathway-informed; no pathway masks or MSigDB logic was used.

## Mean metrics across all folds and seeds

| AUROC | AUPRC | Balanced accuracy | Brier | ECE |
| ---: | ---: | ---: | ---: | ---: |
| 0.660386 | 0.644426 | 0.613443 | 0.342827 | 0.350715 |

## Fold AUROC range

- Best: seed 37, fold 5, AUROC 0.741782
- Worst: seed 11, fold 3, AUROC 0.547483
