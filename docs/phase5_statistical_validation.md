# Phase 5 statistical validation foundation

Limited-resolution production run.

**Production scope:** This development-only run completed the exact predefined permutation range 1 through 50. The reduced permutation count and its statistical-resolution limitation are documented below.

This report is development-only. It evaluates whether the observed development BINN OOF result is distinguishable from label-shuffled development results and documents calibration for the existing development OOF predictions.

- Inputs: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, `pathway_mask.npz`, and `binn_oof_predictions.csv` from development outputs only.
- Permutation labels are shuffled only inside development data.
- No external cohort or held-out NDD data was loaded or used.
- This is not final validation, and it does not freeze a model.
- No biological interpretation or final performance claim is made.

## Summary

- Observed pooled AUROC: 0.702895
- Permutations: 50
- Requested permutation range: 1 to 50
- Requested permutations this run: 50
- Completed permutations in requested range: 50
- Completed unique requested permutations: 50
- Exact requested coverage complete: True
- Skipped existing permutations: 0
- Device used: cuda (requested: cuda)
- Checkpoint/resume: auto_resume=True, resume_path=None, checkpoint_path=/content/pathway-informed-pd-transcriptomics/results/development/statistical_validation_permutation_null_production50.csv, checkpoint_every=1, progress_enabled=True
- Null AUROC mean: 0.536038
- Null AUROC std: 0.022579
- Empirical p-value: 0.019608
- Brier: 0.249992
- ECE: 0.154340

## Bootstrap confidence intervals

| Metric | Estimate | CI lower | CI upper | n bootstrap |
| --- | ---: | ---: | ---: | ---: |
| auroc | 0.702895 | 0.673004 | 0.731080 | 2000 |
| auprc | 0.647955 | 0.605461 | 0.688369 | 2000 |
| balanced_accuracy | 0.649360 | 0.622228 | 0.676291 | 2000 |
| brier | 0.249992 | 0.232254 | 0.267916 | 2000 |
| ece | 0.154340 | 0.138305 | 0.186404 | 2000 |

## Computational limitation

The label-permutation analysis was limited to 50 permutations because each
permutation required repeated BINN training across three random seeds and five
development folds. A 1,000-permutation analysis was not computationally
feasible within the available Google Colab runtime and project time constraints.

Accordingly, this result is reported as a limited-resolution,
development-only statistical check rather than a high-precision significance
test. With 50 permutations, the smallest attainable empirical p-value is
1 / (50 + 1) = 0.019608. Therefore, this analysis does not support a claim of
p < 0.01. No external-validation or biological claim is made from this result.
