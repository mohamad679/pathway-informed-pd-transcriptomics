# Results

## Cohort and feature-space summary

The development PD/HC cohort contained 438 samples. The frozen external PD/HC cohort contained 72 samples, with 22 HC and 50 PD. The held-out NDD stress-test set contained 48 unlabeled samples. The fixed feature space contained 13,908 genes, 1,297 Reactome pathways, and 75,429 retained pathway-mask edges.

## Baseline development results

The baseline table below is consolidated from `results/development/baseline_summary.csv`.

| Model | AUROC | AUPRC | Balanced accuracy | Brier | ECE |
|---|---:|---:|---:|---:|---:|
| logistic_regression | 0.687072 | 0.678433 | 0.615451 | 0.286623 | 0.248903 |
| random_forest | 0.664454 | 0.613488 | 0.625779 | 0.230697 | 0.050093 |
| unconstrained_mlp | 0.650682 | 0.625957 | 0.613266 | 0.342944 | 0.333170 |

## BINN development results

The pathway-constrained BINN mean development-CV metrics from `results/development/binn_cv.csv` were AUROC 0.712859, AUPRC 0.676449, balanced accuracy 0.649416, Brier 0.249931, and ECE 0.208059.

Mean fold metrics and pooled out-of-fold metrics are distinct summaries. The statistical-validation pooled out-of-fold AUROC was 0.702895.

## Statistical validation

The observed pooled development AUROC was 0.702895. Bootstrap confidence intervals from `results/development/statistical_validation_bootstrap_ci.csv` were:

| Metric | Estimate | 95% CI lower | 95% CI upper |
|---|---:|---:|---:|
| auroc | 0.702895 | 0.673004 | 0.731080 |
| auprc | 0.647955 | 0.605461 | 0.688369 |
| balanced_accuracy | 0.649360 | 0.622228 | 0.676291 |
| brier | 0.249992 | 0.232254 | 0.267916 |
| ece | 0.154340 | 0.138305 | 0.186404 |

The permutation validation used 50 permutations. The null AUROC mean was 0.536038 with standard deviation 0.022579. The empirical p-value was 0.019608, the minimum attainable value with 50 permutations. This computationally limited result cannot support `p < 0.01`.

## Attribution agreement

Activation-vs-Integrated-Gradients agreement was evaluated on development-only interpretation artifacts. The mean seed/fold Spearman correlation was 0.808936. The global Spearman correlation was 0.933899, with global top-20 overlap of 12 pathways.

RNA-processing pathways in the activation top 20: 0. RNA-processing pathways in the Integrated Gradients top 20: 0. Absence from the top 20 is a model attribution result, not evidence of biological absence.

## Frozen external evaluation

The frozen external cohort had 72 samples: 22 HC and 50 PD. External metrics from `results/external/external_metrics.json` were AUROC 0.695455, AUPRC 0.782081, balanced accuracy at threshold 0.5 of 0.500000, Brier 0.694444, and ECE 0.694441.

The AUROC shows ranking discrimination only. Fixed-threshold balanced accuracy was 0.5, poor calibration was severe, and no external tuning, threshold adjustment, retraining, or recalibration was performed.

## NDD stress test

The NDD stress-test set contained 48 unlabeled samples. The mean PD probability was 0.489050, and the fraction predicted PD at threshold 0.5 was 0.520833. This is an unlabeled specificity/stress test only, not diagnostic validation.

## Overall interpretation

The workflow produced moderate development discrimination and similar moderate external ranking discrimination. It failed to provide useful fixed-threshold external performance, and external calibration failed severely. Workflow integrity is stronger than predictive performance.
