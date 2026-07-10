# Limitations

## Biological scope

Peripheral blood is not brain tissue. Blood-expression signals are indirect and may reflect systemic inflammation, medication, immune composition, comorbidities, technical factors, or combinations of these factors. This study cannot establish Parkinson's disease mechanisms or causality.

Microarray expression measures steady-state transcript abundance. Static expression is not RNA dynamics: this project does not study RNA dynamics, splicing kinetics, RNA velocity, degradation rates, or isoform-level regulation.

Pathway attribution describes model usage, not biological causation. Activation ranking is not explanation. Integrated Gradients is model attribution, not causal evidence. Agreement between activation and Integrated Gradients rankings does not prove biological truth.

## Dataset and cohort limitations

The development cohort contains 438 Parkinson's disease/control samples. The external cohort contains 72 samples, with 22 controls and 50 Parkinson's disease samples. The held-out NDD cohort contains 48 samples and has no binary Parkinson's disease-versus-control target labels.

There is only one external cohort. The development and external cohorts differ simultaneously in platform, site, study era, recruitment, and population, so platform shift cannot be separated from cohort shift. Within-sample z-scoring and the frozen `StandardScaler` do not eliminate structural batch effects.

Class imbalance affects interpretation of AUPRC and threshold metrics. In the external cohort, the Parkinson's disease prevalence is 50/72, which must be considered when interpreting precision-recall results.

## Model and statistical limitations

Development BINN performance is moderate, not diagnostic. The development pooled observed AUROC was 0.702895.

Permutation testing used only 50 permutations because 1000 repeated three-seed, five-fold BINN training runs were not computationally feasible within available runtime and project constraints. The minimum attainable empirical p-value was 1/51 = 0.019608. The result cannot support p < 0.01.

Bootstrap confidence intervals quantify sampling uncertainty, but they do not fix dataset shift or hidden confounding. Pathway databases are incomplete and biased toward well-studied biology. Genes with no retained Reactome pathway annotation are not represented in the pathway-constrained first layer.

Hyperparameters and the final epoch count were fixed using development data only. The final model was not recalibrated on the external cohort.

## External validation limitations

The one-time frozen-model external results were:

| Metric | Result |
| --- | ---: |
| AUROC | 0.695455 |
| AUPRC | 0.782081 |
| Balanced accuracy at threshold 0.5 | 0.500000 |
| Brier | 0.694444 |
| ECE | 0.694441 |

AUROC represents ranking discrimination only. AUPRC must be interpreted in light of the 50/72 Parkinson's disease prevalence. Fixed-threshold balanced accuracy of 0.500000 indicates no useful discrimination at the frozen 0.5 decision threshold.

The Brier score and ECE show severe external miscalibration. This poor external calibration is a central limitation of the frozen model result. No threshold adjustment, recalibration, or external-data tuning was allowed after freezing.

The external result is research validation only and not clinical validation. It does not support clinical, diagnostic, deployment, causal, or mechanistic claims.

## NDD stress-test limitations

The held-out NDD stress-test-only results were:

| Quantity | Result |
| --- | ---: |
| NDD sample count | 48 |
| Mean PD probability | 0.489050 |
| Fraction predicted PD at threshold 0.5 | 0.520833 |

The NDD cohort is unlabeled for the binary Parkinson's disease-versus-control task. This is only a specificity/stress test, not diagnostic validation.

Approximately half of NDD samples being classified as Parkinson's disease does not establish disease specificity. These results may indicate sensitivity to broader neurodegenerative or cohort signals, but they cannot determine the source.

## Reproducibility and chain-of-custody limitations

The model was frozen at tag `frozen-v1`. `HASH_BEFORE` and `HASH_AFTER` match. This verifies artifact immutability, not scientific validity.

Git LFS is required to retrieve `frozen/model_v1.pt`. Raw and processed GEO-derived data are not committed because of repository size and provenance constraints. Full reproduction therefore requires reacquiring or reconstructing data from the recorded provenance and scripts.

Hardware, library, and floating-point differences may cause small numerical changes in retraining. External scoring must not be rerun as a model-selection experiment.

## Appropriate conclusion

This is a methods and reproducibility study. It demonstrates a controlled pathway-prior modeling workflow and honest external evaluation.

It does not provide a diagnostic tool, clinical evidence, disease mechanism, or deployment-ready model. The strongest result is the auditable workflow and chain of custody, not the absolute predictive performance.
