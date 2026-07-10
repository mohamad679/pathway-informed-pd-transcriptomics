# Phase 4 activation-vs-Integrated-Gradients agreement

Generated at: 2026-07-10T13:49:35.865141+00:00

This is a development-only interpretation summary comparing existing activation and Integrated Gradients pathway attribution outputs. No retraining was performed in this step. This is not final validation and is not a biological claim.

- Inputs: `results/development/pathway_activation_scores.csv`, `results/development/pathway_activation_stability.csv`, `results/development/pathway_ig_scores.csv`, and `results/development/pathway_ig_stability.csv` only.
- No external cohort or held-out NDD data was loaded or used.
- Attention, permutation testing, final validation, and later-phase logic are not included.

## Agreement summary

| Metric | Value |
| --- | ---: |
| Mean seed/fold Spearman | 0.808936 |
| Minimum seed/fold Spearman | 0.671447 |
| Maximum seed/fold Spearman | 0.865073 |
| Mean top-20 overlap | 13.333333 |
| Global Spearman | 0.933899 |
| Global top-20 overlap | 12 |
| Activation top-20 RNA-processing count | 0 |
| IG top-20 RNA-processing count | 0 |
| No external/NDD data used | yes |

## Seed/fold agreement

| Seed | Fold | Spearman | Top-20 overlap | Pathways compared |
| ---: | ---: | ---: | ---: | ---: |
| 11 | 1 | 0.805343 | 15 | 1297 |
| 11 | 2 | 0.781701 | 17 | 1297 |
| 11 | 3 | 0.671447 | 13 | 1297 |
| 11 | 4 | 0.865073 | 11 | 1297 |
| 11 | 5 | 0.805572 | 12 | 1297 |
| 23 | 1 | 0.840274 | 14 | 1297 |
| 23 | 2 | 0.785029 | 12 | 1297 |
| 23 | 3 | 0.808775 | 14 | 1297 |
| 23 | 4 | 0.813343 | 13 | 1297 |
| 23 | 5 | 0.801035 | 13 | 1297 |
| 37 | 1 | 0.837012 | 13 | 1297 |
| 37 | 2 | 0.811177 | 13 | 1297 |
| 37 | 3 | 0.842916 | 14 | 1297 |
| 37 | 4 | 0.829798 | 14 | 1297 |
| 37 | 5 | 0.835545 | 12 | 1297 |

## RNA-processing-flagged tier

This table reports ranking agreement metadata only; it is not a biological claim.

| Pathway | Activation mean rank | IG mean rank | Activation rank range | IG rank range | Activation top 20 | IG top 20 |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| REACTOME_MRNA_SPLICING | 102.733 | 238.867 | 32-361 | 40-899 | false | false |
| REACTOME_MRNA_SPLICING_MINOR_PATHWAY | 518.600 | 606.400 | 204-882 | 132-1072 | false | false |
| REACTOME_SUMOYLATION_OF_RNA_BINDING_PROTEINS | 526.133 | 526.200 | 179-937 | 237-1038 | false | false |
| REACTOME_EXPORT_OF_VIRAL_RIBONUCLEOPROTEINS_FROM_NUCLEUS | 731.400 | 461.600 | 331-1209 | 171-868 | false | false |
