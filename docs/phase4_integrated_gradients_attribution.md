# Phase 4 pathway-level Integrated Gradients attribution

Generated at: 2026-07-10T13:37:47.059609+00:00

This is development-only interpretation retraining. Phase 3 fold models were not checkpointed, so the same predefined development folds and seeds were retrained solely to calculate out-of-fold interpretation artifacts. This is not model selection and is not final validation. It is not a biological claim.

- Inputs: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, `pathway_mask.npz`, `pathway_names.txt`, `gene_space.txt`, and `config/pathways.yaml` only.
- Each validation partition was transformed with its fold's train-only scaler.
- This is pathway-level Integrated Gradients on pathway activations, not raw-gene Integrated Gradients.
- Integrated Gradients uses a zero baseline, targets the downstream pathway-to-hidden-to-logit head, and runs with the model in evaluation mode so dropout is disabled.
- No external cohort or held-out NDD data was loaded or used. Attention, permutation testing, and later-phase logic are not included.
- Off-mask weights were enforced at every optimizer step and were zero after each interpretation retraining.

## Audit

| Metric | Value |
| --- | ---: |
| Pathways | 1297 |
| Seeds | 3 |
| Development folds | 5 |
| IG steps | 16 |
| RNA-processing pathways | 4 |
| Maximum masked weight after interpretation retraining | 0.0 |
| No external/NDD data used | yes |

## Top 20 pathways by mean IG rank

| Pathway | Mean rank | RNA-processing flag |
| --- | ---: | --- |
| REACTOME_DEVELOPMENTAL_BIOLOGY | 5.933 | false |
| REACTOME_POST_TRANSLATIONAL_PROTEIN_MODIFICATION | 7.800 | false |
| REACTOME_RNA_POLYMERASE_II_TRANSCRIPTION | 8.000 | false |
| REACTOME_HEMOSTASIS | 11.267 | false |
| REACTOME_INNATE_IMMUNE_SYSTEM | 13.267 | false |
| REACTOME_SIGNALING_BY_INTERLEUKINS | 18.333 | false |
| REACTOME_CELL_CYCLE | 23.733 | false |
| REACTOME_SIGNALING_BY_RECEPTOR_TYROSINE_KINASES | 23.733 | false |
| REACTOME_VIRAL_INFECTION_PATHWAYS | 25.467 | false |
| REACTOME_VESICLE_MEDIATED_TRANSPORT | 25.733 | false |
| REACTOME_INFECTIOUS_DISEASE | 25.933 | false |
| REACTOME_ADAPTIVE_IMMUNE_SYSTEM | 27.467 | false |
| REACTOME_DISEASES_OF_SIGNAL_TRANSDUCTION_BY_GROWTH_FACTOR_RECEPTORS_AND_SECOND_MESSENGERS | 30.200 | false |
| REACTOME_NEUTROPHIL_DEGRANULATION | 34.933 | false |
| REACTOME_SENSORY_PERCEPTION | 37.467 | false |
| REACTOME_SARS_COV_INFECTIONS | 39.067 | false |
| REACTOME_CLASS_A_1_RHODOPSIN_LIKE_RECEPTORS | 40.067 | false |
| REACTOME_GPCR_LIGAND_BINDING | 40.067 | false |
| REACTOME_RHO_GTPASE_CYCLE | 40.400 | false |
| REACTOME_SIGNALING_BY_RHO_GTPASES_MIRO_GTPASES_AND_RHOBTB3 | 41.400 | false |
