# Phase 4 activation-based pathway attribution

Generated at: 2026-07-10T12:59:45.165779+00:00

This is development-only interpretation retraining. Phase 3 fold models were not checkpointed, so the same predefined development folds and seeds were retrained only to calculate out-of-fold pathway activations. This is not model selection and not final validation. It is not a biological claim.

- Inputs: `dev_X.npy`, `dev_y.npy`, `dev_folds.json`, `pathway_mask.npz`, `pathway_names.txt`, `gene_space.txt`, and `config/pathways.yaml` only.
- Each validation partition was transformed with its fold's train-only scaler and used only for activation computation.
- No external cohort or held-out NDD data was loaded or used.
- Integrated Gradients, Captum, attention, and permutation testing are not included.
- Off-mask weights were enforced at every optimizer step and were zero after each interpretation retraining.

## Audit

| Metric | Value |
| --- | ---: |
| Pathways | 1297 |
| Seeds | 3 |
| Development folds | 5 |
| RNA-processing pathways | 4 |
| Maximum masked weight after interpretation retraining | 0.0 |
| No external/NDD data used | yes |

## Top 20 pathways by mean activation rank

| Pathway | Mean rank | RNA-processing flag |
| --- | ---: | --- |
| REACTOME_POST_TRANSLATIONAL_PROTEIN_MODIFICATION | 5.400 | false |
| REACTOME_DEVELOPMENTAL_BIOLOGY | 5.933 | false |
| REACTOME_RNA_POLYMERASE_II_TRANSCRIPTION | 6.000 | false |
| REACTOME_INFECTIOUS_DISEASE | 7.400 | false |
| REACTOME_INNATE_IMMUNE_SYSTEM | 7.533 | false |
| REACTOME_CYTOKINE_SIGNALING_IN_IMMUNE_SYSTEM | 9.267 | false |
| REACTOME_TRANSPORT_OF_SMALL_MOLECULES | 13.000 | false |
| REACTOME_CELLULAR_RESPONSES_TO_STIMULI | 13.667 | false |
| REACTOME_METABOLISM_OF_LIPIDS | 14.733 | false |
| REACTOME_ADAPTIVE_IMMUNE_SYSTEM | 16.867 | false |
| REACTOME_METABOLISM_OF_RNA | 19.400 | false |
| REACTOME_NERVOUS_SYSTEM_DEVELOPMENT | 19.467 | false |
| REACTOME_CELL_CYCLE | 19.867 | false |
| REACTOME_MEMBRANE_TRAFFICKING | 19.867 | false |
| REACTOME_VESICLE_MEDIATED_TRANSPORT | 19.867 | false |
| REACTOME_VIRAL_INFECTION_PATHWAYS | 20.800 | false |
| REACTOME_HEMOSTASIS | 21.133 | false |
| REACTOME_SIGNALING_BY_RHO_GTPASES_MIRO_GTPASES_AND_RHOBTB3 | 21.600 | false |
| REACTOME_SIGNALING_BY_GPCR | 21.867 | false |
| REACTOME_SIGNALING_BY_RECEPTOR_TYROSINE_KINASES | 23.400 | false |
