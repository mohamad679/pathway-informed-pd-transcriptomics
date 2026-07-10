# Split Audit

This report covers Phase 1 split creation and leakage validation only.
Cross-validation folds were created from the development PD/HC cohort only.
Held-out NDD samples were not used for split creation.
External cohort samples were not used for split creation.
Donor IDs were not available in the processed metadata inputs, so GSM accession was used as the conservative subject identifier for split integrity.

## Inputs

- Development labels: `data/processed/dev_y.npy`
- Development sample IDs: `data/processed/dev_sample_ids.txt`
- Held-out NDD sample IDs: `data/processed/ndd_sample_ids.txt`
- External sample IDs: `data/processed/ext_sample_ids.txt`

## Outputs

- Fold file: `data/processed/dev_folds.json`
- Number of folds: `5`

## Fold Summary

- Fold 1: train=`350`, validation=`88`, train_class_counts=`{'HC': 186, 'PD': 164}`, validation_class_counts=`{'HC': 47, 'PD': 41}`
- Fold 2: train=`350`, validation=`88`, train_class_counts=`{'HC': 186, 'PD': 164}`, validation_class_counts=`{'HC': 47, 'PD': 41}`
- Fold 3: train=`350`, validation=`88`, train_class_counts=`{'HC': 186, 'PD': 164}`, validation_class_counts=`{'HC': 47, 'PD': 41}`
- Fold 4: train=`351`, validation=`87`, train_class_counts=`{'HC': 187, 'PD': 164}`, validation_class_counts=`{'HC': 46, 'PD': 41}`
- Fold 5: train=`351`, validation=`87`, train_class_counts=`{'HC': 187, 'PD': 164}`, validation_class_counts=`{'HC': 46, 'PD': 41}`

## Boundary Confirmation

- Splits were created from development PD/HC only: `yes`
- Held-out NDD used for split creation: `no`
- External cohort used for split creation: `no`
- Modeling performed: `no`
- Baselines implemented: `no`
- Pathway masks implemented: `no`
- MSigDB logic implemented: `no`
- External validation used for model selection: `no`
