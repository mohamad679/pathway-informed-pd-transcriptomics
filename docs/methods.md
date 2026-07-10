# Methods

## Study design

This project used development-only model construction and selection, followed by a frozen one-time external evaluation and a held-out NDD stress test. Development, external, and NDD cohorts were kept separate in purpose and timing: development artifacts drove model construction, the external cohort was scored only after the model and preprocessing were frozen, and NDD samples were used only as an unlabeled stress test.

## Data sources and cohorts

The development cohort was `GSE99039`, with 438 PD/HC samples. The frozen external cohort was `GSE6613`, with 72 PD/HC samples: 22 HC and 50 PD. The held-out NDD set contained 48 samples.

Platform information was derived from project provenance and platform annotation artifacts. The implementation did not invent sample metadata beyond repository evidence.

## Probe annotation and fixed gene space

Probe annotation used platform annotations only. The fixed gene space was defined by the `GPL570` and `GPL96` annotation intersection, yielding 13,908 genes. No outcome-based feature selection was performed. Exact feature ordering was persisted in the frozen `gene_space.txt` payload and development processed artifacts.

## Preprocessing

Processed expression matrices used existing within-sample z-scoring. During development cross-validation, `StandardScaler` was fit on each training partition only. After freeze, the final `StandardScaler` was fit on full development only and stored in `frozen/preprocessing_config.json`. No external fitting, imputation, or post-freeze recalibration was performed.

## Development splits

Development evaluation used subject-disjoint five-fold development cross-validation with seeds 11, 23, and 37. Each sample appeared once in validation per seed. External and NDD cohorts were excluded from development decisions.

## Baselines

Development baselines were Logistic Regression, Random Forest, and an unconstrained MLP.

`src/models/logistic_baseline.py` defines a `StandardScaler` plus `LogisticRegression` pipeline using `solver="liblinear"`, L2 penalty, `class_weight="balanced"`, `max_iter=5000`, and seed-specific `random_state`.

`src/models/random_forest_baseline.py` defines `RandomForestClassifier` with 500 trees, `max_features="sqrt"`, `class_weight="balanced_subsample"`, `n_jobs=-1`, and seed-specific `random_state`.

`src/models/mlp_baseline.py` defines a `StandardScaler` plus `MLPClassifier` pipeline with one 128-unit hidden layer, ReLU activation, Adam solver, `alpha=0.0001`, batch size 64, learning rate 0.001, `max_iter=300`, early stopping, validation fraction 0.15, `n_iter_no_change=20`, and seed-specific `random_state`.

## Pathway prior and BINN

The pathway prior used MSigDB Reactome `2024.1.Hs`. The retained mask contained 1,297 pathways, 13,908 genes, and 75,429 retained mask edges. The BINN implementation in `src/models/binn.py` uses a masked gene-to-pathway first layer, ReLU, dropout, and a downstream hidden layer. Off-mask weights are constrained to zero by `src/models/masked_layers.py` and training-time mask application. Attention was not implemented; activation ranking is not attention.

## Development evaluation

Development metrics included AUROC, AUPRC, balanced accuracy, Brier score, and ECE, using a fixed threshold of 0.5 for threshold metrics. Out-of-fold predictions were saved for pooled development summaries. Statistical validation used bootstrap confidence intervals and a limited 50-permutation empirical test. The empirical p-value formula was:

```text
(1 + count(null >= observed)) / (1 + n_permutations)
```

## Pathway attribution

Development-only interpretation retraining produced activation-based pathway ranking and pathway-level Integrated Gradients from a zero baseline. Agreement was summarized with Spearman correlation, top-20 overlap, and fold stability. These outputs are model attribution only, not causal or mechanistic biological explanation.

## Final model freeze

The final epoch count was selected as the deterministic median of the 15 development-CV `best_epoch` values. The selected count was 16 epochs. The frozen model used seed 11, `hidden_dim=64`, `dropout=0.25`, and 18,123,110 parameters. Full-development retraining wrote six immutable payload files under `frozen/`, stored the model with Git LFS, and produced `HASH_BEFORE.txt` before scoring. `HASH_AFTER.txt` after scoring is identical. The frozen tag is `frozen-v1` at commit `6b150f8f64935e3bbd4b7bb64a5ec59e665a7f22`.

## One-time external evaluation

The frozen external evaluation used only the frozen model and frozen scaler, with threshold fixed at 0.5. No tuning, retraining, recalibration, or model selection was performed. The external cohort was scored once, and the NDD cohort was scored once.

## Software and reproducibility

The project uses Python with NumPy, pandas, SciPy, scikit-learn, PyTorch, Captum, matplotlib, and pytest. Exact pinned versions are in `requirements.txt`. Raw and processed data are ignored by Git. Git LFS is required for `frozen/model_v1.pt`.

Key implementation entry points include `scripts/01_prepare_data.py`, `scripts/02_summarize_baselines.py`, `scripts/03_run_binn_cv.py`, `scripts/04_summarize_attribution_agreement.py`, `scripts/05_run_statistical_validation.py`, `scripts/06_freeze_model.py`, `scripts/06_score_external_once.py`, `scripts/07_verify_reproducibility.py`, `scripts/07_generate_final_figures.py`, and `scripts/07_phase7_gate.py`.
