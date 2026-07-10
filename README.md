# Pathway-Informed Parkinson's Disease Transcriptomics

## Overview

This repository is a methods and reproducibility research project for pathway-informed Parkinson's disease transcriptomics. It evaluates a knowledge-guided biologically informed neural network (BINN) with a Reactome pathway-constrained first layer using a development cohort, a frozen one-time external evaluation, and a held-out neurodegenerative disease (NDD) stress test.

This is not a diagnostic tool, not clinical, and not deployment-ready.

## Key design safeguards

- Fixed annotation-derived gene space.
- Development-only model construction and selection.
- Subject-disjoint five-fold development cross-validation.
- Masked pathway layer with off-mask weights constrained to zero.
- Frozen preprocessing and model payload.
- `frozen-v1` tag at commit `6b150f8f64935e3bbd4b7bb64a5ec59e665a7f22`.
- `HASH_BEFORE.txt` and `HASH_AFTER.txt` chain-of-custody manifests.
- One-time external and NDD scoring after freeze.
- Negative external calibration and fixed-threshold results retained without adjustment.

## Repository structure

- `src/`: data, model, interpretation, and evaluation utilities.
- `scripts/`: phase scripts and final artifact-verification entry points.
- `config/`: project configuration placeholders.
- `docs/`: methods, results, limitations, provenance, gates, and decision log.
- `results/development/`: committed development-result CSV/JSON artifacts.
- `results/external/`: committed frozen external and NDD result artifacts.
- `results/figures/`: final presentation figures.
- `frozen/`: immutable frozen payload files and hash manifests.
- `tests/`: unit, boundary, gate, and artifact-verification tests.

## Data

The project uses GEO accessions `GSE99039` for development and held-out NDD samples, and `GSE6613` for frozen external PD/HC evaluation. Raw and processed data are not committed. They can be reconstructed through the phase scripts and provenance documentation, subject to GEO availability and local processed artifacts.

The frozen model file `frozen/model_v1.pt` is managed through Git LFS. A clean clone must run `git lfs pull` before artifact verification.

## Reproduction

```bash
git clone <repository-url>
cd pathway-informed-pd-transcriptomics
git lfs install
git lfs pull
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/07_verify_reproducibility.py
python scripts/07_generate_final_figures.py
python scripts/07_phase7_gate.py
```

Committed-result verification works without rerunning external scoring. Full data reconstruction requires GEO access, local processed matrices, platform annotations, and the configured MSigDB Reactome GMT file.

## Results summary

| Result | Value |
|---|---:|
| Development pooled AUROC | 0.702895 |
| External AUROC | 0.695455 |
| External balanced accuracy | 0.500000 |
| External Brier | 0.694444 |
| External ECE | 0.694441 |
| Permutation p-value | 0.019608 with 50 permutations |
| Global attribution Spearman | 0.933899 |

External calibration is poor, and fixed-threshold performance is not useful. These results have no clinical interpretation.

## Figures

- [Figure 1: cohort overview](results/figures/fig01_cohort_overview.png)
- [Figure 2: development model comparison](results/figures/fig02_development_model_comparison.png)
- [Figure 3: permutation validation](results/figures/fig03_permutation_validation.png)
- [Figure 4: top pathways](results/figures/fig04_top_pathways.png)
- [Figure 5: attribution agreement](results/figures/fig05_attribution_agreement.png)
- [Figure 6: external validation](results/figures/fig06_external_validation.png)

## Documentation

- [Methods](docs/methods.md)
- [Results](docs/results.md)
- [Limitations](docs/limitations.md)
- [Reproducibility](docs/reproducibility.md)
- [Decision log](docs/decision_log.md)
- [Phase 1 gate](docs/phase1_gate.md)
- [Phase 2 gate](docs/phase2_gate.md)
- [Phase 3 gate](docs/phase3_gate.md)
- [Phase 4 gate](docs/phase4_gate.md)
- [Phase 5 gate](docs/phase5_gate.md)
- [Phase 6 gate](docs/phase6_gate.md)
- [Phase 7 gate](docs/phase7_gate.md)

## Scope and non-claims

- No clinical validation.
- No diagnostic accuracy claim.
- No deployment readiness.
- No mechanistic or causal inference.
- No state-of-the-art claim.

## License

See [LICENSE](LICENSE).
