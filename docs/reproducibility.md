# Reproducibility

## Level 1: Artifact verification from a clean clone

Artifact verification checks committed result files and frozen payload integrity without rescoring.

Required steps:

```bash
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

This level verifies that required committed files exist, `frozen/model_v1.pt` is a real downloaded Git LFS binary rather than a pointer file, `HASH_BEFORE.txt` equals `HASH_AFTER.txt`, both manifests verify the six frozen payload files, external result integrity is internally consistent, phase gate reports contain `PASS`, and presentation figures can be regenerated from committed result artifacts.

No raw arrays, processed arrays, model training, external inference, NDD inference, rescoring, recalibration, or threshold adjustment are required.

## Level 2: Full computational reconstruction

Full reconstruction requires reacquiring GEO data, platform annotations, processed matrices, the MSigDB Reactome GMT file, and substantially more compute. It includes reproducing development baselines, BINN cross-validation, attribution retraining, limited permutation testing, bootstrap, model freeze, and one-time post-freeze scoring.

The external cohort must not be reused for iterative model selection, tuning, threshold adjustment, or recalibration. The frozen external evaluation remains a one-time post-freeze evaluation.
