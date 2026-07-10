# Phase 7 Final Project Gate

## Gate Status

- Status: `PASS`.
- PASS means final methods/reproducibility artifact integrity only.
- PASS does not mean clinical validation, diagnostic utility, deployment readiness, mechanistic inference, or superiority.

## Documentation Audit

- Required documents: `5`.
- Phase 1-6 PASS reports: `6`.

## Figure Audit

- Figures present and nonempty: `6`.
- Final top-20 companion rows: `20`.

## Frozen Hash Audit

- `HASH_BEFORE.txt` equals `HASH_AFTER.txt`.
- Immutable payload files verified: `6`.
- `frozen/model_v1.pt` is not a Git LFS pointer in the working copy.

## Result-Value Audit

- Development, permutation, attribution, external, and NDD authoritative values match committed source artifacts.
- No raw/processed arrays were loaded.
- No result file was changed by the gate.

## Claim-Safety Audit

- No forbidden clinical, diagnostic, deployment, generalization, strong-performance, or SOTA claim text was found.
- External poor calibration and balanced accuracy 0.5 are visible.
- The 50-permutation limitation is visible.
- NDD stress-test-only language is visible.

## Repository-Hygiene Audit

- No tracked caches, notebooks, raw/processed data, or external input arrays were found.
- The expected large model is configured for Git LFS.

## Boundary Confirmation

- Tests are not run inside the gate.
- No training/scoring/recomputation occurred.
