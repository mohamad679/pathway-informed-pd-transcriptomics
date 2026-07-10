# Phase 3 Final Gate Audit

This is a development-only audit of existing BINN outputs, not final validation or a final performance claim.

## Gate Status

- Status: `PASS`
- The Phase 3 gate concerns stable training and pathway-mask integrity, not model superiority.

## Validated Development Outputs

- Mask shape: `(1297, 13908)`; nonzero entries: `75429`.
- Exact model: `pathway_constrained_binn`; seeds: `11`, `23`, `37`; folds: `1`–`5`.
- All 15 metric rows and 1,314 OOF rows passed integrity checks.
- Each seed covers 438 unique development sample indices exactly once.
- Off-mask weights are exactly zero; reported masked/unmasked weight counts match the pathway mask.
- Mean development metrics across fold/seed rows:

| AUROC | AUPRC | Balanced accuracy | Brier | ECE | Max masked weight |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.712859 | 0.676449 | 0.649416 | 0.249931 | 0.208059 | 0.0 |

## Boundary Confirmation

- No external cohort or held-out NDD result was used.
- No model was trained and BINN CV was not rerun inside this gate.
- This is development-only and not final validation.
