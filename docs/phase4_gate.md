# Phase 4 Final Gate Audit

This is a development-only attribution artifact integrity audit, not final validation, not a final performance claim, and not a biological claim.

## Gate Status

- Status: `PASS`
- The Phase 4 gate concerns existing activation and Integrated Gradients artifact integrity only.
- Activation and IG show agreement/stability summaries only.

## Validated Development Outputs

- Activation score rows: `19455`.
- Integrated Gradients score rows: `19455`.
- Pathway count: `1297`.
- Exact seeds: `11`, `23`, `37`; folds: `1`-`5`; pathways per seed/fold: `1,297`.
- Mean seed/fold Spearman: `0.808936`.
- Global Spearman: `0.933899`.
- Global top-20 overlap: `12`.
- Activation top-20 RNA-processing count: `0`.
- IG top-20 RNA-processing count: `0`.
- RNA-processing tier rows: `4`.

## Boundary Confirmation

- No training or retraining was run inside the gate.
- Activation attribution was not run inside the gate.
- Integrated Gradients attribution was not run inside the gate.
- No external cohort or held-out NDD data was used.
- This is not final validation.
- This is not a biological claim.
