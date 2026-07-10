# Phase 3 BINN model audit

Generated at: 2026-07-10T09:33:14.251676+00:00

This is a model-integrity smoke test, not training or evaluation.

| Check | Value |
| --- | ---: |
| Mask shape | `(1297, 13908)` |
| Model parameter count | 18123110 |
| Forward logits shape | `(4,)` |
| Maximum absolute masked weight after enforcement | 0.0 |
| Masked weights | 17963247 |
| Unmasked weights | 75429 |

Only the first four rows of `data/processed/dev_X.npy` were loaded for one forward pass. A synthetic scalar loss was backpropagated solely to verify gradient masking. No training, cross-validation, external cohort, or held-out NDD data was used.
