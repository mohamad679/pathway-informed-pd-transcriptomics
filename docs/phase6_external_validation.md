# Phase 6 frozen-model external validation

This report records the one-time inference-only Phase 6 execution.

## Chain of custody

- Frozen commit: `6b150f8f64935e3bbd4b7bb64a5ec59e665a7f22`
- Frozen tag: `frozen-v1`
- `frozen-v1` was verified before loading external or NDD inputs.
- `HASH_BEFORE.txt` was verified before scoring.
- No model retraining, weight update, or preprocessing fitting occurred.
- The external cohort was scored once and the NDD cohort was scored once.
- The classification threshold was fixed at exactly `0.5`; outcomes were not used for tuning or selection.

## External PD/HC cohort

- Samples: `72` (`22` HC, `50` PD)
- AUROC: `0.695455`
- AUPRC: `0.782081`
- Balanced accuracy at 0.5: `0.500000`
- Brier score: `0.694444`
- ECE: `0.694441`

## Held-out NDD specificity stress test

The NDD cohort has no binary target label in this analysis. This is a specificity/stress-test summary only and is not diagnostic validation.

- Samples: `48`
- Mean PD probability: `0.489050`
- Fraction predicted PD at 0.5: `0.520833`

These results are research validation only, not clinical validation, and do not support clinical, diagnostic, or deployment claims.
