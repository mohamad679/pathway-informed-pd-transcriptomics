# Phase 6 Final Frozen External-Validation Gate

This read-only gate validates chain of custody and internal consistency of the existing frozen bundle, one-time external outputs, NDD stress-test outputs, and scoring audit. No training, preprocessing fitting, inference, rescoring, threshold tuning, or model selection occurs inside the gate.

## Gate Status

- Status: `PASS`.
- PASS means chain-of-custody and artifact integrity only.
- PASS does not mean clinical validation.
- PASS does not mean model deployment readiness.
- PASS does not mean calibration is acceptable.
- No clinical, diagnostic, deployment, or biological claim is made.

## Frozen Chain of Custody

- `HASH_BEFORE.txt` and `HASH_AFTER.txt` are exactly identical.
- Both manifests independently verify the exact six-file immutable payload.
- Frozen commit: `6b150f8f64935e3bbd4b7bb64a5ec59e665a7f22`.
- Frozen tag: `frozen-v1`.
- Frozen payload modified: `false`.

## External PD/HC Artifact

- Rows: `72` (`22` HC, `50` PD).
- Fixed threshold: `0.5`.
- AUROC: `0.695455`; this indicates ranking discrimination only and is not clinical validity.
- AUPRC: `0.782081`.
- Fixed-threshold balanced accuracy: `0.500000`.
- Brier: `0.694444`.
- ECE: `0.694441`.
- The fixed-threshold balanced accuracy and poor calibration remain explicit limitations.

## Held-Out NDD Artifact

- Rows: `48`.
- Fraction predicted PD at 0.5: `0.520833`.
- NDD remains an unlabeled specificity/stress test only, not diagnostic validation.

## Boundary Confirmation

- No model retraining or weight update occurred inside the gate.
- No preprocessing or scaler refit occurred inside the gate.
- No external or NDD inference/rescoring occurred inside the gate.
- No external metric was used for model selection.
- No Phase 7 work was performed.
