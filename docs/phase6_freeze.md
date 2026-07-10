# Phase 6 pre-external frozen BINN bundle

This report documents only pre-external model selection, full-development retraining, frozen bundle creation, and pre-score hashing.

- Final epoch count was selected from development CV `best_epoch` values only.
- The deterministic rule is the integer median across the 15 seed/fold `best_epoch` values.
- The final `StandardScaler` was fit on full development only.
- External and NDD files were not loaded.
- The external cohort has not yet been scored.
- Frozen files must not change after commit/tag; later scoring must verify the pre-score hash manifest first.
- This is not external-validation performance and makes no final/external performance claim.

## Frozen training configuration

- Development sample count: `438`
- Gene count: `13908`
- Pathway count: `1297`
- Selected epoch count: `16`
- Final seed: `11`
- Parameter count: `18123110`
- Max masked weight: `0.0`
- Architecture: `hidden_dim=64`, `dropout=0.25`
- Optimizer: Adam with `learning_rate=1e-3`, `weight_decay=1e-4`, `batch_size=64`
