"""Run a non-training BINN mask-integrity smoke test."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.binn import build_binn_from_npz


MASK_PATH = ROOT / "data" / "processed" / "pathway_mask.npz"
DEV_X_PATH = ROOT / "data" / "processed" / "dev_X.npy"
AUDIT_PATH = ROOT / "docs" / "phase3_binn_model_audit.md"
DECISION_LOG_PATH = ROOT / "docs" / "decision_log.md"
DECISION_ENTRY = """## 2026-07-10 — Phase 3 BINN model foundation audit

- Added a fixed gene-to-pathway masked linear layer and BINN classifier foundation.
- The integrity smoke test uses only four development expression rows for one forward pass and a synthetic backward pass.
- Off-mask weights are excluded from the forward computation, receive zero gradients, and are hard-zeroed after mask application.
- No training, cross-validation, attention, Integrated Gradients, external cohort, or held-out NDD data was used.
"""


def append_if_missing(path: Path, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if block.strip() not in existing:
        separator = "" if not existing or existing.endswith("\n\n") else "\n"
        path.write_text(existing + separator + block.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    mask_shape = sparse.load_npz(MASK_PATH).shape
    dev_batch = torch.from_numpy(
        np.array(np.load(DEV_X_PATH, mmap_mode="r")[:4], dtype=np.float32, copy=True)
    )
    model = build_binn_from_npz(MASK_PATH)
    model.eval()

    logits = model(dev_batch)
    synthetic_loss = logits.sum()
    synthetic_loss.backward()
    masked_layer = model.gene_to_pathway
    if masked_layer.weight.grad is None:
        raise RuntimeError("Masked layer did not receive gradients during the synthetic backward pass.")
    if not torch.all(masked_layer.weight.grad[masked_layer.mask == 0] == 0):
        raise RuntimeError("Off-mask gradients were not zero.")

    model.apply_masks_()
    summary = model.mask_integrity_summary()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    audit = f"""# Phase 3 BINN model audit

Generated at: {datetime.now(timezone.utc).isoformat()}

This is a model-integrity smoke test, not training or evaluation.

| Check | Value |
| --- | ---: |
| Mask shape | `{mask_shape}` |
| Model parameter count | {parameter_count} |
| Forward logits shape | `{tuple(logits.shape)}` |
| Maximum absolute masked weight after enforcement | {summary['max_abs_masked_weight']:.1f} |
| Masked weights | {summary['n_masked_weights']} |
| Unmasked weights | {summary['n_unmasked_weights']} |

Only the first four rows of `data/processed/dev_X.npy` were loaded for one forward pass. A synthetic scalar loss was backpropagated solely to verify gradient masking. No training, cross-validation, external cohort, or held-out NDD data was used.
"""
    AUDIT_PATH.write_text(audit, encoding="utf-8")
    append_if_missing(DECISION_LOG_PATH, DECISION_ENTRY)

    print(f"mask shape: {mask_shape}")
    print(f"model parameter count: {parameter_count}")
    print(f"forward logits shape: {tuple(logits.shape)}")
    print(f"max_abs_masked_weight: {summary['max_abs_masked_weight']:.1f}")
    print("confirmation no training/CV/external/NDD used: yes")


if __name__ == "__main__":
    main()
