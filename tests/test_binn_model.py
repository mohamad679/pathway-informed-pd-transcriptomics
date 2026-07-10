from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.binn import BINNClassifier, build_binn_from_npz


MASK = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)


def test_forward_output_shape() -> None:
    model = BINNClassifier(MASK, hidden_dim=4, dropout=0.0)

    assert model(torch.zeros(5, 3)).shape == (5,)


def test_apply_masks_preserves_zero_masked_weights() -> None:
    model = BINNClassifier(MASK, hidden_dim=4)
    with torch.no_grad():
        model.gene_to_pathway.weight.fill_(7.0)

    model.apply_masks_()

    assert torch.all(model.gene_to_pathway.weight[model.gene_to_pathway.mask == 0] == 0)


def test_mask_integrity_summary_keys() -> None:
    model = BINNClassifier(MASK, hidden_dim=4)
    summary = model.mask_integrity_summary()

    assert set(summary) == {"max_abs_masked_weight", "n_masked_weights", "n_unmasked_weights"}
    assert summary["n_masked_weights"] == 3
    assert summary["n_unmasked_weights"] == 3


def test_build_from_temporary_npz(tmp_path: Path) -> None:
    mask_path = tmp_path / "synthetic_mask.npz"
    sparse.save_npz(mask_path, sparse.csr_matrix(MASK))

    model = build_binn_from_npz(mask_path, hidden_dim=5, dropout=0.0)

    assert model.gene_to_pathway.weight.shape == (2, 3)
    assert model.pathway_to_hidden.out_features == 5
