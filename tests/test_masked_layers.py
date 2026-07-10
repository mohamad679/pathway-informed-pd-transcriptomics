from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.masked_layers import MaskedLinear


def test_masked_forward_ignores_off_mask_weights() -> None:
    layer = MaskedLinear(3, 2, torch.tensor([[1, 0, 1], [0, 1, 0]]), bias=False)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[2.0, 999.0, 3.0], [999.0, 5.0, 999.0]]))

    output = layer(torch.tensor([[1.0, 10.0, 2.0]]))

    assert torch.allclose(output, torch.tensor([[8.0, 50.0]]))


def test_apply_mask_zeros_off_mask_weights() -> None:
    layer = MaskedLinear(2, 2, [[1, 0], [0, 1]])
    with torch.no_grad():
        layer.weight.fill_(4.0)

    layer.apply_mask_()

    assert torch.equal(layer.weight, torch.tensor([[4.0, 0.0], [0.0, 4.0]]))
    assert layer.max_abs_masked_weight() == 0.0


def test_off_mask_gradients_are_zero() -> None:
    layer = MaskedLinear(2, 2, [[1, 0], [0, 1]], bias=False)
    layer(torch.ones(1, 2)).sum().backward()

    assert layer.weight.grad is not None
    assert torch.equal(layer.weight.grad[layer.mask == 0], torch.zeros(2))


def test_invalid_mask_shape_fails() -> None:
    with pytest.raises(ValueError, match="shape"):
        MaskedLinear(3, 2, [[1, 0], [0, 1]])


def test_non_binary_mask_fails() -> None:
    with pytest.raises(ValueError, match="binary"):
        MaskedLinear(2, 2, [[1, 0.5], [0, 1]])
