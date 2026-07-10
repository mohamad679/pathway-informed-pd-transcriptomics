"""Neural-network layers with fixed, biologically informed connectivity masks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class MaskedLinear(nn.Module):
    """A linear layer whose off-mask connections are permanently inactive."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: Tensor | object,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must both be positive.")

        mask_tensor = torch.as_tensor(mask)
        expected_shape = (out_features, in_features)
        if tuple(mask_tensor.shape) != expected_shape:
            raise ValueError(
                f"mask must have shape {expected_shape}, got {tuple(mask_tensor.shape)}."
            )
        if not torch.all((mask_tensor == 0) | (mask_tensor == 1)).item():
            raise ValueError("mask must be binary, containing only 0 and 1 values.")

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("mask", mask_tensor.to(dtype=torch.float32))
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()
        self.weight.register_hook(self._mask_gradient)

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
        self.apply_mask_()

    def _mask_gradient(self, gradient: Tensor) -> Tensor:
        return gradient * self.mask.to(dtype=gradient.dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        masked_weight = self.weight * self.mask.to(dtype=self.weight.dtype)
        return F.linear(inputs, masked_weight, self.bias)

    @torch.no_grad()
    def apply_mask_(self) -> "MaskedLinear":
        """Hard-zero weights for all connections absent from the mask."""
        self.weight.mul_(self.mask.to(dtype=self.weight.dtype))
        return self

    def count_masked_weights(self) -> int:
        """Return the number of permanently disabled (zero-mask) weights."""
        return int((self.mask == 0).sum().item())

    def max_abs_masked_weight(self) -> float:
        """Return the largest absolute value among disabled weights."""
        masked_values = self.weight.detach()[self.mask == 0]
        return float(masked_values.abs().max().item()) if masked_values.numel() else 0.0
