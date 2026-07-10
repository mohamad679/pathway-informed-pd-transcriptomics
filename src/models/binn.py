"""Foundation BINN classifier with a fixed gene-to-pathway connectivity layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch import Tensor, nn

from models.masked_layers import MaskedLinear


class BINNClassifier(nn.Module):
    """Binary classifier constrained by a gene-to-pathway membership mask."""

    def __init__(
        self,
        pathway_mask: Tensor | np.ndarray,
        hidden_dim: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        mask = torch.as_tensor(pathway_mask)
        if mask.ndim != 2:
            raise ValueError("pathway_mask must be two-dimensional: (n_pathways, n_genes).")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1.")

        n_pathways, n_genes = mask.shape
        self.gene_to_pathway = MaskedLinear(n_genes, n_pathways, mask)
        self.pathway_to_hidden = nn.Linear(n_pathways, hidden_dim)
        self.hidden_to_logit = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        x = self.gene_to_pathway(inputs)
        x = self.dropout(self.relu(x))
        x = self.pathway_to_hidden(x)
        x = self.dropout(self.relu(x))
        return self.hidden_to_logit(x).squeeze(-1)

    def forward_with_pathway_activations(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Return logits and post-ReLU, pre-dropout pathway activations."""
        pathway_activations = self.relu(self.gene_to_pathway(inputs))
        x = self.dropout(pathway_activations)
        x = self.pathway_to_hidden(x)
        x = self.dropout(self.relu(x))
        logits = self.hidden_to_logit(x).squeeze(-1)
        return logits, pathway_activations

    def pathway_head_from_activations(self, pathway_activations: Tensor) -> Tensor:
        """Apply only the downstream classifier head to pathway activations."""
        x = self.dropout(pathway_activations)
        x = self.pathway_to_hidden(x)
        x = self.dropout(self.relu(x))
        return self.hidden_to_logit(x).squeeze(-1)

    @torch.no_grad()
    def apply_masks_(self) -> "BINNClassifier":
        """Hard-zero weights outside every fixed-connectivity mask."""
        for module in self.modules():
            if isinstance(module, MaskedLinear):
                module.apply_mask_()
        return self

    def mask_integrity_summary(self) -> dict[str, float | int]:
        """Summarize fixed-connectivity weight integrity across masked layers."""
        masked_layers = [module for module in self.modules() if isinstance(module, MaskedLinear)]
        n_masked_weights = sum(module.count_masked_weights() for module in masked_layers)
        n_total_weights = sum(module.weight.numel() for module in masked_layers)
        max_abs_masked_weight = max(
            (module.max_abs_masked_weight() for module in masked_layers), default=0.0
        )
        return {
            "max_abs_masked_weight": max_abs_masked_weight,
            "n_masked_weights": n_masked_weights,
            "n_unmasked_weights": n_total_weights - n_masked_weights,
        }


def build_binn_from_npz(
    mask_path: str | Path,
    hidden_dim: int = 64,
    dropout: float = 0.25,
) -> BINNClassifier:
    """Load a SciPy sparse pathway mask NPZ and build a BINN classifier."""
    mask = sparse.load_npz(Path(mask_path)).toarray().astype(np.float32, copy=False)
    return BINNClassifier(mask, hidden_dim=hidden_dim, dropout=dropout)
