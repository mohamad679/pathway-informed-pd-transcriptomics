"""Frozen model bundle serialization and pre-score hash-manifest utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from models.final_training import serialize_scaler


PAYLOAD_FILES = (
    "model_v1.pt",
    "preprocessing_config.json",
    "gene_space.txt",
    "pathway_names.txt",
    "pathway_mask.npz",
    "training_metadata.json",
)


def _write_lines(path: Path, values: Iterable[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")


def save_frozen_bundle(
    *,
    frozen_dir: str | Path,
    model: torch.nn.Module,
    scaler: StandardScaler,
    gene_space: list[str],
    pathway_names: list[str],
    pathway_mask: np.ndarray,
    training_metadata: dict[str, object],
    hidden_dim: int = 64,
    dropout: float = 0.25,
    seed: int = 11,
    n_epochs: int,
) -> dict[str, Path]:
    """Write the immutable Phase 6 frozen payload files."""
    output_dir = Path(frozen_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_array = np.asarray(pathway_mask)
    if mask_array.ndim != 2:
        raise ValueError("pathway_mask must be two-dimensional.")
    if mask_array.shape != (len(pathway_names), len(gene_space)):
        raise ValueError("pathway names, gene space, and pathway mask dimensions must agree.")

    payload_paths = {name: output_dir / name for name in PAYLOAD_FILES}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": "BINNClassifier",
            "hidden_dim": int(hidden_dim),
            "dropout": float(dropout),
            "n_genes": int(mask_array.shape[1]),
            "n_pathways": int(mask_array.shape[0]),
            "seed": int(seed),
            "n_epochs": int(n_epochs),
        },
        payload_paths["model_v1.pt"],
    )
    preprocessing_config = {
        "preprocessing_order": [
            "input matrices are already within-sample z-scored before this frozen bundle",
            "align features exactly to gene_space.txt order",
            "apply frozen StandardScaler parameters fitted on full development only",
            "run BINNClassifier forward pass",
        ],
        "source_data_within_sample_z_scored": True,
        "standard_scaler": {
            "fit_scope": "full_development_only",
            "parameters": serialize_scaler(scaler),
        },
        "feature_order": "gene_space.txt",
        "missing_external_genes_policy": "fail_loudly",
        "imputation": "none",
        "feature_selection": "none",
        "binary_class_mapping": {"HC": 0, "PD": 1},
        "development_cohort_only_used_for_fitting": True,
    }
    payload_paths["preprocessing_config.json"].write_text(
        json.dumps(preprocessing_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_lines(payload_paths["gene_space.txt"], gene_space)
    _write_lines(payload_paths["pathway_names.txt"], pathway_names)
    sparse.save_npz(payload_paths["pathway_mask.npz"], sparse.csr_matrix(mask_array))
    payload_paths["training_metadata.json"].write_text(
        json.dumps(training_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload_paths


def compute_bundle_hashes(frozen_dir: str | Path) -> dict[str, str]:
    """Compute SHA-256 hashes for immutable payload files only."""
    base = Path(frozen_dir)
    hashes: dict[str, str] = {}
    for filename in PAYLOAD_FILES:
        path = base / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing frozen payload file: {filename}")
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def write_hash_manifest(hashes: dict[str, str], output_path: str | Path) -> None:
    """Write a deterministic two-column SHA-256 manifest sorted by filename."""
    lines = [f"{hashes[filename]}  {filename}" for filename in sorted(hashes)]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_hash_manifest(manifest_path: Path) -> dict[str, str]:
    manifest_hashes: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid hash manifest line {line_number}.")
        sha256, filename = parts
        if filename in manifest_hashes:
            raise ValueError(f"Duplicate hash manifest entry: {filename}")
        manifest_hashes[filename] = sha256
    return manifest_hashes


def verify_hash_manifest(frozen_dir: str | Path, manifest_path: str | Path) -> bool:
    """Verify a frozen payload hash manifest against the current payload files."""
    expected_names = set(PAYLOAD_FILES)
    manifest_hashes = _read_hash_manifest(Path(manifest_path))
    manifest_names = set(manifest_hashes)
    if manifest_names != expected_names:
        missing = sorted(expected_names - manifest_names)
        extra = sorted(manifest_names - expected_names)
        raise ValueError(f"Hash manifest payload mismatch; missing={missing}, extra={extra}")
    current_hashes = compute_bundle_hashes(frozen_dir)
    mismatched = [
        filename
        for filename in PAYLOAD_FILES
        if current_hashes[filename] != manifest_hashes[filename]
    ]
    if mismatched:
        raise ValueError(f"Frozen payload hash mismatch: {mismatched}")
    return True
