from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.binn import BINNClassifier
from models.frozen_bundle import (
    PAYLOAD_FILES,
    compute_bundle_hashes,
    save_frozen_bundle,
    verify_hash_manifest,
    write_hash_manifest,
)


def _write_bundle(tmp_path: Path) -> Path:
    mask = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    model = BINNClassifier(mask, hidden_dim=4, dropout=0.0)
    scaler = StandardScaler().fit(np.array([[0.0, 1.0, 2.0], [2.0, 3.0, 4.0]], dtype=np.float32))
    save_frozen_bundle(
        frozen_dir=tmp_path,
        model=model,
        scaler=scaler,
        gene_space=["g1", "g2", "g3"],
        pathway_names=["p1", "p2"],
        pathway_mask=mask,
        training_metadata={"seed": 11, "n_epochs": 2},
        hidden_dim=4,
        dropout=0.0,
        seed=11,
        n_epochs=2,
    )
    return tmp_path


def test_bundle_writes_required_payload_files(tmp_path: Path) -> None:
    frozen_dir = _write_bundle(tmp_path)
    assert {path.name for path in frozen_dir.iterdir()} == set(PAYLOAD_FILES)
    loaded_mask = sparse.load_npz(frozen_dir / "pathway_mask.npz").toarray()
    assert loaded_mask.shape == (2, 3)


def test_hash_manifest_verifies(tmp_path: Path) -> None:
    frozen_dir = _write_bundle(tmp_path)
    manifest = frozen_dir / "HASH_BEFORE.txt"
    write_hash_manifest(compute_bundle_hashes(frozen_dir), manifest)
    assert verify_hash_manifest(frozen_dir, manifest) is True


def test_modified_payload_fails_verification(tmp_path: Path) -> None:
    frozen_dir = _write_bundle(tmp_path)
    manifest = frozen_dir / "HASH_BEFORE.txt"
    write_hash_manifest(compute_bundle_hashes(frozen_dir), manifest)
    (frozen_dir / "gene_space.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_hash_manifest(frozen_dir, manifest)


def test_missing_payload_fails_verification(tmp_path: Path) -> None:
    frozen_dir = _write_bundle(tmp_path)
    manifest = frozen_dir / "HASH_BEFORE.txt"
    write_hash_manifest(compute_bundle_hashes(frozen_dir), manifest)
    (frozen_dir / "pathway_names.txt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing frozen payload file"):
        verify_hash_manifest(frozen_dir, manifest)


def test_hash_before_itself_is_not_included_in_payload_hashes(tmp_path: Path) -> None:
    frozen_dir = _write_bundle(tmp_path)
    (frozen_dir / "HASH_BEFORE.txt").write_text("not a payload\n", encoding="utf-8")
    hashes = compute_bundle_hashes(frozen_dir)
    assert "HASH_BEFORE.txt" not in hashes
    assert set(hashes) == set(PAYLOAD_FILES)


def test_deterministic_manifest_ordering(tmp_path: Path) -> None:
    frozen_dir = _write_bundle(tmp_path)
    manifest = frozen_dir / "HASH_BEFORE.txt"
    hashes = compute_bundle_hashes(frozen_dir)
    write_hash_manifest(dict(reversed(list(hashes.items()))), manifest)
    filenames = [line.split()[1] for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert filenames == sorted(PAYLOAD_FILES)
