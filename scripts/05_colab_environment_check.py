"""Check Colab readiness for the Phase 5 development-only permutation run."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "development"

REQUIRED_INPUTS = (
    PROCESSED_DIR / "dev_X.npy",
    PROCESSED_DIR / "dev_y.npy",
    PROCESSED_DIR / "dev_folds.json",
    PROCESSED_DIR / "pathway_mask.npz",
    RESULTS_DIR / "binn_oof_predictions.csv",
)

RECOMMENDED_COMMAND = """python scripts/05_run_statistical_validation.py \\
  --n-permutations 1000 \\
  --start-permutation-index 1 \\
  --n-bootstrap 2000 \\
  --permutation-seed 20260710 \\
  --bootstrap-seed 20260710 \\
  --device cuda \\
  --progress \\
  --checkpoint-every 1 \\
  --auto-resume \\
  --max-epochs 300 \\
  --patience 20 \\
  --hidden-dim 64 \\
  --batch-size 64"""


def main() -> int:
    print(f"Python version: {platform.python_version()} ({sys.executable})")
    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    missing_paths = []
    for path in REQUIRED_INPUTS:
        exists = path.exists()
        print(f"{'OK' if exists else 'MISSING'}: {path.relative_to(ROOT)}")
        if not exists:
            missing_paths.append(path)

    print("\nRecommended production command:")
    print(RECOMMENDED_COMMAND)

    return 1 if missing_paths else 0


if __name__ == "__main__":
    raise SystemExit(main())
