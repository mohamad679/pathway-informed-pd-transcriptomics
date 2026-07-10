from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "07_verify_reproducibility.py"
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase7_gate import is_lfs_pointer  # noqa: E402


def _load_script_module():
    spec = importlib.util.spec_from_file_location("verify_reproducibility", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_boundaries_do_not_load_raw_or_processed_arrays() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden = (
        "np.load",
        "numpy.load",
        ".npy",
        "torch.load",
        "fit(",
        "predict(",
        "predict_proba",
        "data/raw",
        "data/processed",
    )
    assert not [token for token in forbidden if token in source]


def test_required_files_do_not_include_data_array_paths() -> None:
    module = _load_script_module()
    required_text = "\n".join(module.REQUIRED_FILES)
    for forbidden in ("ext_X", "ext_y", "ndd_X", "dev_X", "dev_y", ".npy"):
        assert forbidden not in required_text


def test_lfs_pointer_rejection(tmp_path: Path) -> None:
    pointer = tmp_path / "model_v1.pt"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n"
        "size 123\n",
        encoding="utf-8",
    )
    binary = tmp_path / "real_model.pt"
    binary.write_bytes(b"\x80\x02real-binary-payload")
    assert is_lfs_pointer(pointer)
    assert not is_lfs_pointer(binary)


def test_key_metric_mismatch_fails(tmp_path: Path) -> None:
    module = _load_script_module()
    temp_root = tmp_path / "project"
    for relative in (
        "results/development/binn_cv.csv",
        "results/development/baseline_summary.csv",
        "results/development/pathway_attribution_global_agreement.csv",
        "results/development/pathway_attribution_seed_fold_agreement.csv",
        "results/development/statistical_validation_bootstrap_ci.csv",
    ):
        target = temp_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    phase5_summary = {"observed_auroc": 0.7028949602800749, "null_auroc_mean": 0.5360381729997558, "null_auroc_std": 0.022578589987006906, "empirical_p_value": 0.0196078431372549}
    phase6_summary = {
        "external_metrics": {
            "auroc": 0.6954545454545455,
            "auprc": 0.7820807603992959,
            "balanced_accuracy": 0.5,
            "brier": 0.6944439246508343,
            "ece": 0.6944412781278375,
        },
        "ndd_summary": {
            "mean_pd_probability": 0.48905033407815307,
            "fraction_predicted_pd_at_0_5": 0.5208333333333334,
        },
    }
    baseline_path = temp_root / "results" / "development" / "baseline_summary.csv"
    baseline = pd.read_csv(baseline_path)
    baseline.loc[baseline["model"] == "logistic_regression", "auroc_mean"] = 0.1
    baseline.to_csv(baseline_path, index=False)
    with pytest.raises(ValueError, match="baseline_logistic_auroc"):
        module._verify_authoritative_metrics(temp_root, phase5_summary, phase6_summary)
