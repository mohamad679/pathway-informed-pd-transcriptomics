from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "05_run_statistical_validation.py"


def load_phase5_script():
    spec = importlib.util.spec_from_file_location("phase5_statistical_validation", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_cuda_device_selection_failure_is_clear_when_unavailable(monkeypatch) -> None:
    phase5 = load_phase5_script()
    monkeypatch.setattr(phase5.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="--device cuda was requested"):
        phase5.resolve_device("cuda")


def test_auto_device_uses_cpu_when_cuda_unavailable(monkeypatch) -> None:
    phase5 = load_phase5_script()
    monkeypatch.setattr(phase5.torch.cuda, "is_available", lambda: False)

    assert phase5.resolve_device("auto") == "cpu"


def test_requested_range_empirical_statistics_exclude_unrelated_rows() -> None:
    phase5 = load_phase5_script()
    permutation_df = pd.DataFrame(
        {
            "permutation_index": [1, 2, 99],
            "null_auroc": [0.1, 0.2, 1.0],
        }
    )
    requested_indices = phase5.requested_permutation_indices(1, 2)

    scoped_df = phase5.requested_permutation_subset(permutation_df, requested_indices)
    scoped_p_value = phase5.empirical_p_value(
        0.9,
        scoped_df["null_auroc"].to_numpy(dtype=float),
    )
    unscoped_p_value = phase5.empirical_p_value(
        0.9,
        permutation_df["null_auroc"].to_numpy(dtype=float),
    )

    assert scoped_df["permutation_index"].tolist() == [1, 2]
    assert scoped_p_value == 1 / 3
    assert unscoped_p_value != scoped_p_value


def test_final_coverage_detection_requires_exact_requested_indices() -> None:
    phase5 = load_phase5_script()
    requested_indices = phase5.requested_permutation_indices(1, 3)

    complete_df = pd.DataFrame({"permutation_index": [1, 2, 3], "null_auroc": [0.1, 0.2, 0.3]})
    missing_df = pd.DataFrame({"permutation_index": [1, 3], "null_auroc": [0.1, 0.3]})
    extra_df = pd.DataFrame(
        {"permutation_index": [1, 2, 3, 4], "null_auroc": [0.1, 0.2, 0.3, 0.4]}
    )

    assert phase5.final_index_coverage_complete(complete_df, requested_indices)
    assert not phase5.final_index_coverage_complete(missing_df, requested_indices)
    assert not phase5.final_index_coverage_complete(extra_df, requested_indices)


def test_resume_and_auto_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    phase5 = load_phase5_script()
    resume_path = tmp_path / "resume.csv"
    output_path = tmp_path / "output.csv"
    pd.DataFrame({"permutation_index": [1], "null_auroc": [0.1]}).to_csv(
        resume_path,
        index=False,
    )

    with pytest.raises(ValueError, match="--resume and --auto-resume"):
        phase5.load_resume_dataframe(resume_path, True, output_path)


def test_auto_resume_loads_selected_output_path(tmp_path: Path) -> None:
    phase5 = load_phase5_script()
    output_path = tmp_path / "selected_output.csv"
    expected = pd.DataFrame({"permutation_index": [1], "null_auroc": [0.1]})
    expected.to_csv(output_path, index=False)

    loaded = phase5.load_resume_dataframe(None, True, output_path)

    pd.testing.assert_frame_equal(loaded, expected)
