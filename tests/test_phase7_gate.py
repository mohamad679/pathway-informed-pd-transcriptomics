from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase7_gate import (  # noqa: E402
    audit_claim_safety,
    audit_documentation,
    audit_figures,
    audit_frozen_chain,
    audit_repository_hygiene,
    audit_result_values,
    is_lfs_pointer,
    run_phase7_gate,
)


def _copy_project_subset(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for directory in ("docs", "results", "frozen"):
        shutil.copytree(ROOT / directory, root / directory)
    for filename in ("README.md", ".gitattributes"):
        shutil.copy2(ROOT / filename, root / filename)
    return root


def test_missing_document_fails(tmp_path: Path) -> None:
    root = _copy_project_subset(tmp_path)
    (root / "docs" / "methods.md").unlink()
    with pytest.raises(ValueError, match="missing required documentation"):
        audit_documentation(root)


def test_missing_figure_fails(tmp_path: Path) -> None:
    root = _copy_project_subset(tmp_path)
    (root / "results" / "figures" / "fig02_development_model_comparison.png").unlink()
    with pytest.raises(ValueError, match="missing or empty figures"):
        audit_figures(root)


def test_incorrect_metric_fails(tmp_path: Path) -> None:
    root = _copy_project_subset(tmp_path)
    path = root / "results" / "development" / "baseline_summary.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["model"] == "logistic_regression", "auroc_mean"] = 0.1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="baseline_logistic_auroc"):
        audit_result_values(root)


def test_forbidden_claim_fails(tmp_path: Path) -> None:
    root = _copy_project_subset(tmp_path)
    readme = root / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + "\nThis is clinically validated.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden claim"):
        audit_claim_safety(root)


def test_lfs_pointer_working_tree_file_fails(tmp_path: Path) -> None:
    root = _copy_project_subset(tmp_path)
    (root / "frozen" / "model_v1.pt").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
        "size 456\n",
        encoding="utf-8",
    )
    assert is_lfs_pointer(root / "frozen" / "model_v1.pt")
    with pytest.raises(ValueError, match="Git LFS pointer"):
        audit_frozen_chain(root)


def test_hash_mismatch_fails(tmp_path: Path) -> None:
    root = _copy_project_subset(tmp_path)
    (root / "frozen" / "HASH_AFTER.txt").write_text("different\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must match exactly"):
        audit_frozen_chain(root)


def test_missing_calibration_limitation_fails(tmp_path: Path) -> None:
    root = _copy_project_subset(tmp_path)
    for relative in ("README.md", "docs/methods.md", "docs/results.md", "docs/reproducibility.md"):
        path = root / relative
        text = path.read_text(encoding="utf-8")
        text = text.replace("poor calibration", "calibration issue")
        text = text.replace("calibration was severely poor", "calibration was limited")
        path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="required limitation language"):
        audit_claim_safety(root)


def test_tracked_raw_processed_path_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from eval import phase7_gate

    monkeypatch.setattr(phase7_gate, "_git_lines", lambda root, command: ["data/processed/dev_X.npy"])
    with pytest.raises(ValueError, match="tracked forbidden"):
        audit_repository_hygiene(ROOT)


def test_valid_fixture_passes_where_practical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _copy_project_subset(tmp_path)
    from eval import phase7_gate

    tracked = [
        "README.md",
        ".gitattributes",
        "frozen/model_v1.pt",
        "results/development/baseline_summary.csv",
        "results/external/external_metrics.json",
    ]
    monkeypatch.setattr(phase7_gate, "_git_lines", lambda current_root, command: tracked)
    summary = run_phase7_gate(root)
    assert summary["status"] == "PASS"
    assert json.loads(json.dumps(summary))["training_scoring_recomputation"] is False
