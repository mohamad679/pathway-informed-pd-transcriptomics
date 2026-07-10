from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCORING_SCRIPT = ROOT / "scripts" / "06_score_external_once.py"
POST_HASH_SCRIPT = ROOT / "scripts" / "06_verify_post_score_hash.py"
FROZEN_SCORING_MODULE = ROOT / "src" / "eval" / "frozen_scoring.py"


def _scoring_source() -> str:
    return SCORING_SCRIPT.read_text(encoding="utf-8")


def test_scoring_script_contains_no_optimizer() -> None:
    assert "optimizer" not in _scoring_source().lower()


def test_scoring_script_contains_no_scaler_fit_or_fit_transform() -> None:
    source = _scoring_source()
    assert "StandardScaler" not in source
    assert ".fit(" not in source
    assert "fit_transform" not in source


def test_scoring_script_contains_no_training_function_calls() -> None:
    source = _scoring_source()
    prohibited = (
        "fit_full_development_binn",
        "train_one_binn_fold",
        "run_binn_cv",
        "select_final_epoch_count",
    )
    assert all(name not in source for name in prohibited)


def test_hash_before_verification_precedes_external_and_ndd_array_loading() -> None:
    source = _scoring_source()
    verification = source.index("hash_before_verified = verify_hash_manifest")
    external_load = source.index("external_X = np.load(EXTERNAL_X_PATH")
    ndd_load = source.index("ndd_X = np.load(NDD_X_PATH")
    assert verification < external_load
    assert verification < ndd_load


def test_threshold_is_fixed_at_point_five() -> None:
    module_source = FROZEN_SCORING_MODULE.read_text(encoding="utf-8")
    source = _scoring_source()
    assert "FIXED_THRESHOLD = 0.5" in module_source
    assert "THRESHOLD = 0.5" in source
    assert "threshold tuning" in source


def test_required_frozen_tag_check_exists() -> None:
    source = _scoring_source()
    assert 'REQUIRED_FROZEN_TAG = "frozen-v1"' in source
    assert '"tag", "--points-at", "HEAD"' in source
    assert '"rev-list", "-n", "1", REQUIRED_FROZEN_TAG' in source


def test_result_output_paths_are_under_results_external() -> None:
    source = _scoring_source()
    assert 'RESULTS_DIR = ROOT / "results" / "external"' in source
    assert 'ROOT / "results" / "development"' not in source
    assert "results/development" not in source


def test_post_score_hash_writes_after_and_compares_before() -> None:
    source = POST_HASH_SCRIPT.read_text(encoding="utf-8")
    assert 'HASH_AFTER_PATH = FROZEN_DIR / "HASH_AFTER.txt"' in source
    assert "write_hash_manifest(payload_hashes_after, HASH_AFTER_PATH)" in source
    assert "verify_hash_manifest(FROZEN_DIR, HASH_BEFORE_PATH)" in source
    assert "HASH_BEFORE_PATH.read_text" in source
    assert "HASH_AFTER_PATH.read_text" in source
