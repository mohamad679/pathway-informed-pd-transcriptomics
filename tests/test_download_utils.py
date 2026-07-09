from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.download import ensure_dir, sha256_file


def test_ensure_dir_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    created = ensure_dir(target)

    assert created == target
    assert target.is_dir()


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    target = tmp_path / "example.txt"
    target.write_text("phase-1-download\n", encoding="utf-8")

    assert (
        sha256_file(target)
        == "78904e662e09ceb3f04c8bb321ecc186b26e7b74e86319a639813419acd8d96d"
    )
