"""Write and verify the post-score hash manifest for immutable frozen payloads."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.frozen_bundle import compute_bundle_hashes, verify_hash_manifest, write_hash_manifest


FROZEN_DIR = ROOT / "frozen"
HASH_BEFORE_PATH = FROZEN_DIR / "HASH_BEFORE.txt"
HASH_AFTER_PATH = FROZEN_DIR / "HASH_AFTER.txt"


def main() -> int:
    payload_hashes_after = compute_bundle_hashes(FROZEN_DIR)
    write_hash_manifest(payload_hashes_after, HASH_AFTER_PATH)
    print("HASH_AFTER written")

    verify_hash_manifest(FROZEN_DIR, HASH_BEFORE_PATH)
    verify_hash_manifest(FROZEN_DIR, HASH_AFTER_PATH)
    if HASH_BEFORE_PATH.read_text(encoding="utf-8") != HASH_AFTER_PATH.read_text(
        encoding="utf-8"
    ):
        raise ValueError("HASH_BEFORE and HASH_AFTER payload hashes differ")

    print("HASH_BEFORE vs HASH_AFTER: PASS")
    print("frozen payload modified: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
