from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from eval.phase7_gate import run_phase7_gate  # noqa: E402


def main() -> int:
    try:
        summary = run_phase7_gate(ROOT)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        print("FAIL")
        print(f"Phase 7 gate failure: {error}")
        return 1

    print("PASS")
    print(f"documentation audit: {summary['documentation']}")
    print(f"figure audit: {summary['figures']}")
    print(f"frozen hash audit: {summary['frozen']}")
    print(f"result-value audit: {summary['results']}")
    print(f"claim-safety audit: {summary['claims']}")
    print(f"repository-hygiene audit: {summary['hygiene']}")
    print("tests are not run inside the gate: yes")
    print("no training/scoring/recomputation occurred: yes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
