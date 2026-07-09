from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from GEOparse import utils as geoutils


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    file_path = Path(path)
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_geo_series(accession: str, raw_dir: str | Path) -> list[Path]:
    accession = accession.upper()
    accession_dir = ensure_dir(Path(raw_dir) / accession)
    filename = f"{accession}_series_matrix.txt.gz"
    destination = accession_dir / filename
    range_subdir = re.sub(r"\d{1,3}$", "nnn", accession)
    url = (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/"
        f"{range_subdir}/{accession}/matrix/{filename}"
    )
    geoutils.download_from_url(url, str(destination), force=False, silent=False)
    return [destination]


def collect_download_metadata(
    accession: str, downloaded_paths: list[str | Path]
) -> dict[str, object]:
    paths = [Path(path) for path in downloaded_paths]
    if not paths:
        raise ValueError("downloaded_paths must contain at least one file")

    files = []
    mtimes = []
    for path in sorted(paths):
        stat = path.stat()
        mtimes.append(stat.st_mtime)
        files.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
            }
        )

    downloaded_at = datetime.fromtimestamp(min(mtimes), tz=UTC).isoformat()
    return {
        "accession": accession.upper(),
        "downloaded_at_utc": downloaded_at,
        "files": files,
    }
