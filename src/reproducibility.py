"""Local artifact integrity and source provenance; hashes are not signatures."""

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_provenance():
    files = sorted(
        [
            *ROOT.glob("src/*.py"),
            *ROOT.glob("scripts/*.py"),
            ROOT / "config.yaml",
            ROOT / "requirements.txt",
        ]
    )
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
        )
        commit = result.stdout.strip() if result.returncode == 0 else None
    except FileNotFoundError:
        commit = None
    return {"code_commit": commit, "source_sha256": hashes}


def verify_hashes(directory, expected):
    directory = Path(directory).resolve()
    for name, digest in expected.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Artifact path outside expected directory")
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"Artifact checksum mismatch: {name}")
