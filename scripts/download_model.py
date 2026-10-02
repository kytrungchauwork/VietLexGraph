"""Download the versioned inference-only release, verify it, then install locally."""

import argparse
import hashlib
import io
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1.0.0"
URL = "https://github.com/kytrungchauwork/VietLexGraph/releases/download/v1.0.0/vietlexgraph-model-v1.zip"
SHA256 = "936f0f6b76a862b2f2e8937e62827e83d158062eecc2c359f6987443d999319b"
FILES = {"relation_model.pkl", "scaler.pkl", "metadata.json", "LICENSE", "README.md"}
MAX_BYTES = 5_000_000


def install_archive(payload, destination, expected_sha256=SHA256):
    """Do not deserialize pickles; reject checksum/path/size errors before writing."""
    if len(payload) > MAX_BYTES or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Model archive checksum/size mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != len(FILES) or {m.filename for m in members} != FILES:
            raise ValueError("Unexpected archive members; refusing extraction")
        if sum(m.file_size for m in members) > MAX_BYTES:
            raise ValueError("Expanded model archive is too large")
        content = {m.filename: archive.read(m) for m in members}
    metadata = json.loads(content["metadata.json"])
    for name in ("relation_model.pkl", "scaler.pkl"):
        if hashlib.sha256(content[name]).hexdigest() != metadata["artifact_hashes"][name]:
            raise ValueError("Model file checksum mismatch")
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("Refusing a symlink destination")
    if destination.exists():
        if all(
            (destination / name).is_file()
            and not (destination / name).is_symlink()
            and (destination / name).read_bytes() == blob
            for name, blob in content.items()
        ):
            return
        raise ValueError(
            "Destination already exists; use a new --destination; no files were overwritten"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".vietlexgraph-model-", dir=destination.parent
    ) as folder:
        stage = Path(folder) / "model"
        stage.mkdir()
        for name, blob in content.items():
            (stage / name).write_bytes(blob)
        stage.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    request = urllib.request.Request(URL, headers={"User-Agent": "VietLexGraph-model-downloader"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(MAX_BYTES + 1)
    install_archive(payload, args.destination)
    print(f"Verified {VERSION} inference model installed at {args.destination}")
    print(
        "Encoder weights are fetched separately on first ML use. Only load trusted release pickles."
    )


if __name__ == "__main__":
    main()
