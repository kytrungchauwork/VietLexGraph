import hashlib
import io
import json
import zipfile

import pytest

from scripts.download_model import install_archive


def payload(extra=None):
    files = {
        "relation_model.pkl": b"fake model; never unpickle this fixture",
        "scaler.pkl": b"fake scaler",
        "LICENSE": b"synthetic license fixture",
        "README.md": b"synthetic fixture",
    }
    files["metadata.json"] = json.dumps(
        {
            "artifact_hashes": {
                n: hashlib.sha256(files[n]).hexdigest()
                for n in ("relation_model.pkl", "scaler.pkl")
            }
        }
    ).encode()
    if extra:
        files.update(extra)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_verified_install_is_idempotent_and_never_overwrites(tmp_path):
    blob = payload()
    digest = hashlib.sha256(blob).hexdigest()
    destination = tmp_path / "models"
    install_archive(blob, destination, digest)
    install_archive(blob, destination, digest)
    (destination / "relation_model.pkl").write_bytes(b"existing local research model")
    with pytest.raises(ValueError, match="already exists"):
        install_archive(blob, destination, digest)
    assert (destination / "relation_model.pkl").read_bytes() == b"existing local research model"


def test_bad_checksum_and_archive_paths_write_nothing(tmp_path):
    blob = payload()
    with pytest.raises(ValueError, match="checksum"):
        install_archive(blob, tmp_path / "models", "0" * 64)
    evil = payload({"../leak.txt": b"bad"})
    with pytest.raises(ValueError, match="members"):
        install_archive(evil, tmp_path / "models", hashlib.sha256(evil).hexdigest())
    assert not list(tmp_path.iterdir())
