import json

import pytest

from src.model import RelationPredictor
from src.reproducibility import sha256, verify_hashes


def test_integrity_rejects_modified_artifact_and_path_escape(tmp_path):
    file = tmp_path / "data.bin"
    file.write_bytes(b"original")
    manifest = {"data.bin": sha256(file)}
    verify_hashes(tmp_path, manifest)
    file.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_hashes(tmp_path, manifest)
    with pytest.raises(ValueError, match="outside"):
        verify_hashes(tmp_path, {"../outside": "0" * 64})


def test_invalid_metadata_rejected_before_unpickling(tmp_path):
    (tmp_path / "metadata.json").write_text(json.dumps({"model_type": "unknown"}))
    (tmp_path / "relation_model.pkl").write_bytes(b"not a pickle")
    with pytest.raises(ValueError, match="metadata"):
        RelationPredictor(tmp_path)
