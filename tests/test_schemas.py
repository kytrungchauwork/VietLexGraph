import pytest

from src.schemas import LegalEdge


def test_provenance():
    with pytest.raises(ValueError):
        LegalEdge("a", "b", "PREDICTED_RELATED")
    assert LegalEdge("a", "b", "REFERENCES").confidence == 1
