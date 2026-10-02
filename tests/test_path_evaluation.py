import numpy as np
import pytest

from scripts.gold import validate
from src.graph import build_evidence_graph, build_reading_graph, reading_path
from src.parser import parse_document
from src.path_evaluation import BASELINES, baseline_path, path_metrics
from src.relations import extract_relations
from src.schemas import LegalEdge


def test_formal_reference_is_evidence_not_prerequisite():
    doc = extract_relations(
        parse_document("Điều 1. Quy tắc\nNội dung.\nĐiều 2. Thi hành\nCăn cứ Điều 1.")
    )
    a, b = [n.id for n in doc.articles]
    evidence = build_evidence_graph(doc)
    graph = build_reading_graph(doc)
    assert evidence.has_edge(b, a)
    assert not graph.has_edge(a, b)
    assert any(e.edge_type == "REFERENCES" for e in doc.edges)  # weak signal stays raw
    assert not graph.graph["decisions"][0]["accepted"]
    assert len(baseline_path(doc, b, "CitationOnly")) == 2
    assert len(baseline_path(doc, b, "HybridNoML")) == 1


def test_pruning_preserves_connected_route_and_raw_evidence():
    doc = extract_relations(
        parse_document("Điều 1. A\nA.\nĐiều 2. B\nTheo Điều 1.\nĐiều 3. C\nTheo Điều 2.")
    )
    a, b, c = [n.id for n in doc.articles]
    graph = build_reading_graph(doc)
    path = reading_path(graph, c, max_prerequisites=1)
    assert [s["id"] for s in path] == [b, c]
    assert path[-1]["omitted"][0]["id"] == a
    assert graph.has_edge(a, b)
    assert len(reading_path(graph, c, max_depth=1)) == 2
    assert len(reading_path(graph, c, max_prerequisites=0)) == 1
    assert {d["edge_type"] for _, _, d in graph.edges(data=True)} == {"READ_BEFORE"}
    assert "READ_BEFORE" not in {
        d["edge_type"] for _, _, d in build_evidence_graph(doc).edges(data=True)
    }


def test_all_six_baselines_and_strict_metrics():
    doc = extract_relations(
        parse_document("Điều 1. A\nA.\nĐiều 2. B\nTheo Điều 1.\nĐiều 3. C\nTheo Điều 2.")
    )
    a, b, c = [n.id for n in doc.articles]
    vectors = {a: np.array([1.0, 0.0]), b: np.array([0.0, 1.0]), c: np.array([0.0, 1.0])}
    ml = [
        LegalEdge(
            a,
            c,
            "READ_BEFORE",
            0.9,
            "ml",
            {"model": "synthetic fixture"},
            "Synthetic recommendation",
        )
    ]
    for name in BASELINES:
        path = baseline_path(doc, c, name, embeddings=vectors, predictions=ml)
        assert path[-1]["id"] == c
        assert len({s["id"] for s in path}) == len(path)
    with pytest.raises(ValueError, match="requires"):
        baseline_path(doc, c, "FullSystem")
    result = path_metrics([a, b, c], c, {a: 0, b: 1}, [(b, a)])
    assert result["path_precision"] == 0.5
    assert result["path_recall"] == 1
    assert result["extra_reading_cost"] == 1
    assert result["pairwise_order_accuracy"] == 0
    assert path_metrics([c], c, {a: 0, b: 0})["path_f1"] == 1
    assert path_metrics([b, c], c, {a: 0, b: 1})["pairwise_order_accuracy"] is None
    with pytest.raises(ValueError, match="Unjudged"):
        path_metrics([a, c], c, {b: 1})
    with pytest.raises(ValueError, match="adjudicated"):
        path_metrics([c], c, {a: "UNCERTAIN"})


def test_gold_rejects_missing_human_labels():
    with pytest.raises(ValueError, match="Adjudicate"):
        validate(
            [
                {
                    "document_id": "d",
                    "target_article": "a",
                    "candidate_article": "b",
                    "read_before": "",
                }
            ]
        )


def test_gold_freeze_hashes_and_document_disjointness(tmp_path, monkeypatch):
    import csv
    import hashlib
    import json
    import sys

    import scripts.gold as gold
    from src.reproducibility import verify_hashes

    data = tmp_path / "data"
    data.mkdir()
    docs, rows = [], []
    for doc_id in ["dev-doc", "test-doc"]:
        articles = [
            {"id": f"{doc_id}:a", "text": "Target"},
            {"id": f"{doc_id}:b", "text": "Candidate"},
        ]
        digest = hashlib.sha256(doc_id.encode()).hexdigest()
        docs.append({"id": doc_id, "sha256": digest, "articles": articles})
        rows.append(
            dict(
                zip(
                    gold.FIELDS,
                    [
                        doc_id,
                        "a" * 40,
                        digest,
                        articles[0]["id"],
                        articles[1]["id"],
                        gold.digest("Target"),
                        gold.digest("Candidate"),
                        "0",
                        "1",
                        "1",
                        "SYNTHETIC TEST ONLY",
                        "synthetic-fixture-not-human",
                        "test-v1",
                    ],
                )
            )
        )
    (data / "processed_documents.jsonl").write_text("\n".join(json.dumps(d) for d in docs))
    (data / "split_manifest.json").write_text(
        json.dumps({"revision": "a" * 40, "splits": {"train": []}})
    )
    annotation = tmp_path / "synthetic.csv"
    with annotation.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=gold.FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "frozen"
    monkeypatch.setattr(gold, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gold.py",
            "freeze",
            "--annotations",
            str(annotation),
            "--test-documents",
            "test-doc",
            "--output",
            str(output),
        ],
    )
    gold.main()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"] == {"gold-test": 1, "gold-dev": 1}
    verify_hashes(output, manifest["files"])
    with pytest.raises(ValueError, match="already exists"):
        gold.main()
    (output / "gold-test.csv").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        verify_hashes(output, manifest["files"])
