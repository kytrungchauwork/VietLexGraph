from copy import deepcopy

import networkx as nx
import pytest

from src.explanation import annotate_reading_graph, explain_reading_path
from src.graph import build_evidence_graph, build_reading_graph, reading_path
from src.parser import parse_document
from src.relations import extract_relations


def document(text):
    doc = extract_relations(parse_document(text))
    graph = build_reading_graph(doc)
    return doc, graph, build_evidence_graph(doc)


def test_definition_preserves_actual_term_evidence_and_input_graphs():
    doc, graph, evidence = document(
        'Điều 3. Giải thích từ ngữ\n"Phí" là khoản thu.\nĐiều 10. Quy định\nPhí theo Điều 3.'
    )
    a, b = [n.id for n in doc.articles]
    path = reading_path(graph, b)
    before_graph, before_evidence, before_path = deepcopy(graph), deepcopy(evidence), deepcopy(path)
    steps = explain_reading_path(path, graph, evidence, doc.by_id)
    assert steps[0]["reason_type"] == "definition_prerequisite"
    assert "phí" in steps[0]["explanation"].casefold()
    assert steps[0]["article"] == "Điều 3" and steps[0]["position"] == 1
    assert steps[0]["confidence"] == 0.95
    raw = steps[0]["evidence"]
    assert [e["relation"] for e in raw[:2]] == ["DEFINES", "USES_TERM"]
    assert raw[0]["source_id"] == a and raw[1]["source_id"] == b
    assert raw[0]["target_id"] == raw[1]["target_id"]  # TERM, not fake article→article DEFINES
    annotated = annotate_reading_graph(graph, steps)
    assert annotated.number_of_edges() == graph.number_of_edges()
    for u, v, key, edge in graph.edges(keys=True, data=True):
        enriched = dict(annotated[u][v][key])
        assert enriched.pop("explanation")["text"]
        assert enriched == edge
    assert nx.utils.graphs_equal(graph, before_graph)
    assert nx.utils.graphs_equal(evidence, before_evidence)
    assert path == before_path
    assert steps[-1]["reason_type"] == "target_provision"
    assert steps[-1]["confidence"] is None


def test_reference_direction_and_no_explanation_for_unselected_reference():
    doc, graph, evidence = document("Điều 5. Quy tắc\nNội dung.\nĐiều 20. Áp dụng\nTheo Điều 5.")
    a, b = [n.id for n in doc.articles]
    step = explain_reading_path([a, b], graph, evidence, doc.by_id)[0]
    assert step["reason_type"] == "explicit_reference"
    assert step["evidence"][0]["source_id"] == b
    assert step["evidence"][0]["target_id"] == a
    assert "Điều 20 viện dẫn" in step["explanation"]
    only_target = explain_reading_path([b], graph, evidence, doc.by_id)
    assert len(only_target) == 1 and only_target[0]["reason_type"] == "target_provision"


def custom_graph(origin="rule", role="GENERAL_RULE", target_role="SPECIFIC_RULE"):
    graph = nx.MultiDiGraph()
    graph.add_node("a", type="ARTICLE", number="1", role=role)
    graph.add_node("b", type="ARTICLE", number="2", role=target_role)
    graph.add_edge(
        "a",
        "b",
        edge_type="READ_BEFORE",
        origin=origin,
        confidence=0.72,
        evidence={"model_type": "test-model"},
    )
    return graph


def test_ml_is_predicted_not_role_based_fact():
    graph = custom_graph("ml", "PROCEDURE")
    step = explain_reading_path(["a", "b"], graph, nx.MultiDiGraph(), {})[0]
    assert step["reason_type"] == "predicted_support"
    assert step["model_origin"] == "test-model" and step["confidence"] == 0.72
    assert "được hệ thống dự đoán" in step["explanation"]
    assert "bắt buộc" not in step["explanation"]
    assert step["evidence"][0]["provenance"]["origin"] == "ml"


@pytest.mark.parametrize(
    ("role", "target_role", "expected"),
    [
        ("GENERAL_RULE", "SPECIFIC_RULE", "general_to_specific"),
        ("PROCEDURE", "UNKNOWN", "procedure_prerequisite"),
        ("UNKNOWN", "UNKNOWN", "structural_support"),
    ],
)
def test_metadata_templates_require_existing_edge(role, target_role, expected):
    graph = custom_graph(role=role, target_role=target_role)
    assert (
        explain_reading_path(["a", "b"], graph, nx.MultiDiGraph(), {})[0]["reason_type"] == expected
    )
    graph.remove_edges_from(list(graph.edges(keys=True)))
    assert (
        explain_reading_path(["a", "b"], graph, nx.MultiDiGraph(), {})[0]["reason_type"]
        == "insufficient_evidence"
    )
    assert graph.number_of_edges() == 0


def test_strong_evidence_wins_over_parallel_ml():
    doc, graph, evidence = document("Điều 5. Quy tắc\nNội dung.\nĐiều 20. Áp dụng\nTheo Điều 5.")
    a, b = [n.id for n in doc.articles]
    graph.add_edge(
        a,
        b,
        edge_type="READ_BEFORE",
        origin="ml",
        confidence=0.999,
        evidence={"model_type": "test"},
    )
    step = explain_reading_path([a, b], graph, evidence, doc.by_id)[0]
    assert step["reason_type"] == "explicit_reference" and step["confidence"] == 0.85
    assert [r["reason_type"] for r in step["explanations"]] == [
        "explicit_reference",
        "predicted_support",
    ]


def test_cycle_does_not_justify_reading_before_an_earlier_step():
    graph = custom_graph()
    path = [{"id": "b", "cycle": ["a", "b"]}, {"id": "a", "cycle": ["a", "b"]}]
    steps = explain_reading_path(path, graph, nx.MultiDiGraph(), {})
    assert steps[0]["reason_type"] == "cycle_context"
    assert steps[0]["confidence"] is None
    assert steps[1]["reason_type"] == "target_provision"
    assert not any(s["explanations"] for s in steps)


def test_empty_and_invalid_paths():
    graph = custom_graph()
    assert explain_reading_path([], graph, nx.MultiDiGraph(), {}) == []
    with pytest.raises(ValueError):
        explain_reading_path(["a", "a"], graph, nx.MultiDiGraph(), {})
