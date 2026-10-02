from pathlib import Path

from src.graph import build_reading_graph, reading_path
from src.parser import parse_document
from src.relations import extract_relations
from src.schemas import LegalEdge


def test_sample_chain():
    d = extract_relations(parse_document(Path("samples/sample_legal_text.txt").read_text()))
    g = build_reading_graph(d)
    path = reading_path(g, d.articles[-1].id)
    assert [d.by_id[s["id"]].number for s in path] == ["3", "4", "5", "6"]
    assert all(s["reasons"] for s in path[:-1])


def test_cycle_and_no_prerequisite():
    d = extract_relations(
        parse_document(
            "Điều 1. A\nĐiều 2 được áp dụng.\nĐiều 2. B\nTheo Điều 1.\nĐiều 3. C\nRiêng."
        )
    )
    # In prose, put the reference after text to avoid a heading-shaped line.
    d = extract_relations(
        parse_document("Điều 1. A\nTheo Điều 2.\nĐiều 2. B\nTheo Điều 1.\nĐiều 3. C\nRiêng.")
    )
    g = build_reading_graph(d)
    assert len(reading_path(g, d.articles[0].id)) == 2
    assert reading_path(g, d.articles[0].id)[0]["cycle"]
    assert len(reading_path(g, d.articles[2].id)) == 1


def test_ml_filter_depth_and_preservation():
    d = extract_relations(parse_document("Điều 1. A\nA.\nĐiều 2. B\nB.\nĐiều 3. C\nTheo Điều 1."))
    a, b, c = [n.id for n in d.articles]
    g = build_reading_graph(d, [LegalEdge(b, c, "PREDICTED_RELATED", 0.9, "ml")])
    assert len(reading_path(g, c)) == 3
    assert len(reading_path(g, c, include_ml=False)) == 2
    assert len(reading_path(g, c, min_confidence=0.95)) == 2
    assert len(reading_path(g, c, max_depth=0)) == 1
    assert g[a][c][0]["origin"] == "deterministic_reference_rule"


def test_ml_cycle_preserves_deterministic_order_inside_cluster():
    d = extract_relations(parse_document("Điều 1. A\nA.\nĐiều 2. B\nTheo Điều 1."))
    a, b = [n.id for n in d.articles]
    g = build_reading_graph(d, [LegalEdge(b, a, "PREDICTED_RELATED", 0.9, "ml")])
    assert [step["id"] for step in reading_path(g, a)] == [b, a]
    assert g[a][b][0]["source"] == a


def test_exact_3_10_20_chain_and_structural_direction():
    from src.graph import structural_graph

    d = extract_relations(
        parse_document(
            'Điều 3. Giải thích từ ngữ\n"Dữ liệu" là thông tin.\nĐiều 10. Nghĩa vụ\nDữ liệu phải chính xác.\nĐiều 20. Thủ tục\nTheo Điều 10.'
        )
    )
    g = build_reading_graph(d)
    assert [d.by_id[s["id"]].number for s in reading_path(g, d.articles[-1].id)] == [
        "3",
        "10",
        "20",
    ]
    structure = structural_graph(d)
    for e in d.edges:
        if e.edge_type == "CONTAINS":
            assert structure.has_edge(e.source, e.target)
            assert d.by_id[e.target].parent_id == e.source


def test_duplicate_edges_self_loop_and_unknown_target():
    import pytest

    d = extract_relations(parse_document("Điều 1. A\nA.\nĐiều 2. B\nTheo Điều 1. Theo Điều 1."))
    a, b = [n.id for n in d.articles]
    g = build_reading_graph(d)
    assert [s["id"] for s in reading_path(g, b)] == [a, b]
    g.add_edge(a, a, edge_type="REFERENCES", origin="deterministic", confidence=1.0, evidence={})
    assert reading_path(g, b)[0]["cycle"] == [a]
    with pytest.raises(ValueError):
        reading_path(g, "unknown")


def test_definition_before_explicit_before_ml():
    d = extract_relations(
        parse_document(
            'Điều 1. ML\nA.\nĐiều 2. Giải thích từ ngữ\n"Hệ thống" là A.\nĐiều 3. Explicit\nB.\nĐiều 4. Target\nHệ thống theo Điều 3.'
        )
    )
    a, b, c, t = [n.id for n in d.articles]
    g = build_reading_graph(d, [LegalEdge(a, t, "PREDICTED_RELATED", 0.9, "ml")])
    assert [s["id"] for s in reading_path(g, t)] == [b, c, a, t]
