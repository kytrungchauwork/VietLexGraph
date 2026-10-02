from src.parser import parse_document
from src.relations import extract_relations


def test_reference_patterns_and_unresolved():
    text = "\n".join(f"Điều {i}. Tên\nNội dung." for i in range(1, 8))
    text += "\nĐiều 20. Khác\nTheo Điều 5; khoản 2 Điều 5; điểm a khoản 2 Điều 5; các Điều 5, 6 và 7; từ Điều 1 đến Điều 3; Điều 99; Điều 1 của Luật Khác."
    d = extract_relations(parse_document(text))
    refs = [e for e in d.edges if e.edge_type == "REFERENCES"]
    assert {d.by_id[e.target].number for e in refs} == {"1", "2", "3", "5", "6", "7"}
    assert any(e.evidence["point"] == "a" for e in refs)
    assert len(d.unresolved_references) == 2


def test_terms_boundaries_longest_and_idempotence():
    d = extract_relations(
        parse_document(
            'Điều 3. Giải thích từ ngữ\n1. "Nhà cung cấp" là tổ chức.\n2. "cung" là đơn vị.\nĐiều 10. Nghĩa vụ\nNHÀ CUNG CẤP làm việc.\nĐiều 11. Khác\nCungcấp.'
        )
    )
    uses = [e for e in d.edges if e.edge_type == "USES_TERM"]
    assert len(uses) == 1
    assert uses[0].evidence["term"] == "nhà cung cấp"
    count = len(d.edges)
    extract_relations(d)
    assert len(d.edges) == count


def test_multiline_corpus_definitions():
    d = extract_relations(
        parse_document(
            "Điều 1. Giải thích từ ngữ\n1.\nTrưng cầu ý dân\nlà việc tổ chức bỏ phiếu.\n2.\nCử tri\nlà người bỏ phiếu.\nĐiều 2. Thực hiện\nCử tri tham gia trưng cầu ý dân."
        )
    )
    assert {e.evidence["term"] for e in d.edges if e.edge_type == "DEFINES"} == {
        "trưng cầu ý dân",
        "cử tri",
    }
    assert len([e for e in d.edges if e.edge_type == "USES_TERM"]) == 2


def test_grouped_external_reference_is_not_local():
    d = extract_relations(
        parse_document(
            "Điều 1. Một\nMột.\nĐiều 2. Hai\nHai.\nĐiều 3. Ba\nTheo Điều 1 và Điều 2 của Luật Khác."
        )
    )
    assert not [e for e in d.edges if e.edge_type == "REFERENCES"]
    assert len(d.unresolved_references) == 2


def test_ambiguous_definition_does_not_choose_first_owner():
    d = extract_relations(
        parse_document(
            'Điều 1. Giải thích từ ngữ\n"Hệ thống" là A.\nĐiều 2. Giải thích từ ngữ\n"Hệ thống" là B.\nĐiều 3. Dùng\nHệ thống hoạt động.'
        )
    )
    assert len([e for e in d.edges if e.edge_type == "DEFINES"]) == 2
    assert not [e for e in d.edges if e.edge_type == "USES_TERM"]


def test_multiple_unnumbered_quoted_definitions():
    d = extract_relations(
        parse_document('Điều 1. Giải thích từ ngữ\n"A" là nội dung A.\n"B" là nội dung B.')
    )
    assert len([e for e in d.edges if e.edge_type == "DEFINES"]) == 2
