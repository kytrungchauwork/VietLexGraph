import pytest

from src.parser import parse_document


def test_hierarchy_spacing_and_text():
    d = parse_document(
        "PHẦN THỨ NHẤT\nCHUNG\nCHƯƠNG I\nQUY ĐỊNH\nMỤC 1. Chung\nTIỂU MỤC 1. Nhỏ\n Điều  1 . Tên\n1. Nội dung\nđ) Điểm đ\nTiếp tục\nĐiều 2. Hai\nBody"
    )
    assert [a.number for a in d.articles] == ["1", "2"]
    assert d.articles[0].title == "Tên"
    assert "Tiếp tục" in d.articles[0].text
    point = next(n for n in d.nodes if n.type == "POINT")
    assert point.number == "đ"
    assert d.by_id[point.parent_id].type == "CLAUSE"
    assert d.by_id[d.articles[0].parent_id].type == "SUBSECTION"
    assert "Điều 2" not in d.articles[0].text


def test_no_chapter_multiline_title():
    d = parse_document("Điều 1.\nPhạm vi điều chỉnh\nNội dung.\nĐiều 2. Khác\nKhác.")
    assert d.articles[0].parent_id == d.id
    assert d.articles[0].title == "Phạm vi điều chỉnh"
    assert d.articles[0].body == "Nội dung."


def test_invalid():
    with pytest.raises(ValueError, match="Không phát hiện"):
        parse_document("Văn bản không cấu trúc")


def test_prose_reference_at_start_is_not_heading():
    d = parse_document(
        "Điều 1. Tên\nĐiều 2 được áp dụng trong trường hợp này.\n1.Nội dung\nđ)Điểm\nĐiều 2. Khác\nNội dung."
    )
    assert len(d.articles) == 2
    assert "Điều 2 được áp dụng" in d.articles[0].body
    assert len([n for n in d.nodes if n.type == "POINT"]) == 1


def test_dotless_heading_and_gaps_unicode_and_blank_title():
    import unicodedata

    text = "CHƯƠNG I\nChung\nĐiều 3 Phạm vi\nNội dung A.\nĐiều 10\n\nTên xuống dòng\n1.Nội dung B.\nđ)Điểm B."
    doc = parse_document(unicodedata.normalize("NFD", text))
    assert [a.number for a in doc.articles] == ["3", "10"]
    assert doc.articles[0].title == "Phạm vi"
    assert doc.articles[1].title == "Tên xuống dòng"
    assert "Tên xuống dòng" not in doc.articles[1].body
    assert next(n for n in doc.nodes if n.type == "CLAUSE").body.startswith("Nội dung B.")
