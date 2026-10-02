from scripts.prepare_data import clean_corpus_markdown, split_documents


def test_split_document_disjoint_and_reproducible():
    split = split_documents([str(i) for i in range(100)])
    assert list(map(len, split.values())) == [70, 15, 15]
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["train"]) & set(split["val"])
    assert split == split_documents([str(i) for i in reversed(range(100))])


def test_corpus_cleanup():
    assert clean_corpus_markdown("## **Điều 1. Tên**") == "Điều 1. Tên"


def test_hard_negatives_exclude_all_references_and_self():
    from scripts.prepare_data import make_pairs
    from src.parser import parse_document
    from src.relations import extract_relations

    text = "\n".join(f"Điều {i}. Nghĩa vụ\nNội dung nhà cung cấp." for i in range(1, 9))
    text += "\nĐiều 9. Thực hiện\nTheo các Điều 1, 2 và 3."
    doc = extract_relations(parse_document(text))
    pairs = make_pairs(doc)
    positives = {(p["source"], p["target"]) for p in pairs if p["label"]}
    negatives = {(p["source"], p["target"]) for p in pairs if not p["label"]}
    assert not positives & negatives
    assert all(s != t for s, t in negatives)
    assert len(negatives) == 5
