import numpy as np
import pandas as pd

from src.model import pair_features, ranking_metrics, tune_threshold


def test_threshold_precision_constraint_and_fallback():
    result = tune_threshold([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6])
    assert result["constraint_met"] and result["threshold"] == 0.9
    fallback = tune_threshold([0, 1], [0.9, 0.8])
    assert not fallback["constraint_met"]


def test_ranking_uses_all_relevant_targets():
    pairs = pd.DataFrame(
        {
            "document_id": ["d"] * 4,
            "source": ["a"] * 4,
            "target": ["b", "c", "d", "e"],
            "label": [0, 1, 0, 1],
        }
    )
    result = ranking_metrics(pairs, [0.9, 0.8, 0.7, 0.6])
    assert result["mrr"] == 0.5
    assert result["recall_at_3"] == 0.5
    assert result["recall_at_5"] == 1


def test_feature_direction_and_missing_chapter():
    a = dict(text="A", title="A", number="5", order=5, terms=[], chapter=None, section=None)
    b = dict(text="B", title="B", number="2", order=2, terms=[], chapter=None, section=None)
    f = pair_features(a, b, np.ones(3), np.zeros(3))
    assert f.shape == (20,)
    assert list(f[-7:-3]) == [0, 0, 3, 1]


def test_chunking_retains_tail_and_respects_budget():
    from src.model import E5Encoder

    class Tokenizer:
        cls_token_id = 1
        sep_token_id = 2

        def encode(self, text, **kwargs):
            return [ord(c) for c in text]

        def build_inputs_with_special_tokens(self, ids):
            return [1] + ids + [2]

    encoder = object.__new__(E5Encoder)
    encoder.tokenizer = Tokenizer()
    encoder.chunk_tokens = 32
    article = {"title": "Title", "text": "a" * 130 + "TAIL"}
    chunks = encoder.chunk_ids(article)
    prefix_length = len(encoder.tokenizer.encode("query: Title")[:10])
    recovered = "".join(chr(i) for chunk in chunks for i in chunk[1 + prefix_length : -1])
    assert recovered == article["text"]
    assert all(len(chunk) <= 34 for chunk in chunks)


def test_citations_masked_in_body_and_title():
    from src.model import clean_model_text, model_title

    a = {
        "title": "Thực hiện Điều 15",
        "text": "Điều 20. Thực hiện Điều 15\nTheo điểm a khoản 2 Điều 15.",
    }
    assert "15" not in clean_model_text(a)
    assert "15" not in model_title(a)
    assert "15" in clean_model_text(a, mask_citations=False)
