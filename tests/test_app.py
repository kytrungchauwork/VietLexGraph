from pathlib import Path

from streamlit.testing.v1 import AppTest

import src.model


def test_app_sample_and_invalid_input():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert not app.exception
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert "Đã phân tích 6 Điều" in app.success[0].value
    steps = [m.value for m in app.markdown if m.value.startswith("**")]
    assert [step.split("Điều ")[1].split(".")[0] for step in steps] == ["3", "4", "5", "6"]
    assert sum(c.value == "Why read this first?" for c in app.caption) == 4
    assert any("DEFINES + USES_TERM" in c.value for c in app.caption)
    assert any("Confidence: 0.950" in c.value for c in app.caption)
    assert any("TARGET_SELECTION" in c.value and "N/A" in c.value for c in app.caption)
    app.text_area[0].set_value("Không có cấu trúc").run()
    app.button[0].click().run()
    assert app.error and "Không phát hiện" in app.error[0].value
    assert not app.exception


def test_deterministic_only_without_checkpoint(monkeypatch):
    import streamlit as st

    st.cache_resource.clear()

    class Missing:
        def __init__(self, *args, **kwargs):
            raise FileNotFoundError("No checkpoint")

    monkeypatch.setattr(src.model, "RelationPredictor", Missing)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    assert any("Running deterministic-only mode" in i.value for i in app.info)
    app.button[0].click().run(timeout=30)
    assert not app.exception
    assert next(c for c in app.checkbox if c.label == "Include AI suggestions").disabled
    st.cache_resource.clear()


def test_repeat_analyze_with_ai_enabled(monkeypatch, tmp_path):
    (tmp_path / "metadata.json").write_text("{}")
    monkeypatch.setenv("VIETLEXGRAPH_MODEL_DIR", str(tmp_path))
    import streamlit as st

    st.cache_resource.clear()

    class Fake:
        metadata = {
            "threshold": 0.8,
            "model_type": "logistic_regression",
            "thresholds": {"logistic_regression": {"constraint_met": True}},
        }

        def __init__(self, *args, **kwargs):
            pass

        def predict(self, doc):
            return []

    monkeypatch.setattr(src.model, "RelationPredictor", Fake)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    app.checkbox[0].check().run()
    app.button[0].click().run()
    app.button[0].click().run()
    assert not app.exception


def test_evidence_view_separates_raw_relations():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=30)
    app.button[0].click().run()
    app.radio[0].set_value("Evidence Relations").run()
    assert not app.exception
    assert "REFERENCES" in app.multiselect[0].options
    assert "READ_BEFORE" not in app.multiselect[0].options
