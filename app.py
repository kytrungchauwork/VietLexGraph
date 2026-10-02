"""Streamlit UI for one structured Vietnamese legal text document."""

import html
import os
import pickle
from collections import Counter
from pathlib import Path

import networkx as nx
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.explanation import annotate_reading_graph, explain_reading_path
from src.graph import build_evidence_graph, build_reading_graph, filter_graph, reading_path
from src.model import RelationPredictor
from src.parser import parse_document
from src.relations import extract_relations

ROOT = Path(__file__).resolve().parent
READING = yaml.safe_load((ROOT / "config.yaml").read_text())["reading"]
DISCLAIMER = "VietLexGraph là công cụ thử nghiệm hỗ trợ điều hướng và phân tích cấu trúc văn bản pháp luật. Hệ thống không cung cấp tư vấn pháp lý, không xác định cách hiểu pháp luật có giá trị chính thức và không thay thế nguồn văn bản chính thức hoặc chuyên gia pháp lý."
COLORS = {
    "REFERENCES": "#2563eb",
    "DEFINES": "#059669",
    "USES_TERM": "#0d9488",
    "CONTAINS": "#94a3b8",
    "READ_BEFORE": "#d97706",
    "REFERENCE": "#2563eb",
    "DEFINITION": "#059669",
    "ML": "#d97706",
}


@st.cache_resource
def load_predictor(model_dir, metadata_mtime):
    return RelationPredictor(model_dir)


def label(doc, aid):
    article = doc.by_id[aid]
    return (
        f"{'Điều' if article.type == 'ARTICLE' else article.type} {article.number}. {article.title}"
    )


def graph_figure(graph):
    layout = nx.spring_layout(graph, seed=42)
    figure = go.Figure()
    edges = list(graph.edges(data=True))
    for kind, color in COLORS.items():
        xs, ys, midpoint_x, midpoint_y, custom, hover = [], [], [], [], [], []
        for index, (u, v, edge) in enumerate(edges):
            if edge.get("evidence_kind", edge["edge_type"]) != kind:
                continue
            x0, y0 = layout[u]
            x1, y1 = layout[v]
            xs.extend([x0, x1, None])
            ys.extend([y0, y1, None])
            midpoint_x.append((x0 + x1) / 2)
            midpoint_y.append((y0 + y1) / 2)
            custom.append(index)
            hover.append(f"{kind}<br>{edge['origin']}<br>confidence={edge['confidence']:.3f}")
            figure.add_annotation(
                x=x0 + 0.78 * (x1 - x0),
                y=y0 + 0.78 * (y1 - y0),
                ax=x0 + 0.65 * (x1 - x0),
                ay=y0 + 0.65 * (y1 - y0),
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowcolor=color,
                text="",
            )
        figure.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line={
                    "color": color,
                    "width": 1.7,
                    "dash": "dot" if kind == "ML" else "solid",
                },
                name=kind,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=midpoint_x,
                y=midpoint_y,
                mode="markers",
                marker={"size": 9, "color": color, "symbol": "diamond"},
                customdata=custom,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[layout[n][0] for n in graph],
            y=[layout[n][1] for n in graph],
            mode="markers+text",
            marker={"size": 22, "color": "#dbeafe", "line": {"width": 1, "color": "#1e40af"}},
            text=[f"{graph.nodes[n]['type']} {graph.nodes[n]['number']}" for n in graph],
            textposition="top center",
            hovertext=[html.escape(graph.nodes[n]["title"]) for n in graph],
            hovertemplate="%{text}: %{hovertext}<extra></extra>",
            showlegend=False,
        )
    )
    figure.update_layout(
        height=550,
        clickmode="event+select",
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False},
        legend={"orientation": "h"},
    )
    return figure


def main():
    st.set_page_config(page_title="VietLexGraph", layout="wide")
    st.title("VietLexGraph — Prerequisite-Aware Reading Guidance for Vietnamese Legal Documents")
    st.caption(
        "Turn a Vietnamese legal document into an evidence graph and discover what to read first."
    )
    st.info(DISCLAIMER)
    st.caption(
        "V1: plain text/.txt; không PDF/OCR. Hãy kiểm tra nguồn pháp luật chính thức. Gợi ý ML không đồng nghĩa với phụ thuộc pháp lý."
    )
    model = None
    model_dir = Path(os.environ.get("VIETLEXGRAPH_MODEL_DIR", ROOT / "models"))
    metadata_path = model_dir / "metadata.json"
    try:
        model = load_predictor(str(model_dir), metadata_path.stat().st_mtime_ns)
    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        ImportError,
        OSError,
        EOFError,
        pickle.UnpicklingError,
    ):
        st.info("ML model unavailable. Running deterministic-only mode.")
    threshold = model.metadata["threshold"] if model else 1.0
    include_ml = st.sidebar.checkbox("Include AI suggestions", value=False, disabled=model is None)
    confidence = st.sidebar.slider(
        "Minimum confidence",
        min_value=0.0,
        max_value=1.0,
        value=float(threshold),
        step=0.01,
        disabled=model is None,
    )
    effective_threshold = max(threshold, confidence)
    if model:
        st.sidebar.caption(
            f"Validation threshold: {threshold:.4f}. Effective minimum: {effective_threshold:.4f}."
        )
        if not model.metadata["thresholds"][model.metadata["model_type"]]["constraint_met"]:
            st.sidebar.warning(
                "Mô hình chưa đạt precision 0.85 trên validation. AI suggestions là thử nghiệm và mặc định tắt."
            )
    depth = st.sidebar.slider(
        "Maximum reading depth", 0, max(10, READING["max_depth"]), READING["max_depth"]
    )
    budget = st.sidebar.slider(
        "Maximum prerequisites",
        0,
        max(20, READING["max_prerequisites"]),
        READING["max_prerequisites"],
    )
    st.sidebar.caption("Điểm ML và trọng số quy tắc chưa được hiệu chỉnh bằng human gold.")
    input_tab, structure_tab, graph_tab, guide_tab = st.tabs(
        ["Input", "Structure", "Graph", "Reading Guide"]
    )
    with input_tab:
        st.caption(
            "Chỉ hỗ trợ plain text hoặc .txt UTF-8 có marker Điều và cấu trúc pháp lý. Sample bên dưới là văn bản giả lập."
        )
        uploaded = st.file_uploader("Upload .txt", type=["txt"])
        text = st.text_area(
            "Paste legal text",
            value=(ROOT / "samples/sample_legal_text.txt").read_text(encoding="utf-8"),
            height=320,
        )
        if st.button("Analyze Document", type="primary"):
            try:
                value = uploaded.getvalue().decode("utf-8-sig") if uploaded is not None else text
                if not value.strip():
                    raise ValueError("Văn bản trống. Hãy nhập hoặc tải lên văn bản.")
                doc = extract_relations(parse_document(value))
                st.session_state["document"] = doc
                st.session_state.pop("predictions", None)
                st.session_state.pop("prediction_key", None)
                st.success(f"Đã phân tích {len(doc.articles)} Điều.")
            except (ValueError, UnicodeDecodeError) as exc:
                st.session_state.pop("document", None)
                st.error(
                    "Không đọc được UTF-8. Hãy tải tệp .txt UTF-8."
                    if isinstance(exc, UnicodeDecodeError)
                    else str(exc)
                )
    doc = st.session_state.get("document")
    if doc is None:
        return
    predictions = []
    if include_ml and model:
        try:
            prediction_key = (doc.text, metadata_path.stat().st_mtime_ns)
            if (
                "predictions" not in st.session_state
                or st.session_state.get("prediction_key") != prediction_key
            ):
                with st.spinner("Đang mã hóa các Điều và dự đoán ứng viên ngữ cảnh…"):
                    st.session_state["predictions"] = model.predict(doc)
                    st.session_state["prediction_key"] = prediction_key
            predictions = st.session_state["predictions"]
        except (OSError, RuntimeError, ValueError) as exc:
            st.warning(f"ML model unavailable. Running deterministic-only mode. {exc}")
    graph = filter_graph(
        build_reading_graph(
            doc,
            predictions,
            reference_confidence=READING["reference_weight"],
            definition_confidence=READING["definition_weight"],
        ),
        include_ml,
        effective_threshold,
    )
    with structure_tab:
        st.subheader("Document Structure")
        counts = Counter(n.type for n in doc.nodes)
        st.write(
            {
                kind: counts[kind]
                for kind in [
                    "PART",
                    "CHAPTER",
                    "SECTION",
                    "SUBSECTION",
                    "ARTICLE",
                    "CLAUSE",
                    "POINT",
                    "TERM",
                ]
            }
        )
        st.write(
            {
                "Explicit references": sum(e.edge_type == "REFERENCES" for e in doc.edges),
                "Defined terms": counts["TERM"],
                "ML contextual relations": sum(
                    d["origin"] == "ml" for _, _, d in graph.edges(data=True)
                ),
            }
        )
        with st.expander("Structure tree", expanded=True):
            tree = []
            for n in doc.nodes:
                if n.type in ("DOCUMENT", "TERM"):
                    continue
                parent, indent = n.parent_id, 0
                while parent and parent != doc.id:
                    indent += 1
                    parent = doc.by_id[parent].parent_id
                tree.append("  " * indent + f"{n.type} {n.number} — {n.title}")
            st.code("\n".join(tree), language=None)
        aid = st.selectbox(
            "Article text", [a.id for a in doc.articles], format_func=lambda i: label(doc, i)
        )
        st.text(doc.by_id[aid].text)
        parent = doc.by_id[aid]
        while parent.parent_id and parent.type != "CHAPTER":
            parent = doc.by_id[parent.parent_id]
        st.caption(
            f"Parent chapter: {parent.number} {parent.title}"
            if parent.type == "CHAPTER"
            else "Parent chapter: none"
        )
        details = [
            {
                "direction": "outgoing" if e.source == aid else "incoming",
                "type": e.edge_type,
                "source": e.source,
                "target": e.target,
                "origin": e.origin,
                "confidence": e.confidence,
                "evidence": e.evidence,
            }
            for e in doc.edges
            if e.edge_type != "CONTAINS" and aid in (e.source, e.target)
        ]
        details.extend(
            {"direction": "contextual candidate", **edge}
            for u, v, edge in graph.edges(data=True)
            if edge["origin"] == "ml" and aid in (u, v)
        )
        st.json(details, expanded=False)
        if doc.unresolved_references:
            with st.expander(
                f"Unresolved / external references ({len(doc.unresolved_references)})"
            ):
                st.json(doc.unresolved_references)
    with guide_tab:
        target = st.selectbox(
            "Chọn Điều cần hiểu",
            [a.id for a in doc.articles],
            index=len(doc.articles) - 1,
            format_func=lambda i: label(doc, i),
        )
    evidence_graph = build_evidence_graph(doc)
    path = reading_path(graph, target, include_ml, effective_threshold, depth, budget)
    explained_steps = explain_reading_path(path, graph, evidence_graph, doc.by_id)
    graph = annotate_reading_graph(graph, explained_steps)
    with graph_tab:
        st.caption(
            "Evidence: viện dẫn đi từ Điều dẫn → Điều được dẫn. Reading: ứng viên đọc trước → Điều đích; đây là khuyến nghị, không phải sự thật pháp lý."
        )
        view = st.radio("Graph view", ["Reading Guidance", "Evidence Relations"], horizontal=True)
        displayed = graph if view == "Reading Guidance" else evidence_graph
        kinds = sorted({d["edge_type"] for _, _, d in displayed.edges(data=True)})
        selected = st.multiselect("Edge types", kinds, default=kinds)
        show_ml_edges = st.checkbox(
            "Show ML-only READ_BEFORE edges", value=True, disabled=view != "Reading Guidance"
        )
        visible = filter_graph(displayed, include_ml=show_ml_edges, edge_types=selected)
        event = st.plotly_chart(
            graph_figure(visible),
            width="stretch",
            on_select="rerun",
            selection_mode="points",
            key="legal_graph",
        )
        edges = list(visible.edges(data=True))
        edge_index = 0
        if event and event.selection.points:
            custom = event.selection.points[0].get("customdata")
            if isinstance(custom, int) and 0 <= custom < len(edges):
                edge_index = custom
        if edges:
            selected_edge = st.selectbox(
                "Edge details",
                range(len(edges)),
                index=edge_index,
                format_func=lambda i: (
                    f"{label(doc, edges[i][0])} → {label(doc, edges[i][1])} [{edges[i][2]['edge_type']}]"
                ),
            )
            st.json(edges[selected_edge][2])
        else:
            st.caption("Không có cạnh phù hợp với bộ lọc.")
    with guide_tab:
        st.subheader("Recommended Reading Path")
        st.caption(
            "Đường đọc theo depth và AI controls; bộ lọc Edge types chỉ áp dụng cho hình graph."
        )
        for i, step in enumerate(explained_steps, 1):
            with st.container(border=True):
                st.markdown(f"**{i}. {label(doc, step['id'])}**")
                if step["target"]:
                    st.caption("Target provision")
                    if step["omitted"]:
                        st.warning(
                            f"Giới hạn đường đọc đã bỏ qua {len(step['omitted'])} ứng viên; bằng chứng gốc vẫn được giữ."
                        )
                        with st.expander("Omitted candidates"):
                            st.json(step["omitted"])
                if step["cycle"]:
                    st.warning(
                        "Nhóm có chu trình: "
                        + ", ".join(label(doc, n) for n in step["cycle"])
                        + ". Không có thứ tự tuyến tính thỏa mãn mọi cạnh."
                    )
                st.caption("Why read this first?")
                st.write(step["explanation"])
                relations = list(dict.fromkeys(e["relation"] for e in step["evidence"]))
                score = f"{step['confidence']:.3f}" if step["confidence"] is not None else "N/A"
                st.caption(f"Evidence: {' + '.join(relations)} · Confidence: {score}")
                st.caption(f"Source: {step['origin']} · {step['confidence_semantics']}")
                if step.get("model_origin"):
                    st.caption(f"Model origin: {step['model_origin']}")
                with st.expander("Why read this first? — Evidence", expanded=False):
                    st.json(
                        {
                            "reason_type": step["reason_type"],
                            "evidence": step["evidence"],
                            "all_explanations": step["explanations"],
                        }
                    )
                if len(step["cycle"]) > 1:
                    st.caption(
                        "Trong nhóm chu trình: ưu tiên ràng buộc deterministic; nếu vẫn có chu trình, giữ thứ tự văn bản."
                    )


if __name__ == "__main__":
    main()
