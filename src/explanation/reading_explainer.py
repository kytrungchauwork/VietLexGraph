"""Explain existing selected READ_BEFORE edges without changing reading decisions.

Confidence is copied from the supporting reading edge, not a new probability of
explanation correctness. Role templates describe metadata, never create edges.
Raw evidence keeps its actual endpoints, including TERM nodes and citation direction.
"""

from copy import deepcopy
from dataclasses import asdict, is_dataclass

PRIORITY = {
    "definition_prerequisite": 0,
    "explicit_reference": 1,
    "procedure_prerequisite": 2,
    "general_to_specific": 3,
    "structural_support": 3,
    "predicted_support": 4,
}


def _metadata(node, graph, metadata):
    value = metadata.get(node, graph.nodes[node] if node in graph else {})
    return asdict(value) if is_dataclass(value) else value


def _label(node, graph, metadata):
    data = _metadata(node, graph, metadata)
    if data.get("type") == "TERM":
        return data.get("title") or data.get("text") or node
    return f"Điều {data['number']}" if data.get("number") else node


def _raw_evidence(graph, source, target, relation, metadata):
    if not graph.has_edge(source, target):
        return []
    return [
        {
            "relation": relation,
            "source": _label(source, graph, metadata),
            "target": _label(target, graph, metadata),
            "source_id": source,
            "target_id": target,
            "provenance": deepcopy(edge),
        }
        for edge in graph[source][target].values()
        if edge.get("edge_type") == relation
    ]


def _explain_edge(source, target, key, edge, graph, evidence_graph, metadata):
    a, b = _label(source, graph, metadata), _label(target, graph, metadata)
    source_meta = _metadata(source, graph, metadata)
    target_meta = _metadata(target, graph, metadata)
    support = [
        {
            "relation": "READ_BEFORE",
            "source": a,
            "target": b,
            "source_id": source,
            "target_id": target,
            "provenance": {k: deepcopy(v) for k, v in edge.items() if k != "explanation"},
        }
    ]
    result = {
        "source_id": source,
        "target_id": target,
        "edge_key": key,
        "confidence": edge.get("confidence"),
        "origin": edge.get("origin"),
        "confidence_semantics": "existing uncalibrated reading-edge weight; not explanation accuracy",
        "evidence": support,
    }
    if edge.get("origin") == "ml":
        result.update(
            reason_type="predicted_support",
            explanation=f"{a} được hệ thống dự đoán là có khả năng hỗ trợ hiểu {b}.",
            model_origin=edge.get("evidence", {}).get("model_type")
            or edge.get("evidence", {}).get("model")
            or "ml (model identity unavailable)",
        )
        return result
    definitions = []
    terms = []
    if source in evidence_graph and target in evidence_graph:
        for term in evidence_graph.successors(source):
            defined = _raw_evidence(evidence_graph, source, term, "DEFINES", metadata)
            used = _raw_evidence(evidence_graph, target, term, "USES_TERM", metadata)
            if defined and used:
                terms.append(_label(term, evidence_graph, metadata))
                definitions.extend(defined + used)
    if (
        definitions
        and edge.get("origin") != "deterministic_reference_rule"
        and edge.get("evidence_kind") != "REFERENCE"
    ):
        result.update(
            reason_type="definition_prerequisite",
            explanation=f"Gợi ý đọc {a} trước vì Điều này định nghĩa {', '.join(f'“{term}”' for term in terms)}, thuật ngữ được sử dụng trong {b}.",
            evidence=definitions + support,
        )
        return result
    references = _raw_evidence(evidence_graph, target, source, "REFERENCES", metadata)
    if references:
        result.update(
            reason_type="explicit_reference",
            explanation=f"Gợi ý đọc {a} trước vì {b} viện dẫn trực tiếp nội dung tại {a}.",
            evidence=references + support,
        )
        return result
    if source_meta.get("role") == "PROCEDURE":
        kind = "procedure_prerequisite"
        text = f"{a} được gán vai trò thủ tục; quan hệ đọc trước hiện có gợi ý xem trình tự/thủ tục này trước khi đọc {b}."
    elif source_meta.get("role") == "GENERAL_RULE" and target_meta.get("role") == "SPECIFIC_RULE":
        kind = "general_to_specific"
        text = f"Theo vai trò metadata và quan hệ đọc trước hiện có, {a} cung cấp quy tắc chung có thể làm nền tảng cho quy định cụ thể tại {b}."
    else:
        kind = "structural_support"
        text = f"{a} được đề xuất trước {b} theo quan hệ đọc trước hiện có; chưa có bằng chứng định nghĩa hoặc viện dẫn trực tiếp để giải thích cụ thể hơn."
    support.append(
        {
            "relation": "READING_METADATA",
            "source": a,
            "target": b,
            "source_id": source,
            "target_id": target,
            "source_role": source_meta.get("role", "UNKNOWN"),
            "target_role": target_meta.get("role", "UNKNOWN"),
        }
    )
    result.update(reason_type=kind, explanation=text)
    return result


def explain_reading_path(reading_path, reading_graph, evidence_graph, article_metadata):
    """Return new step dicts in the original order. Never mutate any input.

    Accept the existing list of step dictionaries or a list of article IDs.
    Only edges to later selected steps justify reading something first. A target
    needs no prerequisite explanation; confidence is None for selection/fallback.
    """
    ids = [s["id"] if isinstance(s, dict) else s for s in reading_path]
    if len(set(ids)) != len(ids) or any(node not in reading_graph for node in ids):
        raise ValueError("Reading path must contain unique known articles")
    positions = {node: i for i, node in enumerate(ids)}
    results = []
    for index, source in enumerate(ids):
        original = reading_path[index]
        step = deepcopy(original) if isinstance(original, dict) else {"id": source}
        options = []
        if index < len(ids) - 1:
            for _, target, key, edge in reading_graph.out_edges(source, keys=True, data=True):
                if edge.get("edge_type") == "READ_BEFORE" and positions.get(target, -1) > index:
                    options.append(
                        _explain_edge(
                            source,
                            target,
                            key,
                            edge,
                            reading_graph,
                            evidence_graph,
                            article_metadata,
                        )
                    )
        options.sort(
            key=lambda item: (
                PRIORITY[item["reason_type"]],
                positions[item["target_id"]],
                str(item["edge_key"]),
            )
        )
        if options:
            primary = options[0]
        elif index == len(ids) - 1:
            primary = {
                "reason_type": "target_provision",
                "explanation": "Đây là Điều đích bạn chọn để tìm hiểu, được đặt cuối đường đọc đề xuất.",
                "evidence": [{"relation": "TARGET_SELECTION", "source_id": source}],
                "confidence": None,
                "origin": "user_selection",
                "confidence_semantics": "not applicable to target selection",
            }
        else:
            primary = {
                "reason_type": "cycle_context" if step.get("cycle") else "insufficient_evidence",
                "explanation": "Điều này nằm trong đường đọc đã chọn, nhưng không có cạnh đọc trước đến bước phía sau để giải thích thứ tự này. Hãy xem cảnh báo chu trình nếu có.",
                "evidence": [
                    {
                        "relation": "PATH_MEMBERSHIP",
                        "source_id": source,
                        "cycle": deepcopy(step.get("cycle", [])),
                    }
                ],
                "confidence": None,
                "origin": "path_context",
                "confidence_semantics": "no supporting forward edge",
            }
        step.update(deepcopy(primary))
        step.update(
            article=_label(source, reading_graph, article_metadata),
            position=index + 1,
            explanations=options,
            target=index == len(ids) - 1,
        )
        results.append(step)
    return results


def annotate_reading_graph(reading_graph, explained_steps):
    """Copy the graph and add explanation metadata only to explained existing edges."""
    annotated = deepcopy(reading_graph)
    for step in explained_steps:
        for explanation in step["explanations"]:
            edge = annotated[explanation["source_id"]][explanation["target_id"]][
                explanation["edge_key"]
            ]
            edge["explanation"] = {
                "type": explanation["reason_type"].upper(),
                "text": explanation["explanation"],
            }
    return annotated
