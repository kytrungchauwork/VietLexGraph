"""Separate observed document evidence from derived READ_BEFORE recommendations."""

import re
from dataclasses import asdict

import networkx as nx

EVIDENCE_TYPES = {"CONTAINS", "REFERENCES", "DEFINES", "USES_TERM"}
PRIORITY = {"DEFINITION": 0, "REFERENCE": 1, "ML": 2, "DOCUMENT_ORDER": 3, "SEMANTIC": 3}
ROLE_ORDER = {
    "DEFINITION": 0,
    "SCOPE": 0,
    "SUBJECT_SCOPE": 0,
    "RIGHT": 1,
    "OBLIGATION": 1,
    "PROCEDURE": 2,
    "SANCTION": 3,
    "TRANSITIONAL": 4,
    "EFFECTIVE_DATE": 4,
    "UNKNOWN": 5,
}


def build_evidence_graph(doc):
    """Raw REFERENCES retains citing → cited direction; no reading claims here."""
    graph = nx.MultiDiGraph(graph_kind="evidence")
    graph.add_nodes_from((n.id, asdict(n)) for n in doc.nodes)
    for edge in doc.edges:
        if edge.edge_type not in EVIDENCE_TYPES or edge.origin != "deterministic":
            raise ValueError("Evidence graph accepts observed extraction relations only")
        graph.add_edge(edge.source, edge.target, **asdict(edge))
    return graph


def structural_graph(doc):
    """Compatibility entry point for the raw evidence graph."""
    return build_evidence_graph(doc)


def _reference_decision(doc, relation, policy):
    if policy in {"citation_only", "definition_citation"}:
        return True, "Baseline follows every resolved intra-document citation."
    article = doc.by_id[relation["source"]]
    start = relation["evidence"].get("start", 0)
    end = relation["evidence"].get("end", start)
    # Deliberately narrow cue exclusion, not a legal classification model.
    left = (
        max(
            article.body.rfind("\n", 0, start),
            article.body.rfind(";", 0, start),
            article.body.rfind(".", 0, start),
        )
        + 1
    )
    context = article.body[left:end].casefold()
    if re.search(r"\b(?:căn cứ|sửa đổi|bổ sung|bãi bỏ|thay thế|dẫn chiếu hình thức)\b", context):
        return (
            False,
            "Formal/amendment cue: keep citation evidence but do not infer reading usefulness.",
        )
    return (
        True,
        "Resolved citation accepted by a conservative reading heuristic; usefulness is unverified.",
    )


def candidate_prerequisites(
    doc, evidence_graph=None, policy="hybrid", reference_confidence=0.85, definition_confidence=0.95
):
    """Produce derived rule recommendations plus explicit accept/reject decisions."""
    if policy not in {"hybrid", "citation_only", "definition_citation"}:
        raise ValueError("Unknown reading policy")
    if not all(0 <= c <= 1 for c in (reference_confidence, definition_confidence)):
        raise ValueError("Rule weights must be in [0, 1]")
    evidence = build_evidence_graph(doc) if evidence_graph is None else evidence_graph
    proposed, decisions, definitions = [], [], {}
    for _, _, relation in evidence.edges(data=True):
        if relation["edge_type"] == "DEFINES":
            definitions.setdefault(relation["target"], []).append(relation)
        if relation["edge_type"] != "REFERENCES":
            continue
        source, target = relation["target"], relation["source"]
        accepted, explanation = _reference_decision(doc, relation, policy)
        if source == target:
            accepted, explanation = False, "Self reference supplies no additional reading article."
        decisions.append(
            {
                "source": source,
                "target": target,
                "accepted": accepted,
                "reason": explanation,
                "evidence": [relation],
            }
        )
        if accepted:
            proposed.append(
                (
                    source,
                    target,
                    {
                        "edge_type": "READ_BEFORE",
                        "origin": "deterministic_reference_rule",
                        "confidence": reference_confidence,
                        "evidence_kind": "REFERENCE",
                        "reason": f"Gợi ý đọc Điều {doc.by_id[source].number} trước vì Điều {doc.by_id[target].number} có viện dẫn; đây là suy luận theo quy tắc, chưa được xác nhận về tính hữu ích.",
                        "evidence": {
                            "relations": [relation],
                            "policy": policy,
                            "confidence_semantics": "uncalibrated rule weight",
                        },
                    },
                )
            )
    if policy != "citation_only":
        for _, _, use in evidence.edges(data=True):
            if use["edge_type"] != "USES_TERM":
                continue
            for definition in definitions.get(use["target"], []):
                source, target = definition["source"], use["source"]
                if source == target:
                    continue
                proposed.append(
                    (
                        source,
                        target,
                        {
                            "edge_type": "READ_BEFORE",
                            "origin": "deterministic_definition_rule",
                            "confidence": definition_confidence,
                            "evidence_kind": "DEFINITION",
                            "reason": f"Gợi ý đọc Điều {doc.by_id[source].number} để xem định nghĩa “{use['evidence']['term']}” được dùng tại Điều {doc.by_id[target].number}.",
                            "evidence": {
                                "relations": [definition, use],
                                "term": use["evidence"]["term"],
                                "policy": policy,
                                "confidence_semantics": "uncalibrated rule weight",
                            },
                        },
                    )
                )
    return proposed, decisions


def build_reading_graph(
    doc, predictions=(), policy="hybrid", reference_confidence=0.85, definition_confidence=0.95
):
    """READ_BEFORE direction: candidate to read first → dependent target."""
    graph = nx.MultiDiGraph(graph_kind="reading", policy=policy)
    graph.add_nodes_from((a.id, asdict(a)) for a in doc.articles)
    proposals, decisions = candidate_prerequisites(
        doc,
        policy=policy,
        reference_confidence=reference_confidence,
        definition_confidence=definition_confidence,
    )
    graph.graph["decisions"] = decisions
    for source, target, data in proposals:
        graph.add_edge(source, target, source=source, target=target, **data)
    for edge in predictions:
        # Accept old in-memory prediction objects; never insert them into evidence graph.
        if edge.edge_type not in {"READ_BEFORE", "PREDICTED_RELATED"} or edge.origin != "ml":
            raise ValueError("Only ML reading candidates may be added")
        if edge.source not in graph or edge.target not in graph:
            raise ValueError("Prediction endpoint outside document")
        if edge.source != edge.target and not graph.has_edge(edge.source, edge.target):
            graph.add_edge(
                edge.source,
                edge.target,
                source=edge.source,
                target=edge.target,
                edge_type="READ_BEFORE",
                origin="ml",
                confidence=edge.confidence,
                evidence_kind="ML",
                reason=edge.reason
                or f"Mô hình đề xuất ứng viên đọc trước với điểm {edge.confidence:.3f}; chưa được hiệu chỉnh bằng human gold.",
                evidence=edge.evidence,
            )
    return graph


def filter_graph(graph, include_ml=True, min_confidence=0.0, edge_types=None, evidence_kinds=None):
    result = nx.MultiDiGraph(**graph.graph)
    result.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        if edge_types is not None and data["edge_type"] not in edge_types:
            continue
        if evidence_kinds is not None and data.get("evidence_kind") not in evidence_kinds:
            continue
        if data["origin"] == "ml" and (not include_ml or data["confidence"] < min_confidence):
            continue
        result.add_edge(u, v, **data)
    return result


def _kind(edge):
    return edge.get(
        "evidence_kind",
        {
            "REFERENCES": "REFERENCE",
            "DEFINED_TERM_DEPENDENCY": "DEFINITION",
            "PREDICTED_RELATED": "ML",
        }.get(edge["edge_type"], "ML"),
    )


def reading_path(
    graph, target, include_ml=True, min_confidence=0.0, max_depth=None, max_prerequisites=None
):
    """Bounded useful-set heuristic; does not claim a globally minimal/optimal path."""
    if (
        max_depth is not None
        and max_depth < 0
        or max_prerequisites is not None
        and max_prerequisites < 0
    ):
        raise ValueError("Depth and prerequisite budget must be nonnegative")
    graph = filter_graph(graph, include_ml, min_confidence)
    if target not in graph:
        raise ValueError("Unknown target article")
    all_ancestors = nx.ancestors(graph, target)
    distances = nx.single_source_shortest_path_length(graph.reverse(), target, cutoff=max_depth)
    sub = graph.subgraph(distances).copy()
    deterministic = nx.DiGraph()
    deterministic.add_nodes_from(sub)
    deterministic.add_edges_from((u, v) for u, v, d in sub.edges(data=True) if d["origin"] != "ml")
    det_ancestors = nx.ancestors(deterministic, target) | {target}

    def priority(node):
        outgoing = list(sub.out_edges(node, data=True))
        rank = min((PRIORITY[_kind(d)] for _, _, d in outgoing), default=0)
        strength = max((d["confidence"] for _, _, d in outgoing), default=1.0)
        return (
            node not in det_ancestors,
            rank,
            -strength,
            distances[node],
            ROLE_ORDER.get(graph.nodes[node].get("role", "UNKNOWN"), 5),
            graph.nodes[node]["order"],
        )

    selected = set(sub)
    if max_prerequisites is not None:
        selected = {target}
        weighted = nx.DiGraph()
        weighted.add_nodes_from(sub)
        for u, v, d in sub.edges(data=True):
            weight = 1 + PRIORITY[_kind(d)]
            if not weighted.has_edge(u, v) or weighted[u][v]["weight"] > weight:
                weighted.add_edge(u, v, weight=weight)
        for node in sorted(set(sub) - {target}, key=priority):
            route = set(nx.shortest_path(weighted, node, target, weight="weight"))
            if len(selected | route) - 1 <= max_prerequisites:
                selected |= route
    omitted = [
        {
            "id": node,
            "reason": "max_depth" if node not in distances else "max_prerequisites",
            "has_rule_support": any(
                d["origin"] != "ml" for _, _, d in graph.out_edges(node, data=True)
            ),
        }
        for node in sorted(all_ancestors - selected, key=lambda n: graph.nodes[n]["order"])
    ]
    sub = sub.subgraph(selected).copy()
    components = list(nx.strongly_connected_components(sub))
    dag = nx.condensation(sub, components)
    ordered = nx.lexicographical_topological_sort(
        dag, key=lambda c: min(priority(n) for n in dag.nodes[c]["members"])
    )
    steps = []
    for component in ordered:
        group = dag.nodes[component]["members"]
        internal = deterministic.subgraph(group)
        if nx.is_directed_acyclic_graph(internal):
            members = list(
                nx.lexicographical_topological_sort(
                    internal, key=lambda n: (n == target, graph.nodes[n]["order"])
                )
            )
        else:
            members = sorted(group, key=lambda n: (n == target, graph.nodes[n]["order"]))
        # The target ends the user-facing path. Any conflicting SCC is explicitly disclosed.
        if target in members:
            members = [n for n in members if n != target] + [target]
        for node in members:
            reasons = []
            seen = set()
            for _, dependent, edge in sorted(
                sub.out_edges(node, data=True), key=lambda x: PRIORITY[_kind(x[2])]
            ):
                text = (
                    edge.get("reason")
                    or f"Khuyến nghị dựa trên {_kind(edge)} liên quan đến Điều {graph.nodes[dependent]['number']}."
                )
                key = (dependent, edge["origin"], text)
                if key in seen:
                    continue
                seen.add(key)
                reasons.append(
                    {
                        "text": text,
                        "origin": edge["origin"],
                        "confidence": edge["confidence"],
                        "evidence": edge["evidence"],
                    }
                )
            steps.append(
                {
                    "id": node,
                    "target": node == target,
                    "cycle": members if len(members) > 1 or sub.has_edge(node, node) else [],
                    "reasons": reasons,
                    "omitted": omitted if node == target else [],
                }
            )
    return steps
