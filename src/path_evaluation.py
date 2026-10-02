"""Reading baselines and strict metrics: missing judgments are never negatives."""

from src.graph import build_reading_graph, reading_path

BASELINES = (
    "DocumentOrder",
    "CitationOnly",
    "DefinitionCitation",
    "SemanticTopK",
    "HybridNoML",
    "FullSystem",
)


def baseline_path(
    doc,
    target,
    name,
    *,
    embeddings=None,
    predictions=None,
    window=5,
    max_depth=5,
    max_prerequisites=5,
    min_confidence=0.0,
    reference_confidence=0.85,
    definition_confidence=0.95,
):
    if name not in BASELINES:
        raise ValueError("Unknown baseline")
    articles = doc.articles
    ids = [a.id for a in articles]
    if target not in ids:
        raise ValueError("Unknown target")
    if min(window, max_depth, max_prerequisites) < 0:
        raise ValueError("Nonnegative limits required")
    if name in {"DocumentOrder", "SemanticTopK"}:
        if name == "DocumentOrder":
            index = ids.index(target)
            selected = ids[max(0, index - min(window, max_prerequisites)) : index]
        else:
            if embeddings is None:
                raise ValueError("SemanticTopK requires E5 embeddings")
            selected = sorted(
                (i for i in ids if i != target),
                key=lambda i: (-float(embeddings[i] @ embeddings[target]), ids.index(i)),
            )[:max_prerequisites]
            selected.sort(key=ids.index)
        return [
            {
                "id": i,
                "target": i == target,
                "cycle": [],
                "omitted": [],
                "reasons": []
                if i == target
                else [
                    {
                        "origin": name,
                        "text": "Preceding document window"
                        if name == "DocumentOrder"
                        else "E5 cosine top-K, ordered by document position",
                        "evidence": {
                            "candidate": i,
                            "target": target,
                            "document_order": ids.index(i),
                            **(
                                {"cosine": float(embeddings[i] @ embeddings[target])}
                                if name == "SemanticTopK"
                                else {}
                            ),
                        },
                    }
                ],
            }
            for i in selected + [target]
        ]
    if name == "FullSystem" and predictions is None:
        raise ValueError("FullSystem requires explicit model predictions (possibly empty)")
    policy = {"CitationOnly": "citation_only", "DefinitionCitation": "definition_citation"}.get(
        name, "hybrid"
    )
    graph = build_reading_graph(
        doc,
        predictions or (),
        policy=policy,
        reference_confidence=reference_confidence,
        definition_confidence=definition_confidence,
    )
    return reading_path(
        graph,
        target,
        include_ml=name == "FullSystem",
        min_confidence=min_confidence,
        max_depth=max_depth,
        max_prerequisites=max_prerequisites,
    )


def path_metrics(path, target, judgments, ordering=None):
    """Judgments must cover the entire candidate pool, enforced by the caller.

    Empty recommendations: precision=1 only when gold also empty, else 0.
    Empty gold: recall=1. Path length excludes the target. Ordering is optional.
    """
    ids = [step["id"] if isinstance(step, dict) else step for step in path]
    if not ids or ids[-1] != target or len(set(ids)) != len(ids):
        raise ValueError("Path must be unique and end in target")
    selected = set(ids[:-1])
    if any(v not in (0, 1) for v in judgments.values()) or target in judgments:
        raise ValueError("Complete adjudicated binary judgments required")
    if selected - judgments.keys():
        raise ValueError("Unjudged recommendation: cannot score as a negative")
    useful = {key for key, value in judgments.items() if value == 1}
    tp = len(selected & useful)
    precision = tp / len(selected) if selected else float(not useful)
    recall = tp / len(useful) if useful else 1.0
    result = {
        "path_precision": precision,
        "path_recall": recall,
        "path_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "extra_reading_cost": len(selected) - tp,
        "path_length": len(selected),
        "pairwise_order_accuracy": None,
        "order_pair_coverage": None,
    }
    if ordering:
        comparable = [(a, b) for a, b in ordering if a in ids and b in ids]
        result["order_pair_coverage"] = len(comparable) / len(ordering)
        result["pairwise_order_accuracy"] = (
            (sum(ids.index(a) < ids.index(b) for a, b in comparable) / len(comparable))
            if comparable
            else None
        )
    return result
