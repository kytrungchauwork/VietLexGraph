"""Reproducible dataset invariants and observed regression cases (not human gold)."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import load_data
from src.graph import build_reading_graph, reading_path
from src.parser import parse_document
from src.relations import extract_relations


def dataset_audit():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    docs, articles, frames, manifest = load_data()
    table = []
    for name, frame in frames.items():
        table.append(
            {
                "split": name,
                "documents": frame.document_id.nunique(),
                "pairs": len(frame),
                "positive": int(frame.label.sum()),
                "negative": int((frame.label == 0).sum()),
                "positive_pct": float(frame.label.mean() * 100),
            }
        )
    groups = manifest["splits"]
    checks = {
        "document_overlap": sum(
            len(set(groups[a]) & set(groups[b]))
            for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
        ),
        "duplicate_pairs": sum(
            int(f.duplicated(["source", "target"]).sum()) for f in frames.values()
        ),
        "self_pairs": sum(int((f.source == f.target).sum()) for f in frames.values()),
        "positive_negative_conflicts": sum(
            int((f.groupby(["source", "target"]).label.nunique() > 1).sum())
            for f in frames.values()
        ),
    }
    assert not any(checks.values())
    result = {
        "table": table,
        "checks": checks,
        "articles": len(articles),
        "documents": len(docs),
        "weak_supervision": "All positives validated against extracted references in load_data; nonreferences remain unlabeled negatives.",
    }
    (ROOT / "artifacts/dataset-audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def regression_observations():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    cases = {
        "ambiguous_reference": "Điều 1. A\nA.\nĐiều 1. B\nB.\nĐiều 2. C\nTheo Điều 1.",
        "unresolved_reference": "Điều 1. A\nTheo Điều 99.",
        "definition_miss": "Điều 1. Giải thích từ ngữ\nHệ thống được gọi là một tập hợp thành phần.\nĐiều 2. Dùng\nHệ thống hoạt động.",
        "paraphrased_term": 'Điều 1. Giải thích từ ngữ\n"Nhà cung cấp" là tổ chức cung cấp hệ thống.\nĐiều 2. Dùng\nBên cung ứng phải báo cáo.',
        "graph_cycle": "Điều 1. A\nTheo Điều 2.\nĐiều 2. B\nTheo Điều 1.",
    }
    output = []
    for name, text in cases.items():
        doc = extract_relations(parse_document(text))
        steps = reading_path(build_reading_graph(doc), doc.articles[-1].id)
        output.append(
            {
                "category": name,
                "source": "authored synthetic regression input, not human annotation",
                "input": text,
                "edge_counts": {
                    kind: sum(e.edge_type == kind for e in doc.edges)
                    for kind in ["REFERENCES", "DEFINES", "USES_TERM"]
                },
                "unresolved": doc.unresolved_references,
                "cycles": any(s["cycle"] for s in steps),
                "path": [doc.by_id[s["id"]].number for s in steps],
            }
        )
    (ROOT / "artifacts/regression-observations.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    dataset_audit()
    regression_observations()
