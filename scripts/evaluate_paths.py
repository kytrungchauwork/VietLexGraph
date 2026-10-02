"""Compare six reading systems; annotated metrics require frozen, complete labels with provenance."""

import argparse
import csv
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import yaml

from scripts.gold import digest, validate
from scripts.train import load_data, score_pairs
from src.path_evaluation import BASELINES, baseline_path, path_metrics
from src.relations import extract_relations
from src.reproducibility import source_provenance, verify_hashes
from src.schemas import LegalDocument, LegalEdge, LegalNode


def restore_document(record):
    # Relation extraction needs article bodies, not the original hierarchy. ML uses
    # original stored article records, preserving chapter/section features below.
    names = {f.name for f in fields(LegalNode)}
    nodes = [LegalNode(record["id"], "DOCUMENT")]
    nodes.extend(
        LegalNode(**{k: v for k, v in a.items() if k in names}) for a in record["articles"]
    )
    for node in nodes[1:]:
        node.parent_id = record["id"]
    return extract_relations(LegalDocument(record["id"], "", nodes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path)
    parser.add_argument("--split", choices=["gold-dev", "gold-test"], default="gold-test")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/path-comparison.json")
    args = parser.parse_args()
    required = [
        ROOT / "models" / name for name in ("metadata.json", "baselines.pkl", "embeddings.npz")
    ]
    required += [
        ROOT / "data" / name
        for name in (
            "processed_documents.jsonl",
            "split_manifest.json",
            "train_pairs.parquet",
            "val_pairs.parquet",
            "test_pairs.parquet",
        )
    ]
    if any(not p.is_file() for p in required):
        print(
            "NOT RUN: local evaluation inputs are absent. Run prepare_data.py and train.py first; annotations are never included in the public model release."
        )
        return
    docs, articles, _, manifest = load_data()
    metadata = json.loads((ROOT / "models/metadata.json").read_text())
    from src.reproducibility import sha256

    if sha256(ROOT / "data/split_manifest.json") != metadata["split_manifest_sha256"]:
        raise ValueError("Retrain after dataset changes")
    verify_hashes(ROOT / "models", metadata["artifact_hashes"])
    cache = np.load(ROOT / "models/embeddings.npz")
    embeddings = dict(zip(cache["ids"], cache["vectors"]))
    bundle = joblib.load(ROOT / "models/baselines.pkl")
    reading = yaml.safe_load((ROOT / "config.yaml").read_text())["reading"]
    gold = {}
    if args.gold_dir:
        frozen = json.loads((args.gold_dir / "manifest.json").read_text())
        verify_hashes(args.gold_dir, frozen["files"])
        sets = {}
        for split in ["gold-dev", "gold-test"]:
            with (args.gold_dir / f"{split}.csv").open() as stream:
                rows = validate(list(csv.DictReader(stream)))
            sets[split] = {r["document_id"] for r in rows}
            if split == args.split:
                known = {d["id"]: d for d in docs}
                for row in rows:
                    document = known[row["document_id"]]
                    if document["id"] in manifest["splits"]["train"]:
                        raise ValueError("Gold overlaps weak training")
                    if (
                        row["document_sha256"] != document["sha256"]
                        or row["dataset_revision"] != manifest["revision"]
                    ):
                        raise ValueError("Gold document revision/hash mismatch")
                    for side in ("target", "candidate"):
                        if row[f"{side}_text_sha256"] != digest(
                            articles[row[f"{side}_article"]]["text"]
                        ):
                            raise ValueError("Gold article hash mismatch")
                    gold.setdefault((row["document_id"], row["target_article"]), {})[
                        row["candidate_article"]
                    ] = int(row["read_before"])
        if sets["gold-dev"] & sets["gold-test"]:
            raise ValueError("Gold document leakage")
        if not gold:
            raise ValueError("Empty gold")
    wanted = {k[0] for k in gold} if gold else set(manifest["splits"]["test"])
    details = []
    for record in docs:
        if record["id"] not in wanted:
            continue
        doc = restore_document(record)
        ids = [a.id for a in doc.articles]
        if gold:
            targets = [t for d, t in gold if d == record["id"]]
            for target in targets:
                if set(gold[(record["id"], target)]) != set(ids) - {target}:
                    raise ValueError("Annotated target requires complete candidate judgments")
        else:
            targets = ids
        pairs = pd.DataFrame([{"source": a, "target": b} for a in ids for b in ids if a != b])
        scores = score_pairs(bundle, pairs, articles, embeddings)[metadata["model_type"]]
        raw_refs = {(e.source, e.target) for e in doc.edges if e.edge_type == "REFERENCES"}
        predictions = [
            LegalEdge(
                row.target,
                row.source,
                "READ_BEFORE",
                float(score),
                "ml",
                {
                    "model": metadata["model_type"],
                    "score_semantics": "uncalibrated weak-reference proxy",
                },
                "Preliminary learned read-before hypothesis",
            )
            for row, score in zip(pairs.itertuples(), scores)
            if score >= metadata["threshold"] and (row.source, row.target) not in raw_refs
        ]
        for target in targets:
            for name in BASELINES:
                path = baseline_path(
                    doc,
                    target,
                    name,
                    embeddings=embeddings,
                    predictions=predictions,
                    min_confidence=metadata["threshold"],
                    max_depth=reading["max_depth"],
                    max_prerequisites=reading["max_prerequisites"],
                    window=reading["document_order_window"],
                    reference_confidence=reading["reference_weight"],
                    definition_confidence=reading["definition_weight"],
                )
                row = {
                    "document_id": doc.id,
                    "target": target,
                    "baseline": name,
                    "path_length": len(path) - 1,
                    "omitted_candidates": len(path[-1]["omitted"]),
                    "cycle_steps": sum(bool(s["cycle"]) for s in path),
                    "path": [s["id"] for s in path],
                }
                if gold:
                    judgments = gold[(doc.id, target)]
                    row.update(path_metrics(path, target, judgments))
                    row["gold_prerequisites"] = sum(judgments.values())
                    row["useful_recommended"] = sum(judgments[s["id"]] for s in path[:-1])
                details.append(row)
        print(f"Compared six baselines: {doc.id}, {len(targets)} targets", flush=True)
    if gold and {r["document_id"] for r in details} != wanted:
        raise ValueError("Unknown gold documents")
    frame = pd.DataFrame(details)
    columns = ["path_length", "omitted_candidates", "cycle_steps"] + (
        ["path_precision", "path_recall", "path_f1", "extra_reading_cost"] if gold else []
    )
    summary = frame.groupby("baseline")[columns].mean().to_dict("index")
    report = {
        "status": "ANNOTATED REFERENCE SET EVALUATED" if gold else "RESEARCH EVALUATION INCOMPLETE",
        "metric_semantics": "Reviewed prerequisite labels; see annotation_provenance"
        if gold
        else "Operational statistics ONLY; no usefulness, precision, recall, F1 or cost claims",
        "protocol": "All target articles in held-out weak-test documents; limits/weights from reading_config; semantic top-K ordered by document position. Gold mode uses frozen selected targets.",
        "annotated_pairs": sum(map(len, gold.values())),
        "annotation_provenance": frozen.get(
            "annotation_provenance", {"annotation_method": "unspecified"}
        )
        if gold
        else None,
        "evaluation_split": args.split if gold else "weak-test-operational",
        "gold_manifest_sha256": sha256(args.gold_dir / "manifest.json") if gold else None,
        "weak_split_manifest_sha256": sha256(ROOT / "data/split_manifest.json"),
        "model_metadata_sha256": sha256(ROOT / "models/metadata.json"),
        "model_type": metadata["model_type"],
        "model_threshold": metadata["threshold"],
        "model_artifact_hashes": metadata["artifact_hashes"],
        "evaluation_source": source_provenance(),
        "aggregation": "macro mean across target articles; path length excludes target; empty-gold recall=1; empty-path precision=1 only if gold empty",
        "reading_config": reading,
        "summary": summary,
        "per_document": frame.groupby(["document_id", "baseline"])[columns]
        .mean()
        .reset_index()
        .to_dict("records"),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
