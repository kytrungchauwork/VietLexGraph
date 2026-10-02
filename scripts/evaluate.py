"""Evaluate held-out documents once; never tune thresholds on test data."""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd

from scripts.train import all_candidates, load_data, score_pairs
from src.model import classification_metrics, ranking_metrics


def main():
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
            "NOT RUN: local evaluation inputs are absent. Run prepare_data.py and train.py first; the inference release alone contains no evaluation data."
        )
        return
    metadata = json.loads((ROOT / "models/metadata.json").read_text())
    digest = hashlib.sha256((ROOT / "data/split_manifest.json").read_bytes()).hexdigest()
    if digest != metadata["split_manifest_sha256"]:
        raise ValueError("Dataset changed since training; retrain before evaluation")
    from src.reproducibility import verify_hashes

    verify_hashes(ROOT / "models", metadata.get("artifact_hashes", {}))
    docs, articles, pairs, manifest = load_data()
    cache = np.load(ROOT / "models/embeddings.npz")
    embeddings = dict(zip(cache["ids"], cache["vectors"]))
    bundle = joblib.load(ROOT / "models/baselines.pkl")
    candidates = all_candidates(docs, set(manifest["splits"]["test"]))
    scores = score_pairs(bundle, candidates, articles, embeddings)
    sampled = score_pairs(bundle, pairs["test"], articles, embeddings)
    results = {}
    for name, prediction in scores.items():
        threshold = metadata["thresholds"][name]["threshold"]
        results[name] = {
            **classification_metrics(candidates.label, prediction, threshold),
            **ranking_metrics(candidates, prediction),
            "threshold": threshold,
        }
    report = {
        "selected_model": metadata["model_type"],
        "test_documents": len(manifest["splits"]["test"]),
        "test_candidate_pairs": len(candidates),
        "ranking_population": "all other articles within source document; sources with resolved references",
        "label_semantics": "weak labels from explicit references; unlabeled candidates are not verified negatives",
        "results": results,
        "sampled_1_to_3_classification": {
            name: classification_metrics(
                pairs["test"].label, prediction, metadata["thresholds"][name]["threshold"]
            )
            for name, prediction in sampled.items()
        },
    }
    (ROOT / "models/metrics.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(results).T.to_csv(ROOT / "models/metrics.csv", index_label="model")
    print(pd.DataFrame(results).T.to_string(float_format=lambda n: f"{n:.4f}"))
    print(
        f"Selected using validation only: {metadata['model_type']}; test documents: {report['test_documents']}"
    )


if __name__ == "__main__":
    main()
