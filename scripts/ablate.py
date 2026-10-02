"""Controlled LR feature/citation ablations; results do not select release artifacts."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from scripts.train import all_candidates, load_data
from src.model import (
    E5Encoder,
    classification_metrics,
    feature_matrix,
    ranking_metrics,
    tune_threshold,
)


def main():
    (ROOT / "artifacts").mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    docs, articles, pairs, manifest = load_data()
    val = all_candidates(docs, set(manifest["splits"]["val"]))
    test = all_candidates(docs, set(manifest["splits"]["test"]))
    results = []
    for masked in (True, False):
        encoder = E5Encoder(
            cfg["encoder"],
            cfg["chunk_tokens"],
            cfg["batch_size"],
            revision=cfg["encoder_revision"],
            mask_citations=masked,
        )
        embeddings = dict(zip(articles, encoder.encode(list(articles.values()))))
        matrices = [
            feature_matrix(frame, articles, embeddings, mask_citations=masked)
            for frame in (pairs["train"], val, test)
        ]
        semantic = matrices[0].shape[1] - 7
        groups = {
            "E5 only": list(range(semantic)),
            "E5 + structural": list(range(semantic + 4)),
            "E5 + term features": list(range(semantic)) + list(range(semantic + 4, semantic + 7)),
            "All features": list(range(semantic + 7)),
        }
        for name, columns in groups.items():
            train_x, val_x, test_x = [x[:, columns] for x in matrices]
            scaler = StandardScaler().fit(train_x)
            model = LogisticRegression(C=0.1, max_iter=1500, random_state=cfg["seed"]).fit(
                scaler.transform(train_x), pairs["train"].label
            )
            val_scores = model.predict_proba(scaler.transform(val_x))[:, 1]
            threshold = tune_threshold(val.label, val_scores, cfg["precision_target"])
            scores = model.predict_proba(scaler.transform(test_x))[:, 1]
            row = {
                "citation_setting": "masked" if masked else "visible",
                "features": name,
                "classifier": "LogisticRegression C=0.1",
                **threshold,
                **classification_metrics(test.label, scores, threshold["threshold"]),
                **ranking_metrics(test, scores),
            }
            results.append(row)
            print(json.dumps(row), flush=True)
        cosine_val = np.array(
            [embeddings[a] @ embeddings[b] for a, b in zip(val.source, val.target)]
        )
        cosine_test = np.array(
            [embeddings[a] @ embeddings[b] for a, b in zip(test.source, test.target)]
        )
        tuned = tune_threshold(val.label, cosine_val, cfg["precision_target"])
        results.append(
            {
                "citation_setting": "masked" if masked else "visible",
                "features": "E5 cosine",
                "classifier": "none",
                **tuned,
                **classification_metrics(test.label, cosine_test, tuned["threshold"]),
                **ranking_metrics(test, cosine_test),
            }
        )
    output = {
        "protocol": "Fixed 30-document split; LR hyperparameters held constant; validation-only thresholds; no release-model reselection from ablations",
        "feature_groups": {
            "structural": [
                "same_chapter",
                "same_section",
                "number_distance",
                "candidate_before_source",
            ],
            "term": ["shared_defined_terms", "shared_token_ratio", "title_similarity"],
        },
        "results": results,
    }
    (ROOT / "artifacts/ablations.json").write_text(json.dumps(output, indent=2))
    pd.DataFrame(results).to_csv(ROOT / "artifacts/ablations.csv", index=False)


if __name__ == "__main__":
    main()
