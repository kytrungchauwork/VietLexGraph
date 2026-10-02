"""Train all four baselines; select using validation documents only."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.model import (
    E5Encoder,
    classification_metrics,
    clean_model_text,
    feature_matrix,
    mlp_scores,
    ranking_metrics,
    train_mlp,
    tune_threshold,
)


def load_data():
    docs = [
        json.loads(line)
        for line in (ROOT / "data/processed_documents.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((ROOT / "data/split_manifest.json").read_text())
    from src.reproducibility import verify_hashes

    verify_hashes(ROOT / "data", manifest.get("file_hashes", {}))
    groups = manifest["splits"]
    if any(
        set(groups[a]) & set(groups[b])
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("Document leakage in split manifest")
    pairs = {name: pd.read_parquet(ROOT / f"data/{name}_pairs.parquet") for name in groups}
    articles = {a["id"]: a for d in docs for a in d["articles"]}
    for name, frame in pairs.items():
        if set(frame.document_id) != set(groups[name]):
            raise ValueError("Pair documents differ from split manifest")
        if frame.duplicated(["source", "target"]).any() or (frame.source == frame.target).any():
            raise ValueError("Duplicate or self pairs")
        if not set(frame.label) <= {0, 1}:
            raise ValueError("Invalid weak label")
        positives = {(e["source"], e["target"]) for d in docs for e in d["references"]}
        for row in frame.itertuples():
            if bool(row.label) != ((row.source, row.target) in positives):
                raise ValueError("Weak label does not match explicit references")
            if (
                articles[row.source]["document_id"] != row.document_id
                or articles[row.target]["document_id"] != row.document_id
            ):
                raise ValueError("Cross-document pair")
    return docs, articles, pairs, manifest


def all_candidates(docs, document_ids):
    records = []
    for doc in docs:
        if doc["id"] not in document_ids:
            continue
        references = {}
        for edge in doc["references"]:
            references.setdefault(edge["source"], set()).add(edge["target"])
        for source, targets in references.items():
            for article in doc["articles"]:
                if article["id"] != source:
                    records.append(
                        {
                            "document_id": doc["id"],
                            "source": source,
                            "target": article["id"],
                            "label": int(article["id"] in targets),
                        }
                    )
    return pd.DataFrame(records)


def score_pairs(bundle, pairs, articles, embeddings):
    ids = list(articles)
    index = {aid: i for i, aid in enumerate(ids)}
    tfidf = bundle["tfidf"].transform([clean_model_text(articles[i]) for i in ids])
    ia = [index[s] for s in pairs.source]
    ib = [index[t] for t in pairs.target]
    scores = {
        "tfidf_cosine": np.asarray(tfidf[ia].multiply(tfidf[ib]).sum(axis=1)).ravel(),
        "e5_cosine": np.array(
            [embeddings[a] @ embeddings[b] for a, b in zip(pairs.source, pairs.target)]
        ),
    }
    # Bounded feature batches also support ranking over all document-local candidates.
    lr, mlp = [], []
    for start in range(0, len(pairs), 2048):
        frame = pairs.iloc[start : start + 2048]
        x = feature_matrix(frame, articles, embeddings)
        semantic = x[:, :-7]
        lr.extend(
            bundle["logistic_regression"].predict_proba(bundle["lr_scaler"].transform(semantic))[
                :, 1
            ]
        )
        mlp.extend(mlp_scores(bundle["mlp"], bundle["mlp_scaler"].transform(x).astype(np.float32)))
    scores["logistic_regression"], scores["mlp"] = np.array(lr), np.array(mlp)
    return scores


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    np.random.seed(cfg["seed"])
    docs, articles, pairs, manifest = load_data()
    encoder = E5Encoder(
        cfg["encoder"], cfg["chunk_tokens"], cfg["batch_size"], revision=cfg.get("encoder_revision")
    )
    vectors = encoder.encode(list(articles.values()))
    embeddings = dict(zip(articles, vectors))
    np.savez_compressed(
        ROOT / "models/embeddings.npz", ids=np.array(list(articles)), vectors=vectors
    )
    train_ids = set(manifest["splits"]["train"])
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True)
    tfidf.fit([clean_model_text(a) for a in articles.values() if a["document_id"] in train_ids])
    x = feature_matrix(pairs["train"], articles, embeddings)
    xv = feature_matrix(pairs["val"], articles, embeddings)
    y, yv = pairs["train"].label.to_numpy(), pairs["val"].label.to_numpy()
    lr_scaler = StandardScaler().fit(x[:, :-7])
    lr = LogisticRegression(max_iter=1500, C=0.1, random_state=cfg["seed"])
    lr.fit(lr_scaler.transform(x[:, :-7]), y)
    scaler = StandardScaler().fit(x)
    state, history = train_mlp(scaler.transform(x), y, scaler.transform(xv), yv, cfg)
    bundle = {
        "tfidf": tfidf,
        "logistic_regression": lr,
        "lr_scaler": lr_scaler,
        "mlp": state,
        "mlp_scaler": scaler,
    }
    # Tune on the full candidate population used by the application, not 1:3 sampled prevalence.
    candidates = all_candidates(docs, set(manifest["splits"]["val"]))
    validation_scores = score_pairs(bundle, candidates, articles, embeddings)
    validations, thresholds = {}, {}
    for name, scores in validation_scores.items():
        thresholds[name] = tune_threshold(candidates.label, scores, cfg["precision_target"])
        validations[name] = {
            **classification_metrics(candidates.label, scores, thresholds[name]["threshold"]),
            **ranking_metrics(candidates, scores),
            **thresholds[name],
        }
    lr_result, mlp_result = validations["logistic_regression"], validations["mlp"]
    improvement = cfg["mlp_min_improvement"]
    mlp_wins = (
        mlp_result["constraint_met"]
        and mlp_result["recall"] >= lr_result["recall"] + improvement
        and mlp_result["mrr"] >= lr_result["mrr"]
        and mlp_result["pr_auc"] >= lr_result["pr_auc"]
    )
    selected = "mlp" if mlp_wins else "logistic_regression"
    artifact = (
        {"scaler": scaler, "state": state} if mlp_wins else {"scaler": lr_scaler, "classifier": lr}
    )
    joblib.dump(bundle, ROOT / "models/baselines.pkl")
    joblib.dump(artifact, ROOT / "models/relation_model.pkl")
    joblib.dump(artifact["scaler"], ROOT / "models/scaler.pkl")
    import hashlib

    metadata = {
        "model_type": selected,
        "encoder": cfg["encoder"],
        "encoder_revision": encoder.revision,
        "chunk_tokens": cfg["chunk_tokens"],
        "threshold": thresholds[selected]["threshold"],
        "seed": cfg["seed"],
        "dataset": cfg["dataset"],
        "dataset_split": cfg["dataset_split"],
        "dataset_revision": manifest["revision"],
        "split_manifest_sha256": hashlib.sha256(
            (ROOT / "data/split_manifest.json").read_bytes()
        ).hexdigest(),
        "experiment_documents": sum(len(v) for v in manifest["splits"].values()),
        "validation": validations,
        "thresholds": thresholds,
        "selection_rule": "MLP only if validation precision constraint met, recall improves >= 0.01, and MRR/PR-AUC do not regress; otherwise LR",
        "threshold_population": "all other articles for each validation source with at least one resolved reference",
        "mlp_history": history,
        "config": cfg,
    }
    from src.reproducibility import sha256, source_provenance

    metadata.update(source_provenance())
    metadata["feature_version"] = 2
    metadata["project"] = "VietLexGraph"
    metadata["target_task"] = "read_before: candidate should be read before target"
    metadata["training_signal"] = "weak supervision derived from explicit intra-document references"
    metadata["calibration_status"] = "not calibrated on human read_before labels"
    metadata["selection_status"] = (
        "preliminary weak-surrogate selection only; final system selection blocked on human-gold path metrics"
    )
    metadata["final_selection_priority"] = [
        "human-gold Path Precision",
        "human-gold Path Recall/F1",
        "low Extra Reading Cost",
        "document-level stability",
        "simplicity",
    ]
    metadata["citation_masking"] = "body and title; viện dẫn replacement"
    metadata["artifact_hashes"] = {
        name: sha256(ROOT / "models" / name)
        for name in ["relation_model.pkl", "scaler.pkl", "baselines.pkl", "embeddings.npz"]
    }
    import importlib.metadata

    metadata["environment"] = {
        name: importlib.metadata.version(name)
        for name in [
            "numpy",
            "scikit-learn",
            "torch",
            "transformers",
            "datasets",
            "streamlit",
            "networkx",
        ]
    }
    (ROOT / "models/metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({"selected": selected, "validation": validations}, indent=2))


if __name__ == "__main__":
    main()
