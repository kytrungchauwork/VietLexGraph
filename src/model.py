"""Frozen E5 embeddings and small, validation-selected relation classifiers."""

import hashlib
import json
import re
from importlib.metadata import version
from pathlib import Path

import numpy as np

from src.schemas import LegalEdge

ROOT = Path(__file__).resolve().parents[1]


def tokens(text):
    return set(re.findall(r"\w+", text.casefold()))


def clean_model_text(article, mask_citations=True):
    # Remove explicit reference numbers to reduce trivial label shortcuts.
    from src.relations import REF

    return REF.sub("viện dẫn", article["text"]) if mask_citations else article["text"]


def model_title(article, mask_citations=True):
    return clean_model_text({"text": article["title"]}, mask_citations)


class E5Encoder:
    def __init__(
        self,
        name="intfloat/multilingual-e5-small",
        chunk_tokens=384,
        batch_size=32,
        revision=None,
        local_files_only=False,
        mask_citations=True,
    ):
        self.mask_citations = mask_citations
        if revision is None:
            import yaml

            cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
            if name != cfg["encoder"]:
                raise ValueError("An explicit immutable revision is required for another encoder")
            revision = cfg["encoder_revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Encoder revision must be a full commit hash")
        import torch
        from transformers import AutoModel, AutoTokenizer

        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            name,
            revision=revision,
            cache_dir=ROOT / ".cache" / "huggingface",
            local_files_only=local_files_only,
        )
        self.model = AutoModel.from_pretrained(
            name,
            trust_remote_code=False,
            use_safetensors=True,
            revision=revision,
            cache_dir=ROOT / ".cache" / "huggingface",
            local_files_only=local_files_only,
        )
        self.model.eval().requires_grad_(False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.name, self.revision = name, revision
        self.chunk_tokens = min(chunk_tokens, self.model.config.max_position_embeddings - 2, 510)
        if self.chunk_tokens < 16:
            raise ValueError("chunk_tokens must be at least 16")
        self.batch_size = batch_size

    def chunk_ids(self, article):
        tok = self.tokenizer
        title = tok.encode(
            "query: " + model_title(article, getattr(self, "mask_citations", True)),
            add_special_tokens=False,
        )[: min(64, self.chunk_tokens // 3)]
        content = tok.encode(
            clean_model_text(article, getattr(self, "mask_citations", True)),
            add_special_tokens=False,
            truncation=False,
        )
        budget = self.chunk_tokens - len(title)
        return [
            [tok.cls_token_id] + title + content[start : start + budget] + [tok.sep_token_id]
            for start in range(0, max(1, len(content)), budget)
        ]

    def encode(self, articles):
        cache = ROOT / ".cache" / "embeddings"
        cache.mkdir(parents=True, exist_ok=True)
        results = []
        for i, article in enumerate(articles):
            key = hashlib.sha256(
                json.dumps(
                    [
                        version("transformers"),
                        self.name,
                        self.revision,
                        self.chunk_tokens,
                        model_title(article, self.mask_citations),
                        clean_model_text(article, self.mask_citations),
                    ],
                    ensure_ascii=False,
                ).encode()
            ).hexdigest()
            path = cache / f"{key}.npy"
            if path.exists():
                results.append(np.load(path))
                continue
            chunks = self.chunk_ids(article)
            vectors = []
            for start in range(0, len(chunks), self.batch_size):
                inputs = self.tokenizer.pad(
                    {"input_ids": chunks[start : start + self.batch_size]},
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                with self.torch.inference_mode():
                    output = self.model(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (output * mask).sum(1) / mask.sum(1)
                    pooled = self.torch.nn.functional.normalize(pooled, dim=1)
                vectors.extend(pooled.cpu().numpy())
            vector = np.mean(vectors, axis=0)
            vector = (vector / max(np.linalg.norm(vector), 1e-12)).astype(np.float32)
            np.save(path, vector)
            results.append(vector)
            if (i + 1) % 25 == 0:
                print(f"E5: {i + 1}/{len(articles)} articles ({self.device})", flush=True)
        return np.asarray(results, dtype=np.float32)


def pair_features(a, b, ea, eb, structural=True, mask_citations=True):
    semantic = np.concatenate([ea, eb, np.abs(ea - eb), ea * eb, [ea @ eb]])
    if not structural:
        return semantic.astype(np.float32)
    at, bt = (
        tokens(clean_model_text(a, mask_citations)),
        tokens(clean_model_text(b, mask_citations)),
    )
    ta, tb = tokens(model_title(a, mask_citations)), tokens(model_title(b, mask_citations))
    anum = int(re.match(r"\d+", a["number"])[0])
    bnum = int(re.match(r"\d+", b["number"])[0])
    structure = [
        bool(a.get("chapter") and a["chapter"] == b.get("chapter")),
        bool(a.get("section") and a["section"] == b.get("section")),
        abs(anum - bnum),
        b["order"] < a["order"],
        len(set(a.get("terms", [])) & set(b.get("terms", []))),
        len(at & bt) / max(1, len(at | bt)),
        len(ta & tb) / max(1, len(ta | tb)),
    ]
    return np.concatenate([semantic, structure]).astype(np.float32)


def feature_matrix(pairs, articles, embeddings, structural=True, mask_citations=True):
    return np.asarray(
        [
            pair_features(
                articles[r.source],
                articles[r.target],
                embeddings[r.source],
                embeddings[r.target],
                structural,
                mask_citations,
            )
            for r in pairs.itertuples()
        ],
        dtype=np.float32,
    )


def tune_threshold(y, scores, precision_target=0.85):
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y, scores)
    eligible = np.flatnonzero(precision[:-1] >= precision_target)
    met = bool(len(eligible))
    if met:
        index = max(eligible, key=lambda i: (recall[i], precision[i], thresholds[i]))
    else:
        f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
        index = int(np.argmax(f1))
    return {
        "threshold": float(thresholds[index]),
        "constraint_met": met,
        "validation_precision": float(precision[index]),
        "validation_recall": float(recall[index]),
        "fallback": None if met else "maximum validation F1; precision constraint unattainable",
    }


def classification_metrics(y, scores, threshold):
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    prediction = np.asarray(scores) >= threshold
    return {
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
    }


def ranking_metrics(pairs, scores):
    frame = pairs.copy()
    frame["score"] = scores
    results = []
    for _, group in frame.groupby(["document_id", "source"], sort=True):
        labels = group.sort_values(["score", "target"], ascending=[False, True])["label"].to_numpy()
        count = labels.sum()
        if not count:
            continue
        ranks = np.flatnonzero(labels) + 1
        results.append(
            {
                **{f"recall_at_{k}": float(labels[:k].sum() / count) for k in (1, 3, 5)},
                "mrr": float(1 / ranks[0]),
            }
        )
    return (
        {
            key: float(np.mean([r[key] for r in results]))
            for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")
        }
        if results
        else {key: 0.0 for key in ("recall_at_1", "recall_at_3", "recall_at_5", "mrr")}
    )


def make_mlp(input_dim):
    from torch import nn

    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 1),
    )


def mlp_scores(state, x):
    import torch

    net = make_mlp(x.shape[1])
    net.load_state_dict(state)
    net.eval()
    with torch.inference_mode():
        return torch.cat(
            [
                net(torch.from_numpy(batch)).sigmoid().flatten()
                for batch in np.array_split(x.astype(np.float32), max(1, len(x) // 1024 + 1))
            ]
        ).numpy()


def train_mlp(x, y, xv, yv, cfg):
    import copy

    import torch
    from sklearn.metrics import average_precision_score

    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(4)
    net = make_mlp(x.shape[1])
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x.astype(np.float32)), torch.tensor(y, dtype=torch.float32)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=True, generator=torch.Generator().manual_seed(cfg["seed"])
    )
    best_score, best, stale, history = -1.0, None, 0, []
    for epoch in range(cfg["epochs"]):
        net.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb).flatten(), yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scores = mlp_scores(net.state_dict(), xv)
        ap = float(average_precision_score(yv, scores))
        history.append(
            {"epoch": epoch + 1, "loss": float(np.mean(losses)), "validation_pr_auc": ap}
        )
        print(f"MLP epoch {epoch + 1}: validation PR-AUC {ap:.4f}", flush=True)
        if ap > best_score:
            best_score, best, stale = ap, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= cfg["early_stopping_patience"]:
                break
    return best, history


class RelationPredictor:
    """Score the hypothesis 'candidate should be read before target'.

    The current artifact is trained on weak references, not human read_before labels;
    its scores are uncalibrated proxies for that target event.
    """

    def __init__(self, model_dir=None):
        import joblib

        model_dir = Path(model_dir or ROOT / "models")
        self.metadata = json.loads((model_dir / "metadata.json").read_text())
        from src.reproducibility import verify_hashes

        required = {
            "model_type",
            "encoder",
            "encoder_revision",
            "threshold",
            "thresholds",
            "chunk_tokens",
        }
        if not required <= self.metadata.keys() or self.metadata["model_type"] not in {
            "mlp",
            "logistic_regression",
        }:
            raise ValueError("Invalid model metadata")
        if not 0 <= self.metadata["threshold"] <= 1:
            raise ValueError("Invalid model threshold")
        verify_hashes(
            model_dir,
            {
                k: v
                for k, v in self.metadata.get("artifact_hashes", {}).items()
                if k in {"relation_model.pkl", "scaler.pkl"}
            },
        )
        self.artifact = joblib.load(model_dir / "relation_model.pkl")
        self.encoder = None

    def predict(self, doc, min_confidence=None):
        import pandas as pd

        from scripts.prepare_data import article_records

        threshold = max(self.metadata["threshold"], min_confidence or 0.0)
        records = article_records(doc)
        if len(records) < 2:
            return []
        if self.encoder is None:
            self.encoder = E5Encoder(
                self.metadata["encoder"],
                self.metadata["chunk_tokens"],
                revision=self.metadata["encoder_revision"],
            )
        vectors = self.encoder.encode(records)
        articles = {r["id"]: r for r in records}
        embeddings = dict(zip(articles, vectors))
        explicit = {(e.source, e.target) for e in doc.edges if e.edge_type == "REFERENCES"}
        result = []
        for source in articles:
            pairs = pd.DataFrame(
                [
                    {"source": source, "target": target}
                    for target in articles
                    if target != source and (source, target) not in explicit
                ]
            )
            if pairs.empty:
                continue
            full = self.metadata["model_type"] == "mlp"
            x = feature_matrix(pairs, articles, embeddings, structural=full)
            x = self.artifact["scaler"].transform(x).astype(np.float32)
            scores = (
                mlp_scores(self.artifact["state"], x)
                if full
                else self.artifact["classifier"].predict_proba(x)[:, 1]
            )
            for row, score in zip(pairs.itertuples(), scores):
                if score >= threshold:
                    result.append(
                        LegalEdge(
                            row.target,
                            row.source,
                            "READ_BEFORE",
                            float(score),
                            "ml",
                            {
                                "model_type": self.metadata["model_type"],
                                "cosine": float(embeddings[row.source] @ embeddings[row.target]),
                                "threshold": threshold,
                                "score_semantics": "uncalibrated read-before hypothesis; weak-reference training",
                            },
                            reason=f"Mô hình đề xuất đọc Điều {articles[row.target]['number']} trước Điều {articles[row.source]['number']} với điểm {float(score):.3f}; chưa có human-gold calibration.",
                        )
                    )
        return result
