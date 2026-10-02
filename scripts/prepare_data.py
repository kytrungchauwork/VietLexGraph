"""Download UTS_VLC and create document-disjoint weak supervision."""

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.parser import parse_document
from src.relations import extract_relations


def clean_corpus_markdown(text):
    # Corpus transport cleanup only; the application accepts structured plain text.
    text = re.sub(r"^\s*#{1,6}\s+", "", text, flags=re.M)
    return text.replace("**", "").replace("__", "")


def split_documents(ids, seed=42):
    ids = sorted(set(ids))
    if len(ids) < 7:
        raise ValueError("At least seven distinct usable documents are required")
    random.Random(seed).shuffle(ids)
    ntrain, nval = int(0.7 * len(ids)), max(1, int(0.15 * len(ids)))
    return {"train": ids[:ntrain], "val": ids[ntrain : ntrain + nval], "test": ids[ntrain + nval :]}


def article_records(doc):
    by_id = doc.by_id
    records = []
    for article in doc.articles:
        ancestors = {}
        current = article
        while current.parent_id:
            current = by_id[current.parent_id]
            ancestors[current.type] = current.id
        terms = sorted(
            {
                e.evidence["term"]
                for e in doc.edges
                if e.source == article.id and e.edge_type in ("USES_TERM", "DEFINES")
            }
        )
        record = asdict(article)
        record.update(
            document_id=doc.id,
            chapter=ancestors.get("CHAPTER"),
            section=ancestors.get("SECTION"),
            terms=terms,
        )
        records.append(record)
    return records


def make_pairs(doc, ratio=3):
    from sklearn.feature_extraction.text import TfidfVectorizer

    articles = article_records(doc)
    positions = {a["id"]: i for i, a in enumerate(articles)}
    refs = {}
    for edge in doc.edges:
        if edge.edge_type == "REFERENCES":
            refs.setdefault(edge.source, set()).add(edge.target)
    matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([a["text"] for a in articles])
    rows = []
    for source in sorted(refs):
        positive = refs[source]
        a = articles[positions[source]]
        similarities = (matrix[positions[source]] @ matrix.T).toarray()[0]
        ranked = []
        for b in articles:
            if b["id"] == source or b["id"] in positive:
                continue
            same = bool(a["chapter"] and a["chapter"] == b["chapter"])
            proximity = 1 / (1 + min(abs(positions[b["id"]] - positions[t]) for t in positive))
            shared = len(set(a["terms"]) & set(b["terms"]))
            score = (
                float(similarities[positions[b["id"]]])
                + 0.3 * same
                + 0.3 * proximity
                + 0.1 * min(shared, 3)
            )
            ranked.append((score, b["id"]))
        negatives = [
            bid for _, bid in sorted(ranked, key=lambda x: (-x[0], x[1]))[: ratio * len(positive)]
        ]
        for label, targets in ((1, sorted(positive)), (0, negatives)):
            for target in targets:
                rows.append(
                    {"document_id": doc.id, "source": source, "target": target, "label": label}
                )
    return rows


def main():
    import pandas as pd
    import yaml
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Reproducible smaller experiment; default: all documents",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Dataset commit; saved resolved commit makes subsequent runs reproducible",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    revision = args.revision or cfg["dataset_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Dataset revision must be a full commit hash")
    dataset = load_dataset(
        cfg["dataset"],
        split=cfg["dataset_split"],
        revision=revision,
        cache_dir=str(ROOT / ".cache" / "datasets"),
    )
    rows = list(dataset)
    random.Random(cfg["seed"]).shuffle(rows)
    docs, pairs, skipped, hashes = [], {}, [], set()
    for row in rows:
        text = clean_corpus_markdown(row["content"])
        if re.search(r"sửa đổi,?\s*(?:và\s*)?bổ sung", row["title"], re.I):
            skipped.append(
                {"id": row["id"], "reason": "amending instrument with quoted provisions"}
            )
            continue
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in hashes:
            skipped.append({"id": row["id"], "reason": "duplicate text"})
            continue
        hashes.add(digest)
        try:
            doc = extract_relations(parse_document(text, row["id"]))
            if len(doc.articles) < 10 and re.search(r"sửa đổi|bổ sung", row["title"], re.I):
                raise ValueError("Short amending instrument; ambiguous quoted scope")
            numbers = [a.number for a in doc.articles]
            if len(numbers) != len(set(numbers)):
                raise ValueError("Duplicate article numbers; often quoted amendments")
            generated = make_pairs(doc, cfg["negative_ratio"])
            if not generated or len({r["label"] for r in generated}) < 2:
                raise ValueError("No usable positive and negative pairs")
        except ValueError as exc:
            skipped.append({"id": row["id"], "reason": str(exc)})
            continue
        docs.append(
            {
                "id": doc.id,
                "title": row["title"],
                "sha256": digest,
                "articles": article_records(doc),
                "references": [asdict(e) for e in doc.edges if e.edge_type == "REFERENCES"],
                "unresolved_count": len(doc.unresolved_references),
            }
        )
        pairs[doc.id] = generated
        print(
            f"{len(docs)} documents: {doc.id}, {len(doc.articles)} articles, {len(generated)} pairs",
            flush=True,
        )
        if args.max_documents and len(docs) >= args.max_documents:
            break
    split = split_documents([d["id"] for d in docs], cfg["seed"])
    output = ROOT / "data"
    output.mkdir(exist_ok=True)
    (output / "processed_documents.jsonl").write_text(
        "".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8"
    )
    for name, ids in split.items():
        pd.DataFrame([p for did in ids for p in pairs[did]]).to_parquet(
            output / f"{name}_pairs.parquet", index=False
        )
    manifest = {
        "dataset": cfg["dataset"],
        "dataset_split": cfg["dataset_split"],
        "revision": revision,
        "seed": cfg["seed"],
        "max_documents": args.max_documents,
        "splits": split,
        "hashes": {d["id"]: d["sha256"] for d in docs},
        "skipped": skipped,
        "negative_mining": "document-local TF-IDF similarity + chapter + proximity to positive + defined terms",
        "counts": {
            name: {"documents": len(ids), "pairs": sum(len(pairs[i]) for i in ids)}
            for name, ids in split.items()
        },
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    from src.reproducibility import sha256, source_provenance

    manifest["file_hashes"] = {
        p.name: sha256(p)
        for p in [output / "processed_documents.jsonl", *sorted(output.glob("*_pairs.parquet"))]
    }
    manifest["preparation_code"] = source_provenance()
    for name in split:
        frame = pd.read_parquet(output / f"{name}_pairs.parquet")
        manifest["counts"][name].update(
            positive=int(frame.label.sum()),
            negative=int((frame.label == 0).sum()),
            positive_pct=float(frame.label.mean() * 100),
        )
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
