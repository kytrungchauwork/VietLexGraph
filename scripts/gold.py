"""Prepare unlabeled complete candidate pools and freeze reviewed document splits with annotation provenance."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = [
    "document_id",
    "dataset_revision",
    "document_sha256",
    "target_article",
    "candidate_article",
    "target_text_sha256",
    "candidate_text_sha256",
    "explicit_reference",
    "contextually_helpful",
    "read_before",
    "reason",
    "annotator",
    "annotation_version",
]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate(rows):
    seen = set()
    for row in rows:
        key = (row["document_id"], row["target_article"], row["candidate_article"])
        if key in seen or key[1] == key[2]:
            raise ValueError("Duplicate or self pair")
        seen.add(key)
        if row["read_before"] not in {"0", "1"}:
            raise ValueError("Adjudicate UNCERTAIN / missing labels before freezing")
        if row["explicit_reference"] not in {"0", "1"} or row["contextually_helpful"] not in {
            "0",
            "1",
            "",
        }:
            raise ValueError("Invalid binary label")
        for field in ("reason", "annotator", "annotation_version", "dataset_revision"):
            if not row[field].strip():
                raise ValueError(f"Missing {field}")
        for field in ("document_sha256", "target_text_sha256", "candidate_text_sha256"):
            if len(row[field]) != 64 or any(c not in "0123456789abcdef" for c in row[field]):
                raise ValueError(f"Invalid {field}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "freeze"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--test-documents", nargs="+")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; use a new version directory")
    docs = [
        json.loads(line)
        for line in (ROOT / "data/processed_documents.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((ROOT / "data/split_manifest.json").read_text())
    if args.action == "prepare":
        # Four smallest eligible documents; whole candidate pools, no fabricated judgments.
        eligible = sorted(
            (
                d
                for d in docs
                if 20 <= len(d["articles"]) <= 125 and d["id"] not in manifest["splits"]["train"]
            ),
            key=lambda d: (len(d["articles"]), d["id"]),
        )[:4]
        rows, texts = [], {}
        for doc in eligible:
            for target in (doc["articles"][len(doc["articles"]) // 2], doc["articles"][-1]):
                refs = {(r["source"], r["target"]) for r in doc["references"]}
                texts[doc["id"]] = doc
                for candidate in doc["articles"]:
                    if candidate["id"] == target["id"]:
                        continue
                    rows.append(
                        dict(
                            zip(
                                FIELDS,
                                [
                                    doc["id"],
                                    manifest["revision"],
                                    doc["sha256"],
                                    target["id"],
                                    candidate["id"],
                                    digest(target["text"]),
                                    digest(candidate["text"]),
                                    str(int((target["id"], candidate["id"]) in refs)),
                                    "",
                                    "",
                                    "",
                                    "",
                                    "",
                                ],
                            )
                        )
                    )
        args.output.mkdir(parents=True)
        with (args.output / "annotations.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        (args.output / "document_context.json").write_text(
            json.dumps(texts, ensure_ascii=False, indent=2)
        )
        print(
            f"Prepared {len(rows)} UNLABELED pairs across {len(eligible)} documents; add more complete target pools to reach 200–500 if needed."
        )
    else:
        if not args.annotations or not args.test_documents:
            parser.error("freeze requires --annotations and --test-documents")
        with args.annotations.open() as stream:
            rows = validate(list(csv.DictReader(stream)))
        if not rows:
            raise ValueError("No reviewed annotations")
        provenance_path = args.annotations.parent / "annotation_provenance.json"
        provenance = {"annotation_method": "unspecified", "review": {"status": "unspecified"}}
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if (
                provenance.get("annotation_sha256")
                != hashlib.sha256(args.annotations.read_bytes()).hexdigest()
            ):
                raise ValueError("Annotation provenance checksum mismatch")
        known = {d["id"]: d for d in docs}
        pools = {}
        for row in rows:
            doc = known[row["document_id"]]
            if doc["id"] in manifest["splits"]["train"]:
                raise ValueError("Gold evaluation cannot overlap weak-training documents")
            articles = {a["id"]: a for a in doc["articles"]}
            if (
                row["document_sha256"] != doc["sha256"]
                or row["dataset_revision"] != manifest["revision"]
            ):
                raise ValueError("Document revision/hash mismatch")
            for side in ("target", "candidate"):
                if row[f"{side}_text_sha256"] != digest(articles[row[f"{side}_article"]]["text"]):
                    raise ValueError("Article text hash mismatch")
            pools.setdefault((doc["id"], row["target_article"]), set()).add(
                row["candidate_article"]
            )
        for (doc_id, target), candidates in pools.items():
            if candidates != {a["id"] for a in known[doc_id]["articles"]} - {target}:
                raise ValueError(
                    "Path evaluation requires complete candidate pools; adjudicate missing/UNCERTAIN pairs"
                )
        test_ids = set(args.test_documents)
        split = {
            "gold-test": [r for r in rows if r["document_id"] in test_ids],
            "gold-dev": [r for r in rows if r["document_id"] not in test_ids],
        }
        if any(not part for part in split.values()):
            raise ValueError("Both document-disjoint dev and test must be nonempty")
        args.output.mkdir(parents=True)
        hashes = {}
        for name, part in split.items():
            path = args.output / f"{name}.csv"
            with path.open("w") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(part)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "files": hashes,
                    "annotation_provenance": provenance,
                    "counts": {k: len(v) for k, v in split.items()},
                    "test_documents": sorted(test_ids),
                    "annotation_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
                    "annotators": sorted({r["annotator"] for r in rows}),
                    "annotation_versions": sorted({r["annotation_version"] for r in rows}),
                    "label_distribution": {
                        name: {
                            label: sum(r["read_before"] == label for r in part)
                            for label in ("0", "1")
                        }
                        for name, part in split.items()
                    },
                    "protocol": "Frozen by hash; never tune on test. Filesystem is not write-once storage.",
                },
                indent=2,
            )
        )
        print("Frozen document-disjoint gold-dev / gold-test. Archive manifest independently.")


if __name__ == "__main__":
    main()
