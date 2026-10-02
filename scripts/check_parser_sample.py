"""Measure a frozen article-header fixture; do not represent it as human review."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.parser import parse_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=Path, default=ROOT / "samples/sample_legal_text.txt")
    parser.add_argument("--gold", type=Path, default=ROOT / "samples/parser_headers.json")
    args = parser.parse_args()
    content = args.text.read_bytes()
    gold = json.loads(args.gold.read_text())
    if hashlib.sha256(content).hexdigest() != gold["text_sha256"]:
        raise ValueError("Fixture hash changed; review labels before updating the hash")
    actual = [(a.number, a.title) for a in parse_document(content.decode("utf-8")).articles]
    expected = [(a["number"], a["title"]) for a in gold["articles"]]
    from collections import Counter

    matched = sum((Counter(actual) & Counter(expected)).values())
    report = {
        "detected": len(actual),
        "expected": len(expected),
        "matched": matched,
        "precision": matched / len(actual) if actual else 0.0,
        "recall": matched / len(expected) if expected else 0.0,
        "annotation_provenance": gold["annotation_provenance"],
        "human_reviewed": gold["human_reviewed"],
        "scope": "six-header synthetic fixture only; not >=0.99 real-corpus/manual precision evidence",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
