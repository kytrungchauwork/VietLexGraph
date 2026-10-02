# Exploratory evaluation

**SOFTWARE READY / EXPLORATORY EVALUATION AVAILABLE**

This release is a research prototype evaluated on a limited corpus subset. The tables below are aggregate results from local experiments. Internal data, reviewed annotations, document identifiers, manifests and per-pair outputs are deliberately not published.

## Weak-supervision pair scoring

Positive proxy labels come from explicit intra-document references; non-citations are not human-verified negatives. Citation spans in body and title are masked for the main experiment. Training/scalers use training documents; thresholds use validation only. PR-AUC below denotes average precision.

| Model | Precision | Recall | F1 | PR-AUC | MRR |
|---|---:|---:|---:|---:|---:|
| TF-IDF cosine | 0.0000 | 0.0000 | 0.0000 | 0.0750 | 0.3397 |
| E5 cosine | 0.1111 | 0.0112 | 0.0203 | 0.0894 | 0.3696 |
| E5 + Logistic Regression | 0.0353 | 0.1788 | 0.0589 | 0.0368 | 0.1727 |
| E5 + structural MLP | 0.0441 | 0.2626 | 0.0756 | 0.0534 | 0.2504 |

Neither learned classifier achieved validation precision 0.85. LR remains the preliminary proxy artifact under the predeclared validation gate; stronger MLP test ranking did not override selection. These metrics measure reference recovery, not prerequisite usefulness.

A fixed-C LR ablation compared visible/masked citations and four feature groups. E5-only PR-AUC was 0.0423 visible versus 0.0368 masked; masked additional structure, term features and all features gave 0.0339, 0.0348 and 0.0319 respectively. This reveals sensitivity and no consistent feature-expansion benefit; masking does not prove all shortcuts are eliminated.

## AI-assisted reviewed evaluation

The evaluated set contains **344 pairs: 54 positive and 290 negative**, covering four documents and eight targets. Labels were AI-generated and reviewed by the project user. Review qualifications and detailed adjudication were not independently documented; no inter-annotator agreement or independent human-only assessment is claimed.

Development has 160 pairs (27 positive / 133 negative); test has 184 (27 / 157). Each split has two documents and four targets. Splits reuse the original weak document partition, with no weak-training overlap. Complete candidate pools allow unjudged articles to remain distinct from negative judgments. All six systems used the existing checkpoint, threshold and reading settings without tuning on these labels. The texts and earlier operational outputs had been inspected before this assessment; it is exploratory, not a pristine confirmatory holdout.

All systems end in the target. DocumentOrder uses the preceding window of 5. SemanticTopK takes masked E5 top-5 candidates and outputs them in document order. CitationOnly recursively follows citations; DefinitionCitation adds definition-use links; HybridNoML uses conservative reading rules; FullSystem adds learned support. Graph paths share depth/budget 5 and rule weights 0.95/0.85.

### Test results

Macro means across four targets; path length excludes the target, and extra cost counts recommended articles labeled unhelpful.

| System | Path precision | Path recall | Path F1 | Extra articles / target | Path length |
|---|---:|---:|---:|---:|---:|
| DocumentOrder | 0.1500 | 0.5300 | 0.0500 | 4.25 | 5.00 |
| CitationOnly | 0.5000 | 0.5000 | 0.5000 | 0.25 | 0.25 |
| DefinitionCitation | 0.7083 | 0.6350 | 0.6429 | 0.75 | 1.25 |
| SemanticTopK | 0.2500 | 0.6650 | 0.1381 | 3.75 | 5.00 |
| HybridNoML | 0.7083 | 0.6350 | 0.6429 | 0.75 | 1.25 |
| FullSystem | 0.2000 | 0.6550 | 0.1214 | 4.00 | 5.00 |

F1 is averaged per target, not recomputed from macro precision/recall. Empty-gold recall is 1; an empty path scores precision 1 only if the target also has no positive prerequisites. **Two test targets are empty**, so CitationOnly gets macro F1 0.5 while recovering zero labeled prerequisites. The macro table must not be interpreted alone.

### Sensitivity and cost

| System | F1 on positive-prerequisite targets only | Useful / all recommendations | Micro recall |
|---|---:|---:|---:|
| DocumentOrder | 0.1000 | 3 / 20 | 0.1111 |
| CitationOnly | 0.0000 | 0 / 1 | 0.0000 |
| DefinitionCitation | 0.2857 | 2 / 5 | 0.0741 |
| SemanticTopK | 0.2762 | 5 / 20 | 0.1852 |
| HybridNoML | 0.2857 | 2 / 5 | 0.0741 |
| FullSystem | 0.2429 | 4 / 20 | 0.1481 |

Adding ML recovers two additional labeled prerequisites but adds 13 labeled-unhelpful recommendations. Mean extra cost rises from 0.75 to 4.00; keep the existing ML-off default. DefinitionCitation and HybridNoML coincide on this small sample, so it does not distinguish their formal-citation filtering.

One test target has 25 positive prerequisite labels, exceeding the budget of 5 and imposing a recall ceiling of 0.20 for that target. Future assessment should predeclare budget curves and consider necessity/priority tiers on development data. Do not adjust the budget after seeing test merely to raise scores.

There are no ordering annotations, so no order-accuracy result is reported. Two documents per split cannot establish statistical significance, stable superiority or broad generalization. Independent human-only assessment and any comprehension/time user study remain incomplete.

## Reproduction boundaries

`scripts/evaluate.py` measures weak-reference recovery; `scripts/evaluate_paths.py` supports both operational statistics and reviewed labels. `scripts/gold.py` validates complete candidate pools and freezes document-disjoint splits with checksums. Its historical filename and `gold-dev`/`gold-test` CLI identifiers are compatibility names, not a claim that AI-assisted labels are independent human-only evidence. Supply your own licensed, reviewed local inputs to reproduce the protocol; no private annotation files are included in this repository.
