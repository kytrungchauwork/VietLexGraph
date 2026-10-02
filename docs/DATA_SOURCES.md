# Data and model provenance

The public repository ships **source code and synthetic fixtures only**. Git history does not contain internal corpus files, annotations, annotation provenance, checkpoints, embeddings, split manifests or per-pair results. A separate versioned GitHub Release contains only the project-trained inference coefficients, scaler and sanitized metadata; see the README model section. All generated research outputs belong in ignored `data/`, `models/` and `artifacts/` directories.

| Resource | Source and pinned revision | Use | Redistribution here |
|---|---|---|---|
| UTS_VLC, maintained by Underthesea NLP | [Dataset repository](https://huggingface.co/datasets/undertheseanlp/UTS_VLC/tree/decbf8d7c0e1f18bdf1d4dcb429ff0fc322d4231), revision `decbf8d7c0e1f18bdf1d4dcb429ff0fc322d4231`, split `2026` | Public upstream legal-text source; card declares MIT | No corpus text or derived pair files |
| multilingual-e5-small, published by intfloat | [Model repository](https://huggingface.co/intfloat/multilingual-e5-small/tree/614241f622f53c4eeff9890bdc4f31cfecc418b3), revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` | Frozen encoder; card declares MIT | No weights or embeddings |
| Synthetic sample and tests | Original project fixtures | Deterministic demo and regression testing | Yes, under project MIT license |
| AI-assisted reviewed annotations | Local project assessment | Exploratory reading-path evaluation | No annotation files or per-pair outputs |

A dataset compilation license does not certify every underlying legal document's accuracy, current effect or rights. Consult official legal sources. Retain upstream notices if separately redistributing downloaded assets.

Preparation normalizes Unicode, removes Markdown transport markers, rejects detected amendment/duplicate-number documents and exact duplicate text, and splits by document with seed 42. Weak positives are resolved internal citations. Hard non-citation negatives use document-local lexical, structural and term signals; they are not verified useless prerequisites.

The measured subset contains 30 documents / 2,502 articles. Train/validation/test contain 21/4/5 documents and 3,674/308/706 sampled pairs, with 927/77/179 positives. The subset is a limited research prototype, not the full corpus or a production validation sample.

To generate your own local experiment, install full dependencies and run `python scripts/prepare_data.py --max-documents 30`, then train/evaluate. Without the limit, preparation uses all eligible documents and defines a different experiment. Download availability, upstream licenses and compute requirements still apply. The source code can be rerun; the private reviewed assessment cannot be reproduced solely from public files.
