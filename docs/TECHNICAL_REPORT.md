# Architecture and design rationale

Input is one structured Vietnamese legal document and one target article. Output is a unique, bounded recommended reading path ending at the target, with evidence-grounded explanations. This release is a research prototype evaluated on a limited corpus subset.

## Layers

1. **Parser:** NFC-normalized hierarchy with article text preserved. Supports legal heading markers, clauses and points; ambiguous quoted amendments remain a limitation.
2. **Evidence Graph:** observed CONTAINS, REFERENCES, DEFINES and USES_TERM relations. REFERENCES points from citing to cited article. Unresolved/external references are logged rather than fabricated.
3. **Reading Graph:** existing evidence produces derived READ_BEFORE candidates; conservative cues can reject formal/amendment references. Optional ML adds predicted support. Edges retain origin, confidence and evidence.
4. **Reading Path:** SCC handling, DAG ordering, tie breakers, depth and connected-route budget pruning. Default depth/budget is 5. Cycles and omitted candidates are disclosed; pruning is a heuristic, not proven optimal minimal reading.
5. **Explanation Layer:** `src/explanation/reading_explainer.py` explains existing forward edges between selected steps. It preserves path membership and order. `annotate_reading_graph` enriches a copy, leaving the original graph unchanged.

Explanation priority is definition, explicit reference, procedure, structural context, then ML prediction. General-to-specific and procedure templates require an existing reading edge and appropriate metadata. The current extractor does not emit GENERAL_RULE/SPECIFIC_RULE; their template is available when those roles are supplied downstream. Target selection and cycle steps without forward support have an explicit reason and confidence N/A.

## Why these choices?

| Decision | Rationale |
|---|---|
| Deterministic extraction | Legal markers and explicit citations have auditable text evidence; no generative model is needed for V1. |
| Document-level splits | Article pairs share drafting context; pair-random splitting can leak surrounding information. |
| Hard negatives | Similar, nearby uncited provisions challenge topic matching, although they can still be useful prerequisites. |
| Citation masking | Literal article numbers can expose the weak label; both body and title are masked. |
| Frozen E5 and lightweight models | Keeps the learning experiment manageable; more capacity is not automatically better on a small corpus. |
| Precision and extra reading cost | Unhelpful recommendations consume reading effort and can obscure the target. |
| Separate graph directions | Citing→cited is a textual observation; proposed earlier reading→target is a recommendation. |
| SCC handling | A cycle has no linear ordering satisfying every edge; grouping exposes that conflict. |
| Explicit score semantics | Rule weights and proxy-trained model outputs are not calibrated legal probabilities. |

## Scope and limits

No LLM, RAG, chatbot, PDF/OCR, legal QA or multi-document inference is included. Exact term matching misses paraphrases; formal-reference cues are narrow; weak non-citation negatives are not verified unhelpful provisions. Pair scoring is quadratic, mean chunk pooling can dilute local signals, and role labels are tie breakers rather than edge generators. AI-assisted reviewed evaluation does not replace independent human assessment.

## Evaluation and reproducibility

See [evaluation](evaluation.md) for the limited-corpus weak-supervision and AI-assisted reviewed results, [data sources](DATA_SOURCES.md) for pinned revisions, and [limitations](LIMITATIONS.md). Public release validation is recorded in [release report](PUBLIC_RELEASE_REPORT.md).
