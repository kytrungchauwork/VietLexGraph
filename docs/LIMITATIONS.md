# Limitations

This release is a research prototype evaluated on a limited corpus subset: 30 documents and 2,502 articles. It is not a production legal service.

- The AI-assisted reviewed assessment has only four documents and eight targets; no independent human-only prerequisite assessment or comprehension study is claimed.
- Weak citation supervision is biased: citations are not necessarily useful prerequisites, and uncited pairs can be useful.
- Both learned classifiers miss the validation precision target. ML is optional and default off; its scores are not calibrated legal probabilities.
- Exact term matching, narrow citation cues and structured-text parsing can miss or misinterpret complex legal drafting.
- The fixed budget can miss labeled prerequisites. Cycles cannot always be ordered consistently; pruning is not a globally optimal solution.
- Two test targets have no positive prerequisites, affecting macro averages. One has more positives than the reading budget allows. Read the sensitivity analysis in [evaluation](evaluation.md).
- Pair scoring is quadratic and E5 may be slow on CPU. Pretrained encoder download is separate from the lightweight classifier release.
- Source code and synthetic tests are public; private reviewed annotations and internal experiment artifacts are not. The private assessment is therefore not independently reproducible from this repository alone.

Verify legal content with official sources and qualified professionals. Recommended reading order is not legally required reading order.
