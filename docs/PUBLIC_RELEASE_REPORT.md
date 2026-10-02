# Public release report

- **Version:** v1.0.0.
- **Status:** SOFTWARE READY / EXPLORATORY EVALUATION AVAILABLE.
- **Repository:** https://github.com/kytrungchauwork/VietLexGraph.
- **Audited code commit:** recorded after staging verification below.
- **Release commit:** resolve the immutable release reference with `git rev-parse v1.0.0^{commit}`. The report's own documentation commit may follow the audited code commit.

## Distribution

Git contains only the exact source/public-documentation allowlist. Internal `data/`, `models/`, `artifacts/`, `private/`, caches, annotations, environment secrets and historical experiment archives are excluded. The original local workspace remains intact apart from organizing historical reports under ignored `private/reports/`.

The inference artifact is separate: [v1.0.0 release](https://github.com/kytrungchauwork/VietLexGraph/releases/tag/v1.0.0), `vietlexgraph-model-v1.zip` and `SHA256SUMS`. Its archive contains only classifier/scaler parameters, sanitized metadata, LICENSE and model README. No examples, corpus text, embeddings, annotation records, private identifiers or TF-IDF vocabulary are included.

Model SHA-256: `936f0f6b76a862b2f2e8937e62827e83d158062eecc2c359f6987443d999319b`.

## Reproduction

```bash
git clone https://github.com/kytrungchauwork/VietLexGraph.git
cd VietLexGraph
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_model.py  # optional; deterministic sample works without it
streamlit run app.py
```

The lighter demo uses `requirements-app.txt`. CI uses `requirements-test.txt`, without downloading an encoder or training a model. `requirements-dev.txt` also includes full training and audit tooling. The synthetic sample reads 3 → 4 → 5 → 6 for target article6 and displays reasons, evidence and confidence.

For your own local corpus experiment, run preparation, training and evaluation as documented in the README. Evaluation scripts without required local inputs report NOT RUN explicitly and do not fabricate scores. The private AI-assisted reviewed assessment is not reproducible from public source alone because its annotations are intentionally excluded.

## Validation and scope

Publication checks inspect tracked paths against an exact allowlist, common secret patterns, symlinks, unexpected binaries and size limits. Local unit/app tests, formatting, lint, source-only demo and Docker checks are run before publication. The actual hosted CI result is visible in Actions; consult the release commit's workflow rather than assuming local tests certify the remote run.

The evaluated corpus is limited to30 documents/2,502 articles. The reviewed assessment is AI-assisted, user-reviewed and exploratory. No independent human-only prerequisite validation, production readiness, authoritative legal interpretation or research superiority is claimed. See [limitations](LIMITATIONS.md), [evaluation](evaluation.md), and [related work](RELATED_WORK.md).

## License notes

An actual MIT `LICENSE` is included in both source and inference artifact. Project code, original synthetic fixtures and project-trained inference parameters are MIT. Downloaded dataset/model assets retain their upstream licenses and notices. Third-party dependency metadata is documented in [notices](../THIRD_PARTY_NOTICES.md); bundled wheel licenses must be preserved. No absolute legal clearance claim is made.
