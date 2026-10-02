# Public source release

**SOFTWARE READY / EXPLORATORY EVALUATION AVAILABLE**

The release is an experimental engineering prototype, not a production legal service or an independently human-validated research benchmark. See the aggregate [evaluation](evaluation.md) and [related work](../RELATED_WORK.md).

## What is published

`PUBLIC_FILES.txt` is an exact allowlist of source, tests, configuration, public documentation, MIT LICENSE and original synthetic sample/screenshots. Internal datasets, reviewed annotations, annotation provenance, checkpoints, experiment reports, manifests, caches, archives and credentials are excluded. No existing local Git history is imported. A separately reviewed release asset contains only inference parameters and sanitized metadata, protected by a pinned checksum.

`python scripts/check_public_release.py` verifies that tracked files exactly match this allowlist, rejects private paths, symlinks, unexpected binary/large files and common secret patterns, and checks that LICENSE exists. This is a guardrail, not a guarantee against every possible secret or content leak. Screenshots were taken using the original synthetic sample.

For a new sanitized copy, run:

```bash
python scripts/check_public_release.py --export /tmp/vietlexgraph-public-source
```

The destination must not already exist. This copies allowlisted files only. Review the staged diff and tracked-file list before pushing; never use force-add for ignored local research outputs.

## Verification checklist

- Exact tracked-file allowlist and actual MIT LICENSE.
- Source-only checkout runs tests without private data, annotations or checkpoint files.
- Lint, formatting and dependency consistency checks pass.
- Deterministic Streamlit sample runs without a model or credentials.
- Docker includes the downstream explanation module and runs as a non-root user.
- Public documentation uses AI-assisted reviewed evaluation and discloses the limited corpus.
- Related-work sources are attributed; no novelty or superiority claim is asserted.
- Raw annotation metrics and internal reports remain local. Aggregate results are labeled exploratory.

CI repeats publication checks, lint and tests. Hosted CI status is visible in the repository's Actions tab; local success alone is not a claim that a remote workflow has passed. Independent human-only assessment, a larger fresh holdout and any user study remain future work.
