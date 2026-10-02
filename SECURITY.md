# Security policy

VietLexGraph v1.0.0 is an experimental local application, not an authenticated public legal-data service. Do not deploy sensitive input to an untrusted host. Text is parsed locally; public upstream weights/corpus files are downloaded only for the training/ML workflow.

## Reporting

Report vulnerabilities privately through [GitHub private vulnerability reporting](https://github.com/kytrungchauwork/VietLexGraph/security/advisories/new). If that channel is unavailable, contact the repository owner through a verified private channel. Do not post credentials or confidential legal text in public issues.

## Boundaries

- Never load untrusted pickle/joblib checkpoints: they can execute code. The UI accepts text uploads, not model files.
- SHA-256 checks detect file changes; they are not signatures and cannot prevent replacement of both a file and its manifest.
- The encoder uses pinned revisions, safetensors and `trust_remote_code=False`. No generative API token is needed.
- Internal datasets, annotations, checkpoints, reports and credentials are excluded from the public source allowlist. See `PUBLIC_FILES.txt` and `scripts/check_public_release.py`.
- Docker runs as a non-root user and copies only runtime source plus a synthetic sample. Very large inputs may consume substantial memory/CPU; pair scoring is quadratic.
- Dependabot tracks package, Docker and Actions updates. Review updates with tests before merging.

A clean dependency advisory scan is not proof that software has no vulnerabilities. Run your own dependency/security assessment before any deployment beyond the local research demo.
