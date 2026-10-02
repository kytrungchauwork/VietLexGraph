# VietLexGraph third-party notices

Project code and original synthetic sample: MIT, © 2026 VietLexGraph contributors; see LICENSE. Upstream implementations are installed dependencies, not copied source. Preserve their license files when distributing an environment or container.

| Component | Declared license |
|---|---|
| UTS_VLC dataset compilation / multilingual-e5-small | MIT per pinned cards; separate downloads |
| NetworkX, pandas, scikit-learn, joblib | BSD-3-Clause |
| Streamlit, Transformers, Datasets, PyArrow, SentencePiece | Apache-2.0 |
| PyYAML, Plotly, pytest | MIT |
| NumPy, PyTorch wheels | Multiple bundled licenses; retain complete installed notices |

Installed license metadata was inspected during development. Detailed environment inventories are local and are not published because the public release excludes internal experiment artifacts. Direct dependency versions are pinned in the requirements files. Docker uses the smaller `requirements-app.txt` set. Preserve complete wheel notices, especially for NumPy and PyTorch bundled components. License metadata is not a legal opinion; no corpus or pretrained encoder binaries are distributed in Git. The separate inference-model release includes the project MIT LICENSE and only trained classifier/scaler parameters.

## Installed direct dependencies

| Component | Version | Purpose | License metadata | Source | Redistributed? |
|---|---|---|---|---|---|
| PyYAML | 6.0.3 | Runtime / training | MIT License | [Upstream](https://pyyaml.org/) | Source archive: no; deployment installs wheels |
| datasets | 5.0.1 | Runtime / training | Apache Software License | [Upstream](https://github.com/huggingface/datasets) | Source archive: no; deployment installs wheels |
| joblib | 1.6.0 | Runtime / training | BSD-3-Clause | [Upstream](https://joblib.readthedocs.io) | Source archive: no; deployment installs wheels |
| networkx | 3.7 | Runtime / training | BSD-3-Clause | [Upstream](https://networkx.org/) | Source archive: no; deployment installs wheels |
| numpy | 2.5.3 | Runtime / training | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 | [Upstream](https://numpy.org) | Source archive: no; deployment installs wheels |
| pandas | 2.3.3 | Runtime / training | BSD License | [Upstream](https://pandas.pydata.org) | Source archive: no; deployment installs wheels |
| plotly | 6.9.0 | Runtime / training | MIT | [Upstream](https://plotly.com/python/) | Source archive: no; deployment installs wheels |
| pyarrow | 23.0.1 | Runtime / training | Apache-2.0 | [Upstream](https://arrow.apache.org/) | Source archive: no; deployment installs wheels |
| pytest | 9.1.1 | Testing | MIT | [Upstream](https://docs.pytest.org/en/latest/) | Source archive: no; deployment installs wheels |
| scikit-learn | 1.9.1 | Runtime / training | BSD-3-Clause | [Upstream](https://scikit-learn.org) | Source archive: no; deployment installs wheels |
| sentencepiece | 0.2.2 | Runtime / training | Apache-2.0 | [Upstream](https://github.com/google/sentencepiece) | Source archive: no; deployment installs wheels |
| streamlit | 1.64.0 | Runtime / training | Apache-2.0 | [Upstream](https://streamlit.io) | Source archive: no; deployment installs wheels |
| torch | 2.14.1 | Runtime / training | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT | [Upstream](https://pytorch.org) | Source archive: no; deployment installs wheels |
| transformers | 5.18.0 | Runtime / training | Apache 2.0 License | [Upstream](https://github.com/huggingface/transformers) | Source archive: no; deployment installs wheels |
