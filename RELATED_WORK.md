# Related Work

VietLexGraph is a research prototype for **target-specific prerequisite reading guidance within one Vietnamese legal document**. Existing work already covers legal knowledge representation, hierarchy parsing, citation networks and legal retrieval. This project does not claim to invent these techniques or establish superiority over the works below.

| Work | What it does | VietLexGraph task distinction |
|---|---|---|
| [Graphie: a network-based visual interface for UK primary legislation](https://f1000research.com/articles/12-236/v1) | Supports exploration of legislation through network visualization and cross-reference navigation. | Synthesizes a bounded reading sequence for a selected provision, with reasons for proposed earlier reading. Graph exploration remains supporting UI. |
| [Constructing a Knowledge Graph for Vietnamese Legal Cases with Heterogeneous Graphs](https://arxiv.org/abs/2309.09069) | Organizes Vietnamese legal cases and laws in a heterogeneous graph and explores relevant-law recommendation. | Focuses on article-level reading usefulness and order inside one document rather than a case/law knowledge base. |
| [LexPath: A domain-oriented multi-path framework for legal article retrieval](https://arxiv.org/abs/2605.30205) | Retrieves and reranks legal articles for a query using domain-oriented retrieval and intent-aware reranking. | Takes a target article, not a free-text query, and proposes prerequisite articles followed by an ordered reading path. |

These are distinctions in input, output and evaluation target, not claims that other systems lack every overlapping capability. No head-to-head benchmark against these systems has been performed.

VietLexGraph's intended implementation contributions are: separating extracted evidence from derived READ_BEFORE recommendations; combining rule and optional learned signals with provenance; and explaining a compact ordered reading path. The current AI-assisted reviewed evaluation is exploratory and does not establish independent human-only validation or broad research novelty.

The cited LexPath is a separate third-party project. VietLexGraph's parser, graph logic, explanation templates, synthetic sample and Mermaid diagram are project code/assets; external libraries and pretrained models are acknowledged in [third-party notices](THIRD_PARTY_NOTICES.md).
