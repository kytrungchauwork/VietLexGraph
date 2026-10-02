"""Serializable schemas; raw references point from citing to cited article."""

from dataclasses import dataclass, field
from typing import Any

NODE_TYPES = {
    "DOCUMENT",
    "PART",
    "CHAPTER",
    "SECTION",
    "SUBSECTION",
    "ARTICLE",
    "CLAUSE",
    "POINT",
    "TERM",
}


@dataclass
class LegalNode:
    id: str
    type: str
    number: str = ""
    title: str = ""
    text: str = ""
    parent_id: str | None = None
    order: int = 0
    body: str = ""
    role: str = "UNKNOWN"


@dataclass
class LegalEdge:
    source: str
    target: str
    edge_type: str
    confidence: float = 1.0
    origin: str = "deterministic"
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self):
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be in [0, 1]")
        if self.edge_type == "READ_BEFORE":
            if (
                self.origin
                not in {"ml", "deterministic_reference_rule", "deterministic_definition_rule"}
                or not self.evidence
                or not self.reason
            ):
                raise ValueError("READ_BEFORE requires derived provenance, evidence and reason")
        elif self.edge_type == "PREDICTED_RELATED":
            if self.origin != "ml":
                raise ValueError("Predictions require ML provenance")
        elif self.origin != "deterministic" or self.confidence != 1:
            raise ValueError("Deterministic edges require confidence 1")


@dataclass
class LegalDocument:
    id: str
    text: str
    nodes: list[LegalNode]
    edges: list[LegalEdge] = field(default_factory=list)
    unresolved_references: list[dict] = field(default_factory=list)

    @property
    def articles(self):
        return [n for n in self.nodes if n.type == "ARTICLE"]

    @property
    def by_id(self):
        return {n.id: n for n in self.nodes}
