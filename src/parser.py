"""NFC normalization and a hierarchy-aware legal structure state machine."""

import re
import unicodedata

from src.schemas import LegalDocument, LegalEdge, LegalNode

LEVELS = {
    "DOCUMENT": 0,
    "PART": 1,
    "CHAPTER": 2,
    "SECTION": 3,
    "SUBSECTION": 4,
    "ARTICLE": 5,
    "CLAUSE": 6,
    "POINT": 7,
}
HEADINGS = [
    ("PART", r"PHẦN\s+((?:THỨ\s+)?[\w]+)"),
    ("CHAPTER", r"CHƯƠNG\s+([IVXLCDM]+|\d+)"),
    ("SUBSECTION", r"TIỂU\s+MỤC\s+(\d+)"),
    ("SECTION", r"MỤC\s+(\d+)"),
    ("ARTICLE", r"ĐIỀU\s*(\d+[a-zđ]?)"),
    ("CLAUSE", r"(?:KHOẢN\s+)?(\d+)\s*[.)]"),
    ("POINT", r"(?:ĐIỂM\s+)?([a-zđ])\s*[.)]"),
]


def normalize_text(text):
    text = (
        unicodedata.normalize("NFC", text.lstrip("\ufeff"))
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )
    return "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in text.split("\n"))


def parse_document(text: str, document_id="document") -> LegalDocument:
    text = normalize_text(text)
    root = LegalNode(document_id, "DOCUMENT", text=text)
    nodes, stack = [root], [root]
    lines_by_id = {root.id: []}
    awaiting_title = None
    heading_end = {}
    inline_body = {}
    for line in text.splitlines():
        match = None
        for kind, pattern in HEADINGS:
            if kind in ("CLAUSE", "POINT") and not any(n.type == "ARTICLE" for n in stack):
                continue
            suffix = (
                r"\s*(.*)$" if kind in ("CLAUSE", "POINT") else r"(?=\s|[.:]|$)\s*[.:]?\s*(.*)$"
            )
            candidate = re.match(r"^" + pattern + suffix, line, re.I)
            if candidate:
                if (
                    kind == "ARTICLE"
                    and candidate[2]
                    and not re.match(r"^ĐIỀU\s*\d+[a-zđ]?\s*[.:]", line, re.I)
                    and not candidate[2][0].isupper()
                ):
                    continue
                match = (kind, candidate)
                break
        if match:
            kind, m = match
            while LEVELS[stack[-1].type] >= LEVELS[kind]:
                stack.pop()
            node = LegalNode(
                f"{document_id}:{kind.lower()}:{len(nodes)}",
                kind,
                m[1],
                m[2] if kind not in ("CLAUSE", "POINT") else "",
                parent_id=stack[-1].id,
                order=len(nodes),
            )
            nodes.append(node)
            stack.append(node)
            lines_by_id[node.id] = []
            heading_end[node.id] = 1
            inline_body[node.id] = m[2] if kind in ("CLAUSE", "POINT") else ""
            awaiting_title = node if not node.title and kind not in ("CLAUSE", "POINT") else None
        elif line and awaiting_title is not None:
            awaiting_title.title = line
            heading_end[awaiting_title.id] = len(lines_by_id[awaiting_title.id]) + 1
            awaiting_title = None
        for node in stack[1:]:
            lines_by_id[node.id].append(line)
    for node in nodes[1:]:
        node.text = "\n".join(lines_by_id[node.id]).strip()
        body_lines = node.text.splitlines()[heading_end[node.id] :]
        if inline_body[node.id]:
            body_lines.insert(0, inline_body[node.id])
        node.body = "\n".join(body_lines).strip()
    doc = LegalDocument(document_id, text, nodes)
    doc.edges = [LegalEdge(n.parent_id, n.id, "CONTAINS") for n in nodes[1:]]
    if not doc.articles:
        raise ValueError(
            "Không phát hiện được cấu trúc Điều. V1 yêu cầu văn bản plain text có giữ cấu trúc pháp lý."
        )
    return doc
