"""Conservative internal references, definitions and exact term matching."""

import re

from src.parser import normalize_text
from src.schemas import LegalEdge, LegalNode

ROLE_PREFIXES = {
    "giải thích từ ngữ": "DEFINITION",
    "phạm vi điều chỉnh": "SCOPE",
    "đối tượng áp dụng": "SUBJECT_SCOPE",
    "quyền": "RIGHT",
    "nghĩa vụ": "OBLIGATION",
    "trình tự": "PROCEDURE",
    "thủ tục": "PROCEDURE",
    "xử lý vi phạm": "SANCTION",
    "điều khoản chuyển tiếp": "TRANSITIONAL",
    "hiệu lực thi hành": "EFFECTIVE_DATE",
}
REF = re.compile(
    r"(?<!\w)(?:(?:điểm\s+(?P<point>[a-zđ])\s+)?khoản\s+(?P<clause>\d+)\s+)?(?:từ\s+)?(?:các\s+)?điều\s+(?P<first>\d+[a-zđ]?)(?P<rest>(?:\s*(?:,|và)\s*\d+[a-zđ]?)*)(?:\s+đến\s+điều\s+(?P<end>\d+))?",
    re.I,
)


def semantic_role(title):
    title = title.casefold().strip()
    return next(
        (
            role
            for prefix, role in ROLE_PREFIXES.items()
            if re.match(re.escape(prefix) + r"(?:\b|$)", title)
        ),
        "UNKNOWN",
    )


def extract_relations(doc):
    # Idempotent extraction; do not retain stale term nodes or inferred relations.
    doc.nodes[:] = [n for n in doc.nodes if n.type != "TERM"]
    doc.edges[:] = [e for e in doc.edges if e.edge_type == "CONTAINS"]
    doc.unresolved_references.clear()
    by_number = {}
    for a in doc.articles:
        by_number.setdefault(a.number.casefold(), []).append(a)
        a.role = semantic_role(a.title)
    for a in doc.articles:
        for m in REF.finditer(a.body):
            nums = [m["first"]] + re.findall(r"\d+[a-zđ]?", m["rest"], re.I)
            if m["end"] and m["first"].isdigit():
                start, end = int(m["first"]), int(m["end"])
                nums = [str(i) for i in range(start, end + 1)] if 0 <= end - start <= 1000 else nums
            suffix = a.body[m.end() : m.end() + 200]
            # A shared foreign-document qualifier can follow another listed article.
            suffix = re.sub(r"^(?:\s*(?:,|và)\s*điều\s+\d+[a-zđ]?)+", "", suffix, flags=re.I)
            external = bool(
                re.match(
                    r"\s+(?:của\s+)?(?:Luật|Bộ luật|Nghị định|Thông tư|Hiến pháp|Nghị quyết|Quyết định)\b(?!\s+này\b)",
                    suffix,
                    re.I,
                )
            )
            for num in nums:
                targets = by_number.get(num.casefold(), [])
                evidence = {
                    "text_field": "body",
                    "text": m[0],
                    "start": m.start(),
                    "end": m.end(),
                    "clause": m["clause"],
                    "point": m["point"],
                }
                if external or len(targets) != 1:
                    doc.unresolved_references.append(
                        {
                            "source": a.id,
                            "number": num,
                            "reason": "external" if external else "missing_or_ambiguous",
                            **evidence,
                        }
                    )
                elif targets[0].id != a.id:
                    doc.edges.append(
                        LegalEdge(a.id, targets[0].id, "REFERENCES", evidence=evidence)
                    )
    definitions = []
    seen_definitions = set()
    for a in doc.articles:
        definition_context = (
            a.role == "DEFINITION" or "các từ ngữ dưới đây được hiểu như sau" in a.body.casefold()
        )
        blocks = (
            re.split(r"(?:^|\n)\s*\d+\.\s*|\n(?=[\"“])", a.body)
            if definition_context
            else a.body.splitlines()
        )
        for block in blocks:
            line = re.sub(r"\s+", " ", block).strip()
            line = re.sub(r"^\d+\.\s*", "", line).strip()
            quoted = re.match(r'^["“](.+?)["”]\s+(?:là|được hiểu là)\s+(.+)', line, re.I)
            unquoted = (
                re.match(r"^(.{2,100}?)\s+(?:là|được hiểu là)\s+(.+)", line, re.I)
                if definition_context
                else None
            )
            match = quoted or unquoted
            if not match:
                continue
            term = normalize_text(match[1]).casefold().strip(' "“”')
            if (a.id, term) in seen_definitions:
                continue
            seen_definitions.add((a.id, term))
            tid = f"{doc.id}:term:{len(definitions)}"
            node = LegalNode(
                tid, "TERM", title=term, text=match[2], parent_id=a.id, order=len(doc.nodes)
            )
            doc.nodes.append(node)
            doc.edges.append(
                LegalEdge(a.id, tid, "DEFINES", evidence={"term": term, "definition": match[2]})
            )
            definitions.append((term, tid, a.id))
    term_owners = {}
    for term, _, defining in definitions:
        term_owners.setdefault(term, set()).add(defining)
    for a in doc.articles:
        occupied = []
        for term, tid, defining in sorted(definitions, key=lambda t: -len(t[0])):
            if a.id == defining or len(term_owners[term]) != 1:
                continue
            hits = []
            for m in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", a.text.casefold()):
                if not any(m.start() < end and m.end() > start for start, end in occupied):
                    hits.append(m.span())
            if hits:
                occupied.extend(hits)
                doc.edges.append(
                    LegalEdge(a.id, tid, "USES_TERM", evidence={"term": term, "spans": hits})
                )
    return doc
