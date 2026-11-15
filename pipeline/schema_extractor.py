"""Conservative section excerpts: missing evidence remains explicitly unknown."""
import re
from pydantic import BaseModel, ConfigDict, Field


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    page: int = Field(ge=1)
    quote: str = Field(min_length=1)


class ResearchPaperSchema(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str
    primary_domain: str | None = None
    research_problem: str | None = None
    methodology: str | None = None
    datasets_and_benchmarks: str | None = None
    key_findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: dict[str, list[Evidence]] = Field(default_factory=dict)


class StructuredExtractor:
    SECTION = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?"
        r"(abstract|introduction|background|related work|methodology|methods?|"
        r"proposed (?:method|approach)|model(?: architecture)?|architecture|"
        r"datasets?(?: and benchmarks)?|experimental setup|experiments?|"
        r"results(?: and conclusion)?|evaluation|conclusions?|discussion|"
        r"limitations?(?: and future work)?|references|acknowledg(?:e)?ments)"
        r"\s*(?::\s*(.*))?$", re.IGNORECASE)

    def extract(self, raw_text: str, filename: str, pages: list[dict] | None = None) -> ResearchPaperSchema:
        if pages is None:
            parts = re.split(r"--- Page (\d+) ---\n?", raw_text)
            pages = ([{"page": int(parts[i]), "text": parts[i + 1]} for i in range(1, len(parts), 2)]
                     if len(parts) > 1 else [{"page": 1, "text": raw_text}])
        title = filename.removesuffix(".pdf").replace("_", " ")
        first_lines = [line.strip() for p in pages[:1] for line in p["text"].splitlines() if line.strip()]
        if first_lines and not self.SECTION.match(first_lines[0]):
            title = first_lines[0]
        sections = []
        active = None
        for page in pages:
            buffer = []
            def flush():
                if active and buffer:
                    sections.append((active, Evidence(page=page["page"], quote=" ".join(buffer))))
                buffer.clear()
            for line in page["text"].splitlines():
                line = line.strip()
                if not line or line == title or line.isdigit():
                    continue
                heading = self.SECTION.match(line)
                if heading:
                    flush()
                    active = heading[1].lower()
                    if heading[2]:
                        buffer.append(heading[2])
                elif active:
                    buffer.append(line)
            flush()
        mapping = {
            "research_problem": r"abstract|introduction",
            "methodology": r"method|proposed|model|architecture",
            "datasets_and_benchmarks": r"dataset|experimental setup|experiment",
            "key_findings": r"result|evaluation|conclusion",
            "limitations": r"limitation",
        }
        data = {"title": title, "evidence": {}}
        for field, pattern in mapping.items():
            found = [e for heading, e in sections if re.match(pattern, heading)][:3]
            if found:
                data["evidence"][field] = found
                data[field] = ([e.quote for e in found] if field in {"key_findings", "limitations"}
                               else "\n\n".join(e.quote for e in found))
        return ResearchPaperSchema(**data)
