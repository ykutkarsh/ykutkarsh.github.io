"""Source quality, evidence conflict, and citation-aware discovery handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from aab.memory.database import ExperienceStore
from aab.research.web_reader import WebDocument


@dataclass(frozen=True)
class SourceAssessment:
    url: str
    domain: str
    quality: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Discovery:
    claim: str
    status: str
    confidence: float
    citations: tuple[str, ...]
    conflict_group: str | None


class SourceEvaluator:
    """Evaluate provenance signals without treating web text as instructions."""

    def __init__(self, store: ExperienceStore) -> None:
        self.store = store

    def assess(self, document: WebDocument) -> SourceAssessment:
        parsed = urlparse(document.url)
        domain = parsed.netloc.lower()
        score = 0.4
        reasons = ["public web source"]
        if parsed.scheme == "https":
            score += 0.2
            reasons.append("HTTPS transport")
        if domain.endswith((".gov", ".edu")):
            score += 0.2
            reasons.append("institutional domain")
        if len(document.text) >= 200:
            score += 0.1
            reasons.append("substantive retrieved text")
        return SourceAssessment(document.url, domain, min(score, 1.0), tuple(reasons))

    def store_discovery(
        self,
        claim: str,
        documents: list[WebDocument],
        confidence: float | None = None,
    ) -> Discovery:
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        if not documents:
            raise ValueError("at least one source document is required")
        assessments = [self.assess(document) for document in documents]
        citations = tuple(document.url for document in documents)
        normalized_claim = self._normalize(claim)
        conflict_group = self._conflict_group(normalized_claim)
        status = "hypothesis"
        if len(documents) >= 2:
            supporting = self._supporting_sources(documents, normalized_claim)
            status = "verified" if supporting == len(documents) else "conflicted"
        score = confidence if confidence is not None else sum(item.quality for item in assessments) / len(assessments)
        if not 0 <= score <= 1:
            raise ValueError("confidence must be between 0 and 1")
        discovery_id = self.store.record_discovery(
            claim.strip(), status, score, conflict_group, citations
        )
        self.store.record_citations(discovery_id, documents, assessments)
        return Discovery(claim.strip(), status, score, citations, conflict_group)

    def _conflict_group(self, claim: str) -> str:
        return re.sub(r"\b(not|no|never)\b", "", claim).strip()

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", value.lower()).strip()

    def _supporting_sources(self, documents: list[WebDocument], claim: str) -> int:
        terms = {term for term in claim.split() if len(term) > 3}
        return sum(
            1
            for document in documents
            if terms.issubset(set(self._normalize(document.text).split()))
        )
