"""Resolve reviewed statements against the installed edition of an engine manual."""
import hashlib
import re
from collections.abc import Callable
from functools import cache
from pathlib import Path
from string import Formatter
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ml.contracts import AnswerVerification, CheckedPage, Evidence, EvidenceBundle, FaultCode, SearchOutcome, SearchPlan, TechnicalClaim
from ml.generation.text import number_list
from ml.document_ids import document_id_variants

DEFAULT_CATALOG = Path(__file__).with_name('ingestion') / 'reviewed_evidence.json'
PageKey = Annotated[str, Field(pattern=r'^[^:]+:[1-9]\d*$')]


class ReviewedPage(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sha256: Annotated[str, Field(pattern=r'^[a-f0-9]{64}$')]
    captures: dict[str, re.Pattern[str]] = Field(default_factory=dict)

    def read(self, item: Evidence) -> dict[str, str] | None:
        text = ' '.join(item.text.split())
        if hashlib.sha256(text.encode()).hexdigest() != self.sha256:
            return None
        values = {}
        for name, pattern in self.captures.items():
            match = pattern.search(text)
            if match is None or match[1] is None:
                return None
            values[name] = match[1]
        return values


class ReviewEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    fault_code: FaultCode | None = None
    topic: str | None = None
    conflict_notice: str = ''
    claims: list[TechnicalClaim] = Field(min_length=1)


class ReviewCatalog(BaseModel):
    model_config = ConfigDict(extra='forbid')
    pages: dict[PageKey, ReviewedPage]
    topics: dict[str, Annotated[list[re.Pattern[str]], Field(min_length=1)]]
    entries: list[ReviewEntry]

    @classmethod
    def load(cls, path: Path | None = None):
        return cls.model_validate_json((path or DEFAULT_CATALOG).read_bytes())

    @model_validator(mode='after')
    def check_references(self):
        for page in self.pages.values():
            if any(pattern.groups != 1 for pattern in page.captures.values()):
                raise ValueError('Each capture must extract one value from the reviewed page')
        for entry in self.entries:
            if not (entry.fault_code or entry.topic) or (entry.topic and entry.topic not in self.topics):
                raise ValueError('Review entry needs a known topic or diagnostic code')
            for claim in entry.claims:
                fields = set()
                for key in claim.evidence_ids:
                    if key not in self.pages or fields.intersection(self.pages[key].captures):
                        raise ValueError(f'Unknown source or ambiguous capture in {key}')
                    fields.update(self.pages[key].captures)
                for _, name, spec, conversion in Formatter().parse(claim.text):
                    if name is not None and (name not in fields or not name.isidentifier() or spec or conversion):
                        raise ValueError(f'Unsupported field {name} in reviewed claim')
        return self

    def resolve(self, plan: SearchPlan, lookup: Callable[[str, str, int], Evidence | None], *,
                fault: str | None = None, question: str = '', risk_only: bool = False) -> SearchOutcome:
        @cache
        def read(key):
            document, number = key.rsplit(':', 1)
            page = int(number)
            item = lookup(plan.engine_model, document, page)
            if (item is None or item.document_id not in document_id_variants(document) or item.pdf_page != page
                    or plan.engine_model not in item.engine_models):
                return None
            values = self.pages[key].read(item)
            return (item, values) if values is not None else None

        topics = {name for name, terms in self.topics.items() if all(term.search(question) for term in terms)}
        sources = {}
        claims = []
        notices = []
        checked_pages = {}
        rejected = 0
        for entry in self.entries:
            if not ((fault and entry.fault_code == fault) or entry.topic in topics):
                continue
            # A multi-page procedure is released as a unit, including exceptions and conflicting tables.
            if any(read(key) is None for claim in entry.claims for key in claim.evidence_ids):
                rejected += sum(not risk_only or c.kind == 'risk' for c in entry.claims)
                continue
            if entry.conflict_notice:
                notices.append(entry.conflict_notice)
            for claim in entry.claims:
                if risk_only and claim.kind != 'risk':
                    continue
                fields, ids = {}, []
                used = {name for _, name, _, _ in Formatter().parse(claim.text) if name}
                for key in claim.evidence_ids:
                    item, values = read(key)
                    sources[item.evidence_id] = item
                    ids.append(item.evidence_id)
                    fields.update(values)
                    numbers = checked_pages.setdefault(item.evidence_id, {})
                    for name, value in values.items():
                        if name in used:
                            numbers.update(dict.fromkeys(number_list(value)))
                claims.append(TechnicalClaim(text=claim.text.format_map(fields), kind=claim.kind, evidence_ids=ids))
        return SearchOutcome(plan=plan, claims=claims, evidence=EvidenceBundle(items=list(sources.values())),
                             verification=AnswerVerification(engine_model=plan.engine_model, rejected_claims=rejected,
                                 pages=[CheckedPage(evidence_id=k, numbers=list(v)) for k, v in checked_pages.items()]
                             ) if checked_pages or rejected else None,
                             summary=' '.join(dict.fromkeys(notices)))
