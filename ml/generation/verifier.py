"""Programmatic verification of generated claims against the retrieved manual pages.

A claim passes only when its excerpt is found on the cited page of a manual for this engine, the claim
keeps the excerpt's numbers, prohibitions and conditions, and a repair step concerns the faulty unit.
Every rejected claim gets machine-readable reasons, so verification quality can be measured.
"""
import re
from dataclasses import dataclass

from ml.contracts import Evidence, EvidenceBundle, TechnicalClaim, TechnicalDraft, VerifiedResponse
from ml.generation.claims import MAX_CLAIMS, GroundedClaim, quote_of
from ml.generation.profiles import mentions_unit
from ml.generation.text import (
    Normalized, align, claim_numbers, content_stems, fold, has_condition, has_negation, left_context, number_problems,
    numbers_stay_with_words, sentence_bounds, splits_token, words,
)


MIN_QUOTE_CHARS = 12
MAX_QUOTE_CHARS = 600
MIN_OVERLAP = 0.5
UNIT_BOUND_KINDS = {'operation', 'spare_part'}

REASONS = {
    'no_evidence': 'нет ссылки на источник',
    'unknown_evidence': 'ссылка на фрагмент, которого нет среди найденных',
    'several_sources': 'одна цитата не может подтверждать несколько источников',
    'wrong_engine': 'источник не относится к двигателю этой машины',
    'contents_page': 'оглавление не подтверждает процедуру',
    'quote_too_short': 'цитата слишком короткая',
    'quote_too_long': 'цитата слишком длинная',
    'quote_not_found': 'цитаты нет на указанной странице',
    'quote_splits_word': 'цитата начинается или заканчивается внутри слова или числа',
    'number_moved': 'число цитаты оторвано от своей строки',
    'number_not_in_quote': 'число утверждения отсутствует в цитате',
    'numbers_out_of_order': 'числа утверждения идут не в том порядке, что в цитате',
    'range_truncated': 'диапазон норматива передан не полностью',
    'unknown_engine_in_claim': 'утверждение упоминает модель двигателя, которой нет в источнике',
    'other_engine_in_quote': 'цитата относится к другой модели двигателя',
    'unrelated_to_quote': 'утверждение не следует из цитаты',
    'lost_negation': 'потерян запрет или отрицание источника',
    'invented_negation': 'запрет или отрицание отсутствует в источнике',
    'lost_condition': 'потеряно условие источника',
    'unsupported_permission': 'разрешение эксплуатации не следует из источника',
    'off_unit_operation': 'операция не относится к узлу из диагноза',
    'duplicate': 'повтор утверждения',
    'limit': 'превышено число утверждений',
}

_ENGINE_CODE = re.compile(r'(?:ЯМЗ|ЗМЗ|УМЗ)[\s-]*(\d{3,5})', re.IGNORECASE)
_PERMISSION = re.compile(
    r'(?:можно|допускается|разрешается|разрешено|допустим[а-я]*|безопасн[а-я]*)\s+(?:[а-я]+\s+){0,3}?'
    r'(?:эксплуат|выпуск|выезд|выезж|ехать|перегон|перегн|движени|двигат|продолж|работ)',
)
_LEADER_LINE = re.compile(r'(?:\.\s?){4,}\s*\d+\s*$|…+\s*\d+\s*$')


@dataclass(frozen=True)
class ClaimCheck:
    claim: TechnicalClaim
    reasons: tuple[str, ...]
    evidence_id: str | None = None
    excerpt: str = ''
    sentence: str = ''

    @property
    def accepted(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class VerificationReport:
    checks: tuple[ClaimCheck, ...]

    @property
    def accepted(self) -> list[TechnicalClaim]:
        return [check.claim for check in self.checks if check.accepted]

    @property
    def rejected(self) -> list[ClaimCheck]:
        return [check for check in self.checks if not check.accepted]


def is_contents_page(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    leaders = sum(bool(_LEADER_LINE.search(line)) for line in lines)
    return len(lines) >= 5 and leaders >= 5 and leaders >= 0.3 * len(lines)


def engine_code(engine_model: str | None) -> str | None:
    match = re.search(r'\d{3,5}', engine_model or '')
    return match.group() if match else None


def mentions_other_engine(text: str, engine_model: str | None) -> bool:
    target = engine_code(engine_model)
    mentioned = set(_ENGINE_CODE.findall(text))
    if not target or not mentioned:
        return False
    compatible = {code for code in mentioned if target.startswith(code) or code.startswith(target)}
    return not compatible


def unknown_engine(claim: str, source: str, engine_model: str | None) -> bool:
    """The claim names an engine model that neither the source nor the vehicle card mentions."""
    stated = set(_ENGINE_CODE.findall(claim))
    known = set(_ENGINE_CODE.findall(source)) | {engine_code(engine_model) or ''}
    return any(not any(code.startswith(other) or other.startswith(code) for other in known if other)
               for code in stated)


def grants_permission(text: str) -> bool:
    return bool(_PERMISSION.search(fold(text)))


class EvidenceVerifier:
    """Deterministic ``Verifier``. Optional engine and fault code tighten the checks."""

    def __init__(self, engine_model: str | None = None, anomaly_type: str | None = None, *, max_claims: int = MAX_CLAIMS):
        self.engine_model = engine_model
        self.anomaly_type = anomaly_type
        self.max_claims = max_claims

    def verify(self, draft: TechnicalDraft, evidence: EvidenceBundle) -> VerifiedResponse:
        return VerifiedResponse(claims=self.inspect(draft, evidence).accepted)

    def inspect(
        self, draft: TechnicalDraft, evidence: EvidenceBundle, *,
        engine_model: str | None = None, anomaly_type: str | None = None,
    ) -> VerificationReport:
        engine_model = engine_model or self.engine_model
        anomaly_type = anomaly_type or self.anomaly_type
        sources = {item.evidence_id: item for item in evidence.items}
        checks, seen, accepted = [], set(), 0
        for claim in draft.claims:
            check = self.check(claim, sources, engine_model, anomaly_type)
            if check.accepted:
                key = ' '.join(words(claim.text))
                if key in seen:
                    check = ClaimCheck(claim, ('duplicate',), check.evidence_id, check.excerpt, check.sentence)
                elif accepted >= self.max_claims:
                    check = ClaimCheck(claim, ('limit',), check.evidence_id, check.excerpt, check.sentence)
                else:
                    seen.add(key)
                    accepted += 1
            checks.append(check)
        return VerificationReport(tuple(checks))

    def check(
        self, claim: TechnicalClaim, sources: dict[str, Evidence], engine_model: str | None, anomaly_type: str | None,
    ) -> ClaimCheck:
        if not claim.evidence_ids:
            return ClaimCheck(claim, ('no_evidence',))
        if any(key not in sources for key in claim.evidence_ids):
            return ClaimCheck(claim, ('unknown_evidence',))
        grounded = isinstance(claim, GroundedClaim) and bool(claim.quote)
        if grounded and len(claim.evidence_ids) > 1:
            return ClaimCheck(claim, ('several_sources',))
        result = None
        for key in claim.evidence_ids:
            result = self.check_source(claim, sources[key], engine_model, anomaly_type)
            if not result.accepted:
                return result
        return result

    def check_source(
        self, claim: TechnicalClaim, source: Evidence, engine_model: str | None, anomaly_type: str | None,
    ) -> ClaimCheck:
        def reject(*reasons: str, excerpt: str = '', sentence: str = '') -> ClaimCheck:
            return ClaimCheck(claim, reasons, source.evidence_id, excerpt, sentence)

        if engine_model and engine_model not in source.engine_models:
            return reject('wrong_engine')
        if is_contents_page(source.text):
            return reject('contents_page')
        quote = quote_of(claim)
        if len(quote) > MAX_QUOTE_CHARS:
            return reject('quote_too_long')
        if len(Normalized.of(quote).chars) < MIN_QUOTE_CHARS or len(words(quote)) < 2:
            return reject('quote_too_short')
        match = align(quote, source.text)
        if match is None:
            return reject('quote_not_found')
        page = source.text
        excerpt = page[match.start:match.end]
        start, end = sentence_bounds(page, match.start, match.end)
        sentence = page[start:end]
        scope = left_context(page, start, match.start) + ' ' + excerpt

        reasons = []
        if splits_token(page, match.start, match.end):
            reasons.append('quote_splits_word')
        if not numbers_stay_with_words(quote, match):
            reasons.append('number_moved')
        ignore = (source.manual,)
        stated_numbers = claim_numbers(claim.text, ignore=ignore)
        reasons.extend(number_problems(claim.text, excerpt, ignore=ignore))
        if unknown_engine(claim.text, sentence + ' ' + scope, engine_model):
            reasons.append('unknown_engine_in_claim')
        if mentions_other_engine(scope, engine_model) and (
            stated_numbers or claim.kind in {'reference_value', 'operation', 'check', 'spare_part'}
        ):
            reasons.append('other_engine_in_quote')
        stated = content_stems(claim.text)
        supported = content_stems(sentence)
        if stated and len(stated & supported) < max(1, MIN_OVERLAP * len(stated)):
            reasons.append('unrelated_to_quote')
        if has_negation(scope) and not has_negation(claim.text):
            reasons.append('lost_negation')
        if has_negation(claim.text) and not has_negation(sentence):
            reasons.append('invented_negation')
        if has_condition(scope) and not has_condition(claim.text):
            reasons.append('lost_condition')
        if grants_permission(claim.text) and not grants_permission(sentence):
            reasons.append('unsupported_permission')
        if anomaly_type and claim.kind in UNIT_BOUND_KINDS and not mentions_unit(anomaly_type, claim.text, sentence):
            reasons.append('off_unit_operation')
        return reject(*reasons, excerpt=excerpt, sentence=sentence)


def explain(check: ClaimCheck) -> str:
    return '; '.join(REASONS.get(reason, reason) for reason in check.reasons)
