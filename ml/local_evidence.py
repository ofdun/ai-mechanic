"""Read reviewed rows and complete passages from the locally installed engine manuals."""
import re
from typing import TYPE_CHECKING

from ml.contracts import AnswerVerification, CheckedPage, EvidenceBundle, SearchOutcome, TechnicalClaim
from ml.operability import is_release_question, release_outcome
from ml.planner.local import is_diagnostic_followup
from ml.reviewed_checks import ReviewCatalog

if TYPE_CHECKING:
    from ml.architectures import LLMEvidenceWorkflow


def compact(item):
    return ' '.join(item.text.split()) if item else ''


def topic_terms(text):
    words = re.findall(r'[а-я]{3,}', text.casefold().replace('ё', 'е'))
    ignored = ('как', 'нужн', 'можн', 'провер', 'опред', 'поиск', 'долж', 'руковод', 'двигат',
               'мотор', 'дальш', 'причин', 'следу', 'почему', 'сказат', 'скол', 'значен', 'номин',
               'пожал', 'найти', 'покаж', 'указа', 'этого', 'очень', 'может', 'быть', 'будет',
               'такое', 'чинить', 'почин', 'отремонт', 'ремонтир', 'нормат', 'назван', 'характерист', 'вместе')
    return {w[:4] for w in words if not w.startswith(ignored) and w not in {
        'при', 'что', 'для', 'чем', 'это', 'под', 'него', 'его', 'чего', 'мне', 'меня', 'или', 'тип',
    }}


def candidate_pages(retriever, plan, question):
    terms = topic_terms(question)
    specific = terms - {'моме', 'затя', 'креп', 'болт', 'гаек', 'разм', 'норм', 'допу', 'техн'}
    if not specific:
        return []
    result = []
    for item in retriever.search(plan).items:
        if plan.engine_model not in item.engine_models:
            continue
        fragment = compact(item).casefold().replace('ё', 'е')
        windows = (fragment[i:i + 300] for i in range(0, len(fragment), 100))
        if any(all(term in window for term in specific) for window in windows):
            result.append(item)
            if len(result) == 2:
                break
    return result


def engine_key(name):
    return re.sub(r'[^\w.]', '', name.casefold())


def engine_mentions(models):
    names = []
    for model in models:
        # Recognize other variants of a known family even when the requested variant has no manual.
        family = re.match(r'([^\W\d_]{2,})[\s‑–-]*\d', model)
        if family:
            names.append(re.escape(family[1]) + r'[\s‑–-]*\d[a-zа-яё\d]*(?:[.‑–-][a-zа-яё\d]+)*')
        else:
            names.append(re.escape(model))
    return re.compile(r'(?<!\w)(?:' + '|'.join(sorted(set(names), key=len, reverse=True)) + r')(?!\w)', re.I)


class LocalEvidence:
    def __init__(self, retriever, catalog: ReviewCatalog | None = None, engine_models=(), *,
                 generation: 'LLMEvidenceWorkflow | None' = None):
        self.retriever = retriever
        self.catalog = catalog if catalog is not None else ReviewCatalog.load()
        self.generation = generation
        self.models = engine_mentions(engine_models) if engine_models else None

    def run(self, plan, diagnostic):
        question = plan.original_query.casefold().replace('ё', 'е')
        models = self.models or engine_mentions([plan.engine_model])
        other_engine = any(engine_key(name) != engine_key(plan.engine_model) for name in models.findall(question))
        if other_engine:
            return SearchOutcome(plan=plan, claims=[], evidence=EvidenceBundle())
        if is_release_question(question):
            return release_outcome(plan, self.retriever.search)

        followup = diagnostic is not None and is_diagnostic_followup(question)
        diagnosis_lookup = diagnostic is not None and (plan.anomaly_type is not None or followup)
        if diagnosis_lookup:
            risk_only = followup and bool(re.search(r'опас|гроз|риск|плох', question))
            result = self.catalog.resolve(plan, self.retriever.page, fault=diagnostic.anomaly_type, risk_only=risk_only)
        else:
            question = models.sub(' ', question)
            result = self.catalog.resolve(plan, self.retriever.page, question=question)
        if result.summary:
            return result
        if self.generation is not None:
            if result.claims:
                written = self.generation.outcome(plan, result.evidence, result.claims)
                written.verification = result.verification
                return written
            return self.generation.run(plan, diagnostic)
        if result.claims or diagnosis_lookup:
            return result
        pages, claims = {}, []
        if not re.search(r'сколько|момент|давлен|норматив|допуск|температур|зазор|размер|затяж|затян', question):
            self.passages(plan, question, pages, claims)
        evidence = list(pages.values()) if claims else candidate_pages(self.retriever, plan, question)
        verification = result.verification
        if pages:
            verification = AnswerVerification(engine_model=plan.engine_model,
                pages=[CheckedPage(evidence_id=key) for key in pages],
                rejected_claims=verification.rejected_claims if verification else None)
        return SearchOutcome(plan=plan, claims=claims, evidence=EvidenceBundle(items=evidence), verification=verification)

    def passages(self, plan, source_query, pages, claims):
        # Restrict verbatim excerpts to a single complete instruction about the named component.
        terms = topic_terms(source_query)
        if not terms:
            return
        for item in self.retriever.search(plan).items:
            if plan.engine_model not in item.engine_models:
                continue
            for match in re.finditer(r'(?:^|(?<=[.!?])\s+)([А-ЯЁ][^.!?]{30,280}\.)', compact(item)):
                sentence = match[1]
                lower = sentence.casefold().replace('ё', 'е')
                if not all(t in lower for t in terms) or re.search(r'\d|[();:]|[а-я][А-Я]', sentence):
                    continue
                if re.search(r'приводит к|может привести|приводят к|вызывает|может вызвать', lower):
                    kind = 'observation'
                elif re.match(r'(?:проверить|проверьте|осмотрите|осмотреть)\b|'
                              r'перед проведением проверки .{3,80}? проверить\b', lower):
                    kind = 'check'
                else:
                    continue
                if any(c.text == sentence or c.kind == kind for c in claims):
                    continue
                pages[item.evidence_id] = item
                claims.append(TechnicalClaim(kind=kind, text=sentence, evidence_ids=[item.evidence_id]))
