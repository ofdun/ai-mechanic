"""Final answer for the mechanic.

The finalizer composes the message from verified claims, the diagnostic result and the pipeline state. It
is deterministic: it orders and labels facts but never adds technical facts of its own. The only text not
taken from a manual is the agent's risk assessment of a fault type, which is always labelled as such.
The verdict comes first, so it reads in ten seconds; checks and repairs follow as numbered steps.
"""
import logging
import re
from collections import defaultdict
from typing import Literal

from pydantic import BaseModel, Field

from ml.contracts import AgentResponse, DiagnosticResult, Evidence, SessionRecord, TechnicalClaim
from ml.generation.claims import conflict_group, quote_of
from ml.generation.profiles import PROFILES, URGENCY_LABELS, Urgency, urgency_for
from ml.generation.text import claim_numbers, content_stems, fold, has_negation


logger = logging.getLogger(__name__)

Tone = Literal['ok', 'attention', 'alert', 'repeat', 'info']

SECTION_TITLES = {
    'observation': 'Признаки по руководству',
    'cause': 'Возможные причины — гипотезы, подтвердите проверкой',
    'risk': 'Риски по руководству',
    'check': 'Что проверить',
    'reference_value': 'Нормативы',
    'operation': 'Возможный ремонт',
    'spare_part': 'Запчасти',
}
STEP_KINDS = {'check', 'operation'}
SIGNAL_LABELS = {'good': 'хорошая', 'warning': 'с помехами', 'bad': 'непригодна'}
CONFLICT_TITLE = 'Источники расходятся — агент не выбирает сам'
CONFLICT_NOTE = 'Уточните у специалиста или в документации производителя для вашей модификации двигателя.'
DECISION_NOTE = 'Решение о ремонте и выпуске машины принимает механик или ответственный специалист.'
ML_NOTE = 'Тип неисправности — гипотеза акустической модели, её подтверждает проверка по руководству.'
PAGE_NOTE = 'Номера страниц указаны по PDF.'
NO_EVIDENCE_LIMITATIONS = {
    'Недостаточно доказательств для технических выводов.',
    'Применимых доказательств для выбранного двигателя не найдено.',
    'По найденным фрагментам не удалось подтвердить технические утверждения.',
    'В руководстве по этому двигателю недостаточно сведений для вывода.',
    'В руководстве по этому двигателю ответ не найден.',
    'На найденных страницах руководства нет подтверждения для ответа.',
}
CONFLICT_PREFIX = 'Руководства расходятся'
TICKET_LIMITATION = 'Для заявки нужна хотя бы одна подтверждённая операция.'
_PROHIBITION = re.compile(
    r'(?:запрещ[а-я]*|не\s+допуска[а-я]*|нельзя)\s+(?:[а-я]+\s+){0,3}?'
    r'(?:эксплуатац|работ|пуск|запуск|движени|выезд|выпуск)',
)
_RELEASE_QUESTION = re.compile(r'перег[оа]н|своим ходом|выпус|в рейс|ехать|доехать|эксплуатир')


class AnswerItem(BaseModel):
    text: str
    kind: str | None = None
    manual: str | None = None
    pdf_page: int | None = None
    quote: str | None = None


class AnswerSection(BaseModel):
    title: str
    items: list[AnswerItem] = Field(default_factory=list)
    numbered: bool = False
    note: str | None = None


class MechanicAnswer(BaseModel):
    """Structured answer for a UI. ``text`` is the same answer as plain chat text."""

    tone: Tone
    verdict: str
    details: list[str] = Field(default_factory=list)
    sections: list[AnswerSection] = Field(default_factory=list)
    urgency: Urgency | None = None
    text: str = ''


def percent(value: float | None) -> str:
    return '—' if value is None else f'{value:.0%}'


def decimal(value: float | None) -> str:
    return '—' if value is None else f'{value:.2f}'.replace('.', ',')


def prohibits_operation(claim: TechnicalClaim) -> bool:
    return bool(_PROHIBITION.search(fold(claim.text)))


def find_conflicts(claims: list[TechnicalClaim], sources: dict[str, Evidence]) -> list[list[TechnicalClaim]]:
    """Groups of claims that disagree: marked by the generator or detected between different manuals."""
    groups: dict[str, list[TechnicalClaim]] = defaultdict(list)
    for claim in claims:
        if label := conflict_group(claim):
            groups[label].append(claim)
    conflicts = [group for group in groups.values() if len(group) > 1]
    grouped = {id(claim) for group in conflicts for claim in group}

    def document(claim: TechnicalClaim) -> str | None:
        source = sources.get(claim.evidence_ids[0]) if claim.evidence_ids else None
        return source.document_id if source else None

    comparable = [claim for claim in claims
                  if id(claim) not in grouped and claim.kind in {'check', 'reference_value', 'operation'}]
    for index, first in enumerate(comparable):
        for second in comparable[index + 1:]:
            if id(first) in grouped or id(second) in grouped or document(first) == document(second):
                continue
            a, b = content_stems(first.text), content_stems(second.text)
            if not a or not b or len(a & b) / len(a | b) < 0.5:
                continue
            values_a, values_b = claim_numbers(first.text), claim_numbers(second.text)
            differs = (values_a and values_b and values_a != values_b) or (
                has_negation(first.text) != has_negation(second.text))
            if differs:
                conflicts.append([first, second])
                grouped.update({id(first), id(second)})
    return conflicts


class MechanicFinalizer:
    """Turns an ``AgentResponse`` into a short, verified answer for the mechanic."""

    def compose(self, response: AgentResponse) -> str:
        try:
            return self.answer(response).text
        except Exception:  # The original pipeline message is a safe fallback for the chat.
            logger.exception('Finalizer failed; keeping the pipeline message')
            return response.message

    def answer(self, response: AgentResponse) -> MechanicAnswer:
        if response.action == 'get_diagnostic_history':
            answer = self.history(response)
        elif response.action in {'diagnose', 'draft_ticket'} and response.diagnostic is not None:
            answer = self.diagnosis(response)
        elif response.action == 'search_manuals':
            answer = self.manual_answer(response)
        else:
            answer = MechanicAnswer(tone='info', verdict=response.message)
        answer.text = render(answer)
        return answer

    # Diagnosis ---------------------------------------------------------------------------------

    def diagnosis(self, response: AgentResponse) -> MechanicAnswer:
        result = response.diagnostic
        if response.requires_remeasurement:
            details = [f'Причина: {response.message}']
            if result.status == 'check_failed' and result.description:
                details.append(f'Запись: {result.description}')
            details.append('Что сделать: повторите запись на холостом ходу и снова запустите проверку.')
            return MechanicAnswer(tone='repeat', verdict='🔁 Сбой проверки — заключение о двигателе не формируется.',
                                  details=details)
        if result.status == 'normal':
            return MechanicAnswer(tone='ok', verdict='✅ Норма — значимых отклонений не обнаружено.', details=[
                'Можно продолжать эксплуатацию в обычном режиме.', self.metrics(result),
            ])
        title = f'{result.anomaly_title} ({result.anomaly_type})'
        if response.search_plan is None:
            return MechanicAnswer(tone='attention', verdict=f'⚠️ Отклонение «{title}» требует проверки данных.',
                                  details=[response.message, self.metrics(result),
                                           'Заявку не формирую, пока данные не уточнены.'])

        sources = {item.evidence_id: item for item in response.evidence.items}
        prohibited = [claim for claim in response.claims if prohibits_operation(claim)]
        urgency = urgency_for(result.anomaly_type, result.anomaly_score, prohibited=bool(prohibited))
        profile = PROFILES.get(result.anomaly_type or '')
        details = [f'Приоритет: {URGENCY_LABELS[urgency]}.', self.metrics(result)]
        if result.description:
            details.append(f'Признаки по записи: {result.description}')
        if profile:
            details.append(f'Риск (оценка агента, не из руководства): {profile.risk}')
        for claim in prohibited:
            details.append(f'⛔ По руководству: {claim.text} {citation(claim, sources)}'.rstrip())
        if response.action == 'diagnose' and response.can_create_ticket:
            details.append('Можно сформировать заявку на дефектовку.')
        shown = [claim for claim in response.claims if not any(claim is other for other in prohibited)]
        sections = self.claim_sections(shown, sources)
        if not response.claims:
            unit = f' «{profile.unit}»' if profile else ''
            sections.append(AnswerSection(title='Что делать', items=[AnswerItem(
                text=f'В руководствах для {response.engine_model} не нашлось подтверждённых проверок по этому '
                     f'отклонению. Не буду гадать: рекомендую дефектовку узла{unit} у специалиста.')]))
        if response.action == 'draft_ticket':
            sections.append(self.ticket_section(response, sources, urgency, profile.unit if profile else None))
        sections.extend(self.closing_sections(response, sources, extra=[ML_NOTE, DECISION_NOTE]))
        tone = 'alert' if urgency == 'high' else 'attention'
        return MechanicAnswer(tone=tone, verdict=f'⚠️ Отклонение: {title}', details=details, sections=sections,
                              urgency=urgency)

    @staticmethod
    def metrics(result: DiagnosticResult) -> str:
        parts = [f'оценка аномалии {decimal(result.anomaly_score)}', f'уверенность модели {percent(result.confidence)}',
                 f'запись: {SIGNAL_LABELS.get(result.signal_quality, result.signal_quality)}']
        return ' · '.join(parts).capitalize()

    def ticket_section(self, response: AgentResponse, sources: dict[str, Evidence], urgency: Urgency,
                       unit: str | None) -> AnswerSection:
        draft = response.ticket_draft
        if draft is None:
            reason = (f'в руководстве для {response.engine_model} нет подтверждённой процедуры'
                      f'{f" по узлу «{unit}»" if unit else ""}')
            return AnswerSection(title='Заявка не сформирована', items=[
                AnswerItem(text=f'Причина: {reason}.'),
                AnswerItem(text='Оформите дефектовку вручную по рекомендации выше.'),
            ])
        payload = draft.payload
        items = [AnswerItem(text=f'Приоритет: {URGENCY_LABELS[payload.urgency]}.')]
        items += [AnswerItem(text=f'Работа: {operation}') for operation in payload.operations]
        items.append(AnswerItem(text='Запчасти: ' + (', '.join(payload.spare_parts) if payload.spare_parts
                                                      else 'в руководстве не указаны.')))
        items.append(AnswerItem(text='Источники: ' + '; '.join(
            f'{reference.manual}, стр. {reference.pdf_page}' for reference in payload.references)))
        return AnswerSection(title='📝 Черновик заявки на дефектовку', items=items,
                             note='Заявка уйдёт в ТОиР только после вашего подтверждения.')

    # Questions -------------------------------------------------------------------------------

    def manual_answer(self, response: AgentResponse) -> MechanicAnswer:
        sources = {item.evidence_id: item for item in response.evidence.items}
        question = response.search_plan.original_query if response.search_plan else ''
        extra = [DECISION_NOTE] if _RELEASE_QUESTION.search(fold(question)) else []
        if not response.claims:
            details = [f'В руководствах для {response.engine_model} нет подтверждённого ответа на этот вопрос.',
                       'Не буду придумывать: уточните у специалиста или проведите дефектовку.']
            return MechanicAnswer(tone='info', verdict='📘 Данных недостаточно.', details=details,
                                  sections=self.closing_sections(response, sources, extra=extra))
        sections = self.claim_sections(response.claims, sources)
        sections.extend(self.closing_sections(response, sources, extra=extra))
        return MechanicAnswer(tone='info', verdict=f'📘 Ответ по руководствам для {response.engine_model}',
                              sections=sections)

    # Shared parts -----------------------------------------------------------------------------

    def claim_sections(self, claims: list[TechnicalClaim], sources: dict[str, Evidence]) -> list[AnswerSection]:
        conflicts = find_conflicts(claims, sources)
        in_conflict = {id(claim) for group in conflicts for claim in group}
        grouped: dict[str, list[AnswerItem]] = defaultdict(list)
        for claim in claims:
            if id(claim) not in in_conflict:
                grouped[claim.kind].append(item(claim, sources))
        sections = [AnswerSection(title=SECTION_TITLES[kind], items=grouped[kind], numbered=kind in STEP_KINDS)
                    for kind in SECTION_TITLES if grouped.get(kind)]
        for group in conflicts:
            sections.append(AnswerSection(title=f'⚖️ {CONFLICT_TITLE}', note=CONFLICT_NOTE,
                                          items=[item(claim, sources) for claim in group]))
        return sections

    def closing_sections(self, response: AgentResponse, sources: dict[str, Evidence], *,
                         extra: list[str]) -> list[AnswerSection]:
        sections = []
        cited = defaultdict(set)
        for claim in response.claims:
            for key in claim.evidence_ids:
                if key in sources:
                    cited[sources[key].manual].add(sources[key].pdf_page)
        if cited:
            sections.append(AnswerSection(title='Источники', note=PAGE_NOTE, items=[
                AnswerItem(text=f'{manual} — стр. {", ".join(map(str, sorted(pages)))}')
                for manual, pages in cited.items()]))
        limitations = [text for text in response.limitations if text not in NO_EVIDENCE_LIMITATIONS
                       and text != TICKET_LIMITATION and not text.startswith(CONFLICT_PREFIX)]
        limitations += extra
        if limitations:
            sections.append(AnswerSection(title='Ограничения', items=[
                AnswerItem(text=text) for text in dict.fromkeys(limitations)]))
        return sections

    # History ------------------------------------------------------------------------------------

    def history(self, response: AgentResponse) -> MechanicAnswer:
        sessions = sorted(response.history, key=lambda session: session.checked_at)
        if not sessions:
            return MechanicAnswer(tone='info', verdict='📈 История проверок пуста.')
        items = [AnswerItem(text=describe_session(session)) for session in sessions]
        return MechanicAnswer(tone='info', verdict=f'📈 Последние проверки {response.vehicle_id}',
                              details=[trend(sessions)], sections=[AnswerSection(title='Проверки', items=items)])

    # Ticket fields ----------------------------------------------------------------------------

    def ticket_fields(self, result: DiagnosticResult, claims: list[TechnicalClaim]) -> tuple[str, Urgency]:
        return ticket_fields(result, claims)


def ticket_fields(result: DiagnosticResult, claims: list[TechnicalClaim]) -> tuple[str, Urgency]:
    """Conclusion and urgency for the maintenance request."""
    prohibited = any(prohibits_operation(claim) for claim in claims)
    urgency = urgency_for(result.anomaly_type, result.anomaly_score, prohibited=prohibited)
    profile = PROFILES.get(result.anomaly_type or '')
    parts = [f'Предполагаемая неисправность: {result.anomaly_title} ({result.anomaly_type}) по акустической '
             f'проверке {result.session_id}: оценка аномалии {decimal(result.anomaly_score)}, '
             f'уверенность модели {percent(result.confidence)}.']
    if result.description:
        parts.append(f'Признаки: {result.description.rstrip(".")}.')
    if profile:
        parts.append(f'Требуется дефектовка: {profile.unit}.')
    if prohibited:
        parts.append('Руководство запрещает эксплуатацию до устранения неисправности.')
    return ' '.join(parts), urgency


def conflict_notes(response: AgentResponse) -> list[str]:
    """One line per disagreement between sources, for any client that shows ``limitations``."""
    sources = {item.evidence_id: item for item in response.evidence.items}
    notes = []
    for group in find_conflicts(response.claims, sources):
        wordings = '; '.join(f'«{claim.text.rstrip(".")}» {citation(claim, sources)}'.rstrip() for claim in group)
        notes.append(f'{CONFLICT_PREFIX}: {wordings}. {CONFLICT_NOTE}')
    return notes


def citation(claim: TechnicalClaim, sources: dict[str, Evidence]) -> str:
    source = sources.get(claim.evidence_ids[0]) if claim.evidence_ids else None
    return f'({source.manual}, стр. {source.pdf_page})' if source else ''


def item(claim: TechnicalClaim, sources: dict[str, Evidence]) -> AnswerItem:
    source = sources.get(claim.evidence_ids[0]) if claim.evidence_ids else None
    quote = quote_of(claim)
    return AnswerItem(text=claim.text, kind=claim.kind, manual=source.manual if source else None,
                      pdf_page=source.pdf_page if source else None, quote=quote if quote != claim.text else None)


STATUS_LABELS = {'normal': 'норма', 'anomaly': 'отклонение', 'check_failed': 'сбой записи'}


def describe_session(session: SessionRecord) -> str:
    text = f'{session.checked_at:%d.%m.%Y} · {session.session_id} — {STATUS_LABELS[session.status]}'
    if session.status == 'anomaly':
        text += f' {session.anomaly_type} «{session.anomaly_title}»'
    if session.anomaly_score is not None:
        text += f', оценка {decimal(session.anomaly_score)}'
    return text


TREND_STEP = 0.1


def trend(sessions: list[SessionRecord]) -> str:
    """Describe the anomaly score dynamics over successful checks (bonus B2)."""
    scores = [session.anomaly_score for session in sessions if session.anomaly_score is not None]
    if len(scores) < 2:
        return 'Для оценки динамики нужны хотя бы две успешные проверки.'
    path = ' → '.join(decimal(score) for score in scores)
    steps = [later - earlier for earlier, later in zip(scores, scores[1:])]
    if all(step >= 0 for step in steps) and scores[-1] - scores[0] >= TREND_STEP:
        return f'Оценка аномалии растёт: {path}. Состояние ухудшается — стоит запланировать дефектовку.'
    if all(step <= 0 for step in steps) and scores[0] - scores[-1] >= TREND_STEP:
        return f'Оценка аномалии снижается: {path}.'
    if steps[-1] >= TREND_STEP:
        return f'Оценка аномалии выросла по сравнению с прошлой проверкой: {path}.'
    if steps[-1] <= -TREND_STEP:
        return f'Оценка аномалии снизилась по сравнению с прошлой проверкой: {path}.'
    if max(scores) - min(scores) >= TREND_STEP:
        return f'Оценка аномалии колеблется: {path}.'
    return f'Оценка аномалии стабильна: {path}.'


def render(answer: MechanicAnswer) -> str:
    lines = [answer.verdict, *answer.details]
    for section in answer.sections:
        lines.append('')
        lines.append(f'{section.title}:')
        for number, entry in enumerate(section.items, 1):
            marker = f'{number}.' if section.numbered else '•'
            reference = f' ({entry.manual}, стр. {entry.pdf_page})' if entry.manual and entry.pdf_page else ''
            lines.append(f'{marker} {entry.text}{reference}')
        if section.note:
            lines.append(section.note)
    return '\n'.join(lines).strip()
