"""Grounded responses to questions about releasing the selected vehicle."""
import re
from collections.abc import Callable, Sequence

from ml.contracts import AgentError, DiagnosticResult, EvidenceBundle, SearchOutcome, SearchPlan

_MOVE = re.compile(
    r'\b(?:перег(?:он|на)\w*|доех\w*|ехать|поехать|выеха\w*|выезжа\w*|ездить|'
    r'продолж\w*\s+(?:движение|эксплуатацию)|выпус\w*\s+(?:(?:машин\w*|тс)\s+)?в рейс)\b'
)
_ASK = re.compile(
    r'\b(?:можно|безопасно|допустимо|допускается|разрешено|стоит ли|нужно|'
    r'смогу ли|сможет ли|своим ходом|на базу|в рейс|критично)\b'
)
_OIL = re.compile(r'\b(?:вкладыш\w*|давлени\w*\s+масла)\b')
_FUEL = re.compile(r'\bтопливоподач\w*\b')
RELEASE_LIMIT = 'Данных недостаточно, чтобы подтвердить возможность перегона или выезда своим ходом.'
DECISION_NOTE = 'Решение о ремонте и выпуске машины принимает механик или ответственный специалист.'


def is_release_question(text: str) -> bool:
    text = text.casefold().replace('ё', 'е')
    return bool(_MOVE.search(text) and _ASK.search(text))


def release_query(message: str, recent_messages: Sequence[str], diagnostic: DiagnosticResult | None) -> str:
    context = ' '.join((message, *recent_messages)).casefold()
    fault = diagnostic.anomaly_type if diagnostic else None
    if fault == 'F02' or _OIL.search(context):
        return f'{message} Износ вкладышей, давление масла, проверка системы смазки.'
    if fault == 'F08' or _FUEL.search(context):
        return f'{message} Дизельная топливоподача, проверка топливной системы.'
    return message


def release_details(plan: SearchPlan) -> str:
    context = f'{plan.original_query} {plan.rewritten_query}'.casefold()
    if _OIL.search(context):
        return ('При подозрении на износ вкладышей нужны очный осмотр и фактическое давление масла '
                'с указанием температуры и оборотов при замере.')
    if _FUEL.search(context):
        return ('Отклонение в топливоподаче остаётся гипотезой. '
                'Для решения нужна очная проверка топливной системы.')
    return 'Для решения нужны фактическое состояние машины и результаты очной проверки у механика.'


def release_outcome(plan: SearchPlan, lookup: Callable[[SearchPlan], EvidenceBundle] | None) -> SearchOutcome:
    # Manuals can describe safety conditions. A matching page alone cannot establish the state of this vehicle.
    summary = [RELEASE_LIMIT]
    try:
        evidence = lookup(plan) if lookup else EvidenceBundle()
    except AgentError as error:
        if error.code not in {'RAG_BACKEND_UNAVAILABLE', 'RAG_INDEX_UNAVAILABLE'}:
            raise
        summary.append('Сейчас поиск по руководствам недоступен.')
    else:
        if any(plan.engine_model in item.engine_models for item in evidence.items):
            summary.append('Страницы руководства не подтверждают состояние этой машины.')
    summary.extend((release_details(plan), DECISION_NOTE))
    return SearchOutcome(plan=plan, evidence=EvidenceBundle(), claims=[], summary=' '.join(summary))
