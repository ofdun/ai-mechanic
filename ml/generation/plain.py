"""A short plain-language lead for mechanics, written only from verified claims and checked again."""

import json
import logging
import re

from pydantic import BaseModel, ConfigDict, Field

from ml.contracts import AgentError, TechnicalClaim
from ml.generation import prompts
from ml.generation.claims import conflict_group
from ml.generation.text import claim_numbers, fold, has_negation
from ml.generation.verifier import grants_permission, mentions_other_engine
from ml.providers.openrouter import OpenRouterClient

logger = logging.getLogger(__name__)

MAX_LINES = 3
LEAD_CHARS = 120
LINE_CHARS = 140
ASSESSMENT_CHARS = 200
ASSESSMENT_LABEL = 'Оценка агента, не из руководства'
# The schema leaves room on purpose: a provider that enforces maxLength cuts the text mid-word («регули-»).
# Lengths are enforced in code by whole sentences instead.
OUTPUT_CHARS = 600
_SENTENCE_END = re.compile(r'[.!?…](?=\s|$)')


class PlainOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    lead: str = Field(max_length=OUTPUT_CHARS)
    lines: list[str] = Field(max_length=MAX_LINES)
    assessment: str = Field(max_length=OUTPUT_CHARS)


def whole_sentences(text: str, limit: int) -> str:
    """The longest run of complete sentences within ``limit``; empty when even the first is cut or too long."""
    text = ' '.join(text.split())
    if not text:
        return ''
    if len(text) <= limit and not text.endswith(('-', '‐', '–', ',')):
        return text
    ends = [match.end() for match in _SENTENCE_END.finditer(text) if match.end() <= limit]
    return text[:ends[-1]] if ends else ''


def safe_line(line: str, claims: list[TechnicalClaim], engine_model: str) -> bool:
    """A plain line may only restate claims: same numbers, no new permission, negation or engine."""
    text = ' '.join(line.split())
    if not text or len(text) > LINE_CHARS:
        return False
    stated = set().union(*(claim_numbers(claim.text) for claim in claims))
    if not claim_numbers(text) <= stated:
        return False
    if grants_permission(text) and not any(grants_permission(claim.text) for claim in claims):
        return False
    if has_negation(text) and not any(has_negation(claim.text) for claim in claims) and 'по-разному' not in fold(text):
        return False
    return not mentions_other_engine(text, engine_model)


def safe_assessment(text: str, claims: list[TechnicalClaim], engine_model: str) -> bool:
    """The agent's own view may warn, but never permits operation or brings numbers the claims lack."""
    text = ' '.join(text.split())
    if not text or len(text) > ASSESSMENT_CHARS or grants_permission(text) or _CATEGORICAL.search(fold(text)):
        return False
    stated = set().union(*(claim_numbers(claim.text) for claim in claims))
    return claim_numbers(text) <= stated and not mentions_other_engine(text, engine_model)


CONFLICT_LEAD = 'Руководства пишут по-разному. Ниже обе инструкции.'
# Only the manual may forbid outright (customer rule); the agent's own view stays a warning.
_CATEGORICAL = re.compile(r'нельзя|запрещ|категорич|недопустим|ни в коем')
# With disagreeing manuals the agent does not pick a side: no yes/no, no «suitable option».
_CHOOSES = re.compile(r'(?:^|\s)(?:да|нет|можно|нельзя|подходит|выбирайте|правильн[а-я]*|верн[а-я]*)(?=[\s,.:!—-]|$)')


def plain_text(output: PlainOutput, claims: list[TechnicalClaim], engine_model: str, *, conflict: bool = False) -> str:
    """The checked lead, lines and labelled assessment as chat text; an unsafe lead drops the whole summary."""
    lead = ' '.join(output.lead.split())
    if conflict:
        lead = CONFLICT_LEAD
    elif not safe_line(lead, claims, engine_model):
        return ''
    lines = [' '.join(line.split()) for line in output.lines[:MAX_LINES] if safe_line(line, claims, engine_model)
             and not (conflict and (_CHOOSES.search(fold(line)) or 'по-разному' in fold(line)))]
    if safe_assessment(output.assessment, claims, engine_model):
        lines.append(f'⚠️ {ASSESSMENT_LABEL}: {" ".join(output.assessment.split())}')
    return '\n'.join([lead, *dict.fromkeys(lines)])


class PlainWriter:
    """Writes the human part of an answer. Any failure leaves the structured answer as it is."""

    def __init__(self, client: OpenRouterClient, model: str, *, language: str = 'ru'):
        self.client = client
        self.model = model
        self.language = language

    def write(self, question: str, engine_model: str, claims: list[TechnicalClaim], *, conflict: bool = False) -> str:
        if not claims:
            return ''
        payload = {
            'question': question, 'engine_model': engine_model, 'language': self.language,
            'claims': [{'kind': claim.kind, 'text': claim.text, 'conflict_group': conflict_group(claim)}
                       for claim in claims],
        }
        try:
            output = self.client.complete_json('plain_answer', self.model, [
                {'role': 'system', 'content': prompts.PLAIN},
                {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
            ], PlainOutput, max_tokens=800)
        except AgentError:
            logger.warning('Plain answer failed; the structured answer stays without a lead', exc_info=True)
            return ''
        output = output.model_copy(update={'assessment': whole_sentences(output.assessment, ASSESSMENT_CHARS)})
        return plain_text(output, claims, engine_model, conflict=conflict)
