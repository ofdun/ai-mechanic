"""A short plain-language lead for mechanics, written only from verified claims and checked again."""

import json
import logging

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


class PlainOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    lead: str = Field(max_length=LEAD_CHARS)
    lines: list[str] = Field(max_length=MAX_LINES)


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


def plain_text(output: PlainOutput, claims: list[TechnicalClaim], engine_model: str) -> str:
    """The checked lead and lines as chat text; an unsafe lead drops the whole summary."""
    if not safe_line(output.lead, claims, engine_model):
        return ''
    lines = [' '.join(line.split()) for line in output.lines[:MAX_LINES] if safe_line(line, claims, engine_model)]
    return '\n'.join([' '.join(output.lead.split()), *dict.fromkeys(lines)])


class PlainWriter:
    """Writes the human part of an answer. Any failure leaves the structured answer as it is."""

    def __init__(self, client: OpenRouterClient, model: str, *, language: str = 'ru'):
        self.client = client
        self.model = model
        self.language = language

    def write(self, question: str, engine_model: str, claims: list[TechnicalClaim]) -> str:
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
        return plain_text(output, claims, engine_model)
