"""Technical reasoner, architecture B: a stronger model builds the symptom → cause → check → repair chain."""
from ml.generation import prompts
from ml.generation.generator import ClaimGenerator
from ml.providers.openrouter import OpenRouterClient


class TechnicalReasoner(ClaimGenerator):
    instruction = prompts.REASONER
    reason = True
    max_tokens = 6000

    def __init__(self, client: OpenRouterClient, model: str, *, role: str = 'b_technical_reasoner', note: str = ''):
        super().__init__(client, model, role=role, note=note)
