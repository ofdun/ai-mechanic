"""LLM generator of grounded claims, architecture A.

The generator only drafts. Each claim carries the excerpt it relies on, and ``EvidenceVerifier`` decides
what reaches the mechanic.
"""
import json
from typing import Any

from ml.contracts import DiagnosticResult, EvidenceBundle, SearchPlan, TechnicalDraft
from ml.generation import prompts
from ml.generation.claims import ClaimsOutput, GroundedClaim, to_claims
from ml.providers.openrouter import OpenRouterClient


def diagnosis_context(diagnostic: DiagnosticResult | None) -> dict[str, Any] | None:
    """Only what the generator needs. Scores are interpreted by code, not by the model."""
    if diagnostic is None:
        return None
    return diagnostic.model_dump(include={'status', 'anomaly_type', 'anomaly_title', 'description'})


def source_items(evidence: EvidenceBundle) -> list[dict[str, Any]]:
    return [item.model_dump(include={'evidence_id', 'manual', 'pdf_page', 'section', 'engine_models', 'text'})
            for item in evidence.items]


def search_query(plan: SearchPlan) -> str:
    """The planner's query restores the subject of a follow-up such as «почему это опасно?»."""
    return plan.rewritten_query if plan.rewritten_query.strip() != plan.original_query.strip() else ''


def messages(instruction: str, plan: SearchPlan, evidence: EvidenceBundle,
             diagnostic: DiagnosticResult | None) -> list[dict[str, str]]:
    payload = {
        'question': plan.original_query, 'search_query': search_query(plan), 'engine_model': plan.engine_model,
        'diagnosis': diagnosis_context(diagnostic), 'sources': source_items(evidence),
    }
    return [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
    ]


class ClaimGenerator:
    """``Generator`` that drafts claims with quotes through OpenRouter."""

    instruction = prompts.GENERATOR
    reason = False
    max_tokens = 3600

    def __init__(self, client: OpenRouterClient, model: str, *, role: str = 'a_generator', note: str = ''):
        self.client = client
        self.model = model
        self.role = role
        self.note = note

    def draft(self, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None) -> list[GroundedClaim]:
        if not evidence.items:
            return []
        output = self.client.complete_json(
            self.role, self.model, messages(self.instruction + self.note, plan, evidence, diagnostic), ClaimsOutput,
            max_tokens=self.max_tokens, reason=self.reason,
        )
        return to_claims(output)

    def generate(self, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None) -> TechnicalDraft:
        return TechnicalDraft(claims=self.draft(plan, evidence, diagnostic))
