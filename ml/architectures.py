import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml.config import Settings
from ml.contracts import (
    AgentError,
    DiagnosticResult,
    EvidenceBundle,
    Generator,
    Retriever,
    SearchOutcome,
    SearchPlan,
    TechnicalClaim,
    TechnicalDraft,
    Text,
)
from ml.generation import prompts
from ml.generation.claims import MAX_CLAIMS, GroundedClaim, conflict_group
from ml.generation.finalizer import find_conflicts
from ml.generation.generator import ClaimGenerator, diagnosis_context, search_query
from ml.generation.plain import PlainWriter
from ml.generation.reasoner import TechnicalReasoner
from ml.generation.verifier import EvidenceVerifier
from ml.pipeline import scope_evidence
from ml.providers.openrouter import OpenRouterClient

# Selection replies are only indexes, but a model that reasons by default can spend the whole budget thinking.
SELECTION_TOKENS = 4000
# The disagreement check may add a few claims from other manuals on top of the selected answer.
COVERAGE_CLAIMS = 4


class QueryOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    rewritten_query: Text = Field(max_length=700)
    subqueries: list[Text] = Field(max_length=5)


class ApprovalOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    indexes: list[int] = Field(max_length=MAX_CLAIMS)


class CriticOutput(ApprovalOutput):
    corrective_query: str = Field(max_length=700)


def messages(instruction: str, **payload: Any) -> list[dict[str, str]]:
    return [
        {'role': 'system', 'content': instruction},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
    ]


def approved_indexes(claims: list[TechnicalClaim], indexes: list[int]) -> list[TechnicalClaim]:
    return [claims[index] for index in dict.fromkeys(indexes) if 0 <= index < len(claims)]


def diagnostic_context(diagnostic: DiagnosticResult | None) -> dict[str, Any] | None:
    return diagnosis_context(diagnostic)


class LLMEvidenceWorkflow:
    def __init__(self, settings: Settings, client: OpenRouterClient, retriever: Retriever, reviewed: Generator):
        self.settings = settings
        self.client = client
        self.retriever = retriever
        self.reviewed = reviewed

    def search(self, plan: SearchPlan) -> EvidenceBundle:
        return scope_evidence(self.retriever.search(plan), plan.engine_model)

    def valid_claims(
        self, claims: list[TechnicalClaim], plan: SearchPlan, evidence: EvidenceBundle,
        diagnostic: DiagnosticResult | None, *, include_reviewed: bool = False,
    ) -> list[TechnicalClaim]:
        """Keep manually reviewed claims and claims that pass the programmatic evidence check."""
        trusted = self.reviewed.generate(plan, evidence, diagnostic).claims
        accepted = list(trusted) if include_reviewed else []
        unchecked = [claim for claim in claims if claim not in trusted]
        verifier = EvidenceVerifier(plan.engine_model, plan.anomaly_type)
        verified = verifier.verify(TechnicalDraft(claims=unchecked), evidence).claims
        for claim in claims:
            if (claim in trusted or claim in verified) and claim not in accepted:
                accepted.append(claim)
        return accepted[:MAX_CLAIMS]

    def generate(
        self, role: str, model: str, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None,
        *, note: str = '',
    ) -> list[TechnicalClaim]:
        drafter = TechnicalReasoner if role == 'b_technical_reasoner' else ClaimGenerator
        return drafter(self.client, model, role=role, note=note).draft(plan, evidence, diagnostic)

    def review_payload(self, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None,
                       claims: list[TechnicalClaim]) -> dict[str, Any]:
        return {
            'question': plan.original_query, 'search_query': search_query(plan), 'engine_model': plan.engine_model,
            'diagnosis': diagnostic_context(diagnostic), 'sources': evidence.model_dump()['items'],
            'claims': [claim.model_dump() for claim in claims],
        }

    def find_disagreements(
        self, role: str, model: str, plan: SearchPlan, evidence: EvidenceBundle, diagnostic: DiagnosticResult | None,
        claims: list[TechnicalClaim],
    ) -> list[TechnicalClaim]:
        """Look in the other manuals for what contradicts the answer, so both instructions reach the mechanic."""
        document = {item.evidence_id: item.document_id for item in evidence.items}
        manual = {item.evidence_id: item.manual for item in evidence.items}
        if not claims or len(set(document.values())) < 2:
            return claims
        known = [{'index': index, 'kind': claim.kind, 'text': claim.text,
                  'manual': manual.get(claim.evidence_ids[0], '') if claim.evidence_ids else ''}
                 for index, claim in enumerate(claims)]
        note = prompts.DISAGREEMENT_NOTE + json.dumps(known, ensure_ascii=False)
        found = self.valid_claims(self.generate(role, model, plan, evidence, diagnostic, note=note),
                                  plan, evidence, diagnostic)
        merged = list(claims)
        for claim in found:
            match = re.fullmatch(r'k(\d+)', conflict_group(claim))
            if not match or int(match[1]) >= len(claims) or len(merged) - len(claims) >= COVERAGE_CLAIMS:
                continue
            index = int(match[1])
            target = merged[index]
            own = {document.get(key) for key in target.evidence_ids}
            if not claim.evidence_ids or document.get(claim.evidence_ids[0]) in own or claim in merged:
                continue  # Only another manual can disagree with an instruction.
            label = conflict_group(target) or f'k{index}'
            merged[index] = GroundedClaim.model_validate({**target.model_dump(), 'conflict_group': label})
            merged.append(claim.model_copy(update={'conflict_group': label}))
        return merged

    def outcome(self, plan: SearchPlan, evidence: EvidenceBundle, claims: list[TechnicalClaim]) -> SearchOutcome:
        summary = ''
        if claims:
            sources = {item.evidence_id: item for item in evidence.items}
            writer = PlainWriter(self.client, self.settings.a_generator_model, language=self.settings.answer_language)
            summary = writer.write(plan.original_query, plan.engine_model, claims,
                                   conflict=bool(find_conflicts(claims, sources)), allow_assessment=plan.anomaly_type is not None)
        return SearchOutcome(plan=plan, evidence=evidence, claims=claims, summary=summary)

    def run(self, plan: SearchPlan, diagnostic: DiagnosticResult | None) -> SearchOutcome:
        if not self.settings.allow_external_evidence:
            raise AgentError('LLM_EVIDENCE_NOT_ALLOWED', 'Для передачи источников в OpenRouter нужно разрешение. Для локального PDF включите AGENT_ARCHITECTURE=baseline.')
        if self.settings.agent_architecture == 'b':
            return self.run_evidence_grounded(plan, diagnostic)
        evidence = self.search(plan)
        if not evidence.items:
            return SearchOutcome(plan=plan, evidence=evidence, claims=[])
        draft = self.valid_claims(self.generate('a_generator', self.settings.a_generator_model, plan, evidence, diagnostic),
                                  plan, evidence, diagnostic, include_reviewed=True)
        if not draft:
            return SearchOutcome(plan=plan, evidence=evidence, claims=[])
        approval = self.client.complete_json('a_verifier_finalizer', self.settings.a_verifier_model, messages(
            prompts.REVIEW, **self.review_payload(plan, evidence, diagnostic, draft),
        ), ApprovalOutput, max_tokens=SELECTION_TOKENS)
        final = approved_indexes(draft, approval.indexes)
        final = self.find_disagreements('a_disagreement', self.settings.a_generator_model, plan, evidence, diagnostic, final)
        return self.outcome(plan, evidence, final)

    def run_evidence_grounded(self, plan: SearchPlan, diagnostic: DiagnosticResult | None) -> SearchOutcome:
        strategy = self.client.complete_json('b_query_strategist', self.settings.b_query_model, messages(
            'Сформируй русскоязычный поисковый запрос и до пяти подзапросов по переданным данным. '
            'Сохрани смысл исходного вопроса и используй сведения текущей диагностики только по теме.',
            plan=plan.model_dump(), diagnosis=diagnostic_context(diagnostic),
        ), QueryOutput)
        plan = SearchPlan.model_validate({**plan.model_dump(), **strategy.model_dump()})
        evidence = self.search(plan)
        if not evidence.items:
            return SearchOutcome(plan=plan, evidence=evidence, claims=[])
        draft = self.valid_claims(self.generate('b_technical_reasoner', self.settings.b_reasoner_model, plan, evidence, diagnostic),
                                  plan, evidence, diagnostic, include_reviewed=True)
        critic = self.client.complete_json('b_evidence_critic', self.settings.b_critic_model, messages(
            prompts.CRITIC, **self.review_payload(plan, evidence, diagnostic, draft),
        ), CriticOutput)
        approved = approved_indexes(draft, critic.indexes)
        if query := critic.corrective_query.strip():
            extra = self.search(plan.model_copy(update={'rewritten_query': query, 'subqueries': []}))
            known = {item.evidence_id for item in evidence.items}
            additions = [item for item in extra.items if item.evidence_id not in known]
            if additions:
                evidence = EvidenceBundle(items=[*evidence.items, *additions])
                repaired = self.generate('b_repair', self.settings.b_repair_model, plan, evidence, diagnostic,
                                         note=prompts.REPAIR_NOTE)
                approved = self.valid_claims(repaired, plan, evidence, diagnostic, include_reviewed=True)
        if not approved:
            return SearchOutcome(plan=plan, evidence=evidence, claims=[])
        final = self.client.complete_json('b_finalizer', self.settings.b_finalizer_model, messages(
            prompts.FINAL_SELECTION, question=plan.original_query, search_query=search_query(plan),
            verified_claims=[claim.model_dump() for claim in approved],
        ), ApprovalOutput, max_tokens=SELECTION_TOKENS)
        chosen = approved_indexes(approved, final.indexes)
        chosen = self.find_disagreements('b_disagreement', self.settings.b_repair_model, plan, evidence, diagnostic, chosen)
        return self.outcome(plan, evidence, chosen)
