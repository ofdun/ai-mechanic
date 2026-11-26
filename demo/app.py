"""Run the host with the checked-in training guide and the team's agent pipeline."""
import hashlib
import re
from pathlib import Path

from demo.fixtures import DOCUMENT_ID, ENGINE, GUIDE_ENTRIES, MANUAL, PDF_NAME
from ml.contracts import TechnicalClaim, TechnicalDraft, VerifiedResponse
from ml.pdf_baseline import PdfPageRetriever
from ml.pipeline import Pipeline
from ml.tools.tickets import TicketStore
from demo.host import create_app as create_host


def words(text):
    return re.findall(r'\w+', text.casefold())


class TrainingGuide:
    """Only the two literal statements on the training guide's second page are approved."""

    @staticmethod
    def claims(evidence, question=None):
        claims = []
        for item in evidence.items:
            if (item.document_id, item.manual, item.pdf_page) != (DOCUMENT_ID, MANUAL, 2):
                continue
            if item.engine_models != [ENGINE]:
                continue
            for heading, kind, text in GUIDE_ENTRIES:
                if question is not None and words(heading) != words(question):
                    continue
                if f'{heading}\n{text}' in item.text:
                    claims.append(TechnicalClaim(kind=kind, text=text, evidence_ids=[item.evidence_id]))
        return claims

    def generate(self, plan, evidence, diagnostic):
        if plan.engine_model != ENGINE or not diagnostic or diagnostic.anomaly_type != 'F08':
            return TechnicalDraft()
        question = None if plan.anomaly_type == 'F08' else plan.original_query
        return TechnicalDraft(claims=self.claims(evidence, question))

    def verify(self, draft, evidence):
        approved = self.claims(evidence)
        return VerifiedResponse(claims=[claim for claim in draft.claims if claim in approved])


class TrainingPipeline(Pipeline):
    def ticket_store_for(self, actor_id):
        # Each visitor runs the same synthetic exercise independently.
        actor = hashlib.sha256(actor_id.encode()).hexdigest()
        return TicketStore(self.settings.artifacts_dir / 'training-tickets' / f'{actor}.sqlite3',
                           self.settings.http_timeout_seconds * 4 + 10)


def training_pipeline(settings, http, index):
    guide = TrainingGuide()
    return TrainingPipeline(settings, http, retriever=PdfPageRetriever(index), generator=guide, verifier=guide)


def create_app():
    digest = hashlib.sha256(Path(__file__).with_name(PDF_NAME).read_bytes()).hexdigest()
    return create_host(pipeline_factory=training_pipeline, documents={DOCUMENT_ID: (PDF_NAME, digest)})
