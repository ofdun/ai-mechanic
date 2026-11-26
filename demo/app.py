"""Run the host with the checked-in training guide and the team's agent pipeline."""
import hashlib
from pathlib import Path

from demo.fixtures import DOCUMENT_ID, ENGINE, MANUAL, OPERATION, PDF_NAME, RISK
from ml.contracts import TechnicalClaim, TechnicalDraft, VerifiedResponse
from ml.pdf_baseline import PdfPageRetriever
from ml.pipeline import Pipeline
from ml.tools.tickets import TicketStore
from web.app import create_app as create_host


class TrainingGuide:
    """Only the two literal statements on the training guide's second page are approved."""

    @staticmethod
    def claims(evidence):
        return [TechnicalClaim(kind=kind, text=text, evidence_ids=[item.evidence_id])
                for item in evidence.items
                if (item.document_id, item.manual, item.pdf_page) == (DOCUMENT_ID, MANUAL, 2)
                and item.engine_models == [ENGINE]
                for kind, text in [('risk', RISK), ('operation', OPERATION)] if text in item.text]

    def generate(self, plan, evidence, diagnostic):
        if plan.anomaly_type == 'F08' and diagnostic and diagnostic.anomaly_type == 'F08':
            return TechnicalDraft(claims=self.claims(evidence))
        return TechnicalDraft()

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
