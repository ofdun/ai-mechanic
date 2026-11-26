"""Generation and verification: grounded claims, their programmatic check and the mechanic's answer.

Owner: Алексей. Pipeline hooks: ``Generator`` (``ClaimGenerator``, ``TechnicalReasoner``), ``Verifier``
(``EvidenceVerifier``) and the final message (``MechanicFinalizer``).
"""
from ml.generation.claims import GroundedClaim
from ml.generation.finalizer import MechanicAnswer, MechanicFinalizer
from ml.generation.generator import ClaimGenerator
from ml.generation.reasoner import TechnicalReasoner
from ml.generation.verifier import EvidenceVerifier, VerificationReport

__all__ = [
    'ClaimGenerator', 'EvidenceVerifier', 'GroundedClaim', 'MechanicAnswer', 'MechanicFinalizer',
    'TechnicalReasoner', 'VerificationReport',
]
