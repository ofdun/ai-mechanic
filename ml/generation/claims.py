"""Claims with the verbatim page excerpt that supports them."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ml.contracts import TechnicalClaim


ClaimKind = Literal['observation', 'cause', 'risk', 'check', 'reference_value', 'operation', 'spare_part']
KIND_ORDER: tuple[str, ...] = ('observation', 'cause', 'risk', 'check', 'reference_value', 'operation', 'spare_part')
MAX_CLAIMS = 8


class GroundedClaim(TechnicalClaim):
    """A claim plus the excerpt that proves it.

    ``quote`` is copied from the cited page. Claims that belong to one disagreement between sources share
    a non-empty ``conflict_group``. The shared API keeps serialising the base ``TechnicalClaim`` fields.
    """

    quote: str = ''
    conflict_group: str = ''


class ClaimOutput(BaseModel):
    """One claim as an LLM returns it. All fields are required by the strict JSON schema."""

    model_config = ConfigDict(extra='forbid')

    kind: ClaimKind
    text: str
    evidence_ids: list[str]
    quote: str
    conflict_group: str


class ClaimsOutput(BaseModel):
    model_config = ConfigDict(extra='forbid')

    claims: list[ClaimOutput] = Field(max_length=MAX_CLAIMS)


def to_claims(output: ClaimsOutput) -> list[GroundedClaim]:
    """Convert LLM output, dropping malformed items instead of failing the whole answer."""
    claims = []
    for item in output.claims[:MAX_CLAIMS]:
        try:
            claims.append(GroundedClaim(
                kind=item.kind, text=' '.join(item.text.split()), quote=item.quote.strip(),
                evidence_ids=list(dict.fromkeys(key.strip() for key in item.evidence_ids if key.strip())),
                conflict_group=item.conflict_group.strip(),
            ))
        except ValidationError:
            continue
    return claims


def quote_of(claim: TechnicalClaim) -> str:
    """The excerpt to verify. A claim from an older generator is checked by its own text."""
    return claim.quote if isinstance(claim, GroundedClaim) and claim.quote else claim.text


def conflict_group(claim: TechnicalClaim) -> str:
    return claim.conflict_group if isinstance(claim, GroundedClaim) else ''
