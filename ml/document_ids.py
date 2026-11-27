YAMZ_236_OPERATIONS_ID = 'yamz-236-operations-2010'
LEGACY_YAMZ_236_OPERATIONS_ID = 'yamz-236_re_2010'


def document_id_variants(document_id: str) -> tuple[str, str | None]:
    """Return the published ID and the key used by older indexes."""
    if document_id in (YAMZ_236_OPERATIONS_ID, LEGACY_YAMZ_236_OPERATIONS_ID):
        return YAMZ_236_OPERATIONS_ID, LEGACY_YAMZ_236_OPERATIONS_ID
    return document_id, None
