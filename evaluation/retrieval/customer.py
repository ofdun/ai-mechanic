"""Adapter from the team benchmark `evaluation/datasets/customer_questions.jsonl` to retrieval golden questions.

Only cases routed to `search_manuals` with an engine model are scored. Expected pages come from
`expected_evidence` with scope `passage` or `module`; `context` pages only frame an answer and are not
counted. `abstain` cases become questions without an answer in the corpus. The benchmark names manuals
by source code (`y236re`), which is matched through the PDF file of the customer's kit to the canonical
`document_id` of the ingestion manifest, as in `evaluation.generation.oracle_context`. Renaming an ID
does not touch the dataset.

Rewrites for the `rewritten` and `combined` query modes come from a JSONL file made by
`python -m evaluation.retrieval.rewrite` (one Planner call per question, reused by every run), with the
dataset's own `reference_rewrite` as a fallback.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from evaluation.retrieval.metrics import GoldenQuestion, PageRef
from ml.contracts import AgentError, Text
from ml.retrieval.chunking import nfc

SCORED_SCOPES = ('passage', 'module')
# Source codes of the benchmark (evaluation/datasets/README.md) → PDF files of the customer's kit.
DOCUMENT_FILES = {
    'y236re': 'ЯМЗ_236_руководство по эксплуатации.pdf',
    'y236rk': 'ЯМЗ_236_руководство по РЕМОНТУ.pdf',
    'y536re': 'yamz-536_re_2016.pdf',
    'y536rk': 'yamz-536_rk_2013.pdf',
    'z5234': 'zmz-5234_rk_2025.pdf',
    'z409': 'zmz-40904-40905_rk_2025.pdf',
}


class Rewrite(BaseModel):
    id: Text
    rewritten_query: Text | None = None
    subqueries: list[Text] = Field(default_factory=list, max_length=5)


@dataclass
class Adapted:
    questions: list[GoldenQuestion]
    skipped: dict[str, list[str]] = field(default_factory=dict)


def read_jsonl(path: Path, code: str) -> list[dict]:
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as error:
        raise AgentError(code, f'Не удалось прочитать {path}.') from error


def manual_ids(manifest: Path) -> dict[str, str]:
    """Benchmark source code, designation and title of each manual → canonical document_id."""
    try:
        documents = json.loads(manifest.read_text(encoding='utf-8'))['documents']
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise AgentError('RAG_GOLDEN_INVALID', f'Не удалось прочитать каталог руководств {manifest}.') from error
    names = {}
    by_file = {nfc(document['file']): document['document_id'] for document in documents if document.get('file')}
    for code, file in DOCUMENT_FILES.items():
        if nfc(file) in by_file:
            names[code] = by_file[nfc(file)]
    for document in documents:
        for name in (document.get('designation'), document.get('title')):
            if name:
                names[nfc(name)] = document['document_id']
    return names


def load_rewrites(path: Path) -> dict[str, Rewrite]:
    rewrites = {}
    for number, row in enumerate(read_jsonl(path, 'RAG_GOLDEN_INVALID'), 1):
        try:
            rewrite = Rewrite.model_validate(row)
        except ValidationError as error:
            raise AgentError('RAG_GOLDEN_INVALID', f'{path.name}:{number} не соответствует формату переформулировки.') from error
        rewrites[rewrite.id] = rewrite
    return rewrites


def reference_rewrite(row: dict, engine_model: str) -> str | None:
    reference = row.get('reference_rewrite')
    if not reference:
        return None
    template = reference['template']
    return template.replace(reference['placeholder'], engine_model) if reference.get('placeholder') else template


def is_search_case(row: dict) -> bool:
    return row.get('expected_tool') == 'search_manuals' and bool((row.get('context') or {}).get('engine_model'))


def load_customer_questions(
    path: Path, manifest: Path, split: str | None = None, rewrites: dict[str, Rewrite] | None = None,
) -> Adapted:
    names = manual_ids(manifest)
    rewrites = rewrites or {}
    adapted = Adapted(questions=[])
    for row in read_jsonl(path, 'RAG_GOLDEN_INVALID'):
        if not is_search_case(row) or (split is not None and row.get('split') != split):
            continue
        engine_model = row['context']['engine_model']
        evidence = row.get('expected_evidence')
        if evidence is None:
            adapted.skipped.setdefault('нет разметки страниц', []).append(row['id'])
            continue
        relevant = []
        for item in evidence:
            if item.get('scope') not in SCORED_SCOPES:
                continue
            document_id = names.get(item.get('document') or '') or names.get(nfc(item.get('manual') or ''))
            if document_id is None:
                raise AgentError('RAG_GOLDEN_INVALID',
                                 f'{row["id"]}: руководства {item.get("document")} «{item.get("manual")}» нет в каталоге.')
            relevant.extend(PageRef(document_id=document_id, pdf_page=page) for page in item.get('pdf_pages', []))
        abstain = row.get('answer_outcome') == 'abstain'
        if abstain:
            relevant = []
        elif not relevant:
            adapted.skipped.setdefault('только контекстные страницы', []).append(row['id'])
            continue
        rewrite = rewrites.get(row['id'])
        rewritten = rewrite.rewritten_query if rewrite else reference_rewrite(row, engine_model)
        adapted.questions.append(GoldenQuestion(
            id=row['id'], engine_model=engine_model, question=row['text'],
            relevant=list({(ref.document_id, ref.pdf_page): ref for ref in relevant}.values()), split=row.get('split', 'dev'),
            rewritten_query=rewritten, subqueries=rewrite.subqueries if rewrite else [],
            tags=[row.get('suite', ''), row.get('answer_outcome') or '', *row.get('focus', [])],
        ))
    return adapted


def is_customer_format(path: Path) -> bool:
    """customer_questions.jsonl uses `text`, the simplified retrieval format uses `question`."""
    try:
        with path.open(encoding='utf-8') as file:
            first = next((line for line in file if line.strip()), '{}')
        return 'text' in json.loads(first)
    except (OSError, json.JSONDecodeError):
        return False
