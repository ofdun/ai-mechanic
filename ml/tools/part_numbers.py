"""Part designations (236-1307010-А3, 5340.1002246-01, 406.1004045-01) from the manual pages behind an answer.

Only pages already scoped to the vehicle's engine and cited by a verified claim are read, so a part of another
engine's manual never appears. Nothing comes from external catalogues: no prices, stock or compatibility."""
import re

from ml.contracts import AgentResponse, PartNumber

# A 3–4 digit group, then «-» with 6–8 digits or «.» with 4–8 digits, then an optional suffix (-01, -10, -А3).
# The second group excludes ranges such as 1500-1800 and values such as 100-125.
DESIGNATION = re.compile(r'(?<![\w.,\-])([1-9]\d{2,3}(?:-\d{6,8}|\.\d{4,8})(?:-(?:\d{2,3}|[А-ЯA-Z]\d?))?)(?![\w.,]?\d)')
LIMIT = 10
CONTEXT_CHARS = 140
PART_QUESTION = re.compile(r'запчаст|детал|обозначени|каталожн|номер\w* детал|артикул|что заказ', re.IGNORECASE)
NONE_FOUND = 'На страницах руководства, подтвердивших ответ, обозначений деталей нет.'


def context(lines: list[str], index: int) -> str:
    """The manual line with the designation; a bare code line takes its name from the line above."""
    line = lines[index].strip()
    if not re.search(r'[А-Яа-яA-Za-z]{3}', DESIGNATION.sub('', line)) and index > 0:
        line = f'{lines[index - 1].strip()} {line}'
    line = ' '.join(line.split())
    return line if len(line) <= CONTEXT_CHARS else line[:CONTEXT_CHARS - 1].rstrip() + '…'


def part_numbers(response: AgentResponse) -> list[PartNumber]:
    cited = list(dict.fromkeys(key for claim in response.claims for key in claim.evidence_ids))
    pages = {item.evidence_id: item for item in response.evidence.items
             if response.engine_model in item.engine_models}
    found: dict[str, PartNumber] = {}
    for key in cited:
        page = pages.get(key)
        if page is None:
            continue
        lines = page.text.splitlines()
        for index, line in enumerate(lines):
            for match in DESIGNATION.finditer(line):
                code = match.group(1)
                if code not in found and len(found) < LIMIT:
                    found[code] = PartNumber(designation=code, context=context(lines, index), manual=page.manual,
                                             document_id=page.document_id, pdf_page=page.pdf_page,
                                             evidence_id=page.evidence_id)
    return list(found.values())


def asks_for_parts(response: AgentResponse, question: str | None) -> bool:
    return any(claim.kind == 'spare_part' for claim in response.claims) or bool(question and PART_QUESTION.search(question))
