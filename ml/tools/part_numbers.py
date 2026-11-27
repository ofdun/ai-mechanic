"""Part designations (236-1307010-А3, 5340.1002246-01, 406.1004045-01) from the manual pages behind an answer.

Only pages already scoped to the vehicle's engine and cited by a verified claim are read, so a part of another
engine's manual never appears. Nothing comes from external catalogues: no prices, stock or compatibility."""
import re

from ml.contracts import AgentResponse, PartNumber
from ml.retrieval.sparse import tokenizer

# A 3–4 digit group, then «-» with 6–8 digits or «.» with 4–8 digits, then an optional suffix (-01, -10, -А3).
# The second group excludes ranges such as 1500-1800 and values such as 100-125.
DESIGNATION = re.compile(r'(?<![\w.,\-])([1-9]\d{2,3}(?:-\d{6,8}|\.\d{4,8})(?:-(?:\d{2,3}|[А-ЯA-Z]\d?))?)(?![\w-]|[.,]\d)')
LIMIT = 10
CONTEXT_CHARS = 140
PART_QUESTION = re.compile(r'запчаст|детал|обозначени|каталожн|номер\w* детал|артикул|что заказ', re.IGNORECASE)
NONE_FOUND = 'Подтвердить обозначение детали по руководству этого двигателя не удалось.'


_tokenize = tokenizer('stem')
_QUERY_WORDS = set(_tokenize('какой какая какие каков подскажи скажи покажи найти нужен запчасть деталь '
                             'обозначение каталожный номер артикул пожалуйста по руководству'))


def requested_names(question: str) -> set[str]:
    if not PART_QUESTION.search(question):
        return set()
    return set(_tokenize(question)) - _QUERY_WORDS


def named_designation(line: str, match: re.Match, names: set[str]) -> str | None:
    """Require the requested component name immediately before this catalogue code."""
    if not names:
        return None
    words = list(re.finditer(r'[^\W\d_]+', line[:match.start()]))
    if len(words) >= len(names):
        start = words[-len(names)].start()
        if set(_tokenize(line[start:match.start()])) == names:
            return line[start:match.end()]
    return None


def context(lines: list[str], index: int) -> str:
    """The manual line with the designation; a bare code line takes its name from the line above."""
    line = lines[index].strip()
    if (index > 0 and not DESIGNATION.search(lines[index - 1])
            and not re.search(r'[А-Яа-яA-Za-z]{3}', DESIGNATION.sub('', line))):
        line = f'{lines[index - 1].strip()} {line}'
    line = ' '.join(line.split())
    return line if len(line) <= CONTEXT_CHARS else line[:CONTEXT_CHARS - 1].rstrip() + '…'


def part_numbers(response: AgentResponse, names: set[str] | None = None) -> list[PartNumber]:
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
                if names and not named_designation(line, match, names):
                    continue
                code = match.group(1)
                if code not in found and len(found) < LIMIT:
                    found[code] = PartNumber(designation=code, context=context(lines, index), manual=page.manual,
                                             document_id=page.document_id, pdf_page=page.pdf_page,
                                             evidence_id=page.evidence_id)
    return list(found.values())


def asks_for_parts(response: AgentResponse, question: str | None) -> bool:
    return any(claim.kind == 'spare_part' for claim in response.claims) or bool(question and PART_QUESTION.search(question))
