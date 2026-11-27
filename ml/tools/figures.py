"""Tightening order shown only on a figure: link the figure, never read a sequence off a drawing by guess.

A numbered sequence is given only from a catalog that people checked against the manual page (a local file,
REVIEWED_SEQUENCES_PATH, not part of the repository: it would carry manual content). The catalog entry is
used only when the page text still has the same SHA-256, so another edition or engine never gets it."""
import hashlib
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ml.contracts import AgentResponse, Evidence, FigureReference

QUESTION = re.compile(r'(?:порядо?к|последовательност|очерёдност|очередност)\w*.{0,50}(?:затяж|затягив|заворач|закручив)'
                      r'|(?:затяж|затягив|заворач)\w*.{0,60}(?:порядо?к|последовательност)', re.IGNORECASE | re.DOTALL)
# A caption may go on to the next line when that line starts in lower case («…шпилек\nкрепления … ЯМЗ-238М2»).
CAPTION = re.compile(r'Рис\.?\s*(\d+[а-я]?)\.?\s*((?:Последовательность|Порядок)\s+(?:затяжки|затягивания)[^\n]*'
                     r'(?:\n[а-яё][^\n]*)?)')
# «в порядке, указанном на рисунке 3», «в последовательности, указанной на рис.142», «на рисунках 6 и 7».
MENTION = re.compile(r'(?:порядке|последовательности)[,\s]+указанно\w*\s+на\s+рис(?:\.|унк\w*)\s*(\d+[а-я]?)'
                     r'(?:\s+и\s+(\d+[а-я]?))?', re.IGNORECASE)
ENGINE = re.compile(r'\b(?:ЯМЗ|ЗМЗ|КАМАЗ|ММЗ|Д)-?\s?\d{3,5}(?:\.\d+)?[А-ЯA-Z0-9]*', re.IGNORECASE)
SUBJECTS = ('голов', 'картер', 'маховик', 'крышк', 'коллектор', 'шатун', 'коренн', 'дроссел', 'клапан')
NOTE = ('Порядок затяжки показан на рисунке {figure} ({manual}, стр. {page}) — откройте его. '
        'Последовательность по рисунку агент не перечисляет: проверенных данных для неё нет.')


class ReviewedSequence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    document_id: str
    figure: str
    pdf_page: int
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    steps: list[str] = Field(min_length=2)
    reviewed_by: str = Field(min_length=1)


def page_hash(item: Evidence) -> str:
    return hashlib.sha256(' '.join(item.text.split()).encode()).hexdigest()


def load_sequences(path: Path | None) -> list[ReviewedSequence]:
    if path is None or not path.is_file():
        return []
    return [ReviewedSequence.model_validate(row) for row in json.loads(path.read_text(encoding='utf-8'))]


def subject_of(question: str) -> str | None:
    lowered = question.casefold()
    return next((stem for stem in SUBJECTS if stem in lowered), None)


def sequence_figures(response: AgentResponse, question: str | None) -> list[FigureReference]:
    """Figures with a tightening order on the engine's pages behind the answer, closest to the question first."""
    if not question or not QUESTION.search(question):
        return []
    subject = subject_of(question)
    found: dict[tuple[str, str], FigureReference] = {}
    pages = [item for item in response.evidence.items if response.engine_model in item.engine_models]
    about: dict[tuple[str, str], str] = {}           # the words around a figure, to match the question's subject
    excluded: set[tuple[str, str]] = set()
    for item in pages:
        for match in CAPTION.finditer(item.text):
            caption = ' '.join(match[2].split()).rstrip('.')
            if other_engine(caption, response.engine_model):
                excluded.add((item.document_id, match[1]))
                continue                              # «Рис. 7 … двигателей ЯМЗ-238М2» on a ЯМЗ-236М2 question
            key = (item.document_id, match[1])
            found.setdefault(key, FigureReference(figure=match[1], caption=caption, manual=item.manual,
                                                  document_id=item.document_id, pdf_page=item.pdf_page))
            about[key] = caption
    for item in pages:              # the text names the figure; its caption may be elsewhere or absent
        for match in MENTION.finditer(item.text):
            before = item.text[max(0, match.start() - 200):match.start()]
            for number in filter(None, (match[1], match[2])):
                key = (item.document_id, number)
                if key not in found and key not in excluded:
                    found[key] = FigureReference(figure=number, manual=item.manual, document_id=item.document_id,
                                                 pdf_page=item.pdf_page)
                    about[key] = before
    keys = list(found)
    if subject:
        keys = [key for key in keys if subject in about[key].casefold()]
    return [found[key] for key in keys][:2]


def other_engine(text: str, engine_model: str) -> bool:
    named = {match.group(0).replace(' ', '').upper() for match in ENGINE.finditer(text)}
    return bool(named) and not any(engine_model.upper().startswith(name) or name.startswith(engine_model.upper())
                                   for name in named)


def attach_sequences(figures: list[FigureReference], pages: list[Evidence],
                     reviewed: list[ReviewedSequence]) -> list[FigureReference]:
    """A reviewed sequence for the same document, figure and unchanged page; otherwise the figure link only."""
    hashes = {(item.document_id, item.pdf_page): page_hash(item) for item in pages}
    result = []
    for figure in figures:
        entry = next((row for row in reviewed if (row.document_id, row.figure, row.pdf_page) ==
                      (figure.document_id, figure.figure, figure.pdf_page)
                      and hashes.get((row.document_id, row.pdf_page)) == row.sha256), None)
        result.append(figure.model_copy(update={'sequence': entry.steps}) if entry else figure)
    return result


def figure_note(figure: FigureReference) -> str:
    return NOTE.format(figure=figure.figure, manual=figure.manual, page=figure.pdf_page)
