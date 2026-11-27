"""Text helpers for grounding claims in PDF text.

PDF extraction breaks words across lines, interleaves table columns and glues neighbouring cells
(``пределы0,20-0,40``). Quotes are therefore matched on a normalised stream of letters and digits with
short, bounded gaps, while every number has to stay intact next to its own words on the page.
"""
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation


# Latin and Cyrillic look-alikes are folded together, so ``M12x1,25`` matches ``М12х1,25``.
_LOOKALIKES = str.maketrans({
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у', 'k': 'к', 'm': 'м', 't': 'т',
    'b': 'в', 'h': 'н', 'ё': 'е', '×': 'х',
})

MIN_SEGMENT = 3
MAX_GAP = 60
MAX_ANCHORS = 64


def fold(value: str) -> str:
    return unicodedata.normalize('NFKC', value).casefold().translate(_LOOKALIKES)


@dataclass(frozen=True)
class Normalized:
    """Letters and digits of a text with the original index of every character."""

    chars: str
    origin: tuple[int, ...]

    @classmethod
    def of(cls, text: str) -> 'Normalized':
        chars, origin = [], []
        for index, symbol in enumerate(text):
            for char in fold(symbol):
                if char.isalnum():
                    chars.append(char)
                    origin.append(index)
        return cls(''.join(chars), tuple(origin))


@dataclass(frozen=True)
class Segment:
    quote_start: int
    quote_end: int
    page_start: int
    page_end: int


@dataclass(frozen=True)
class Alignment:
    """Where a quote occurs in a page. ``start`` and ``end`` index the original page text."""

    start: int
    end: int
    segments: tuple[Segment, ...]

    @property
    def is_exact(self) -> bool:
        return len(self.segments) == 1


def _longest_prefix(quote: str, at: int, page: str, low: int, last_start: int) -> tuple[int, int]:
    """Longest prefix of ``quote[at:]`` that starts in ``page[low:last_start + 1]``."""
    remaining = len(quote) - at
    shortest = min(MIN_SEGMENT, remaining)

    def position(length: int) -> int:
        return page.find(quote[at:at + length], low, last_start + length)

    if position(shortest) < 0:
        return 0, -1
    good, bad = shortest, remaining + 1
    while bad - good > 1:
        middle = (good + bad) // 2
        if position(middle) >= 0:
            good = middle
        else:
            bad = middle
    return good, position(good)


def _clean_gap(text: str, haystack: Normalized, previous_end: int, next_start: int) -> bool:
    """Page text skipped between two matched pieces lies between whole words and changes no meaning.

    Without this ``в|ы|пускного`` would match ``впускного`` and a skipped ``не`` would flip a statement.
    """
    gap = text[haystack.origin[previous_end - 1] + 1:haystack.origin[next_start]]
    if not gap or gap[0].isalnum() or gap[-1].isalnum():
        return False
    return not (has_negation(gap) or has_condition(gap))


def _follow(quote: str, page_text: str, haystack: Normalized, start: int) -> list[Segment] | None:
    page = haystack.chars
    segments, at, cursor = [], 0, start
    while at < len(quote):
        last_start = cursor if not segments else cursor + MAX_GAP
        length, found = _longest_prefix(quote, at, page, cursor, last_start)
        if length == 0:
            return None
        if segments and found > cursor and not _clean_gap(page_text, haystack, cursor, found):
            return None
        segments.append(Segment(at, at + length, found, found + length))
        at += length
        cursor = found + length
    return segments


def align(quote: str, page: str) -> Alignment | None:
    """Locate ``quote`` in ``page`` despite PDF line breaks, hyphenation and interleaved columns.

    Every letter and digit of the quote must be found, in order. The page may contain extra words between
    matched pieces (another table column, a displaced subscript): at most ``MAX_GAP`` characters each,
    only between whole words and never a negation or condition. The match stays close to the quote length.
    """
    needle, haystack = Normalized.of(quote), Normalized.of(page)
    q, p = needle.chars, haystack.chars
    if len(q) < MIN_SEGMENT or not p:
        return None
    max_segments = 2 + len(q) // 20
    max_span = 2 * len(q) + 80
    anchor = q[:min(6, len(q))]
    best = None
    start, anchors = p.find(anchor), 0
    while start >= 0 and anchors < MAX_ANCHORS:
        anchors += 1
        segments = _follow(q, page, haystack, start)
        if segments is not None:
            span = segments[-1].page_end - segments[0].page_start
            if len(segments) <= max_segments and span <= max_span:
                key = (len(segments), span)
                if best is None or key < best[0]:
                    best = key, segments
                    if len(segments) == 1:
                        break
        start = p.find(anchor, start + 1)
    if best is None:
        return None
    segments = best[1]
    origin = haystack.origin
    return Alignment(
        start=origin[segments[0].page_start], end=origin[segments[-1].page_end - 1] + 1,
        segments=tuple(segments),
    )


LABEL_CHARS = 3
GLUE_GAP = 3


def numbers_stay_with_words(quote: str, alignment: Alignment) -> bool:
    """Each number of the quote sits on the page right after the words that precede it in the quote.

    This blocks a quote that takes a value from another table row and attaches it to a label. A number
    may open a new piece of the match only after a tiny gap, such as a displaced subscript in ``Mкр``.
    """
    q = Normalized.of(quote).chars
    owner = [-1] * len(q)
    for index, segment in enumerate(alignment.segments):
        for position in range(segment.quote_start, segment.quote_end):
            owner[position] = index
    for match in re.finditer(r'\d+', q):
        first, end = match.start(), match.end()
        index = owner[first]
        if any(owner[position] != index for position in range(first, end)):
            return False
        segment = alignment.segments[index]
        if first > 0 and first - segment.quote_start >= min(LABEL_CHARS, first):
            continue
        has_unit = segment.quote_end - end >= 2 or end == len(q)
        if first == 0 and has_unit:
            continue
        glued = index > 0 and segment.page_start - alignment.segments[index - 1].page_end <= GLUE_GAP
        if glued and has_unit:
            continue
        return False
    return True


# Numbers ------------------------------------------------------------------------------------------

# «1 500» is one number, «М2 100» is not: the left group is 1–3 digits that do not follow a letter or digit.
_THOUSANDS = re.compile(r'(?<![^\W_])(\d{1,3})[    ](?=\d{3}(?!\d))')
_NUMBER = re.compile(r'\d+(?:[.,]\d+)?')
# A unit exponent (``мин-1``, ``мин⁻¹``, ``с -1``) is not a value; codes such as ``ЕВРО-1`` stay, and next to a number it would be read as a
# thousands group: ``мин-1 550…650`` must not become 1550.
_UNIT_EXPONENT = re.compile(r'\b(мин|об|с|ч)\s*[-−–⁻]\s*[1¹](?!\d|[.,]\d)')
_ENGINE_NAME = re.compile(r'(?<!\w)(?:ЯМЗ|ЗМЗ|УМЗ|КамАЗ|МАЗ|ГАЗ|УАЗ)[\s-]*\d[\w.\-]*')
_FAULT_CODE = re.compile(r'\bF\d{2}\b')
_PAGE_REFERENCE = re.compile(r'\b(?:стр|с)\.\s*\d+(?:\s*[-–,]\s*\d+)*', re.IGNORECASE)
_ENUMERATOR = re.compile(r'^\s*\d{1,2}[.)]\s+')


def canonical_number(value: str) -> str:
    try:
        number = Decimal(value.replace(',', '.'))
    except InvalidOperation:
        return value
    text = format(number.normalize(), 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def without_exponents(text: str) -> str:
    return _UNIT_EXPONENT.sub(r'\1 ', text)


def number_list(text: str, *, join_thousands: bool = True) -> list[str]:
    """Numbers in reading order. ``1 500`` is one number unless ``join_thousands`` is off."""
    text = without_exponents(text)
    if join_thousands:
        text = _THOUSANDS.sub(r'\1', text)
    return [canonical_number(match) for match in _NUMBER.findall(text)]


def numbers(text: str) -> set[str]:
    return set(number_list(text))


def _strip_names(text: str, ignore: tuple[str, ...]) -> str:
    for value in sorted(ignore, key=len, reverse=True):
        if value:
            text = re.sub(rf'(?<!\w){re.escape(value)}(?!\w)', ' ', text)
    for pattern in (_ENGINE_NAME, _FAULT_CODE, _PAGE_REFERENCE):
        text = pattern.sub(' ', text)
    return _ENUMERATOR.sub('', text)


def claim_numbers(text: str, *, ignore: tuple[str, ...] = ()) -> set[str]:
    """Numbers stated by a claim, without engine names, fault codes, page references or list markers."""
    return numbers(_strip_names(text, ignore))


def _is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    remaining = iter(haystack)
    return all(any(value == candidate for candidate in remaining) for value in needle)


def all_numbers(text: str) -> set[str]:
    return set(number_list(text)) | set(number_list(text, join_thousands=False))


_RANGE = re.compile(
    r'(\d+(?:[.,]\d+)?)\s*(?:[-–—]|\.{2,3}|…)\s*(\d+(?:[.,]\d+)?)'
    r'|от\s+(\d+(?:[.,]\d+)?)\s+до\s+(\d+(?:[.,]\d+)?)',
    re.IGNORECASE,
)


def ranges(text: str) -> list[tuple[str, str]]:
    """Value ranges such as ``196-216``, ``0,25...0,30`` or ``от 118 до 147``."""
    found = []
    for match in _RANGE.finditer(_THOUSANDS.sub(r'\1', without_exponents(text))):
        low, high = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        found.append((canonical_number(low), canonical_number(high)))
    return found


def ranges_kept(claim: str, source: str, *, ignore: tuple[str, ...] = (), join_thousands: bool = True) -> bool:
    """A range of the source is not cut in the claim.

    Both ends of a range are stated or neither is. When a value of the source occurs only as an end of a
    range (``5-8``), each mention of it in the claim must also stand next to the other end.
    """
    stated = number_list(_strip_names(claim, ignore), join_thousands=join_thousands)
    listed = number_list(source)
    for low, high in ranges(source):
        if low == high:
            continue
        if (low in stated) != (high in stated):
            return False
        low_paired = all(index + 1 < len(listed) and listed[index + 1] == high
                         for index, value in enumerate(listed) if value == low)
        high_paired = all(index > 0 and listed[index - 1] == low
                          for index, value in enumerate(listed) if value == high)
        for index, value in enumerate(stated):
            if low_paired and value == low and (index + 1 >= len(stated) or stated[index + 1] != high):
                return False
            if high_paired and value == high and (index == 0 or stated[index - 1] != low):
                return False
    return True


_LIST_VALUE = re.compile(r'^\s*[−–—-]\s*[^\W\d_][^\d\n]*?\s(\d+(?:[.,]\d+)?)\s*$', re.MULTILINE)
_LIST_RANGE = re.compile(r'^\s*[−–—-]\s*[^\W\d_][^\d\n]*?\s\d+(?:[.,]\d+)?\s*[-–]\s*\d+|'
                         r'^\s*[−–—-]\s*[^\W\d_][^\d\n]*?\s\d+(?:[.,]\d+)?\s*\(', re.MULTILINE)


def number_fragments(claim: str, excerpt: str, *, ignore: tuple[str, ...] = ()) -> bool:
    """A broken table left a lone digit where a range stood: ``− вертикальные 4`` next to
    ``− горизонтальные 120 (10-12)``. A single-digit value of such a list is not stated by the claim."""
    stated = set(number_list(_strip_names(claim, ignore)))
    if not stated or not _LIST_RANGE.search(excerpt):
        return False
    lone = {canonical_number(match) for match in _LIST_VALUE.findall(excerpt) if len(match) == 1}
    return bool(stated & lone)


def number_problems(claim: str, excerpt: str, *, ignore: tuple[str, ...] = ()) -> list[str]:
    """Reasons why the claim's numbers are not backed by the excerpt; empty when they are.

    A space between digits is ambiguous after PDF extraction and whitespace collapsing (``1 500`` or the
    columns ``1050-74`` and ``169``), so the claim passes if either reading of it passes every check.
    """
    available = all_numbers(excerpt)
    stated_text = _strip_names(claim, ignore)
    attempts = []
    for join in (True, False):
        problems = []
        stated = number_list(stated_text, join_thousands=join)
        if not set(stated) <= available:
            problems.append('number_not_in_quote')
        elif not any(_is_subsequence(stated, number_list(excerpt, join_thousands=side)) for side in (True, False)):
            problems.append('numbers_out_of_order')
        if not ranges_kept(claim, excerpt, ignore=ignore, join_thousands=join):
            problems.append('range_truncated')
        if not problems:
            return []
        attempts.append(problems)
    return attempts[0]


# Words ------------------------------------------------------------------------------------------

_WORD = re.compile(r'[^\W\d_]+')
STEM_LENGTH = 5
_GENERIC = {
    'возмо', 'причи', 'прове', 'норма', 'норми', 'реком', 'следу', 'необх', 'требу', 'руков', 'согла',
    'указа', 'этого', 'также', 'котор', 'должн', 'может', 'могут', 'чтобы', 'всего', 'более', 'менее',
    'нужно', 'можно', 'данны', 'приве', 'значе', 'выпол', 'произ',
}


def words(text: str) -> list[str]:
    return _WORD.findall(fold(text))


def stems(text: str) -> set[str]:
    return {word[:STEM_LENGTH] for word in words(text) if len(word) >= 4}


def content_stems(text: str) -> set[str]:
    return stems(text) - _GENERIC


# Negation and conditions ----------------------------------------------------------------------

_NEGATION_WORDS = {'не', 'ни', 'нет', 'нельзя'}
_NEGATION_STEMS = ('запрещ', 'недопуст', 'недопуск', 'воспрещ')
_CONDITION_WORDS = {
    'если', 'при', 'после', 'только', 'перед', 'пока', 'когда', 'без', 'кроме', 'через', 'случае', 'ранее',
    'позднее', 'лишь', 'исключительно',
}


def has_negation(text: str) -> bool:
    return any(word in _NEGATION_WORDS or word.startswith(_NEGATION_STEMS) for word in words(text))


def has_condition(text: str) -> bool:
    return any(word in _CONDITION_WORDS for word in words(text))


# Sentence boundaries --------------------------------------------------------------------------

_ABBREVIATIONS = {
    'рис', 'стр', 'см', 'мин', 'табл', 'п', 'пп', 'т', 'е', 'д', 'гл', 'разд', 'прим', 'ок', 'max', 'min', 'макс',
    'тыс', 'им', 'г', 'гг', 'кгс', 'об',
}
_BULLETS = '-–—•*·▪●'


def _starts_sentence(text: str, index: int) -> bool:
    """A line break at ``index - 1`` starts a sentence when the next line opens like one."""
    while index < len(text) and text[index] in ' \t':
        index += 1
    if index >= len(text):
        return True
    char = text[index]
    return char.isupper() or char.isdigit() or char in _BULLETS


def _ends_sentence(text: str, index: int) -> bool:
    """``text[index]`` is ``.!?;`` and ends a sentence, not an abbreviation or a decimal number."""
    char = text[index]
    if char in '!?;':
        return True
    if char != '.':
        return False
    before = re.search(r'([^\W\d_]+)\s*$', text[max(0, index - 12):index])
    if before and fold(before.group(1)) in _ABBREVIATIONS:
        return False
    after = index + 1
    if after < len(text) and text[after].isdigit():
        return False
    return True


def sentence_bounds(text: str, start: int, end: int, *, limit: int = 500) -> tuple[int, int]:
    """Expand ``text[start:end]`` to the surrounding sentence, table cell or list item."""
    left = start
    while left > max(0, start - limit):
        previous = text[left - 1]
        if previous == '\n' and _starts_sentence(text, left):
            break
        if previous in '.!?;' and _ends_sentence(text, left - 1):
            break
        left -= 1
    while left < start and text[left].isspace():
        left += 1
    right = end
    while right < min(len(text), end + limit):
        char = text[right]
        if char in '.!?;' and _ends_sentence(text, right):
            right += 1
            break
        if char == '\n' and _starts_sentence(text, right + 1):
            break
        right += 1
    return left, right


def left_context(text: str, sentence_start: int, start: int, *, width: int = 60) -> str:
    """Words of the same sentence right before a quote. A wide gap marks another table column."""
    context = text[max(sentence_start, start - width):start]
    column = list(re.finditer(r'[ \t]{3,}', context))
    return context[column[-1].end():] if column else context


def splits_token(text: str, start: int, end: int) -> bool:
    """The match begins or ends inside a word or inside a number."""
    def same_kind(a: str, b: str) -> bool:
        return (a.isalpha() and b.isalpha()) or (a.isdigit() and b.isdigit())

    if start > 0 and same_kind(text[start - 1], text[start]):
        return True
    return end < len(text) and same_kind(text[end - 1], text[end])
