"""Local nearest-neighbour tool selection over Russian mechanic requests."""
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache

from ml.operability import is_release_question, release_query
from ml.planner import Decision, TOOL_DESCRIPTIONS
from ml.retrieval.expansion import FAULT_QUERIES
from ml.retrieval.sparse import tokenizer


EXAMPLES = {
    'diagnose': (
        'проверь двигатель', 'запусти диагностику по этому замеру', 'покажи результат проверки',
        'что показала запись двигателя', 'проведи диагностику', 'проанализируй последний замер',
    ),
    'draft_ticket': (
        'создать заявку', 'создай заявку на дефектовку', 'составь черновик заявки', 'оформи наряд на работы',
        'подготовь обращение в ТОиР', 'заведи заявку', 'хочу оформить заявку на эту машину',
    ),
    'get_diagnostic_history': (
        'покажи историю проверок', 'какие результаты были раньше', 'предыдущие замеры по машине',
        'что изменилось с прошлого раза', 'динамика по этому автомобилю', 'покажи прошлые результаты замеров',
        'повторялась ли эта неисправность', 'какие заявки уже были по этой машине', 'была ли такая поломка раньше',
    ),
    'search_manuals': (
        'какой момент затяжки', 'как проверить зазор', 'что написано в руководстве',
        'какой артикул водяного насоса', 'какой каталожный номер детали', 'как проводится регулировка', 'можно ли своим ходом доехать до базы',
        'чем это опасно', 'почему это опасно', 'что проверить дальше', 'откуда взялся такой вывод',
        'что входит в заявку', 'почему нужна диагностика', 'где найти допуски',
    ),
    'clarify': ('спасибо', 'здравствуй', 'ничего делать не надо', 'отмена'),
}

DIAGNOSTIC_REQUEST = re.compile(
    r'^\s*(?:(?:пожалуйста|давай|можешь|можно)\s+)*'
    r'(?:(?:запуст|начн|провед|сдела)[а-я]*\b.{0,40}\bдиагностик|'
    r'(?:повтор|проанализиру)[а-я]*\b.{0,35}\bзамер|'
    r'(?:покаж|получ)[а-я]*\b.{0,35}\bрезультат[а-я]*\s+(?:проверк|диагностик|замер)|'
    r'что показал[а-я]*\s+(?:запис|замер)|'
    r'провер[ьт][а-я]*\s+(?:этот\s+)?(?:двигател[ьяею]|мотор(?:а|е|у|ом)?)\b)',
)
TICKET_REQUEST = re.compile(
    r'^\s*(?:(?:пожалуйста|давай|хочу|можешь|можно)\s+)*'
    r'(?:созда|оформ|подготов|состав|завед)[а-я]*'
    r'(?:\s+(?:мне|нам|пожалуйста|нов[а-я]*|еще|одн[а-я]*))*\s+(?:заявк|черновик|наряд|обращени)',
)
DIAGNOSTIC_HISTORY = re.compile(
    r'^\s*(?:покаж|посмотр|вывед)[а-я]*\b.{0,45}\b(?:истори[яию]|динамик[а-я]*|'
    r'(?:прошл|предыдущ)[а-я]*\s+(?:результат|замер|проверк))|'
    r'^\s*(?:динамик[а-я]*\b|предыдущ[а-я]*\s+замер|какие результаты были раньше|что изменилось с прошлого раза)',
)
ACTION_REQUESTS = {
    'diagnose': DIAGNOSTIC_REQUEST,
    'draft_ticket': TICKET_REQUEST,
    'get_diagnostic_history': DIAGNOSTIC_HISTORY,
}
TICKET_HISTORY = re.compile(
    r'\b(?:истори[яию]|список|статус|номер|прошл[а-я]*|предыдущ[а-я]*|стар[а-я]*|'
    r'созданн[а-я]*|открыт[а-я]*)\s+(?:[а-я]+\s+){0,2}заявк[а-я]*\b|'
    r'\b(?:какие|сколько)\s+заявк[а-я]*\s+(?:уже\s+)?(?:был|созда|оформ)[а-я]*\b',
)

FAULT_HISTORY = re.compile(
    r'\bповторял[а-я]*\s+ли\b|\bбыл[аои]?\s+ли.{0,45}\b(?:поломк[а-я]*|неисправност[а-я]*)\s+раньше\b'
)


def vehicle_history_decision(message: str) -> Decision | None:
    text = message.casefold().replace('ё', 'е')
    if not TICKET_REQUEST.match(text) and (TICKET_HISTORY.search(text) or FAULT_HISTORY.search(text)):
        return Decision('get_diagnostic_history')
    return None


def features(text):
    words = re.findall(r'[а-яa-z0-9]+', text.casefold().replace('ё', 'е'))
    counts = Counter()
    for word in words:
        counts[word] += 2
        padded = '^' + word + '$'
        counts.update(padded[i:i + 3] for i in range(len(padded) - 2))
    return counts


@lru_cache(maxsize=1)
def examples():
    corpus = [(action, features(phrase)) for action, phrases in EXAMPLES.items()
              for phrase in (*phrases, TOOL_DESCRIPTIONS[action])]
    frequency = Counter(token for _, row in corpus for token in row)
    weights = {token: 1 + math.log((1 + len(corpus)) / (1 + n)) for token, n in frequency.items()}
    return weights, [(action, vector(row, weights)) for action, row in corpus]


def vector(counts, weights):
    weighted = {key: count * weights.get(key, 1) for key, count in counts.items()}
    norm = math.sqrt(sum(v * v for v in weighted.values()))
    return {key: value / norm for key, value in weighted.items()} if norm else {}


class LocalPlanner:
    @classmethod
    def from_postings(cls, postings: Mapping[str, Sequence[tuple[int, int]]]) -> 'LocalPlanner':
        # An isolated PDF extraction artifact should not establish that a request is about an engine.
        return cls(term for term, pages in postings.items() if len(pages) >= 3)

    def __init__(self, manual_terms=()):
        self.tokenize = tokenizer('stem')
        self.manual_terms = set(manual_terms)
        self.manual_terms.update(self.tokenize(' '.join(FAULT_QUERIES.values())))
        self.manual_terms.update(self.tokenize(
            'двигатель мотор норматив момент затяжки дефектовка маркировка износ допуск ремонт '
            'руководство регламент заявка перегнать перегон рейс'
        ))
        self.generic_terms = set(self.tokenize('дай дайте покажи расскажи напиши подскажи'
            ' какой какая какие сколько сделать делать можно нужно почему проверить найти получить дома пожалуйста'
            ' работа способ действие порядок результаты помощь рецепт погода завтра сегодня приготовить грамм'))

    def plan(self, message, engine_model, diagnostic, *, recent_messages=()):
        if is_release_question(message):
            return Decision('search_manuals', release_query(message, recent_messages, diagnostic))
        if history := vehicle_history_decision(message):
            return history
        followup = is_diagnostic_followup(message)
        if followup and (not diagnostic or diagnostic.status != 'anomaly'):
            return Decision('clarify', clarification='О каком отклонении речь? Назовите признак или получите результат диагностики выбранной проверки.')
        weights, rows = examples()
        # A negated clause cannot request an action, even when its vocabulary matches perfectly.
        clauses = re.split(r'[,;.!?]|\b(?:а|но|лучше|вместо этого)\b', message.casefold())
        positive = ' '.join(c for c in clauses if c.strip() and not re.search(r'\b(?:не|без)\b', c))
        query = vector(features(positive), weights)
        matches = [(sum(value * query.get(key, 0) for key, value in row.items()), action)
                   for action, row in rows]
        score, action = max(matches)
        explicit = ACTION_REQUESTS.get(action)
        if explicit is not None and explicit.search(positive.replace('ё', 'е')):
            return Decision(action)

        if not positive.strip() and re.search(r'заявк|черновик|оформ|отправ|созда', message.casefold()):
            return Decision('clarify')
        terms = {term for term in self.tokenize(message) if not term[0].isdigit()} - self.generic_terms
        in_scope = (followup or bool(terms.intersection(self.manual_terms))
                    or engine_model.casefold() in message.casefold())
        if not in_scope:
            if action == 'clarify' and score >= 0.28:
                return Decision('clarify')
            return Decision('clarify', clarification='Я помогаю с диагностикой двигателей. Что хотите узнать о выбранной машине?')

        rewrite = message
        if diagnostic and diagnostic.status == 'anomaly' and followup:
            rewrite = f'{message} {diagnostic.anomaly_title}. {diagnostic.description}'
        return Decision('search_manuals', rewrite)


def is_diagnostic_followup(message):
    return bool(re.fullmatch(
        r'(?:(?:а|и|тогда|подскажи|скажи)\s+)*(?:'
        r'(?:почему|чем)\s+(?:это\s+)?(?:опасно|грозит|плохо)|каков(?:ы)?\s+риск[и]?|'
        r'какие\s+риски|какой\s+риск|что\s+(?:делать|проверить|посмотреть)(?:\s+(?:дальше|теперь|сначала))?'
        r')', message.casefold().strip(' ,.?!\t\n')
    ))
