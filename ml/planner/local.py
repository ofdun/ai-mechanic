"""Local nearest-neighbour tool selection over Russian mechanic requests."""
import math
import re
from collections import Counter
from functools import lru_cache

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
    ),
    'search_manuals': (
        'какой момент затяжки', 'как проверить зазор', 'что написано в руководстве',
        'какой артикул водяного насоса', 'какой каталожный номер детали', 'как проводится регулировка', 'можно ли своим ходом доехать до базы',
        'чем это опасно', 'почему это опасно', 'что проверить дальше', 'откуда взялся такой вывод',
        'что входит в заявку', 'почему нужна диагностика', 'где найти допуски',
    ),
    'clarify': ('спасибо', 'здравствуй', 'ничего делать не надо', 'отмена'),
}


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
    def __init__(self, manual_terms=()):
        self.tokenize = tokenizer('stem')
        self.manual_terms = set(manual_terms)
        self.manual_terms.update(self.tokenize(' '.join(FAULT_QUERIES.values())))
        self.manual_terms.update(self.tokenize('двигатель мотор норматив момент затяжки дефектовка маркировка износ допуск ремонт'))
        self.generic_terms = set(self.tokenize('дай дайте покажи расскажи напиши подскажи'
            ' какой какая какие сколько сделать делать можно нужно почему проверить найти получить дома пожалуйста'))

    def plan(self, message, engine_model, diagnostic, *, recent_messages=()):
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
        if not positive.strip():
            action = 'clarify' if re.search(r'заявк|черновик|оформ|отправ|созда', message.casefold()) else 'search_manuals'
        elif score < 0.28:
            terms = set(self.tokenize(message)) - self.generic_terms
            if not terms.intersection(self.manual_terms) and engine_model.casefold() not in message.casefold():
                return Decision('clarify', clarification='Я помогаю с диагностикой двигателей. Что хотите узнать о выбранной машине?')
            action = 'search_manuals'
        elif (action == 'draft_ticket' and re.search(r'\b(?:как|почему|зачем|когда)\b', positive)):
            action = 'search_manuals'
        if action != 'search_manuals':
            return Decision(action)
        rewrite = message
        if diagnostic and diagnostic.status == 'anomaly' and followup:
            rewrite = f'{message} {diagnostic.anomaly_title}. {diagnostic.description}'
        return Decision(action, rewrite)


def is_diagnostic_followup(message):
    return bool(re.fullmatch(
        r'(?:(?:а|и|тогда|подскажи|скажи)\s+)*(?:'
        r'(?:почему|чем)\s+(?:это\s+)?(?:опасно|грозит|плохо)|каков(?:ы)?\s+риск[и]?|'
        r'какие\s+риски|какой\s+риск|что\s+(?:делать|проверить|посмотреть)(?:\s+(?:дальше|теперь|сначала))?'
        r')', message.casefold().strip(' ,.?!\t\n')
    ))
