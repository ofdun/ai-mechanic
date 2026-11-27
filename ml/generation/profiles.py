"""Fault profiles F01–F14: unit vocabulary, agent risk wording and base urgency.

Codes are conventions of the Mock ML API, the manuals do not contain them. The vocabulary links a code to
the unit that a manual procedure must concern. Risk wording and base urgency are the agent's own
assessment, always shown as such and never as a manual quotation. The customer approved the scheme
"base criticality of the fault type plus a bounded correction by the anomaly score" on 26.11.2026.
"""
from dataclasses import dataclass
from typing import Literal

from ml.generation.text import words


Urgency = Literal['low', 'medium', 'high']
LEVELS: tuple[Urgency, ...] = ('low', 'medium', 'high')


@dataclass(frozen=True)
class FaultProfile:
    code: str
    unit: str
    vocabulary: tuple[str, ...]
    risk: str
    base_urgency: Urgency


PROFILES = {profile.code: profile for profile in (
    FaultProfile('F01', 'клапанный механизм',
                 ('клапан', 'зазор', 'коромысл', 'толкател', 'штанг', 'распределительн', 'газораспредел', 'седл',
                  'головк цилиндр', 'головок цилиндр'),
                 'Нарушение работы клапанного механизма снижает мощность и может привести к прогару клапанов.',
                 'medium'),
    FaultProfile('F02', 'система смазки',
                 ('масл', 'смазк', 'смазочн', 'вкладыш', 'подшипник', 'редукционн', 'манометр', 'давлени'),
                 'Низкое давление масла грозит задиром вкладышей и повреждением подшипников коленчатого вала.',
                 'high'),
    FaultProfile('F03', 'система охлаждения',
                 ('охлажд', 'перегрев', 'термостат', 'радиатор', 'жидкост', 'антифриз', 'водян', 'помп',
                  'прокладк', 'температур', 'головк цилиндр'),
                 'Перегрев и попадание охлаждающей жидкости в масло грозят повреждением головки цилиндров и подшипников.',
                 'high'),
    FaultProfile('F04', 'ремённый привод агрегатов',
                 ('ремен', 'ремн', 'ремён', 'натяжн', 'натяжени', 'шкив', 'ролик', 'привод'),
                 'Обрыв ремня может остановить генератор, насос охлаждающей жидкости и другие агрегаты.',
                 'medium'),
    FaultProfile('F05', 'генератор',
                 ('генератор', 'заряд', 'аккумулятор', 'напряжени', 'щетк', 'щёт', 'регулятор напряжения'),
                 'Неисправность генератора ведёт к разряду аккумуляторной батареи и остановке в рейсе.',
                 'low'),
    FaultProfile('F06', 'сцепление',
                 ('сцеплени', 'ведом', 'нажимн', 'выжимн', 'выключени сцеплени', 'маховик', 'корзин'),
                 'Неисправность сцепления затрудняет трогание и переключение передач.',
                 'medium'),
    FaultProfile('F07', 'коробка передач',
                 ('коробк', 'передач', 'шестерн', 'синхрониз', 'переключени', 'кпп'),
                 'Неисправность коробки передач может привести к её заклиниванию в движении.',
                 'medium'),
    FaultProfile('F08', 'дизельная топливная аппаратура',
                 ('топлив', 'форсун', 'впрыск', 'тнвд', 'насос высокого давления', 'плунжер', 'распылител',
                  'подкачивающ', 'нагнетательн', 'опережени'),
                 'Нарушение топливоподачи снижает мощность, повышает дымность и нагрузку на цилиндро-поршневую группу.',
                 'medium'),
    FaultProfile('F09', 'турбокомпрессор и наддув',
                 ('турбокомпрессор', 'наддув', 'турбин', 'ротор', 'надувочн', 'интеркулер', 'охладител наддув'),
                 'Неисправность турбокомпрессора снижает мощность, а разрушение ротора опасно для двигателя.',
                 'medium'),
    FaultProfile('F10', 'компрессор пневмотормозов и насос ГУР',
                 ('компрессор', 'пневм', 'тормоз', 'гур', 'гидроусилител', 'рулев', 'насос гидро'),
                 'Неисправность компрессора пневмотормозов или насоса ГУР влияет на безопасность: тормоза и рулевое управление.',
                 'high'),
    FaultProfile('F11', 'система рециркуляции отработавших газов',
                 ('рециркуляц', 'egr', 'отработавш', 'выпускн', 'охладител рециркуляц'),
                 'Неисправность рециркуляции отработавших газов ухудшает работу двигателя и экологические показатели.',
                 'low'),
    FaultProfile('F12', 'система зажигания',
                 ('зажигани', 'свеч', 'катушк', 'искр', 'высоковольтн', 'датчик'),
                 'Пропуски зажигания снижают мощность и увеличивают расход топлива.',
                 'medium'),
    FaultProfile('F13', 'система подачи бензина',
                 ('топлив', 'бензин', 'форсун', 'рамп', 'бензонасос', 'дроссел', 'инжектор', 'регулятор давления'),
                 'Нарушение подачи бензина вызывает перебои в работе и потерю мощности.',
                 'medium'),
    FaultProfile('F14', 'цепь ГРМ и гидронатяжитель',
                 ('цеп', 'натяжител', 'башмак', 'успокоител', 'звездочк', 'звёздочк', 'газораспредел',
                  'распределительн'),
                 'Растяжение цепи или отказ гидронатяжителя могут привести к перескоку цепи и встрече клапанов с поршнями.',
                 'high'),
)}

URGENCY_LABELS: dict[Urgency, str] = {
    'high': 'высокий — дефектовка до выпуска в рейс',
    'medium': 'средний — дефектовка на ближайшем ТО, эксплуатация под наблюдением',
    'low': 'низкий — проверить узел при следующей проверке',
}

HIGH_SCORE = 0.85
LOW_SCORE = 0.40


def mentions_unit(code: str | None, *texts: str) -> bool:
    """A text concerns the unit that the fault type points to."""
    profile = PROFILES.get(code or '')
    if profile is None:
        return True
    present = set(words(' '.join(texts)))

    def starts(prefix: str) -> bool:
        return any(word.startswith(prefix) for word in present)

    return any(all(starts(part) for part in words(term)) for term in profile.vocabulary)


def urgency_for(code: str | None, anomaly_score: float | None, *, prohibited: bool = False) -> Urgency:
    """Base urgency of the fault type, one level up or down by the anomaly score.

    ``prohibited`` means a verified manual excerpt forbids operation, which always gives ``high``.
    """
    if prohibited:
        return 'high'
    profile = PROFILES.get(code or '')
    level = LEVELS.index(profile.base_urgency if profile else 'medium')
    if anomaly_score is not None and anomaly_score >= HIGH_SCORE:
        level += 1
    elif anomaly_score is not None and anomaly_score < LOW_SCORE:
        level -= 1
    return LEVELS[max(0, min(level, len(LEVELS) - 1))]
