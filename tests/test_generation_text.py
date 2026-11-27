import pytest

from ml.generation.text import (
    align, canonical_number, claim_numbers, has_condition, has_negation, left_context, number_problems, numbers,
    numbers_stay_with_words, ranges, ranges_kept, sentence_bounds, splits_token,
)


# Synthetic page fragments that reproduce PDF extraction artefacts. They are not taken from the manuals.
LAYOUT_TABLE = (
    'Засорение фильтра тонкой       Заменить фильтрующий эле-\n'
    'очистки топлива                мент и прокачать систему\n'
    'Нарушение регулировки   или    Отрегулировать форсунку и, если\n'
    'засорение форсунки             необходимо, промыть ее\n'
)
INTERLEAVED = 'Наличие воздуха в топливной Прокачать систему, устранить негер\nсистеме\n-\nметичность\n'
GLUED = 'тепловые зазоры должны укладываться в пределы0,20-0,40 мм.'
SUBSCRIPT = 'затяжка должна быть М от 118 до 147 Н·м (от кр\n12 до 15 кгс·м).'


def matched(quote, page):
    found = align(quote, page)
    return page[found.start:found.end] if found else None


@pytest.mark.parametrize(('quote', 'page', 'expected'), [
    ('Отрегулировать форсунку и, если необходимо, промыть её', LAYOUT_TABLE,
     'Отрегулировать форсунку и, если\nзасорение форсунки             необходимо, промыть ее'),
    ('Заменить фильтрующий элемент и прокачать систему', LAYOUT_TABLE,
     'Заменить фильтрующий эле-\nочистки топлива                мент и прокачать систему'),
    ('Прокачать систему, устранить негерметичность', INTERLEAVED,
     'Прокачать систему, устранить негер\nсистеме\n-\nметичность'),
    ('должны укладываться в пределы 0,20–0,40 мм', GLUED, 'должны укладываться в пределы0,20-0,40 мм'),
    ('M от 118 до 147 H·м (от 12 до 15 кгс·м)', SUBSCRIPT, 'М от 118 до 147 Н·м (от кр\n12 до 15 кгс·м'),
])
def test_quotes_survive_pdf_extraction_artefacts(quote, page, expected):
    assert matched(quote, page) == expected


@pytest.mark.parametrize(('quote', 'page'), [
    ('Зазор впускного клапана 0,40 мм', 'Зазор выпускного клапана 0,40 мм.'),
    ('Клапан должен быть закрыт', 'Клапан не должен быть закрыт.'),
    ('Затянуть гайку моментом', 'Затянуть гайку при необходимости моментом'),
    ('Отрегулировать форсунку и промыть её', 'Отрегулировать форсунку и прочистить её'),
    ('Затянуть гайку 110 Н·м', 'Затянуть гайку 10 Н·м'),
])
def test_alignment_refuses_changes_of_meaning(quote, page):
    assert align(quote, page) is None


def test_number_cannot_move_to_another_label():
    page = 'Зазор впускного клапана 0,25 мм. Зазор выпускного клапана 0,40 мм.'
    taken = 'Зазор впускного клапана 0,40 мм'
    found = align(taken, page)
    assert found is not None and not numbers_stay_with_words(taken, found)
    honest = 'Зазор выпускного клапана 0,40 мм'
    assert numbers_stay_with_words(honest, align(honest, page))


def test_number_next_to_displaced_subscript_stays_attached():
    quote = 'М от 118 до 147 Н·м (от 12 до 15 кгс·м)'
    assert numbers_stay_with_words(quote, align(quote, SUBSCRIPT))


@pytest.mark.parametrize(('raw', 'canonical'), [
    ('0,30', '0.3'), ('0.25', '0.25'), ('015', '15'), ('196', '196'), ('2,0', '2'), ('0', '0'),
])
def test_canonical_numbers(raw, canonical):
    assert canonical_number(raw) == canonical


def test_numbers_and_thousands():
    assert numbers('при 1 500 об/мин давление 0,40–0,55 МПа') == {'1500', '0.4', '0.55'}


def test_claim_numbers_ignore_engine_codes_pages_and_list_markers():
    text = '1. Для ЯМЗ-236М2 при F08 см. стр. 52: зазор 0,25 мм'
    assert claim_numbers(text) == {'0.25'}
    assert claim_numbers('насос. 15 мин') == {'15'}


def test_negation_and_condition_markers():
    assert has_negation('Запрещается пуск') and has_negation('не допускается') and not has_negation('Неисправность')
    assert has_condition('при необходимости') and has_condition('не ранее чем через 15 мин')
    assert not has_condition('Отрегулировать форсунку')


def test_sentence_bounds_respect_abbreviations_and_cells():
    text = 'Первое предложение. Совместить метки (рис. 17) и затянуть гайки. Следующее.'
    start = text.index('затянуть')
    left, right = sentence_bounds(text, start, start + 8)
    assert text[left:right] == 'Совместить метки (рис. 17) и затянуть гайки.'
    cell = 'строка таблицы\nОтрегулировать форсунку и, если\nнеобходимо, промыть\nНовая строка'
    start = cell.index('Отрегулировать')
    left, right = sentence_bounds(cell, start, start + 5)
    assert cell[left:right] == 'Отрегулировать форсунку и, если\nнеобходимо, промыть'


def test_left_context_stops_at_table_column():
    text = 'Не работает насос       Разобрать насос'
    assert left_context(text, 0, text.index('Разобрать')) == ''
    text = 'Запрещено запускать двигатель.'
    assert left_context(text, 0, text.index('запускать')) == 'Запрещено '


def test_split_tokens_ignore_glued_cells():
    text = 'моментом 110 Н·м; пределы0,20'
    assert splits_token(text, text.index('10 Н'), text.index('10 Н') + 6)
    assert not splits_token(text, text.index('0,20'), len(text))


def test_claim_numbers_strip_whole_engine_names_only():
    assert claim_numbers('двигатели ЯМЗ-536, ЯМЗ-5361 и ЯМЗ-5363: зазор 0,3 мм') == {'0.3'}
    assert claim_numbers('Fixture 12 РЭ, зазор 12 мм', ignore=('Fixture 12 РЭ',)) == {'12'}


def test_ranges_are_detected_in_common_notations():
    assert ranges('0,25-0,30 мм; 550…650; от 118 до 147 Н·м') == [('0.25', '0.3'), ('550', '650'), ('118', '147')]


@pytest.mark.parametrize(('claim', 'excerpt', 'problems'), [
    ('момент 5–8 Н·м', 'клемма (8), затяжка моментом Мкр=5-8 Н·м', []),
    ('момент 5 Н·м', 'клемма (8), затяжка моментом Мкр=5-8 Н·м', ['range_truncated']),
    ('клемма (8), момент 5 Н·м', 'клемма (8), затяжка моментом Мкр=5-8 Н·м', ['range_truncated']),
    ('гайка M12x1,75, 3 болта', 'отвернув 3 болта (2) M12x1,75', ['numbers_out_of_order']),
    ('3 болта (2) M12x2', 'отвернув 3 болта (2) M12x1,75', ['numbers_out_of_order']),
    ('3 болта (2) M14x1,75', 'отвернув 3 болта (2) M12x1,75', ['number_not_in_quote']),
    ('сталь 45 ГОСТ 1050-74 169', 'Сталь 45 ГОСТ 1050-74\n169', []),
    ('частота 1500 об/мин', 'частота 1 500 об/мин', []),
])
def test_number_problems(claim, excerpt, problems):
    assert number_problems(claim, excerpt) == problems


def test_ranges_kept_ignores_values_that_also_stand_alone():
    assert ranges_kept('зазор 0,25 мм', 'зазор 0,25 мм, при проверке 0,25-0,40 мм') is False
    assert ranges_kept('зазор 0,25–0,40 мм', 'зазор 0,25 мм, при проверке 0,25-0,40 мм')
