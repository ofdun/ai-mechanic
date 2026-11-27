import test from 'node:test'
import assert from 'node:assert/strict'
import { exactQuoteRange } from './quoteHighlight.js'

test('selects only the unique literal excerpt, keeping line breaks and Unicode offsets', () => {
  const page = '🔧 Введение.\nПроверить учебный датчик.\nПримечание.'
  const quote = 'Проверить учебный датчик.'
  const range = exactQuoteRange(page, quote)
  assert.equal(page.slice(range.start, range.end), quote)
  assert.equal(range.start, page.indexOf(quote))
})

test('does not normalize changed numbers, punctuation, or case', () => {
  const page = 'Учебная нагрузка 7 Н·м.\nСледующая строка.'
  for (const quote of ['Учебная нагрузка 8 Н·м.', 'учебная нагрузка 7 Н·м.',
    'Учебная нагрузка 7 Н*м.']) {
    assert.equal(exactQuoteRange(page, quote), null)
  }
})

test('does not guess between repeated or overlapping matches', () => {
  assert.equal(exactQuoteRange('Повтор. Повтор.', 'Повтор.'), null)
  assert.equal(exactQuoteRange('aaa', 'aa'), null)
})

test('old responses without quote and empty pages do not highlight', () => {
  for (const [page, quote] of [['Текст', undefined], [undefined, 'Текст'], ['Текст', '  '], ['', 'Текст']]) {
    assert.equal(exactQuoteRange(page, quote), null)
  }
})


test('ignores layout whitespace and maps the match back to the unchanged original page', () => {
  const page = 'Вступление.\nУчебная   нагрузка 7\r\nН·м.\nКонец.'
  const range = exactQuoteRange(page, 'Учебная нагрузка 7 Н·м.')
  assert.equal(page.slice(range.start, range.end), 'Учебная   нагрузка 7\r\nН·м.')
  assert.equal(exactQuoteRange('7 Н·м.\n7  Н·м.', '7 Н·м.'), null)
})

test('never selects part of another number or designation', () => {
  assert.equal(exactQuoteRange('Нагрузка 117 Н·м.', '17 Н·м.'), null)
  assert.equal(exactQuoteRange('Нагрузка 0,17 Н·м.', '17 Н·м.'), null)
  assert.equal(exactQuoteRange('Температура −17 °C.', '17 °C.'), null)
  assert.equal(exactQuoteRange('деталь ABC1234', 'ABC123'), null)
  const range = exactQuoteRange('117 Н·м; 17 Н·м.', '17 Н·м.')
  assert.deepEqual(range, { start: 9, end: 16 })
})
