import test from 'node:test'
import assert from 'node:assert/strict'
import { pickVoice, speechChunks, speechParts } from './speech.js'

const voice = (lang, localService = true, name = lang) => ({ lang, localService, name })

test('prefers a local voice of the exact language, then any voice of the same language', () => {
  const voices = [voice('en-US'), voice('ru-RU', false, 'cloud'), voice('ru_RU', true, 'device'), voice('ru-UA')]
  assert.equal(pickVoice(voices, 'ru').name, 'device')
  assert.equal(pickVoice([voice('ru-RU', false, 'cloud'), voice('ru-UA', true, 'ua')], 'ru-RU').name, 'cloud')
  assert.equal(pickVoice([voice('hi-IN')], 'hi').lang, 'hi-IN')
})

test('no voice for the language means nothing to read', () => {
  assert.equal(pickVoice([voice('ru-RU'), voice('en-US')], 'tg'), null)
  assert.equal(pickVoice([], 'ru'), null)
  assert.equal(speechParts([{ text: 'Javob.', lang: 'uz' }], [voice('ru-RU')]), null)
  assert.equal(speechParts([{ text: 'Ответ.', lang: 'ru' }, { text: 'Javob.', lang: 'uz' }], [voice('ru-RU')]), null)
})

test('empty lines are skipped and do not require a voice', () => {
  assert.equal(speechParts([{ text: '  ', lang: 'uz' }], [voice('ru-RU')]), null)
  const parts = speechParts([{ text: '', lang: 'uz' }, { text: 'Норма.', lang: 'ru' }], [voice('ru-RU')])
  assert.deepEqual(parts.map(part => [part.text, part.lang]), [['Норма.', 'ru-RU']])
})

test('drops emoji and dot separators that engines would spell out', () => {
  assert.deepEqual(speechChunks('📘 Ответ по руководствам для D245'), ['Ответ по руководствам для D245'])
  assert.deepEqual(speechChunks('🔧 D245 · стр. 12 · ⚖️ Руководства расходятся'), ['D245, стр. 12, Руководства расходятся'])
})

test('keeps numbers and units and splits long text into short sentence chunks', () => {
  assert.deepEqual(speechChunks('Момент 7 Н·м. Проверить зазор 0,25 мм!'), ['Момент 7 Н·м. Проверить зазор 0,25 мм!'])
  const long = Array.from({ length: 60 }, (_, index) => 'слово' + index).join(' ') + '.'
  const chunks = speechChunks(long)
  assert.ok(chunks.length > 1)
  assert.ok(chunks.every(chunk => chunk.length <= 220))
  assert.equal(chunks.join(' '), long)
})

test('short sentences are merged into one utterance', () => {
  assert.deepEqual(speechChunks('Норма. Можно ехать.'), ['Норма. Можно ехать.'])
  assert.deepEqual(speechChunks('Без точки в конце'), ['Без точки в конце'])
})
