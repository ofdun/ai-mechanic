// Reading a short answer aloud with the browser's own speech synthesis (issue #44).
// Starts only on a click, never while dictation is recording, and stays hidden
// when the device has no voice for the answer language.

const MAX_CHUNK = 220

function normalizeLang(lang) {
  return String(lang || '').trim().toLowerCase().replace(/_/g, '-')
}

export function pickVoice(voices, lang) {
  const wanted = normalizeLang(lang)
  const primary = wanted.split('-')[0]
  if (!primary) return null
  const candidates = voices.filter(voice => normalizeLang(voice.lang).split('-')[0] === primary)
  const rank = voice => (normalizeLang(voice.lang) === wanted ? 0 : 2) + (voice.localService ? 0 : 1)
  return candidates.sort((a, b) => rank(a) - rank(b))[0] || null
}

// Long utterances are cut off in some engines, so split on sentence ends and keep each piece short.
export function speechChunks(text) {
  // Emoji and the "·" separator are visual only: engines would spell out their names.
  const clean = String(text || '').replace(/[\p{Extended_Pictographic}️‍]/gu, '')
    .replace(/\s+·\s+/g, ', ').replace(/\s+/g, ' ').replace(/^[\s,]+/, '').trim()
  if (!clean) return []
  const sentences = clean.match(/[^.!?…]+(?:[.!?…]+|$)/g).map(part => part.trim()).filter(Boolean)
  const chunks = []
  for (const sentence of sentences) {
    let rest = sentence
    while (rest.length > MAX_CHUNK) {
      const cut = rest.lastIndexOf(' ', MAX_CHUNK)
      const at = cut > 0 ? cut : MAX_CHUNK
      chunks.push(rest.slice(0, at).trim())
      rest = rest.slice(at).trim()
    }
    const last = chunks[chunks.length - 1]
    if (last && last.length + rest.length + 1 <= MAX_CHUNK && /[.!?…]$/.test(last)) chunks[chunks.length - 1] = last + ' ' + rest
    else if (rest) chunks.push(rest)
  }
  return chunks
}

// lines: [{ text, lang }] → utterance parts with a voice each, or null when some language has no voice.
export function speechParts(lines, voices) {
  const parts = []
  for (const { text, lang } of lines) {
    const chunks = speechChunks(text)
    if (!chunks.length) continue
    const voice = pickVoice(voices, lang)
    if (!voice) return null
    for (const chunk of chunks) parts.push({ text: chunk, lang: voice.lang, voice })
  }
  return parts.length ? parts : null
}

const synth = typeof window !== 'undefined' && window.speechSynthesis && window.SpeechSynthesisUtterance
  ? window.speechSynthesis : null
const listeners = new Set()
let state = { voices: synth ? synth.getVoices() : [], playing: null, dictating: false }
let current = null

function update(change) {
  state = { ...state, ...change }
  for (const listener of listeners) listener()
}

if (synth) {
  const refresh = () => update({ voices: synth.getVoices() })
  if (synth.addEventListener) synth.addEventListener('voiceschanged', refresh)
  else synth.onvoiceschanged = refresh
}

export function subscribe(listener) {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function speechState() {
  return state
}

export function stopReading() {
  current = null
  synth?.cancel()
  if (state.playing) update({ playing: null })
}

export function startReading(id, parts) {
  if (!synth || state.dictating || !parts?.length) return
  synth.cancel()
  const token = {}
  current = token
  const finish = () => {
    if (current !== token) return
    current = null
    update({ playing: null })
  }
  update({ playing: id })
  parts.forEach((part, index) => {
    const utterance = new window.SpeechSynthesisUtterance(part.text)
    utterance.lang = part.lang
    utterance.voice = part.voice
    utterance.onerror = finish
    if (index === parts.length - 1) utterance.onend = finish
    synth.speak(utterance)
  })
}

// Dictation must never hear the speaker: starting it silences the reading and blocks new readings.
export function setDictating(dictating) {
  if (dictating) stopReading()
  if (state.dictating !== dictating) update({ dictating })
}
