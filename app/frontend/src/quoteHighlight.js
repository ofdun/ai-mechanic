// Layout whitespace may differ between the index and PDF reader. Keep all other characters literal.
function pageMap(text) {
  const chars = [], starts = [], ends = []
  for (let index = 0; index < text.length;) {
    const start = index
    const whitespace = /\s/u.test(text[index])
    if (whitespace) while (index < text.length && /\s/u.test(text[index])) index++
    else index++
    chars.push(whitespace ? ' ' : text[start])
    starts.push(start)
    ends.push(index)
  }
  return { text: chars.join(''), starts, ends }
}

const word = char => !!char && /[\p{L}\p{N}_]/u.test(char)

export function exactQuoteRange(text, quote) {
  if (typeof text !== 'string' || typeof quote !== 'string') return null
  const excerpt = quote.trim().replace(/\s+/gu, ' ')
  if (!excerpt) return null
  const page = pageMap(text)
  let match = null
  for (let start = page.text.indexOf(excerpt); start >= 0; start = page.text.indexOf(excerpt, start + 1)) {
    const end = start + excerpt.length
    // Never select a suffix of a different number, part name, or word.
    if ((word(excerpt[0]) && word(page.text[start - 1])) ||
        (word(excerpt.at(-1)) && word(page.text[end]))) continue
    if ((/\d/u.test(excerpt[0]) && /[.,+−-]/u.test(page.text[start - 1] || '')) ||
        (/\d/u.test(excerpt.at(-1)) && /[.,]/u.test(page.text[end] || ''))) continue
    if (match) return null // An ambiguous excerpt cannot select the correct occurrence reliably.
    match = { start: page.starts[start], end: page.ends[end - 1] }
  }
  return match
}
