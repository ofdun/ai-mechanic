import React, { useState } from 'react'
import { Check, Copy } from 'lucide-react'
import { Source } from './ManualReader'
import { useUiText } from './uiLanguage'

function CopyButton({ text }) {
  const t = useUiText()
  const [copied, setCopied] = useState(false)
  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* No clipboard over plain HTTP: the code stays selectable. */ }
  }
  return <button type="button" className="copy-code" onClick={copy} aria-label={t('Скопировать {code}', { code: text })}>
    {copied ? <Check aria-hidden="true" /> : <Copy aria-hidden="true" />}{copied ? t('Скопировано') : t('Копировать')}
  </button>
}

/** Part designations printed on the manual pages behind the answer. Stock, price and fit are not known here. */
export function PartNumbers({ result, className = 'claims' }) {
  const t = useUiText()
  const parts = result.part_numbers || []
  if (!parts.length && !result.part_numbers_note) return null
  return <section className={className + ' part-numbers'}><h4>{t('Обозначения деталей из руководства')}</h4>
    <div>{parts.length ? <>
      <ul>{parts.map(part => <li key={part.designation}>
        <code>{part.designation}</code> <CopyButton text={part.designation} />
        <p className="meta">{part.context}</p>
        <Source source={part} quote={part.designation} />
      </li>)}</ul>
      <p className="meta">{t('Наличие и совместимость проверьте у поставщика: агент знает только руководство.')}</p>
    </> : <p>{t(result.part_numbers_note)}</p>}</div>
  </section>
}
