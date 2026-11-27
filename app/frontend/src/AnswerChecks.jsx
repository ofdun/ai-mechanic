import React from 'react'
import { Source } from './ManualReader'
import { useUiText } from './uiLanguage'

export function AnswerChecks({ result }) {
  const t = useUiText()
  const audit = result.verification
  if (!audit || audit.engine_model !== result.engine_model || result.requires_remeasurement || result.recognition_error) return null

  const sources = new Map(result.evidence?.items.map(s => [s.evidence_id, s]))
  const pages = new Map()
  for (const entry of audit.pages) {
    const source = sources.get(entry.evidence_id)
    if (!source?.engine_models.includes(audit.engine_model)) continue
    const key = JSON.stringify([source.document_id, source.pdf_page])
    if (!pages.has(key)) pages.set(key, { source, numbers: new Set() })
    for (const number of entry.numbers) pages.get(key).numbers.add(number)
  }
  if (!pages.size && !audit.rejected_claims) return null
  return <details className="answer-checks">
    <summary>{t('Что проверено в этом ответе')}</summary>
    <p>{t('Двигатель')} · {audit.engine_model}</p>
    {Array.from(pages, ([key, { source, numbers }]) => <div key={key}>
      <Source source={source} />
      {!!numbers.size && <p>{t('Числа, сверенные с этой страницей')} · {Array.from(numbers).join(' · ')}</p>}
    </div>)}
    {audit.rejected_claims != null && <p>{t('Утверждений отсеяно при проверке источников')} · {audit.rejected_claims}</p>}
  </details>
}
