import React, { useContext, useState } from 'react'
import { Download } from 'lucide-react'
import { formatDate } from './api'
import { UiLanguage, useUiText } from './uiLanguage'

const outcomes = {
  confirmed: 'Гипотеза ИИ подтвердилась',
  not_confirmed: 'Гипотеза ИИ не подтвердилась',
  other_cause: 'Причина другая',
}
const marks = { '': 'Метки не проверялись', matched: 'Метки совпали', not_matched: 'Метки не совпадают' }

/** What the mechanic found after the defect check. Stored for the model team; it changes no verdict or ticket. */
export function InspectionForm({ disabled, onSave }) {
  const t = useUiText()
  const [outcome, setOutcome] = useState('')
  const [mark, setMark] = useState('')
  const [comment, setComment] = useState('')
  function submit(event) {
    event.preventDefault()
    onSave({ outcome, ...(mark && { marks: mark }), ...(comment.trim() && { comment: comment.trim() }) })
    setOutcome('')
    setMark('')
    setComment('')
  }
  return <form className="inspection-form" onSubmit={submit}>
    <h4>{t('Результат осмотра')}</h4>
    <fieldset><legend>{t('Что показал осмотр')}</legend>
      {Object.entries(outcomes).map(([value, label]) => <label key={value}>
        <input type="radio" name="inspection-outcome" value={value} required checked={outcome === value}
          disabled={disabled} onChange={() => setOutcome(value)} />{t(label)}</label>)}
    </fieldset>
    <fieldset><legend>{t('Для проверки по меткам')}</legend>
      {Object.entries(marks).map(([value, label]) => <label key={value}>
        <input type="radio" name="inspection-marks" value={value} checked={mark === value}
          disabled={disabled} onChange={() => setMark(value)} />{t(label)}</label>)}
    </fieldset>
    <label className="inspection-comment">{t('Комментарий')}
      <textarea value={comment} maxLength={1000} rows={2} disabled={disabled} onChange={e => setComment(e.target.value)} />
    </label>
    <p className="meta">{t('Запись не меняет диагноз и заявку. «Метки совпали» не значит, что двигатель исправен.')}</p>
    <button type="submit" className="secondary" disabled={disabled || !outcome}>{t('Записать результат осмотра')}</button>
  </form>
}

export function InspectionList({ records, vehicleId }) {
  const t = useUiText()
  const language = useContext(UiLanguage)
  if (!records.length) return <p className="meta">{t('Результатов осмотра по этой машине пока нет.')}</p>
  function download() {
    const blob = new Blob([JSON.stringify(records, null, 2)], { type: 'application/json' })
    const link = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(blob), download: `inspections-${vehicleId}.json`,
    })
    link.click()
    URL.revokeObjectURL(link.href)
  }
  return <>
    <ol className="inspection-history">{[...records].reverse().map(record => <li key={record.inspection_id}>
      <div><time dateTime={record.recorded_at}>{formatDate(record.recorded_at, language)}</time>
        <small>{t('Замер {id}', { id: record.session_id })}</small></div>
      <div><strong>{t(outcomes[record.outcome])}</strong>
        {record.marks && <span className="meta"> · {t(marks[record.marks])}</span>}
        {record.comment && <p>{record.comment}</p>}</div>
    </li>)}</ol>
    <button type="button" className="secondary" onClick={download}><Download aria-hidden="true" />{t('Скачать JSON')}</button>
  </>
}
