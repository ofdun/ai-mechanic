import { UiLanguage, useUiText } from './uiLanguage'
import React, { useContext, useEffect, useState } from 'react'
import { createPortal, flushSync } from 'react-dom'
import { CheckCircle2, CircleAlert, Printer, RotateCcw } from 'lucide-react'
import { formatDate } from './api'
import { AnswerChecks } from './AnswerChecks'
import { Source } from './ManualReader'
import { ReadAloud } from './ReadAloud'
import { ScoreChart, score } from './ScoreChart'
import { PartNumbers } from './Parts'
import { TighteningFigures } from './Figures'
import { InspectionList } from './Inspection'

const outcomes = { normal: 'Норма', anomaly: 'Отклонение', check_failed: 'Сбой проверки', recognition_error: 'Ошибка распознавания' }
const recordingWarnings = { warning: 'Запись с помехами', bad: 'Запись непригодная' }
const sections = {
  observation: 'Наблюдения', cause: 'Возможные причины', risk: 'Риски',
  check: 'Что проверить', reference_value: 'Нормативы', operation: 'Работы', spare_part: 'Запчасти',
}

function pageKey(source) {
  return JSON.stringify([source.document_id, source.pdf_page])
}

function Entries({ items, render = item => item }) {
  if (items.length === 1) return render(items[0])
  return <ul>{items.map((item, index) => <li key={index}>{render(item)}</li>)}</ul>
}

function verifiedClaims(result) {
  const evidence = (result.evidence?.items || []).filter(item => item.engine_models.includes(result.engine_model))
  const sources = new Map(evidence.map(item => [item.evidence_id, item]))
  const verified = (result.claims || []).filter(claim => claim.evidence_ids.length && claim.evidence_ids.every(id => sources.has(id)))
  const summary = verified.length && verified.length === result.claims.length &&
    !result.requires_remeasurement && !result.recognition_error ? result.summary?.trim() : ''
  return { sources, verified, summary }
}

function ManualAnswer({ summary, children }) {
  if (!summary) return children
  return <>
    <p className="manual-answer">{summary}</p>
    {children}
  </>
}

// The short answer as it is spoken: verdict and recording warning first, then the lead,
// the summary (or the checked claims when there is no summary) and the limitations.
function spokenAnswer({ title, warning, details, lead, summary, claims = [], limitations = [] }) {
  return [title, warning, details, lead, ...(summary.length ? summary : claims), ...limitations].filter(line => line?.text)
}

// B1: earlier checks and tickets of this vehicle, written by the agent from its own records.
function VehicleMemory({ result, className = 'report-row' }) {
  const t = useUiText()
  if (!result.memory?.length) return null
  return <section className={className}><h4>{t('По этой машине')}</h4><div><Entries items={result.memory} /></div></section>
}

function verdict(result) {
  if (result.recognition_error) return { status: 'recognition_error', title: outcomes.recognition_error }
  if (result.requires_remeasurement && result.diagnostic.status !== 'check_failed')
    return { status: 'check_failed', title: 'Требуется повторный замер' }
  const status = result.diagnostic.status
  return { status, title: outcomes[status] }
}

function anomalyDetails(result, status, t) {
  const { diagnostic, message } = result
  if (status !== 'anomaly') return ''
  const confidence = diagnostic.confidence == null ? '' : Math.round(diagnostic.confidence * 100) + '%'
  return [
    diagnostic.anomaly_title && !message.includes(diagnostic.anomaly_title) && diagnostic.anomaly_title,
    confidence && !message.includes(confidence) && t('Уверенность ИИ {value}', { value: confidence }),
    // The threshold the verdict was checked against (MIN_DIAGNOSTIC_CONFIDENCE), so the number above can be read.
    result.min_confidence != null && t('Порог уверенности {value}', { value: Math.round(result.min_confidence * 100) + '%' }),
  ].filter(Boolean).join(' · ')
}

function AnswerLine({ line, className }) {
  return <p className={className} lang={line.translated === false ? 'ru' : undefined}>{line.text}</p>
}

function LocalizedResponse({ result, compact = false, report = false }) {
  const t = useUiText()
  const { localized: answer, diagnostic } = result
  const { labels } = answer
  const { sources, verified, summary } = verifiedClaims(result)
  const status = diagnostic && verdict(result)
  const showClaims = !result.requires_remeasurement && !result.recognition_error
  const pages = new Map()
  if (!verified.length && result.action === 'search_manuals') {
    for (const source of sources.values()) pages.set(pageKey(source), source)
  }
  const details = status && anomalyDetails({ ...result, message: answer.message.original || answer.message.text }, status.status, t)
  const lines = summary ? answer.summary : []
  const Icon = status?.status === 'normal' ? CheckCircle2 : status?.status === 'check_failed' ? RotateCcw : CircleAlert
  const uiLang = useContext(UiLanguage)
  const spoken = line => ({ text: line.text, lang: line.translated === false ? 'ru' : answer.language })
  const shownMessage = compact || !lines.length || result.action !== 'search_manuals'
  const readLines = !compact && spokenAnswer({
    title: status && { text: t(status.title), lang: uiLang },
    warning: diagnostic && { text: t(recordingWarnings[diagnostic.signal_quality] || ''), lang: uiLang },
    details: details && { text: details, lang: uiLang },
    lead: shownMessage && spoken(answer.message),
    summary: lines.map(spoken),
    claims: showClaims && !diagnostic ? verified.map(claim => spoken(answer.claims[result.claims.indexOf(claim)] || { text: claim.text, translated: false })) : [],
    limitations: answer.limitations.map(spoken),
  })
  return <article className={report ? 'report report-' + status.status : compact ? 'diagnostic-verdict' : 'reply'} lang={answer.language}>
    {status && <div className={report ? 'report-heading' : 'verdict ' + status.status}><h3>{report && <Icon aria-hidden="true" />}{t(status.title)}</h3>
      {report ? <p className="report-reading">{[t(recordingWarnings[diagnostic.signal_quality]), details].filter(Boolean).join(' · ')}</p>
        : recordingWarnings[diagnostic.signal_quality] && <span>{t(recordingWarnings[diagnostic.signal_quality])}</span>}
    </div>}
    {shownMessage &&
      <AnswerLine className={report ? 'report-lead' : undefined} line={answer.message} />}
    {!report && details && <p className="meta">{details}</p>}
    {!compact && <>
      {lines.map((line, index) => <AnswerLine key={index} line={line} />)}
      {showClaims && Object.entries(sections).map(([kind, title]) => {
        const claims = verified.filter(claim => claim.kind === kind)
        return claims.length > 0 && <section className={report ? ['operation', 'check'].includes(kind) ? 'report-row report-action' : 'report-row' : 'claims'} key={kind}>
          <h4>{labels['section.' + kind] || title}</h4>
          <div>{claims.map(claim => {
            const index = result.claims.indexOf(claim)
            return <div key={index}>
              <AnswerLine line={answer.claims[index] || { text: claim.text, translated: false }} />
              {claim.evidence_ids.map(id => <Source key={id} source={sources.get(id)} quote={claim.quote || claim.text} />)}
            </div>
          })}</div>
        </section>
      })}
      {labels['conflict.title'] && result.message.includes('⚖️ ' + labels['conflict.title']) &&
        <p className="limitations"><strong>{labels['conflict.title']}</strong> {labels['conflict.note']}</p>}
      {Array.from(pages, ([key, source]) => <Source key={key} source={source} />)}
      {result.action === 'get_diagnostic_history' && <HistoryReply result={result} />}
      <PartNumbers result={result} className={report ? 'report-row' : 'claims'} />
      <TighteningFigures result={result} className={report ? 'report-row' : 'claims'} />
    </>}
    {answer.memory?.length > 0 && <section className={report ? 'report-row' : 'claims'}>
      <h4>{labels.memory || t('По этой машине')}</h4>
      <div>{answer.memory.map((line, index) => <AnswerLine key={index} line={line} />)}</div>
    </section>}
    {answer.limitations.filter(line => !compact || line.text === labels.untranslated)
      .map((line, index) => <AnswerLine key={index} line={line} />)}
    {readLines && <ReadAloud lines={readLines} />}
    {!compact && <AnswerChecks result={result} />}
  </article>
}

export function DiagnosticReport({ result }) {
  const t = useUiText()
  const uiLang = useContext(UiLanguage)
  if (result.localized) return <LocalizedResponse result={result} report />
  const { diagnostic } = result
  const { sources, verified, summary } = verifiedClaims(result)
  const { status, title: statusTitle } = verdict(result)
  const claims = status === 'anomaly' ? verified : []
  const narrative = claims.filter(c => ['observation', 'cause', 'risk'].includes(c.kind))
  const actions = ['check', 'reference_value', 'operation', 'spare_part'].flatMap(kind =>
    claims.filter(c => c.kind === kind))
  const references = []
  const referenceNumbers = new Map()
  for (const claim of [...narrative, ...actions]) {
    for (const id of claim.evidence_ids) {
      const source = sources.get(id)
      const key = pageKey(source)
      if (!referenceNumbers.has(key)) {
        referenceNumbers.set(key, references.push(source))
      }
    }
  }
  function citedClaim(claim) {
    const numbers = new Set(claim.evidence_ids.map(id => {
      const source = sources.get(id)
      return referenceNumbers.get(pageKey(source))
    }))
    const lastSpace = claim.text.trimEnd().lastIndexOf(' ')
    return <p key={claim.kind + claim.text}>
      {claim.kind === 'reference_value' && t('Норматив:') + ' '}
      {claim.kind === 'spare_part' && t('Запчасти:') + ' '}
      {claim.text.slice(0, lastSpace + 1)}<span className="cited-end">{claim.text.slice(lastSpace + 1)}{[...numbers].map(number => {
        const source = references[number - 1]
        return <sup key={number}><Source source={source} number={number} quote={claim.quote || claim.text} /></sup>
      })}</span>
    </p>
  }
  const reading = [t(recordingWarnings[diagnostic.signal_quality]), anomalyDetails(result, status, t)].filter(Boolean).join(' · ')
  const OutcomeIcon = { anomaly: CircleAlert, recognition_error: CircleAlert, check_failed: RotateCcw, normal: CheckCircle2 }[status]
  const lead = status === 'normal' ? t('По результату этой проверки можно продолжить эксплуатацию.') : result.message
  const title = status === 'anomaly' && diagnostic.anomaly_title
  const titleAt = title ? lead.indexOf(title) : -1
  const readLines = spokenAnswer({
    title: { text: t(statusTitle), lang: uiLang },
    details: { text: reading, lang: uiLang },
    lead: { text: lead, lang: status === 'normal' ? uiLang : 'ru' },
    summary: [status === 'normal' ? result.message : status === 'anomaly' ? summary : ''].map(text => ({ text, lang: 'ru' })),
    limitations: (result.limitations || []).map(text => ({ text, lang: 'ru' })),
  })
  return <article className={'report report-' + status}>
    <header className="report-heading"><h3><OutcomeIcon aria-hidden="true" />{t(statusTitle)}</h3>
      {reading && <p className="report-reading">{reading}</p>}
    </header>
    <p className="report-lead">{titleAt < 0 ? lead : <>
      {lead.slice(0, titleAt)}<strong>{title}</strong>{lead.slice(titleAt + title.length)}
    </>}</p>
    {status === 'normal' && <p>{result.message}</p>}
    <ManualAnswer summary={status === 'anomaly' && summary}>
      {[['observation', 'Что заметили'], ['cause', 'Возможная причина'], ['risk', 'Чем это грозит']].map(([kind, label]) => {
        const group = narrative.filter(claim => claim.kind === kind)
        return group.length ? <section key={kind} className="report-row"><h4>{t(label)}</h4><div>{group.map(citedClaim)}</div></section> : null
      })}
      {!!actions.length && <section className="report-row report-action"><h4>{t('Что делать дальше')}</h4><div>{actions.map(citedClaim)}</div></section>}
    </ManualAnswer>
    <VehicleMemory result={result} />
    <PartNumbers result={result} className="report-row" />
    {!!result.limitations?.length && <section className="report-row"><h4>{t('Что осталось уточнить')}</h4><div><Entries items={result.limitations} /></div></section>}
    {!!references.length && <footer className="report-row report-sources"><h4>{t('Руководство')}</h4>
      <div>{references.map((source, index) => <div key={pageKey(source)}>
        {references.length > 1 && <span>[{index + 1}] </span>}<Source source={source} />
      </div>)}</div>
    </footer>}
    <ReadAloud lines={readLines} />
    <AnswerChecks result={result} />
  </article>
}

export function DiagnosticVerdict({ result }) {
  const t = useUiText()
  if (result.localized) return <LocalizedResponse result={result} compact />
  const { diagnostic, message } = result
  const { status, title } = verdict(result)
  const details = anomalyDetails(result, status, t)
  const warning = recordingWarnings[diagnostic.signal_quality]
  return <div className="diagnostic-verdict">
    <div className={'verdict ' + status}>
      <h3>{t(title)}</h3>
      {warning && <span>{t(warning)}</span>}
    </div>
    <p>{message}</p>
    {status === 'normal' && <p>{t('По результату этой проверки можно продолжить эксплуатацию.')}</p>}
    {details && <p className="meta">{details}</p>}
    <VehicleMemory result={result} className="claims" />
  </div>
}

export function Response({ result }) {
  const t = useUiText()
  const uiLang = useContext(UiLanguage)
  if (result.localized) return <LocalizedResponse result={result} />
  const { diagnostic, message, requires_remeasurement: repeat } = result
  const { sources, verified, summary } = verifiedClaims(result)
  const limitations = result.limitations || []
  const insufficient = result.action === 'search_manuals' && verified.length === 0
  const referencePages = new Map()
  if (insufficient) {
    for (const source of sources.values()) referencePages.set(pageKey(source), source)
  }
  const russian = text => ({ text, lang: 'ru' })
  const shown = diagnostic && verdict(result)
  const warning = diagnostic && recordingWarnings[diagnostic.signal_quality]
  const readLines = spokenAnswer({
    title: shown && { text: t(shown.title), lang: uiLang },
    warning: warning && { text: t(warning), lang: uiLang },
    details: shown && { text: anomalyDetails(result, shown.status, t), lang: uiLang },
    lead: (diagnostic || !['search_manuals', 'get_diagnostic_history'].includes(result.action)) && russian(message),
    summary: [summary].filter(Boolean).map(russian),
    claims: !diagnostic && !repeat && !result.recognition_error ? verified.map(claim => russian(claim.text)) : [],
    limitations: insufficient
      ? [...(limitations.length ? limitations.map(russian) : [{ text: t('Ответ по руководству подтвердить не удалось.'), lang: uiLang }]),
        { text: t('Рекомендуется дефектовка.'), lang: uiLang }]
      : limitations.map(russian),
  })
  return <article className="reply">
    {diagnostic ? <DiagnosticVerdict result={result} /> : !['search_manuals', 'get_diagnostic_history'].includes(result.action) && <p>{message}</p>}
    <ManualAnswer summary={summary}>
      {!repeat && !result.recognition_error && Object.entries(sections).map(([kind, title]) => {
        const group = verified.filter(claim => claim.kind === kind)
        if (!group.length) return null
        return <section key={kind} className="claims"><h4>{t(title)}</h4>
          <Entries items={group} render={claim => <>{claim.text}
            <div>{claim.evidence_ids.map(id => <Source key={id} source={sources.get(id)} quote={claim.quote || claim.text} />)}</div>
          </>} /></section>
      })}
    </ManualAnswer>
    {!repeat && !result.recognition_error && <PartNumbers result={result} />}
    <TighteningFigures result={result} />
    {!insufficient && limitations.length > 0 && <section className="limitations"><h4>{t('Ограничения')}</h4>
      <Entries items={limitations} /></section>}
    {insufficient && <div className="search-fallback">
      <Entries items={limitations.length ? limitations : [t('Ответ по руководству подтвердить не удалось.')]} />
      {!limitations.length && ' ' + t('Рекомендуется дефектовка.')}
    </div>}
    {referencePages.size > 0 && <section className="claims"><h4>{insufficient ? t('Страницы по теме для самостоятельной проверки') : t('Источники ответа')}</h4>
      {Array.from(referencePages, ([key, source]) => <div key={key}><Source source={source} /></div>)}
    </section>}
    {result.action === 'get_diagnostic_history' && <VehicleMemory result={result} className="claims" />}
    {result.action === 'get_diagnostic_history' && <HistoryReply result={result} />}
    <ReadAloud lines={readLines} />
    <AnswerChecks result={result} />
  </article>
}

function HistoryReply({ result }) {
  const t = useUiText()
  return <>
    <VehicleHistory vehicleId={result.vehicle_id} sessions={[...(result.history || [])].reverse()} />
    {!!result.inspections?.length && <section className="history-report">
      <h3>{t('Результаты осмотра')}</h3>
      <InspectionList records={result.inspections} vehicleId={result.vehicle_id} />
    </section>}
  </>
}

export function VehicleHistory({ vehicleId, sessions }) {
  const t = useUiText()
  const language = useContext(UiLanguage)
  const meanings = {
    normal: 'Проверка не выявила отклонений.',
    anomaly: 'В записи обнаружены признаки отклонения. Причину уточняют при диагностике.',
    check_failed: 'Оценить двигатель по этой записи не удалось. Нужен повторный замер.',
  }
  return <section className="history-report"><h3>{t('История проверок ·')} {vehicleId}</h3>
    {sessions.length ? <>
      <ScoreChart sessions={sessions} id={'score-trend-' + vehicleId} />
      <ol className="vehicle-history">{sessions.map(row =>
        <li key={row.session_id}><div><time dateTime={row.checked_at}>{formatDate(row.checked_at, language)}</time>
          <small>{t('Замер {id}', { id: row.session_id })}</small></div>
          <div><strong>{t(outcomes[row.status])}</strong>
            {row.anomaly_score != null && <span className="meta"> · {t('оценка аномалии {value}', { value: score(row.anomaly_score) })}</span>}
            <p>{t(meanings[row.status])}</p></div></li>)}</ol>
    </> : <p>{t('У этой машины пока нет проверок.')}</p>}
  </section>
}

const priorities = { low: 'Низкий', medium: 'Средний', high: 'Высокий' }
const schedules = {
  low: 'Проверить узел при следующей проверке.',
  medium: 'Запланировать дефектовку на ближайшее ТО. До него машина работает под наблюдением.',
  high: 'Запланировать дефектовку до выпуска в рейс.',
}
function vehicleName(vehicle) {
  if (!vehicle) return ''
  const { make, model } = vehicle
  return model?.startsWith(make || '') ? model : [make, model].filter(Boolean).join(' ')
}

function PrintSheet({ draft, urgency, vehicle, diagnosis }) {
  const t = useUiText()
  const language = useContext(UiLanguage)
  const [printedAt, setPrintedAt] = useState(() => new Date())
  useEffect(() => {
    const refresh = () => flushSync(() => setPrintedAt(new Date()))
    window.addEventListener('beforeprint', refresh)
    return () => window.removeEventListener('beforeprint', refresh)
  }, [])
  const { payload } = draft
  const session = vehicle?.sessions.find(item => item.session_id === payload.session_id)
  const engine = draft.engine_model || vehicle?.engine_model
  const sameCheck = diagnosis?.diagnostic && diagnosis.vehicle_id === payload.vehicle_id && diagnosis.session_id === payload.session_id
  const status = sameCheck && verdict(diagnosis)
  const details = status && anomalyDetails({ diagnostic: diagnosis.diagnostic, message: payload.conclusion }, status.status, t)
  return createPortal(<article className="print-sheet" lang={language}>
    <header className="print-heading">
      <h1>{t('Лист дефектовки')}</h1>
      <p>{formatDate(printedAt, language)}</p>
    </header>
    <dl className="print-fields">
      <dt>{t('Транспортное средство')}</dt><dd><strong>{payload.vehicle_id}</strong>{vehicleName(vehicle) && ' · ' + vehicleName(vehicle)}</dd>
      {engine && <><dt>{t('Двигатель')}</dt><dd>{engine}</dd></>}
      <dt>{t('Проверка')}</dt><dd>{payload.session_id}{session && ' · ' + formatDate(session.checked_at, language)}</dd>
      <dt>{t('Диагноз')}</dt><dd>
        {status && <p><strong>{[t(status.title), details].filter(Boolean).join(' · ')}</strong></p>}
        <p>{payload.conclusion}</p>
      </dd>
      <dt>{t('Приоритет')}</dt><dd><strong>{t(priorities[urgency])}</strong></dd>
    </dl>
    <section className="print-section">
      <h2>{t('Работы')}</h2>
      <ul className="print-checklist">{payload.operations.map((operation, index) =>
        <li key={index}><span className="print-box" aria-hidden="true" />{operation}</li>)}</ul>
    </section>
    <section className="print-section">
      <h2>{t('Запчасти')}</h2>
      {payload.spare_parts.length ? <ul className="print-checklist">{payload.spare_parts.map((part, index) =>
        <li key={index}><span className="print-box" aria-hidden="true" />{part}</li>)}</ul>
        : <p>{t('Нет подтверждённых позиций')}</p>}
    </section>
    <section className="print-section">
      <h2>{t('Страницы руководства')}</h2>
      <ol className="print-references">{payload.references.map((reference, index) =>
        <li key={index}>{reference.manual}, {t('стр.') + ' ' + reference.pdf_page}</li>)}</ol>
    </section>
    <footer className="print-signoff">
      {['Исполнитель', 'Дата', 'Подпись'].map(label => <p key={label}><span>{t(label)}</span></p>)}
    </footer>
  </article>, document.body)
}

export function Draft({ draft, evidence, urgency, onUrgencyChange, disabled, vehicle, diagnosis, printable = false }) {
  const t = useUiText()
  const { payload } = draft
  return <section className="draft" aria-label={t('Черновик заявки')}>
    <div className="draft-heading">
      <h3>{t('Черновик заявки на дефектовку')}</h3>
      <button type="button" onClick={() => window.print()}><Printer aria-hidden="true" />{t('Печать')}</button>
    </div>
    <dl className="fields">
      <dt>{t('ТС / проверка')}</dt><dd>{payload.vehicle_id} / {payload.session_id}</dd>
      <dt><label htmlFor="ticket-urgency">{t('Приоритет')}</label></dt>
      <dd>
        <select id="ticket-urgency" value={urgency} onChange={e => onUrgencyChange(e.target.value)}
          disabled={disabled} aria-describedby={draft.urgency_reason ? 'ticket-schedule ticket-priority-reason' : 'ticket-schedule'}>
          {Object.entries(priorities).map(([value, name]) => <option key={value} value={value}>{t(name)}</option>)}
        </select>
        <p id="ticket-schedule">{t(schedules[urgency])}</p>
        {draft.urgency_reason && <p id="ticket-priority-reason" className="meta">{draft.urgency_reason} — {t(priorities[payload.urgency]).toLocaleLowerCase()}.</p>}
      </dd>
      <dt>{t('Заключение')}</dt><dd>{payload.conclusion}</dd>
      <dt>{t('Работы')}</dt><dd><Entries items={payload.operations} /></dd>
      <dt>{t('Запчасти')}</dt><dd>{payload.spare_parts.length ? payload.spare_parts.join(', ') : t('Нет подтверждённых позиций')}</dd>
      <dt>{t('Источники')}</dt><dd>{payload.references.map((reference, index) => {
        const source = evidence.find(item => item.manual === reference.manual && item.pdf_page === reference.pdf_page)
        return source ? <Source key={index} source={source} /> : <p key={index}>{reference.manual}, {t('стр.')} {reference.pdf_page}</p>
      })}</dd>
    </dl>
    {printable && <PrintSheet draft={draft} urgency={urgency} vehicle={vehicle} diagnosis={diagnosis} />}
  </section>
}
