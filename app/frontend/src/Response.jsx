import React from 'react'
import { CheckCircle2, CircleAlert, RotateCcw } from 'lucide-react'
import { formatDate } from './api'
import { Source } from './ManualReader'

const outcomes = { normal: 'Норма', anomaly: 'Отклонение', check_failed: 'Сбой проверки', recognition_error: 'Ошибка распознавания' }
const qualities = { good: 'хорошая', warning: 'с помехами', bad: 'непригодная' }
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
    <details className="manual-claims"><summary>Подробности из руководства</summary>{children}</details>
  </>
}

function verdict(result) {
  if (result.recognition_error) return { status: 'recognition_error', title: outcomes.recognition_error }
  if (result.requires_remeasurement && result.diagnostic.status !== 'check_failed')
    return { status: 'check_failed', title: 'Требуется повторный замер' }
  const status = result.diagnostic.status
  return { status, title: outcomes[status] }
}

export function DiagnosticReport({ result }) {
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
      {claim.kind === 'reference_value' && 'Норматив: '}
      {claim.kind === 'spare_part' && 'Запчасти: '}
      {claim.text.slice(0, lastSpace + 1)}<span className="cited-end">{claim.text.slice(lastSpace + 1)}{[...numbers].map(number => {
        const source = references[number - 1]
        return <sup key={number}><Source source={source} number={number} /></sup>
      })}</span>
    </p>
  }
  const quality = qualities[diagnostic.signal_quality]
  const confidence = diagnostic.confidence == null ? '' : Math.round(diagnostic.confidence * 100) + '%'
  const OutcomeIcon = { anomaly: CircleAlert, recognition_error: CircleAlert, check_failed: RotateCcw, normal: CheckCircle2 }[status]
  const lead = status === 'normal' ? 'По результату этой проверки можно продолжить эксплуатацию.' : result.message
  const title = status === 'anomaly' && diagnostic.anomaly_title
  const titleAt = title ? lead.indexOf(title) : -1
  return <article className={'report report-' + status}>
    <header className="report-heading"><h3><OutcomeIcon aria-hidden="true" />{statusTitle}</h3>
      <p className="report-reading">{[
        quality && 'Запись ' + quality,
        status === 'anomaly' && diagnostic.anomaly_type && 'Код ' + diagnostic.anomaly_type,
        status === 'anomaly' && !result.message.includes(diagnostic.anomaly_title) && diagnostic.anomaly_title,
        status === 'anomaly' && confidence && !result.message.includes(confidence) && 'Уверенность ИИ ' + confidence,
      ].filter(Boolean).join(' · ')}</p>
    </header>
    <p className="report-lead">{titleAt < 0 ? lead : <>
      {lead.slice(0, titleAt)}<strong>{title}</strong>{lead.slice(titleAt + title.length)}
    </>}</p>
    {status === 'normal' && <p>{result.message}</p>}
    <ManualAnswer summary={status === 'anomaly' && summary}>
      {[['observation', 'Что заметили'], ['cause', 'Возможная причина'], ['risk', 'Чем это грозит']].map(([kind, label]) => {
        const group = narrative.filter(claim => claim.kind === kind)
        return group.length ? <section key={kind} className="report-row"><h4>{label}</h4><div>{group.map(citedClaim)}</div></section> : null
      })}
      {!!actions.length && <section className="report-row report-action"><h4>Что делать дальше</h4><div>{actions.map(citedClaim)}</div></section>}
    </ManualAnswer>
    {!!result.limitations?.length && <section className="report-row"><h4>Что осталось уточнить</h4><div><Entries items={result.limitations} /></div></section>}
    {!!references.length && <footer className="report-row report-sources"><h4>Руководство</h4>
      <div>{references.map((source, index) => <div key={pageKey(source)}>
        {references.length > 1 && <span>[{index + 1}] </span>}<Source source={source} />
      </div>)}</div>
    </footer>}
  </article>
}

export function DiagnosticVerdict({ result }) {
  const { diagnostic, message } = result
  const { status, title } = verdict(result)
  const confidence = diagnostic.confidence == null ? '' : Math.round(diagnostic.confidence * 100) + '%'
  return <div className="diagnostic-verdict">
    <div className={'verdict ' + status}>
      <h3>{title}</h3>
      <span>Запись {qualities[diagnostic.signal_quality]}</span>
    </div>
    <p>{message}</p>
    {status === 'normal' && <p>По результату этой проверки можно продолжить эксплуатацию.</p>}
    {status === 'anomaly' && <p className="meta">
      Код {diagnostic.anomaly_type}
      {!message.includes(diagnostic.anomaly_title) && ' · ' + diagnostic.anomaly_title}
      {confidence && !message.includes(confidence) && ' · Уверенность ИИ ' + confidence}
    </p>}
  </div>
}

export function Response({ result }) {
  const { diagnostic, message, requires_remeasurement: repeat } = result
  const { sources, verified, summary } = verifiedClaims(result)
  const limitations = result.limitations || []
  const insufficient = result.action === 'search_manuals' && verified.length === 0
  const referencePages = new Map()
  if (summary) {
    for (const claim of verified) {
      for (const id of claim.evidence_ids) {
        const source = sources.get(id)
        referencePages.set(pageKey(source), source)
      }
    }
  } else if (insufficient) {
    for (const source of sources.values()) referencePages.set(pageKey(source), source)
  }
  return <article className="reply">
    {diagnostic ? <DiagnosticVerdict result={result} /> : result.action !== 'search_manuals' && <p>{message}</p>}
    <ManualAnswer summary={summary}>
      {!repeat && !result.recognition_error && Object.entries(sections).map(([kind, title]) => {
        const group = verified.filter(claim => claim.kind === kind)
        if (!group.length) return null
        return <section key={kind} className="claims"><h4>{title}</h4>
          <Entries items={group} render={claim => <>{claim.text}
            <div>{claim.evidence_ids.map(id => <Source key={id} source={sources.get(id)} />)}</div>
          </>} /></section>
      })}
    </ManualAnswer>
    {!insufficient && limitations.length > 0 && <section className="limitations"><h4>Ограничения</h4>
      <Entries items={limitations} /></section>}
    {insufficient && <div className="search-fallback">
      <Entries items={limitations.length ? limitations : ['Ответ по руководству подтвердить не удалось.']} />
      {' Рекомендуется дефектовка.'}
    </div>}
    {referencePages.size > 0 && <section className="claims"><h4>{insufficient ? 'Страницы по теме для самостоятельной проверки' : 'Источники ответа'}</h4>
      {Array.from(referencePages, ([key, source]) => <div key={key}><Source source={source} /></div>)}
    </section>}
    {!!result.history?.length && <section><h4>История проверок</h4><ul>{result.history.map(row =>
      <li key={row.session_id}>{formatDate(row.checked_at)} · {outcomes[row.status]}</li>)}</ul></section>}
  </article>
}

const priorities = { low: 'Низкий', medium: 'Средний', high: 'Высокий' }
const schedules = {
  low: 'Проверить узел при следующей проверке.',
  medium: 'Запланировать дефектовку на ближайшее ТО. До него машина работает под наблюдением.',
  high: 'Запланировать дефектовку до выпуска в рейс.',
}
export function Draft({ draft, evidence, urgency, onUrgencyChange, disabled }) {
  const { payload } = draft
  return <section className="draft" aria-label="Черновик заявки">
    <h3>Черновик заявки на дефектовку</h3>
    <dl className="fields">
      <dt>ТС / проверка</dt><dd>{payload.vehicle_id} / {payload.session_id}</dd>
      <dt><label htmlFor="ticket-urgency">Приоритет</label></dt>
      <dd>
        <select id="ticket-urgency" value={urgency} onChange={e => onUrgencyChange(e.target.value)}
          disabled={disabled} aria-describedby={draft.urgency_reason ? 'ticket-schedule ticket-priority-reason' : 'ticket-schedule'}>
          {Object.entries(priorities).map(([value, name]) => <option key={value} value={value}>{name}</option>)}
        </select>
        <p id="ticket-schedule">{schedules[urgency]}</p>
        {draft.urgency_reason && <p id="ticket-priority-reason" className="meta">{draft.urgency_reason} — {priorities[payload.urgency].toLowerCase()}.</p>}
      </dd>
      <dt>Заключение</dt><dd>{payload.conclusion}</dd>
      <dt>Работы</dt><dd><Entries items={payload.operations} /></dd>
      <dt>Запчасти</dt><dd>{payload.spare_parts.length ? payload.spare_parts.join(', ') : 'Нет подтверждённых позиций'}</dd>
      <dt>Источники</dt><dd>{payload.references.map((reference, index) => {
        const source = evidence.find(item => item.manual === reference.manual && item.pdf_page === reference.pdf_page)
        return source ? <Source key={index} source={source} /> : <p key={index}>{reference.manual}, стр. {reference.pdf_page}</p>
      })}</dd>
    </dl>
  </section>
}
