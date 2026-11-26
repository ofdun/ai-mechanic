import React from 'react'
import { formatDate } from './api'

const outcomes = { normal: 'Норма', anomaly: 'Отклонение', check_failed: 'Сбой проверки' }
const qualities = { good: 'хорошая', warning: 'с помехами', bad: 'непригодная' }
const sections = {
  observation: 'Наблюдения', cause: 'Возможные причины', risk: 'Риски',
  check: 'Что проверить', reference_value: 'Нормативы', operation: 'Работы', spare_part: 'Запчасти',
}

function pageLink(source) {
  return '/api/documents/' + encodeURIComponent(source.document_id) + '#page=' + source.pdf_page
}

export function Source({ source }) {
  return <a className="source" href={pageLink(source)}
    target="_blank" rel="noreferrer">{source.manual}, стр.{'\u00a0'}{source.pdf_page}</a>
}

function Entries({ items, render = item => item }) {
  if (items.length === 1) return render(items[0])
  return <ul>{items.map((item, index) => <li key={index}>{render(item)}</li>)}</ul>
}

function verifiedClaims(result) {
  const evidence = (result.evidence?.items || []).filter(item => item.engine_models.includes(result.engine_model))
  const sources = new Map(evidence.map(item => [item.evidence_id, item]))
  const verified = (result.claims || []).filter(claim => claim.evidence_ids.length && claim.evidence_ids.every(id => sources.has(id)))
  return { sources, verified }
}

export function DiagnosticReport({ result }) {
  const { diagnostic, requires_remeasurement: repeat } = result
  const { sources, verified } = verifiedClaims(result)
  const status = repeat ? 'check_failed' : diagnostic.status
  const claims = status === 'anomaly' ? verified : []
  const narrative = claims.filter(c => ['observation', 'cause', 'risk'].includes(c.kind))
  const actions = ['check', 'reference_value', 'operation', 'spare_part'].flatMap(kind =>
    claims.filter(c => c.kind === kind))
  const references = []
  const referenceNumbers = new Map()
  for (const claim of [...narrative, ...actions]) {
    for (const id of claim.evidence_ids) {
      const source = sources.get(id)
      const key = pageLink(source)
      if (!referenceNumbers.has(key)) {
        referenceNumbers.set(key, references.push(source))
      }
    }
  }
  function citedClaim(claim) {
    const numbers = new Set(claim.evidence_ids.map(id => {
      const source = sources.get(id)
      return referenceNumbers.get(pageLink(source))
    }))
    return <p key={claim.kind + claim.text}>
      {claim.kind === 'cause' && 'Возможная причина: '}
      {claim.kind === 'risk' && 'Риск: '}
      {claim.kind === 'reference_value' && 'Норматив: '}
      {claim.kind === 'spare_part' && 'Запчасти: '}
      {claim.text}{[...numbers].map(number => {
        const source = references[number - 1]
        return <sup key={number}><a href={pageLink(source)} target="_blank" rel="noreferrer"
          aria-label={`Источник ${number}, ${source.manual}, страница ${source.pdf_page}`}>[{number}]</a></sup>
      })}
    </p>
  }
  const quality = qualities[diagnostic.signal_quality]
  const confidence = diagnostic.confidence == null ? '' : Math.round(diagnostic.confidence * 100) + '%'
  return <article className={'report report-' + status}>
    <header><h3>{repeat && diagnostic.status !== 'check_failed' ? 'Требуется повторный замер' : outcomes[status]}</h3>
      <p className="report-lead">{status === 'normal' ? 'По результату этой проверки можно продолжить эксплуатацию.'
        : result.message}</p>
      {status === 'normal' && <p>{result.message}</p>}
      <p className="report-reading">{[
        quality && 'Запись ' + quality,
        status === 'anomaly' && diagnostic.anomaly_type,
        status === 'anomaly' && !result.message.includes(diagnostic.anomaly_title) && diagnostic.anomaly_title,
        status === 'anomaly' && confidence && !result.message.includes(confidence) && 'Уверенность модели ' + confidence,
      ].filter(Boolean).join(' · ')}</p>
    </header>
    {narrative.map(citedClaim)}
    {!!actions.length && <section className="report-action"><h4>Что делать дальше</h4>{actions.map(citedClaim)}</section>}
    {!!result.limitations?.length && <section className="report-limits"><h4>Что осталось уточнить</h4><Entries items={result.limitations} /></section>}
    {!!references.length && <footer className="report-sources"><h4>По руководству</h4>
      <ol>{references.map(source => <li key={pageLink(source)}><Source source={source} /></li>)}</ol>
    </footer>}
  </article>
}

export function Response({ result }) {
  const { diagnostic, message, requires_remeasurement: repeat } = result
  const { sources, verified } = verifiedClaims(result)
  const limitations = result.limitations || []
  const insufficient = result.action === 'search_manuals' && verified.length === 0
  const confidence = diagnostic?.confidence == null ? '' : Math.round(diagnostic.confidence * 100) + '%'
  return <article className="reply">
    {diagnostic && <div className={'verdict ' + (repeat ? 'check_failed' : diagnostic.status)}>
      <h3>{repeat && diagnostic.status !== 'check_failed' ? 'Требуется повторный замер' : outcomes[diagnostic.status]}</h3>
      <span>Запись {qualities[diagnostic.signal_quality]}</span>
    </div>}
    <p>{message}</p>
    {diagnostic?.status === 'normal' && !repeat && <p>По результату этой проверки можно продолжить эксплуатацию.</p>}
    {!repeat && diagnostic?.status === 'anomaly' && <p className="meta">
      Код {diagnostic.anomaly_type}
      {!message.includes(diagnostic.anomaly_title) && ' · ' + diagnostic.anomaly_title}
      {confidence && !message.includes(confidence) && ' · Уверенность ' + confidence}
    </p>}
    {!repeat && Object.entries(sections).map(([kind, title]) => {
      const group = verified.filter(claim => claim.kind === kind)
      if (!group.length) return null
      return <section key={kind} className="claims"><h4>{title}</h4>
        <Entries items={group} render={claim => <>{claim.text}
          <div>{claim.evidence_ids.map(id => <Source key={id} source={sources.get(id)} />)}</div>
        </>} /></section>
    })}
    {limitations.length > 0 && <section className="limitations"><h4>Ограничения</h4>
      <Entries items={limitations} /></section>}
    {insufficient && <p>Подтверждённых данных по этому двигателю недостаточно. Рекомендуется дефектовка.</p>}
    {!!result.history?.length && <section><h4>История проверок</h4><ul>{result.history.map(row =>
      <li key={row.session_id}>{formatDate(row.checked_at)} · {outcomes[row.status]}</li>)}</ul></section>}
  </article>
}

const priorities = { low: 'Низкий', medium: 'Средний', high: 'Высокий' }
export function Draft({ draft, evidence }) {
  const { payload } = draft
  return <section className="draft" aria-label="Черновик заявки">
    <h3>Черновик заявки на дефектовку</h3>
    <dl className="fields">
      <dt>ТС / проверка</dt><dd>{payload.vehicle_id} / {payload.session_id}</dd>
      <dt>Приоритет</dt><dd>{priorities[payload.urgency]}</dd>
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
