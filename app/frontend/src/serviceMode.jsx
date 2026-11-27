import React, { useState } from 'react'

// Service view of the cycle cost (#41). Hidden from the mechanic: open the page with ?service=1 to switch it on
// for this browser and ?service=0 to switch it off.
const serviceKey = 'ai-mechanic.serviceMode'

function readServiceMode() {
  try {
    const flag = new URLSearchParams(location.search).get('service')
    if (flag === '1' || flag === '0') localStorage.setItem(serviceKey, flag)
    return localStorage.getItem(serviceKey) === '1'
  } catch {
    return new URLSearchParams(location.search).get('service') === '1'
  }
}

export function useServiceMode() {
  const [enabled] = useState(readServiceMode)
  return enabled
}

const requestLabels = {
  diagnose: 'Результат диагностики', refresh: 'Перевод результата', chat: 'Вопрос в чате',
  draft_ticket: 'Черновик заявки', confirm: 'Отправка заявки',
}
const stepLabels = {
  planner: 'Планировщик', diagnosis: 'ML API: диагностика', retrieval: 'Поиск по руководствам',
  embeddings: 'Эмбеддинг запроса', rerank: 'Реранкинг', cloudflare: 'Cloudflare Workers AI', toir_ticket: 'ТОиР: заявка',
  a_generator: 'Генератор A', a_verifier_finalizer: 'Проверка A', a_disagreement: 'Расхождения A',
  b_query_strategist: 'Стратег запроса B', b_technical_reasoner: 'Технический анализ B', b_evidence_critic: 'Критик B',
  b_repair: 'Исправление B', b_disagreement: 'Расхождения B', b_finalizer: 'Финализатор B',
  plain_answer: 'Простой ответ', answer_translation: 'Перевод ответа',
}

// A cycle runs from the diagnostic result to the answer or the ticket: a new diagnosis starts the next one.
export function useCycleMetrics() {
  const [cycle, setCycle] = useState(null)
  function start(kind) {
    const startedAt = Date.now()
    const begun = performance.now()
    function finish(fields, newCycle) {
      const entry = { kind, startedAt, finishedAt: Date.now(), clientMs: Math.round(performance.now() - begun), ...fields }
      setCycle(old => newCycle || !old ? { startedAt, requests: [entry] } : { ...old, requests: [...old.requests, entry] })
    }
    return {
      done: (usage, { newCycle = false } = {}) => finish({ ok: true, usage: usage || null }, newCycle),
      fail: error => finish({ ok: false, usage: null, error: error?.code || error?.message || 'Ошибка' }, false),
    }
  }
  return { cycle, start }
}

const integer = new Intl.NumberFormat('ru-RU')
const seconds = ms => ms == null ? 'нет данных' : ms < 1000 ? `${ms} мс` : `${(ms / 1000).toLocaleString('ru-RU', { maximumFractionDigits: 2 })} с`
const tokens = value => value == null ? '—' : integer.format(value)
const dollars = value => '$' + value.toLocaleString('en-US', { minimumFractionDigits: 6, maximumFractionDigits: 6 })

function sumKnown(values) {
  const known = values.filter(value => value != null)
  return known.length ? known.reduce((total, value) => total + value, 0) : null
}

export function cycleTotals(cycle) {
  const requests = cycle.requests
  const usages = requests.map(request => request.usage).filter(Boolean)
  const steps = usages.flatMap(usage => usage.steps)
  const llm = steps.filter(step => step.kind === 'llm')
  return {
    requests: requests.length,
    failed: requests.filter(request => !request.ok).length,
    unmeasured: requests.filter(request => request.ok && !request.usage).length,
    llmCalls: llm.length,
    input: sumKnown(usages.map(usage => usage.input_tokens)),
    output: sumKnown(usages.map(usage => usage.output_tokens)),
    reasoning: sumKnown(usages.map(usage => usage.reasoning_tokens)),
    cost: sumKnown(usages.map(usage => usage.cost_usd)),
    // A failed request returns no measurements, so its spend is unknown too.
    costComplete: usages.every(usage => usage.cost_complete) && requests.every(request => request.ok && request.usage),
    retries: usages.reduce((total, usage) => total + usage.retries, 0),
    failedCalls: usages.reduce((total, usage) => total + usage.failed_calls, 0),
    agentMs: sumKnown(usages.map(usage => usage.latency_ms)),
    clientMs: requests.reduce((total, request) => total + request.clientMs, 0),
    wallMs: requests.at(-1).finishedAt - cycle.startedAt,
    ticket: requests.some(request => request.kind === 'confirm' && request.ok),
  }
}

function costText(cost, complete) {
  if (cost == null) return complete ? '$0 — платных вызовов не было' : 'нет данных'
  return complete ? dollars(cost) : `не менее ${dollars(cost)}: по части вызовов стоимость неизвестна`
}

function Step({ step }) {
  const name = stepLabels[step.step] || step.step
  const paid = step.model != null && step.kind !== 'service'
  return <tr className={step.ok ? undefined : 'failed'}>
    <th scope="row">{name}{!step.ok && ' · ошибка'}</th>
    <td>{step.model || '—'}{step.provider && <><br /><small>{step.provider}</small></>}</td>
    <td>{tokens(step.input_tokens)} / {tokens(step.output_tokens)}{step.reasoning_tokens != null && <><br /><small>рассуждения {tokens(step.reasoning_tokens)}</small></>}</td>
    <td>{step.cost_usd != null ? dollars(step.cost_usd) : paid ? 'нет данных' : '—'}</td>
    <td>{seconds(step.latency_ms)}</td>
    <td>{step.retries}</td>
  </tr>
}

export function CycleCost({ cycle }) {
  if (!cycle) {
    return <section className="service-metrics" aria-label="Служебный режим">
      <h3>Стоимость и время цикла</h3>
      <p className="meta">Нет данных: в этом окне ещё не было запросов к агенту.</p>
    </section>
  }
  const total = cycleTotals(cycle)
  return <section className="service-metrics" aria-label="Служебный режим">
    <h3>Стоимость и время цикла</h3>
    <p className="meta">{total.ticket ? 'Цикл завершён заявкой в ТОиР.' : 'Цикл открыт: от результата диагностики до последнего ответа.'}{' '}
      Значения взяты из ответов провайдеров для запросов в этом окне; если провайдер значение не вернул, указано «нет данных».</p>
    <dl className="service-totals">
      <div><dt>Стоимость LLM</dt><dd>{costText(total.cost, total.costComplete)}</dd></div>
      <div><dt>Токены вход / выход</dt><dd>{total.input == null && total.output == null ? (total.llmCalls ? 'нет данных' : 'LLM не вызывалась')
        : `${tokens(total.input)} / ${tokens(total.output)}`}{total.reasoning != null && ` · рассуждения ${tokens(total.reasoning)}`}</dd></div>
      <div><dt>Время цикла</dt><dd>{seconds(total.wallMs)}</dd></div>
      <div><dt>Ожидание ответов</dt><dd>{seconds(total.clientMs)} · агент {seconds(total.agentMs)}</dd></div>
      <div><dt>Запросы / LLM-вызовы</dt><dd>{total.requests} / {total.llmCalls}</dd></div>
      <div><dt>Повторы и ошибки</dt><dd>повторов в вызовах {total.retries} · ошибок вызовов {total.failedCalls} · неудачных запросов {total.failed}</dd></div>
    </dl>
    {total.unmeasured > 0 && <p className="meta">Для {total.unmeasured} запрос(ов) сервер не вернул замеры.</p>}
    {cycle.requests.map((request, index) => <details key={index} className="service-request" open={index === cycle.requests.length - 1}>
      <summary>{requestLabels[request.kind] || request.kind} · {seconds(request.clientMs)}
        {request.usage?.cost_usd != null && ` · ${dollars(request.usage.cost_usd)}`}{!request.ok && ` · не выполнен (${request.error})`}</summary>
      {request.usage?.steps.length ? <div className="service-table"><table>
        <thead><tr><th scope="col">Шаг</th><th scope="col">Модель</th><th scope="col">Токены вх / вых</th>
          <th scope="col">Стоимость</th><th scope="col">Задержка</th><th scope="col">Повторы</th></tr></thead>
        <tbody>{request.usage.steps.map((step, stepIndex) => <Step key={stepIndex} step={step} />)}</tbody>
      </table></div> : <p className="meta">{request.ok ? 'Нет данных о вызовах.' : 'Ответ с ошибкой не содержит замеров.'}</p>}
    </details>)}
  </section>
}
