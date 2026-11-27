import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ArrowRight, CheckCheck, ClipboardList, FileText, Gauge, History, MessageSquare, ScanLine, Send, Wrench } from 'lucide-react'
import { conversationId, formatDate, request } from './api'
import { DiagnosticReport, DiagnosticVerdict, Draft, Response, VehicleHistory } from './Response'
import { ManualNavigation } from './ManualReader'
import { useChatHistory } from './chatHistory'
import './style.css'

function App() {
  const [view, setView] = useState('result')
  const [catalog, setCatalog] = useState(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [vehicleId, setVehicleId] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [engineFilter, setEngineFilter] = useState('')
  const vehiclesByEngine = useMemo(() => {
    const groups = new Map()
    for (const vehicle of catalog || []) {
      if (!groups.has(vehicle.engine_model)) groups.set(vehicle.engine_model, [])
      groups.get(vehicle.engine_model).push(vehicle)
    }
    return groups
  }, [catalog])
  const selectionControl = useRef(null)
  useLayoutEffect(() => {
    if (selectionControl.current) {
      document.getElementById(selectionControl.current)?.focus({ preventScroll: true })
      selectionControl.current = null
    }
  }, [vehicleId, sessionId])
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    request('/api/vehicles', null, controller.signal).then(({ vehicles }) => {
      if (controller.signal.aborted) return
      const sorted = vehicles.map(vehicle => ({
        ...vehicle,
        sessions: [...vehicle.sessions].sort((a, b) => Date.parse(b.checked_at) - Date.parse(a.checked_at)),
      }))
      setCatalog(sorted)
    }).catch(err => {
      if (!controller.signal.aborted) setError(err.message)
    })
    return () => controller.abort()
  }, [attempt])
  const vehicle = catalog?.find(item => item.vehicle_id === vehicleId)
  function changeVehicle(id, control = 'vehicle') {
    selectionControl.current = control
    setVehicleId(id)
    setSessionId(catalog.find(item => item.vehicle_id === id)?.sessions[0]?.session_id || '')
  }
  function changeEngine(engine) {
    setEngineFilter(engine)
    changeVehicle('', 'engine')
  }
  function changeSession(id) {
    selectionControl.current = 'session'
    setSessionId(id)
  }
  return <>
    <header className="mast"><div className="mast-inner">
      <span className="wordmark"><span className="brand-mark"><Wrench aria-hidden="true" /></span>Цифровой механик</span>
      <h1>Диагностика двигателя</h1>
    </div></header>
    <main>
      {error ? <div role="alert" className="error"><p>{error}</p><button onClick={() => setAttempt(n => n + 1)}>Повторить загрузку</button></div>
        : !catalog ? <p role="status">Загружаем список транспорта…</p>
        : !catalog.length ? <p role="status">В справочнике пока нет транспортных средств.</p>
        : <Workspace key={JSON.stringify([vehicleId, sessionId])}
          {...{ view, setView, vehiclesByEngine, engineFilter, vehicle, sessionId, changeEngine, changeVehicle, changeSession }} />}
    </main>
  </>
}

const pendingLabels = { diagnose: 'Получаем результат', chat: 'Ищем ответ', draft_ticket: 'Готовим черновик', confirm: 'Отправляем заявку' }
const views = [
  { id: 'result', label: 'Результат', icon: FileText },
  { id: 'chat', label: 'Чат', icon: MessageSquare },
  { id: 'ticket', label: 'Заявка', icon: ClipboardList },
  { id: 'history', label: 'История', icon: History },
]

function EmptyView({ icon: Icon, title, children }) {
  return <div className="empty"><Icon aria-hidden="true" /><h3>{title}</h3><p>{children}</p></div>
}

function ticketUnavailableReason(result) {
  if (result.requires_remeasurement || result.recognition_error) return result.message
  if (result.diagnostic?.status === 'normal') return 'Замер в норме. Заявка на дефектовку не требуется.'
  if (result.diagnostic?.status === 'anomaly' && result.search_plan) {
    return 'Для этого отклонения в руководстве пока не найдены подтверждённые работы для заявки.'
  }
  return result.limitations?.join(' ') || result.message
}

function VehicleOptions({ vehicles }) {
  return vehicles.map(({ vehicle_id, make, model }) => {
    const name = model?.startsWith(make || '') ? model : [make, model].filter(Boolean).join(' ')
    return <option key={vehicle_id} value={vehicle_id}>{[vehicle_id, name].filter(Boolean).join(' · ')}</option>
  })
}

function Workspace({ view, setView, vehiclesByEngine, engineFilter, vehicle, sessionId, changeEngine, changeVehicle, changeSession }) {
  const { id, turns, setTurns, reset, warning } = useChatHistory(vehicle?.vehicle_id, sessionId)
  const sendingTurn = useRef(null)
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState('')
  const [error, setError] = useState('')
  const [canDraft, setCanDraft] = useState(false)
  const [draft, setDraft] = useState(null)
  const [ticketStatus, setTicketStatus] = useState(null)
  const [diagnosis, setDiagnosis] = useState(null)
  const [ticketHint, setTicketHint] = useState('')
  const chatBody = useRef(null)
  const composer = useRef(null)
  const focusQuestion = useRef(false)
  useLayoutEffect(() => {
    if (view === 'chat' && focusQuestion.current) composer.current?.focus()
    focusQuestion.current = false
  }, [view])
  useLayoutEffect(() => {
    const pane = chatBody.current
    const turn = pane?.querySelector('.turn:last-child')
    if (turn) pane.scrollTop += turn.getBoundingClientRect().top - pane.getBoundingClientRect().top
  }, [turns.length, turns.at(-1)?.kind])
  function selectTab(next) {
    setView(next)
    document.getElementById('tab-' + next)?.focus()
  }
  function navigateTabs(event, index) {
    const last = views.length - 1
    const next = { ArrowRight: (index + 1) % views.length, ArrowLeft: (index + last) % views.length, Home: 0, End: last }[event.key]
    if (next === undefined) return
    event.preventDefault()
    selectTab(views[next].id)
  }
  const running = useRef(null)
  useEffect(() => () => running.current?.abort(), [])
  const pair = { vehicle_id: vehicle?.vehicle_id, session_id: sessionId }
  async function run(kind, turn) {
    const text = turn?.question ?? question.trim()
    const retryId = turn?.id
    if (running.current || !sessionId || !id) return
    const controller = new AbortController()
    running.current = controller
    setPending(kind)
    setError('')
    const turnId = kind === 'chat' ? retryId ?? conversationId() : null
    sendingTurn.current = turnId
    function updateTurn(fields) {
      setTurns(old => old.map(turn => turn.id === turnId ? { ...turn, ...fields } : turn))
    }
    if (kind === 'chat') {
      if (retryId) updateTurn({ error: '' })
      else {
        setTurns(old => [...old, { id: turnId, question: text }])
        if (!turn) setQuestion('')
      }
    }
    try {
      if (kind === 'confirm') {
        const { draft_id, draft_version } = draft.value
        const receipt = await request('/agent/tickets/confirm', {
          ...pair, draft_id, draft_version, confirmed: true, urgency: draft.urgency,
        }, controller.signal)
        if (controller.signal.aborted) return
        setTicketStatus({ ticketId: receipt.ticket_id })
        setDraft(null)
        setCanDraft(false)
        return
      }
      const body = kind === 'chat' ? { message: text } : { command: kind }
      const result = await request('/agent/run', { ...pair, conversation_id: id, ...body }, controller.signal)
      if (controller.signal.aborted) return
      if (result.vehicle_id !== pair.vehicle_id || result.session_id !== sessionId || result.conversation_id !== id || result.engine_model !== vehicle.engine_model)
        throw new Error('Ответ относится к другой проверке. Получите результат заново.')
      if (result.diagnostic) setDiagnosis(result)
      if (kind === 'diagnose' && result.diagnostic) {
        setTurns(old => {
          const verdict = { kind: 'verdict', result }
          return old.some(turn => turn.kind === 'verdict')
            ? old.map(turn => turn.kind === 'verdict' ? verdict : turn)
            : [...old, verdict]
        })
      }
      if (kind === 'chat') updateTurn({ result })
      if (result.diagnostic || result.action === 'draft_ticket') {
        setTicketHint(result.ticket_draft || result.can_create_ticket ? '' : ticketUnavailableReason(result))
        setCanDraft(Boolean(result.can_create_ticket && !result.requires_remeasurement))
        setDraft(result.ticket_draft && !ticketStatus ? {
          value: result.ticket_draft, evidence: result.evidence.items, urgency: result.ticket_draft.payload.urgency,
        } : null)
      }
    } catch (err) {
      if (controller.signal.aborted) return
      if (kind === 'confirm') {
        if (!err.code || ['TICKET_OUTCOME_UNKNOWN', 'DUPLICATE_TICKET', 'TICKET_IN_PROGRESS'].includes(err.code)) {
          setTicketStatus({ message: err.code === 'DUPLICATE_TICKET' ? 'Для этой проверки уже есть заявка или попытка отправки. Сверьте её статус в ТОиР.'
            : 'Статус отправки требует сверки с ТОиР. Повторная отправка заблокирована.' })
          setCanDraft(false)
          setDraft(null)
          return
        }
        if (['DRAFT_EXPIRED', 'DRAFT_STALE', 'DRAFT_NOT_FOUND', 'REMEASUREMENT_REQUIRED'].includes(err.code)) {
          setDraft(null)
          setCanDraft(false)
          setError(err.message + ' Получите результат диагностики заново.')
          return
        }
      }
      const message = err.message || 'Сервис недоступен. Повторите запрос.'
      if (kind === 'chat') updateTurn({ error: message })
      else setError(message)
    } finally {
      if (!controller.signal.aborted) setPending('')
      running.current = null
    }
  }

  const returnLabels = { result: 'К результату', chat: 'К чату', history: 'К истории', ticket: 'К заявке' }
  return <ManualNavigation sessionKey={id} returnLabel={returnLabels[view]}><div className="workspace">
    <aside className="controls" aria-labelledby="checking">
      <div className="controls-heading"><h2 id="checking"><Gauge aria-hidden="true" />Параметры замера</h2>
      </div>
      <div id="selection">
      <div><label htmlFor="engine">Двигатель</label>
      <select id="engine" required value={engineFilter} disabled={pending === 'confirm'} onChange={e => changeEngine(e.target.value)}>
        <option value="" disabled>Выберите двигатель</option>
        {Array.from(vehiclesByEngine.keys(), engine => <option key={engine} value={engine}>{engine}</option>)}
      </select></div>
      <div><label htmlFor="vehicle">Транспортное средство</label>
      <select id="vehicle" required value={vehicle?.vehicle_id || ''} disabled={!engineFilter || pending === 'confirm'} onChange={e => changeVehicle(e.target.value)}>
        <option value="" disabled>Выберите машину</option>
        <VehicleOptions vehicles={vehiclesByEngine.get(engineFilter) || []} />
      </select></div>
      <div className="session-picker"><label htmlFor="session">Проверка</label>
      <select id="session" value={sessionId} disabled={!vehicle?.sessions.length || pending === 'confirm'} onChange={e => changeSession(e.target.value)}>
        {!vehicle?.sessions.length && <option value="">{vehicle ? 'Нет проверок' : 'Сначала выберите машину'}</option>}
        {vehicle?.sessions.map(s => <option key={s.session_id} value={s.session_id}>{s.session_id} · {formatDate(s.checked_at)}</option>)}
      </select></div>
      <button className="primary" disabled={!id || !sessionId || !!pending} onClick={() => run('diagnose')}>Получить результат диагностики<ArrowRight aria-hidden="true" /></button>
      </div>
    </aside>
    <section className="chat" aria-label="Протокол проверки">
      <div className="view-switch" role="tablist" aria-label="Разделы проверки"
        style={{ '--tab-count': views.length, '--active-tab': views.findIndex(tab => tab.id === view) }}>
        {views.map(({ id, label, icon: Icon }, index) =>
          <button key={id} type="button" role="tab" id={'tab-' + id} aria-controls={'panel-' + id}
            aria-selected={view === id} tabIndex={view === id ? 0 : -1}
            onKeyDown={e => navigateTabs(e, index)} onClick={() => setView(id)}><Icon aria-hidden="true" />{label}</button>)}
      </div>
      {!vehicle ? <div className="pane" id={'panel-' + view} role="tabpanel" aria-labelledby={'tab-' + view} tabIndex={0}>
        <EmptyView icon={Gauge} title={engineFilter ? 'Выберите машину' : 'Выберите двигатель'}>
          {engineFilter ? 'В списке транспорта доступны машины с выбранным двигателем.' : 'Затем выберите машину, чтобы открыть её проверки.'}
        </EmptyView>
      </div> : <>
      <div className="pane" hidden={view !== 'result'} id="panel-result" aria-labelledby="tab-result" role="tabpanel" tabIndex={0}>
        {diagnosis ? <DiagnosticReport result={diagnosis} /> : <EmptyView icon={ScanLine}
          title={sessionId ? 'Результат ещё не запрошен' : 'У этой машины пока нет проверок'}>
          {sessionId ? 'Проверьте параметры замера и нажмите «Получить результат диагностики».' : 'Выберите другую машину в списке транспорта.'}
        </EmptyView>}
      </div>
      <div className="pane" hidden={view !== 'chat'} ref={chatBody} id="panel-chat" aria-labelledby="tab-chat" role="tabpanel" tabIndex={0}>
        {warning && <p role="status">{warning}</p>}
        {!!turns.length && <button className="reset-chat" disabled={!!pending} onClick={() => {
          if (window.confirm('Сбросить переписку по этой машине и замеру? Сообщения исчезнут из этого браузера.')) {
            reset()
            setQuestion('')
            setError('')
            composer.current?.focus()
          }
        }}>Сбросить диалог</button>}
        {!turns.length && <EmptyView icon={MessageSquare} title="Что хотите уточнить?">
          Спросите о замере или нормативе из руководства для двигателя {vehicle.engine_model}.
        </EmptyView>}
        <div role="log" aria-label="Сообщения">
        {turns.map(({ id, kind, result, question, error: turnError }) => kind === 'verdict'
          ? <div className="turn" key="verdict"><DiagnosticVerdict result={result} /></div>
          : <div className="turn" key={id}>
          <div className="question"><strong>Вы</strong><p>{question}</p></div>
          {turnError || (!result && id !== sendingTurn.current) ? <div className="chat-error"><p>Ответ не получен. {turnError || 'Отправка прервалась.'}</p>
            <button disabled={!!pending} onClick={() => run('chat', { id, question })}>Повторить</button></div>
            : !result ? <p className="meta">Ищем ответ<span className="pending-dots">…</span></p>
            : result.ticket_draft
            ? <p>Черновик подготовлен в разделе «Заявка».</p>
            : <Response result={result} />}
        </div>)}</div>
      </div>
      <div className="pane" hidden={view !== 'history'} id="panel-history" aria-labelledby="tab-history" role="tabpanel" tabIndex={0}>
        <VehicleHistory vehicleId={vehicle.vehicle_id} sessions={vehicle.sessions.slice(0, 5)} />
      </div>
      <div className="pane" hidden={view !== 'ticket'} id="panel-ticket" aria-labelledby="tab-ticket" role="tabpanel" tabIndex={0}>
        {ticketStatus ? ticketStatus.ticketId
          ? <div className="receipt" role="status"><CheckCheck aria-hidden="true" /><h3>Заявка создана</h3><p>Номер в ТОиР <strong>{ticketStatus.ticketId}</strong></p></div>
          : <p className="limitations" role="status">{ticketStatus.message}</p>
          : draft ? <Draft draft={draft.value} evidence={draft.evidence} urgency={draft.urgency}
              disabled={!!pending} onUrgencyChange={urgency => setDraft(current => ({ ...current, urgency }))} />
          : diagnosis?.manual_referral ? <section className="draft" aria-label="Рекомендация на дефектовку">
            <h3>Рекомендация на дефектовку</h3>
            <p>Заявка в ТОиР не отправлена. Для ручного оформления используйте этот текст.</p>
            <p className="manual-referral">{diagnosis.manual_referral}</p>
          </section>
          : <EmptyView icon={ClipboardList} title="Заявка на дефектовку">
            {canDraft ? 'Создайте черновик. Вы сможете проверить работы и запчасти перед отправкой в ТОиР.'
              : ticketHint || (sessionId ? 'Сначала получите результат диагностики.' : 'У этой машины пока нет проверок. Выберите другое ТС.')}</EmptyView>}
      </div>
      <div className="panel-footer" hidden={view === 'history'}>
        {view === 'result' && <div className="footer-actions">
          <button className="secondary" onClick={() => { focusQuestion.current = true; setView('chat') }}><MessageSquare aria-hidden="true" />Задать вопрос</button>
          <button type="button" aria-controls="panel-ticket" onClick={() => selectTab('ticket')}>К заявке<ArrowRight aria-hidden="true" /></button>
        </div>}
        {view === 'chat' && <form onSubmit={e => { e.preventDefault(); if (question.trim()) { composer.current?.focus(); run('chat') } }}>
          <div className="compose-row"><input ref={composer} aria-label="Вопрос по двигателю" placeholder="Вопрос по двигателю…" value={question}
            maxLength={8000} disabled={!sessionId} onChange={e => setQuestion(e.target.value)} autoComplete="off" />
            <button type="submit" className="submit" disabled={!id || !question.trim() || !!pending || !sessionId}><Send aria-hidden="true" /><span>Отправить</span></button></div>
        </form>}
        {view === 'ticket' && <button className="primary" disabled={!!pending || !!ticketStatus || (!draft && !canDraft)}
          onClick={() => run(draft ? 'confirm' : 'draft_ticket')}>{ticketStatus ? ticketStatus.ticketId ? 'Отправлено в ТОиР' : 'Отправка заблокирована'
            : draft ? 'Подтвердить заявку' : 'Создать заявку'}</button>}
      </div>
      </>}
      <div className="panel-status" role="status">{error || (pending && pending !== 'chat' &&
        <>{pendingLabels[pending]}<span className="pending-dots">…</span></>)}</div>
    </section>
  </div></ManualNavigation>
}

createRoot(document.getElementById('root')).render(<App />)
