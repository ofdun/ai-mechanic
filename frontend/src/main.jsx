import React, { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { conversationId, formatDate, request } from './api'
import { Draft, Response } from './Response'
import './style.css'

function App() {
  const [catalog, setCatalog] = useState(null)
  const [error, setError] = useState('')
  const [attempt, setAttempt] = useState(0)
  const [vehicleId, setVehicleId] = useState('')
  const [sessionId, setSessionId] = useState('')
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
      setVehicleId(sorted[0]?.vehicle_id || '')
      setSessionId(sorted[0]?.sessions[0]?.session_id || '')
    }).catch(err => {
      if (!controller.signal.aborted) setError(err.message)
    })
    return () => controller.abort()
  }, [attempt])
  const vehicle = catalog?.find(item => item.vehicle_id === vehicleId)
  function changeVehicle(id) {
    selectionControl.current = 'vehicle'
    setVehicleId(id)
    setSessionId(catalog.find(item => item.vehicle_id === id)?.sessions[0]?.session_id || '')
  }
  function changeSession(id) {
    selectionControl.current = 'session'
    setSessionId(id)
  }
  return <>
    <header className="mast"><div><span className="wordmark">Цифровой механик</span><span className="mast-note">Диагностический пост</span></div></header>
    <main><div className="page-title"><h1>Диагностика двигателя</h1></div>
      {error ? <div role="alert" className="error"><p>{error}</p><button onClick={() => setAttempt(n => n + 1)}>Повторить загрузку</button></div>
        : !catalog ? <p role="status">Загружаем список транспорта…</p>
        : !vehicle ? <p role="status">В справочнике пока нет транспортных средств.</p>
        : <Workspace key={JSON.stringify([vehicleId, sessionId])} {...{ catalog, vehicle, sessionId, changeVehicle, changeSession }} />}
    </main>
  </>
}

const pendingLabels = { diagnose: 'Получаем результат…', chat: 'Ищем ответ…', draft_ticket: 'Готовим черновик…', confirm: 'Отправляем заявку…' }

function Workspace({ catalog, vehicle, sessionId, changeVehicle, changeSession }) {
  const [id] = useState(conversationId)
  const [turns, setTurns] = useState([])
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState('')
  const [error, setError] = useState('')
  const [canDraft, setCanDraft] = useState(false)
  const [draft, setDraft] = useState(null)
  const [ticketStatus, setTicketStatus] = useState('')
  const [view, setView] = useState('Результат')
  const [diagnosis, setDiagnosis] = useState(null)
  const [ticketHint, setTicketHint] = useState('')
  const chatBody = useRef(null)
  useLayoutEffect(() => {
    if (chatBody.current) chatBody.current.scrollTop = chatBody.current.scrollHeight
  }, [turns])
  const running = useRef(null)
  useEffect(() => () => running.current?.abort(), [])
  const pair = { vehicle_id: vehicle.vehicle_id, session_id: sessionId }
  async function run(kind) {
    if (running.current || !sessionId) return
    const controller = new AbortController()
    running.current = controller
    if (kind === 'diagnose') setView('Результат')
    setPending(kind)
    setError('')
    const text = question.trim()
    try {
      if (kind === 'confirm') {
        const { draft_id, draft_version } = draft.value
        const receipt = await request('/agent/tickets/confirm', { ...pair, draft_id, draft_version, confirmed: true }, controller.signal)
        if (controller.signal.aborted) return
        setTicketStatus('Заявка создана. Номер в ТОиР: ' + receipt.ticket_id)
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
      if (kind === 'chat') {
        setTurns(old => [...old, { result, question: text }])
        setQuestion('')
      }
      if (result.diagnostic || result.action === 'draft_ticket') {
        setTicketHint(result.ticket_draft || result.can_create_ticket ? ''
          : result.limitations?.join(' ') || result.message)
        setCanDraft(Boolean(result.can_create_ticket && !result.requires_remeasurement))
        setDraft(result.ticket_draft && !ticketStatus ? { value: result.ticket_draft, evidence: result.evidence.items } : null)
      }
    } catch (err) {
      if (controller.signal.aborted) return
      if (kind === 'confirm') {
        if (!err.code || ['TICKET_OUTCOME_UNKNOWN', 'DUPLICATE_TICKET', 'TICKET_IN_PROGRESS'].includes(err.code)) {
          setTicketStatus(err.code === 'DUPLICATE_TICKET' ? 'Для этой проверки уже есть заявка или попытка отправки. Сверьте её статус в ТОиР.'
            : 'Статус отправки требует сверки с ТОиР. Повторная отправка заблокирована.')
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
      setError(err.message || 'Сервис недоступен. Повторите запрос.')
    } finally {
      if (!controller.signal.aborted) setPending('')
      running.current = null
    }
  }

  return <div className="workspace">
    <aside className="controls" aria-labelledby="checking">
      <h2 id="checking">Параметры замера</h2>
      <div id="selection">
      <div><label htmlFor="vehicle">Транспортное средство</label>
      <select id="vehicle" value={vehicle.vehicle_id} disabled={pending === 'confirm'} onChange={e => changeVehicle(e.target.value)}>
        {catalog.map(v => <option key={v.vehicle_id} value={v.vehicle_id}>{v.vehicle_id}</option>)}
      </select>
      </div>
      <dl className="engine"><dt>Двигатель</dt><dd>{vehicle.engine_model}</dd></dl>
      <div><label htmlFor="session">Проверка</label>
      <select id="session" value={sessionId} disabled={!vehicle.sessions.length || pending === 'confirm'} onChange={e => changeSession(e.target.value)}>
        {!vehicle.sessions.length && <option>Нет проверок</option>}
        {vehicle.sessions.map(s => <option key={s.session_id} value={s.session_id}>{s.session_id} · {formatDate(s.checked_at)}</option>)}
      </select></div>
      <button className="primary" disabled={!sessionId || !!pending} onClick={() => run('diagnose')}>Получить результат диагностики</button>
      </div>
    </aside>
    <section className="chat" aria-label="Протокол проверки">
      <nav className="view-switch" aria-label="Разделы проверки">
        {['Результат', 'Чат', 'Заявка'].map(name =>
          <button key={name} type="button" aria-pressed={view === name} onClick={() => setView(name)}>{name}</button>)}
      </nav>
      <div className="pane" hidden={view !== 'Результат'} aria-label="Результат проверки" role="region" tabIndex={0}>
        {diagnosis ? <Response result={diagnosis} /> : <div className="empty">
          <h3>{sessionId ? 'Результат ещё не запрошен' : 'Проверок пока нет'}</h3>
          <p>{sessionId ? 'Получите результат выбранного замера.' : 'Выберите другую машину.'}</p></div>}
      </div>
      <div className="pane" hidden={view !== 'Чат'} ref={chatBody} role="log" aria-label="Сообщения" tabIndex={0}>
        {!turns.length && <div className="empty"><h3>Вопрос по двигателю</h3>
          <p>{diagnosis ? diagnosis.message : `Спросите о замере или о руководстве для двигателя ${vehicle.engine_model}.`}</p></div>}
        {turns.map(({ result, question }) => <div className="turn" key={result.request_id}>
          <div className="question"><strong>Вы</strong><p>{question}</p></div>
          {result.ticket_draft
            ? <p>Черновик подготовлен в разделе «Заявка».</p>
            : <Response result={result} />}
        </div>)}
      </div>
      <div className="pane ticket-pane" hidden={view !== 'Заявка'} role="region" aria-label="Заявка на дефектовку" tabIndex={0}>
        {ticketStatus ? <p className="notice">{ticketStatus}</p>
          : draft ? <Draft draft={draft.value} evidence={draft.evidence} />
          : <div className="empty"><h3>Заявка на дефектовку</h3>
            <p>{canDraft ? 'Подготовьте черновик и проверьте состав работ перед отправкой.'
              : ticketHint || 'Сначала получите результат диагностики.'}</p></div>}
      </div>
      <div className="panel-footer">
        {view === 'Результат' && <div className="footer-actions">
          <button className="secondary" onClick={() => setView('Чат')}>Задать вопрос</button>
          <button disabled={!canDraft && !draft && !ticketStatus} onClick={() => setView('Заявка')}>К заявке</button>
        </div>}
        {view === 'Чат' && <form onSubmit={e => { e.preventDefault(); if (question.trim()) run('chat') }}>
          <div className="compose-row"><input aria-label="Вопрос по двигателю" placeholder="Вопрос по двигателю" value={question}
            maxLength={8000} disabled={!sessionId || !!pending} onChange={e => setQuestion(e.target.value)} autoComplete="off" />
            <button type="submit" className="submit" disabled={!question.trim() || !!pending || !sessionId}>Отправить</button></div>
        </form>}
        {view === 'Заявка' && <button className="primary" disabled={!!pending || !!ticketStatus || (!draft && !canDraft)}
          onClick={() => run(draft ? 'confirm' : 'draft_ticket')}>{draft ? 'Подтвердить заявку' : 'Создать заявку'}</button>}
      </div>
      <div className="panel-status" role="status">{error || pendingLabels[pending]}</div>
    </section>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
