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
  const [folded, setFolded] = useState(false)
  const lastResponse = useRef(null)
  const revealResult = useRef(false)
  const reopenSelection = useRef(false)
  useLayoutEffect(() => {
    if (reopenSelection.current) {
      document.getElementById('vehicle')?.focus()
      reopenSelection.current = false
    }
  }, [folded])
  useLayoutEffect(() => {
    if (revealResult.current) {
      lastResponse.current?.focus()
      lastResponse.current?.scrollIntoView({ block: 'nearest' })
      revealResult.current = false
    }
  }, [turns])
  const running = useRef(null)
  useEffect(() => () => running.current?.abort(), [])
  const pair = { vehicle_id: vehicle.vehicle_id, session_id: sessionId }
  async function run(kind) {
    if (running.current || !sessionId) return
    const controller = new AbortController()
    running.current = controller
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
      if (kind === 'diagnose' && window.matchMedia('(max-width: 47rem)').matches) {
        setFolded(true)
        revealResult.current = true
      }
      setTurns(old => [...old, { result, question: kind === 'chat' ? text : '' }])
      if (kind === 'chat') setQuestion('')
      if (result.diagnostic || result.action === 'draft_ticket') {
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
    <aside className={'controls' + (folded ? ' folded' : '')} aria-labelledby="checking">
      <h2 id="checking">Параметры замера</h2>
      <button type="button" className="fold-toggle" aria-controls="selection" aria-expanded={!folded}
        onClick={() => { reopenSelection.current = true; setFolded(false) }}><span>{vehicle.vehicle_id} / {sessionId}</span><strong>Изменить</strong></button>
      <div id="selection">
      <label htmlFor="vehicle">Транспортное средство</label>
      <select id="vehicle" value={vehicle.vehicle_id} disabled={pending === 'confirm'} onChange={e => changeVehicle(e.target.value)}>
        {catalog.map(v => <option key={v.vehicle_id} value={v.vehicle_id}>{v.vehicle_id}</option>)}
      </select>
      <dl className="engine"><dt>Двигатель</dt><dd>{vehicle.engine_model}</dd></dl>
      <label htmlFor="session">Проверка</label>
      <select id="session" value={sessionId} disabled={!vehicle.sessions.length || pending === 'confirm'} onChange={e => changeSession(e.target.value)}>
        {!vehicle.sessions.length && <option>Нет проверок</option>}
        {vehicle.sessions.map(s => <option key={s.session_id} value={s.session_id}>{s.session_id} · {formatDate(s.checked_at)}</option>)}
      </select>
      <button className="primary" disabled={!sessionId || !!pending} onClick={() => run('diagnose')}>Получить результат диагностики</button>
      </div>
    </aside>
    <section className="chat" aria-labelledby="chat-title"><div className="chat-head"><h2 id="chat-title">Протокол проверки</h2>
      <span className="serial">{sessionId}</span></div>
      <div className="thread" role="log" aria-label="Сообщения" aria-live="polite">
        {!turns.length && <div className="empty"><h3>{sessionId ? 'Результат ещё не запрошен' : 'Проверок пока нет'}</h3>
          <p>{sessionId ? 'Запустите диагностику или задайте вопрос по руководству для двигателя ' + vehicle.engine_model + '.' : 'Выберите другое транспортное средство.'}</p></div>}
        {turns.map(({ result, question }, index) => <div className="turn" tabIndex={-1}
          ref={index === turns.length - 1 ? lastResponse : null} key={result.request_id}>
          {question && <div className="question"><strong>Вы</strong><p>{question}</p></div>}
          <Response result={result} />
        </div>)}
      </div>
      <div className="actions">
        {!!ticketStatus && <p className="notice" role="status">{ticketStatus}</p>}
        {draft && !ticketStatus && <Draft draft={draft.value} evidence={draft.evidence} disabled={!!pending} onConfirm={() => run('confirm')} />}
        {canDraft && !draft && !ticketStatus && <button className="secondary" disabled={!!pending} onClick={() => run('draft_ticket')}>Создать заявку</button>}
        {error && <p className="error" role="alert">{error}</p>}
        <div className="progress" role="status">{pendingLabels[pending] || ''}</div>
      </div>
      <form className="composer" onSubmit={e => { e.preventDefault(); if (question.trim()) run('chat') }}>
        <label htmlFor="question">Вопрос по двигателю</label>
        <div className="compose-row"><input id="question" placeholder="Что проверить в первую очередь?" value={question}
          maxLength={8000} disabled={!sessionId || !!pending} onChange={e => setQuestion(e.target.value)} autoComplete="off" />
          <button type="submit" className="submit" disabled={!question.trim() || !sessionId || !!pending}>Отправить</button></div>
      </form>
    </section>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
