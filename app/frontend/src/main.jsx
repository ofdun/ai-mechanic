import { UiLanguage, translator, useUiText } from './uiLanguage'
import React, { useEffect, useEffectEvent, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { ArrowRight, CheckCheck, ClipboardList, FileText, Gauge, History, MessageSquare, ScanLine, Trash2, Wrench } from 'lucide-react'
import { conversationId, formatDate, request } from './api'
import { DiagnosticReport, DiagnosticVerdict, Draft, Response, VehicleHistory } from './Response'
import { ManualNavigation } from './ManualReader'
import { FleetOverview } from './FleetOverview'
import { InspectionForm, InspectionList } from './Inspection'
import { useChatHistory } from './chatHistory'
import { ChatComposer } from './ChatComposer'
import { CycleCost, useCycleMetrics, useServiceMode } from './serviceMode'
import './style.css'

const lastVehicleKey = 'ai-mechanic.lastVehicleId'

function readPreference(key) {
  try { return localStorage.getItem(key) } catch { return null }
}

function savePreference(key, value) {
  try { localStorage.setItem(key, value) } catch { /* Storage may be disabled. */ }
}

function sessionKey(vehicleId) {
  return 'ai-mechanic.lastSession.' + vehicleId
}

function selectedSession(vehicle) {
  if (!vehicle) return ''
  const saved = readPreference(sessionKey(vehicle.vehicle_id))
  return vehicle.sessions.some(session => session.session_id === saved) ? saved : vehicle.sessions[0]?.session_id || ''
}

const modeKey = 'ai-mechanic.mode'
const answerLanguages = { ru: 'Русский', uz: 'Oʻzbekcha', tg: 'Тоҷикӣ', ky: 'Кыргызча', kk: 'Қазақша', hi: 'हिन्दी' }
const languageKey = 'ai-mechanic.answerLanguage'

function App() {
  const [mode, setMode] = useState(() => readPreference(modeKey) === 'chief' ? 'chief' : 'mechanic')
  const [language, setLanguage] = useState(() => {
    const saved = readPreference(languageKey)
    return Object.hasOwn(answerLanguages, saved) ? saved : 'ru'
  })
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
      const savedId = readPreference(lastVehicleKey)
      const savedVehicle = sorted.find(item => item.vehicle_id === savedId)
      if (savedVehicle) {
        setEngineFilter(savedVehicle.engine_model)
        setVehicleId(savedVehicle.vehicle_id)
        setSessionId(selectedSession(savedVehicle))
      }
      setCatalog(sorted)
    }).catch(err => {
      if (!controller.signal.aborted) setError(err.message)
    })
    return () => controller.abort()
  }, [attempt])
  const vehicle = catalog?.find(item => item.vehicle_id === vehicleId)
  function changeVehicle(id, control = 'vehicle') {
    if (id) savePreference(lastVehicleKey, id)
    selectionControl.current = control
    setVehicleId(id)
    setSessionId(selectedSession(catalog.find(item => item.vehicle_id === id)))
  }
  function changeEngine(engine) {
    setEngineFilter(engine)
    changeVehicle('', 'engine')
  }
  function changeSession(id) {
    savePreference(sessionKey(vehicleId), id)
    selectionControl.current = 'session'
    setSessionId(id)
  }
  const t = translator(language)
  useEffect(() => {
    document.documentElement.lang = language
    document.title = translator(language)('Диагностика · Цифровой механик')
  }, [language])
  function changeMode(next) {
    savePreference(modeKey, next)
    setMode(next)
  }
  // From the fleet list: open the vehicle's latest check on the History tab.
  function openVehicle(next) {
    savePreference(lastVehicleKey, next.vehicle_id)
    if (next.sessions[0]) savePreference(sessionKey(next.vehicle_id), next.sessions[0].session_id)
    setEngineFilter(next.engine_model)
    setVehicleId(next.vehicle_id)
    setSessionId(next.sessions[0]?.session_id || '')
    setView('history')
    changeMode('mechanic')
  }
  function openResult(event) {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    changeMode('mechanic')
    setView('result')
    if (history.state?.mechanicReader && document.getElementById('manual-title')) history.back()
    else document.getElementById('tab-result')?.focus()
  }
  return <UiLanguage.Provider value={language}>
    <header className="mast"><div className="mast-inner">
      <h1><a className="wordmark" href="#panel-result" onClick={openResult}><span className="brand-mark"><Wrench aria-hidden="true" /></span>{t('Цифровой механик')}</a></h1>
      <div className="mode-switch" role="group" aria-label={t('Режим')}>
        <button type="button" aria-pressed={mode === 'mechanic'} onClick={() => changeMode('mechanic')}>{t('Механик')}</button>
        <button type="button" aria-pressed={mode === 'chief'} onClick={() => changeMode('chief')}>{t('Главный механик')}</button>
      </div>
      <label className="answer-language">{t('Язык')}
        <select id="ui-language" value={language} onChange={event => {
          setLanguage(event.target.value)
          savePreference(languageKey, event.target.value)
        }}>
          {Object.entries(answerLanguages).map(([code, name]) => <option key={code} lang={code} value={code}>{name}</option>)}
        </select>
      </label>
    </div></header>
    <main>
      {error ? <div role="alert" className="error"><p>{t(error)}</p><button onClick={() => setAttempt(n => n + 1)}>{t('Повторить загрузку')}</button></div>
        : !catalog ? <p role="status">{t('Загружаем список транспорта…')}</p>
        : !catalog.length ? <p role="status">{t('В справочнике пока нет транспортных средств.')}</p>
        : mode === 'chief' ? <FleetOverview catalog={catalog} onOpen={openVehicle} />
        : <Workspace key={JSON.stringify([vehicleId, sessionId])}
          {...{ language, view, setView, vehiclesByEngine, engineFilter, vehicle, sessionId, changeEngine, changeVehicle, changeSession }} />}
    </main>
  </UiLanguage.Provider>
}

const pendingLabels = {
  diagnose: 'Получаем результат', chat: 'Ищем ответ', draft_ticket: 'Готовим черновик', confirm: 'Отправляем заявку',
  record_inspection: 'Сохраняем результат осмотра', get_diagnostic_history: 'Загружаем результаты осмотра',
}
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
  if (result.requires_remeasurement || result.recognition_error) return result.localized?.message.text || result.message
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

function sameDiagnostic(previous, current) {
  return previous && Object.keys(previous).length === Object.keys(current).length &&
    Object.keys(current).every(key => previous[key] === current[key])
}

function Workspace({ language, view, setView, vehiclesByEngine, engineFilter, vehicle, sessionId, changeEngine, changeVehicle, changeSession }) {
  const t = useUiText()
  const { id, turns, setTurns, reset, warning } = useChatHistory(vehicle?.vehicle_id, sessionId)
  const [openedDialog, setOpenedDialog] = useState(null)
  const historyLocked = turns.length > 0 && openedDialog !== id
  const sendingTurn = useRef(null)
  const [question, setQuestion] = useState('')
  const [pending, setPending] = useState('')
  const [error, setError] = useState('')
  const [canDraft, setCanDraft] = useState(false)
  const [draft, setDraft] = useState(null)
  const [ticketStatus, setTicketStatus] = useState(null)
  const [diagnosis, setDiagnosis] = useState(null)
  const [ticketHint, setTicketHint] = useState('')
  // «Понимаю, что запись с помехами»: sent with every turn of this check once given. The worker still refuses
  // bad and failed records and checks the draft again when the ticket is sent.
  const acceptWarning = useRef(false)
  const [inspections, setInspections] = useState(null)
  const serviceMode = useServiceMode()
  const metrics = useCycleMetrics()
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
  }, [view, historyLocked, turns.length, turns.at(-1)?.kind])
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
  const lastRequestedLanguage = useRef(language)
  const refreshDiagnosis = useEffectEvent(() => run('diagnose', undefined, true))
  useEffect(() => {
    if (!pending && diagnosis && lastRequestedLanguage.current !== language) refreshDiagnosis()
  }, [language, pending, diagnosis])

  async function run(kind, turn, languageRefresh = false, inspection = null) {
    const text = turn?.question ?? question.trim()
    const retryId = turn?.id
    if (running.current || !sessionId || !id || (kind === 'chat' && historyLocked)) return
    const controller = new AbortController()
    if (kind === 'diagnose') lastRequestedLanguage.current = language
    running.current = controller
    const measure = metrics.start(languageRefresh ? 'refresh' : kind)
    setPending(kind)
    setError('')
    const turnId = kind === 'chat' ? retryId ?? conversationId() : null
    sendingTurn.current = turnId
    function updateTurn(fields) {
      setTurns(old => old.map(turn => turn.id === turnId ? { ...turn, ...fields } : turn))
    }
    if (kind === 'chat') {
      setOpenedDialog(id)
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
        measure.done(receipt.usage)
        setTicketStatus({ ticketId: receipt.ticket_id })
        setDraft(null)
        setCanDraft(false)
        return
      }
      const body = kind === 'chat' ? { message: text } : { command: kind }
      if (acceptWarning.current) body.accept_signal_warning = true
      if (kind === 'record_inspection') body.inspection = inspection
      const result = await request('/agent/run', { ...pair, conversation_id: id, ...(!['draft_ticket', 'get_diagnostic_history', 'record_inspection'].includes(kind) && language !== 'ru' && { language }), ...body }, controller.signal)
      if (controller.signal.aborted) return
      if (result.vehicle_id !== pair.vehicle_id || result.session_id !== sessionId || result.conversation_id !== id || result.engine_model !== vehicle.engine_model)
        throw new Error('Ответ относится к другой проверке. Получите результат заново.')
      measure.done(result.usage, { newCycle: kind === 'diagnose' && !languageRefresh && Boolean(result.diagnostic) })
      if (result.diagnostic && !(kind === 'draft_ticket' && sameDiagnostic(diagnosis?.diagnostic, result.diagnostic))) {
        if (kind !== 'diagnose') lastRequestedLanguage.current = result.language || 'ru'
        setDiagnosis(result)
      }
      if (kind === 'diagnose' && result.diagnostic) {
        const verdict = { id: conversationId(), kind: 'verdict', result }
        setTurns(old => {
          const last = old.findLast(turn => turn.kind === 'verdict')
          const previous = last?.result.diagnostic
          const current = result.diagnostic
          if (sameDiagnostic(previous, current)) {
            if (last.result.localized?.language === result.localized?.language &&
                last.result.signal_warning_accepted === result.signal_warning_accepted &&
                last.result.requires_remeasurement === result.requires_remeasurement) return old
            return old.map(turn => turn === last ? { ...turn, result } : turn)
          }
          return [...old, verdict]
        })
        setOpenedDialog(id)
      }
      if (kind === 'chat') updateTurn({ result })
      if (kind === 'record_inspection' || kind === 'get_diagnostic_history') {
        setInspections(result.inspections || [])
        return true
      }
      if (result.diagnostic || result.action === 'draft_ticket') {
        setTicketHint(result.ticket_draft || result.can_create_ticket ? '' : ticketUnavailableReason(result))
        setCanDraft(Boolean(result.can_create_ticket && !result.requires_remeasurement))
        if (languageRefresh && draft && result.can_create_ticket && sameDiagnostic(diagnosis?.diagnostic, result.diagnostic)) return
        setDraft(result.ticket_draft && !ticketStatus ? {
          value: result.ticket_draft, evidence: result.evidence.items, urgency: result.ticket_draft.payload.urgency,
        } : null)
      }
    } catch (err) {
      if (controller.signal.aborted) return
      measure.fail(err)
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
          setError([err.message, 'Получите результат диагностики заново.'])
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

  const waitingForTranslation = diagnosis && (diagnosis.language || 'ru') !== language
  const latestVerdict = turns.findLast(turn => turn.kind === 'verdict')
  const translationStatus = <p className="meta translation-status" role="status">
    {t(error && !pending ? 'Получите результат диагностики заново.' : 'Получаем результат')}
  </p>
  function confirmWarning(event) {
    event.preventDefault()
    acceptWarning.current = true
    run('diagnose')
  }
  const warningForm = diagnosis?.warning_confirmable && <form className="warning-confirm" onSubmit={confirmWarning}>
    <p><strong>{t('Запись с помехами.')}</strong> {t('Помехи могут исказить вывод. Лучше повторить замер.')}</p>
    <label><input type="checkbox" required disabled={!!pending} /> {t('Понимаю, что запись с помехами и вывод может быть неточным')}</label>
    <button type="submit" className="secondary" disabled={!!pending}>{t('Продолжить с этой записью')}</button>
  </form>
  const returnLabels = { result: 'К результату', chat: 'К чату', history: 'К истории', ticket: 'К заявке' }
  return <ManualNavigation sessionKey={id} returnLabel={t(returnLabels[view])}><div className="workspace">
    <aside className="controls" aria-labelledby="checking">
      <div className="controls-heading"><h2 id="checking"><Gauge aria-hidden="true" />{t('Параметры замера')}</h2>
      </div>
      <div id="selection">
      <div><label htmlFor="engine">{t('Двигатель')}</label>
      <select id="engine" required value={engineFilter} disabled={pending === 'confirm'} onChange={e => changeEngine(e.target.value)}>
        <option value="" disabled>{t('Выберите двигатель')}</option>
        {Array.from(vehiclesByEngine.keys(), engine => <option key={engine} value={engine}>{engine}</option>)}
      </select></div>
      <div><label htmlFor="vehicle">{t('Транспортное средство')}</label>
      <select id="vehicle" required value={vehicle?.vehicle_id || ''} disabled={!engineFilter || pending === 'confirm'} onChange={e => changeVehicle(e.target.value)}>
        <option value="" disabled>{t('Выберите машину')}</option>
        <VehicleOptions vehicles={vehiclesByEngine.get(engineFilter) || []} />
      </select></div>
      <div className="session-picker"><label htmlFor="session">{t('Проверка')}</label>
      <select id="session" value={sessionId} disabled={!vehicle?.sessions.length || pending === 'confirm'} onChange={e => changeSession(e.target.value)}>
        {!vehicle?.sessions.length && <option value="">{vehicle ? t('Нет проверок') : t('Сначала выберите машину')}</option>}
        {vehicle?.sessions.map(s => <option key={s.session_id} value={s.session_id}>{s.session_id} · {formatDate(s.checked_at, language)}</option>)}
      </select></div>
      <button className="primary" disabled={!id || !sessionId || !!pending} onClick={() => run('diagnose')}>{t('Получить результат диагностики')}<ArrowRight aria-hidden="true" /></button>
      </div>
    </aside>
    <section className="chat" aria-label={t('Протокол проверки')}>
      <div className="view-switch" role="tablist" aria-label={t('Разделы проверки')}
        style={{ '--tab-count': views.length, '--active-tab': views.findIndex(tab => tab.id === view) }}>
        {views.map(({ id, label, icon: Icon }, index) =>
          <button key={id} type="button" role="tab" id={'tab-' + id} aria-controls={'panel-' + id}
            aria-selected={view === id} tabIndex={view === id ? 0 : -1}
            onKeyDown={e => navigateTabs(e, index)} onClick={() => setView(id)}><Icon aria-hidden="true" />{t(label)}</button>)}
      </div>
      {!vehicle ? <div className="pane" id={'panel-' + view} role="tabpanel" aria-labelledby={'tab-' + view} tabIndex={0}>
        <EmptyView icon={Gauge} title={engineFilter ? t('Выберите машину') : t('Выберите двигатель')}>
          {engineFilter ? t('В списке транспорта доступны машины с выбранным двигателем.') : t('Затем выберите машину, чтобы открыть её проверки.')}
        </EmptyView>
      </div> : <>
      <div className="pane" hidden={view !== 'result'} id="panel-result" aria-labelledby="tab-result" role="tabpanel" tabIndex={0}>
        {waitingForTranslation ? translationStatus : diagnosis ? <DiagnosticReport result={diagnosis} /> : <EmptyView icon={ScanLine}
          title={sessionId ? t('Результат ещё не запрошен') : t('У этой машины пока нет проверок')}>
          {sessionId ? t('Проверьте параметры замера и нажмите «Получить результат диагностики».') : t('Выберите другую машину в списке транспорта.')}
        </EmptyView>}
        {/* After the verdict: the mechanic first reads why a repeat is needed, then may override it. */}
        {!waitingForTranslation && warningForm}
        {diagnosis?.diagnostic?.status === 'anomaly' && !diagnosis.requires_remeasurement && !diagnosis.recognition_error &&
          <InspectionForm report={diagnosis} disabled={!!pending} onSave={found => run('record_inspection', null, false, found)} />}
        {serviceMode && <CycleCost cycle={metrics.cycle} />}
      </div>
      <div className="chat-panel" hidden={view !== 'chat'} id="panel-chat" aria-labelledby="tab-chat" role="tabpanel" tabIndex={0}>
        <header className="dialog-bar">
          <span><strong>{vehicle.vehicle_id}</strong> · {sessionId || t('Нет замеров')}</span>
          <button type="button" className="reset-chat" disabled={!turns.length || !!pending} onClick={() => {
          if (window.confirm(t('Сбросить переписку по этой машине и замеру? Сообщения исчезнут из этого браузера.'))) {
            reset()
            setQuestion('')
            setError('')
            composer.current?.focus()
          }
        }}><Trash2 aria-hidden="true" />{t('Сбросить диалог')}</button>
        </header>
        <div className="pane dialog-scroll" ref={chatBody}>
        {warning && <p role="status">{t(warning)}</p>}
        {historyLocked && <EmptyView icon={MessageSquare} title={t('Переписка сохранена')}>
          {t('Чтобы открыть её, нажмите «Получить результат диагностики».')}
        </EmptyView>}
        {id && !turns.length && <EmptyView icon={MessageSquare} title={t('Что хотите уточнить?')}>
          {t('Спросите о замере или нормативе из руководства для двигателя {engine}.', { engine: vehicle.engine_model })}
        </EmptyView>}
        <div role="log" aria-label={t('Сообщения')}>
        {!historyLocked && turns.map(({ id, kind, result, question, error: turnError }) => kind === 'verdict'
          ? <div className="turn" key={id || 'verdict'}>{latestVerdict?.result === result && waitingForTranslation ? translationStatus : <DiagnosticVerdict result={result} />}</div>
          : <div className="turn" key={id}>
          <div className="question"><strong>{t('Вы')}</strong><p>{question}</p></div>
          {turnError || (!result && id !== sendingTurn.current) ? <div className="chat-error"><p>{t('Ответ не получен.')} {t(turnError) || t('Отправка прервалась.')}</p>
            <button disabled={!!pending} onClick={() => run('chat', { id, question })}>{t('Повторить')}</button></div>
            : !result ? <p className="meta">{t('Ищем ответ')}<span className="pending-dots">…</span></p>
            : result.ticket_draft
            ? <p>{t('Черновик подготовлен в разделе «Заявка».')}</p>
            : <Response result={result} />}
        </div>)}</div>
        </div>
      </div>
      <div className="pane" hidden={view !== 'history'} id="panel-history" aria-labelledby="tab-history" role="tabpanel" tabIndex={0}>
        <VehicleHistory vehicleId={vehicle.vehicle_id} sessions={vehicle.sessions.slice(0, 5)} />
        <section className="history-report"><h3>{t('Результаты осмотра')}</h3>
          {/* Loaded on request: the History tab itself works from the vehicle list without calling the agent. */}
          {inspections ? <InspectionList records={inspections} vehicleId={vehicle.vehicle_id} />
            : <button type="button" className="secondary" disabled={!id || !sessionId || !!pending}
              onClick={() => run('get_diagnostic_history')}>{t('Показать результаты осмотра')}</button>}
        </section>
      </div>
      <div className="pane" hidden={view !== 'ticket'} id="panel-ticket" aria-labelledby="tab-ticket" role="tabpanel" tabIndex={0}>
        {ticketStatus ? ticketStatus.ticketId
          ? <div className="receipt" role="status"><CheckCheck aria-hidden="true" /><h3>{t('Заявка создана')}</h3><p>{t('Номер в ТОиР')} <strong>{ticketStatus.ticketId}</strong></p></div>
          : <p className="limitations" role="status">{t(ticketStatus.message)}</p>
          : draft ? <Draft draft={draft.value} evidence={draft.evidence} urgency={draft.urgency}
              disabled={!!pending} onUrgencyChange={urgency => setDraft(current => ({ ...current, urgency }))}
              vehicle={vehicle} diagnosis={diagnosis} printable={view === 'ticket'} />
          : diagnosis?.manual_referral ? <section className="draft" aria-label={t('Рекомендация на дефектовку')}>
            <h3>{t('Рекомендация на дефектовку')}</h3>
            <p>{t('Заявка в ТОиР не отправлена. Для ручного оформления используйте этот текст.')}</p>
            <p className="manual-referral">{diagnosis.manual_referral}</p>
          </section>
          : warningForm || <EmptyView icon={ClipboardList} title={t('Заявка на дефектовку')}>
            {canDraft ? t('Создайте черновик. Вы сможете проверить работы и запчасти перед отправкой в ТОиР.')
              : t(ticketHint) || (sessionId ? t('Сначала получите результат диагностики.') : t('У этой машины пока нет проверок. Выберите другое ТС.'))}</EmptyView>}
      </div>
      <div className="panel-footer" hidden={view === 'history'}>
        {view === 'result' && <div className="footer-actions">
          <button className="secondary" onClick={() => { focusQuestion.current = true; setView('chat') }}><MessageSquare aria-hidden="true" />{t('Задать вопрос')}</button>
          <button type="button" aria-controls="panel-ticket" onClick={() => selectTab('ticket')}>{t('К заявке')}<ArrowRight aria-hidden="true" /></button>
        </div>}
        {view === 'chat' && <ChatComposer key={language} input={composer} question={question} setQuestion={setQuestion}
          language={language} disabled={!id || !sessionId || historyLocked} pending={!!pending} send={() => run('chat')} />}
        {view === 'ticket' && <button className="primary" disabled={!!pending || !!ticketStatus || (!draft && !canDraft)}
          onClick={() => run(draft ? 'confirm' : 'draft_ticket')}>{ticketStatus ? ticketStatus.ticketId ? t('Отправлено в ТОиР') : t('Отправка заблокирована')
            : draft ? t('Подтвердить заявку') : t('Создать заявку')}</button>}
      </div>
      </>}
      <div className="panel-status" role="status">{(Array.isArray(error) ? error.map(part => t(part)).join(' ') : t(error)) || (pending && pending !== 'chat' &&
        <>{t(pendingLabels[pending])}<span className="pending-dots">…</span></>)}</div>
    </section>
  </div></ManualNavigation>
}

createRoot(document.getElementById('root')).render(<App />)
