import React, { createContext, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ArrowLeft, BookOpen, ChevronLeft, ChevronRight } from 'lucide-react'
import { request } from './api'

const OpenManual = createContext(null)

export function Source({ source, number }) {
  const open = useContext(OpenManual)
  const title = source.manual + ', стр.\u00a0' + source.pdf_page
  return <button type="button" className={number ? 'citation' : 'source'}
    aria-label={number ? `Источник ${number}, ${title}` : title}
    onClick={event => open(source, event.currentTarget)}>
    {number ? `[${number}]` : <><BookOpen aria-hidden="true" />{title}</>}
  </button>
}

export function ManualNavigation({ children, sessionKey, returnLabel }) {
  const [source, setSource] = useState(null)
  const origin = useRef(null)
  const back = useRef(null)
  function open(value, trigger) {
    const pane = trigger.closest('.pane')
    origin.current = { source: value, trigger, pane, paneTop: pane.scrollTop, top: window.scrollY }
    history.pushState({ ...history.state, mechanicReader: sessionKey }, '')
    setSource(value)
  }
  useEffect(() => {
    const navigate = () => setSource(history.state?.mechanicReader === sessionKey ? origin.current?.source : null)
    window.addEventListener('popstate', navigate)
    return () => window.removeEventListener('popstate', navigate)
  }, [sessionKey])
  useLayoutEffect(() => {
    if (source) {
      window.scrollTo(0, 0)
      back.current.focus({ preventScroll: true })
    } else if (origin.current) {
      const { trigger, pane, paneTop, top } = origin.current
      pane.scrollTop = paneTop
      trigger.focus({ preventScroll: true })
      window.scrollTo(0, top)
    }
  }, [source])

  return <OpenManual.Provider value={open}>
    <div className="inspection" hidden={!!source}>{children}</div>
    {source && <section className="manual" aria-labelledby="manual-title">
      <header className="manual-heading">
        <button type="button" ref={back} onClick={() => history.back()}><ArrowLeft aria-hidden="true" />{returnLabel}</button>
        <h2 id="manual-title">{source.manual}</h2>
      </header>
      <ManualPage source={source} />
    </section>}
  </OpenManual.Provider>
}

function ManualPage({ source }) {
  const [number, setNumber] = useState(source.pdf_page)
  const [mode, setMode] = useState('original')
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [imageFailed, setImageFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const url = '/api/documents/' + encodeURIComponent(source.document_id) + '/pages/' + number
  const body = useRef(null)
  useEffect(() => {
    const controller = new AbortController()
    setError('')
    setImageFailed(false)
    request(url, null, controller.signal).then(value => {
      if (!controller.signal.aborted) setData(value)
    }).catch(err => {
      if (!controller.signal.aborted) setError(err.message)
    })
    return () => controller.abort()
  }, [url, attempt])
  useLayoutEffect(() => {
    body.current.scrollTop = 0
  }, [number, mode])
  const current = data?.page === number ? data : null
  const textMode = mode === 'text' && !!current?.text.trim()
  return <>
    <div className="manual-tools">
      <div className="page-turn">
        <button aria-label="Предыдущая страница" disabled={number <= 1} onClick={() => setNumber(number - 1)}><ChevronLeft aria-hidden="true" /></button>
        <span aria-live="polite">Страница {number}{data && ' из ' + data.total}</span>
        <button aria-label="Следующая страница" disabled={!data || number >= data.total} onClick={() => setNumber(number + 1)}><ChevronRight aria-hidden="true" /></button>
      </div>
      <div className="manual-mode" role="group" aria-label="Вид страницы">
        <button aria-pressed={!textMode} onClick={() => setMode('original')}>Оригинал</button>
        <button aria-pressed={textMode} disabled={!current?.text.trim()} onClick={() => setMode('text')}>Текст</button>
      </div>
    </div>
    <div ref={body} className="manual-body" tabIndex={0} aria-label="Содержимое страницы">
      {error || (imageFailed && !textMode) ? <div className="error" role="alert">
        <p>{error || 'Изображение страницы не загрузилось.'}</p>
        <button onClick={() => { setImageFailed(false); setError(''); setAttempt(n => n + 1) }}>Повторить</button>
      </div> : !current ? <p role="status">Открываем страницу…</p>
        : textMode ? <div className="manual-text">{current.text}</div>
        : <img key={url + ':' + attempt} src={url + '?view=image'} alt={source.manual + ', страница ' + number}
          onError={() => setImageFailed(true)} />}
    </div>
  </>
}
