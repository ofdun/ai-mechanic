import { useUiText } from './uiLanguage'
import React, { createContext, useContext, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { ArrowLeft, BookOpen, ChevronLeft, ChevronRight } from 'lucide-react'
import { request } from './api'
import { exactQuoteRange } from './quoteHighlight'

const OpenManual = createContext(null)

export function Source({ source, number, pageLabel, quote = '' }) {
  const t = useUiText()
  const open = useContext(OpenManual)
  const title = source.manual + ', ' + (pageLabel || t('стр.')) + '\u00a0' + source.pdf_page
  return <button type="button" className={number ? 'citation' : 'source'}
    aria-label={number ? t('Источник {number}, {title}', { number, title }) : title}
    onClick={event => open({ ...source, quote }, event.currentTarget)}>
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
      <ManualPage key={JSON.stringify([source.document_id, source.pdf_page, source.quote])} source={source} />
    </section>}
  </OpenManual.Provider>
}

function ManualPage({ source }) {
  const t = useUiText()
  const [number, setNumber] = useState(source.pdf_page)
  const [mode, setMode] = useState('original')
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [imageFailed, setImageFailed] = useState(false)
  const [attempt, setAttempt] = useState(0)
  const url = '/api/documents/' + encodeURIComponent(source.document_id) + '/pages/' + number
  const body = useRef(null)
  const highlighted = useRef(null)
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
  const current = data?.page === number ? data : null
  const textMode = mode === 'text' && !!current?.text.trim()
  const match = number === source.pdf_page ? exactQuoteRange(current?.text, source.quote) : null
  useLayoutEffect(() => {
    const pane = body.current
    pane.scrollTop = 0
    if (textMode && match && highlighted.current) {
      const offset = highlighted.current.getBoundingClientRect().top - pane.getBoundingClientRect().top
      pane.scrollTop = Math.max(0, offset - Math.min(80, pane.clientHeight / 3))
    }
  }, [number, textMode, current?.text, match?.start, match?.end])
  return <>
    <div className="manual-tools">
      <div className="page-turn">
        <button aria-label={t('Предыдущая страница')} disabled={number <= 1} onClick={() => setNumber(number - 1)}><ChevronLeft aria-hidden="true" /></button>
        <span aria-live="polite">{data ? t('Страница {page} из {total}', { page: number, total: data.total }) : t('Страница') + ' ' + number}</span>
        <button aria-label={t('Следующая страница')} disabled={!data || number >= data.total} onClick={() => setNumber(number + 1)}><ChevronRight aria-hidden="true" /></button>
      </div>
      <div className="manual-mode" role="group" aria-label={t('Вид страницы')}>
        <button aria-pressed={!textMode} onClick={() => setMode('original')}>{t('Оригинал')}</button>
        <button aria-pressed={textMode} disabled={!current?.text.trim()} onClick={() => setMode('text')}>{t('Текст')}</button>
      </div>
    </div>
    <div ref={body} className="manual-body" tabIndex={0} aria-label={t('Содержимое страницы')}>
      {error || (imageFailed && !textMode) ? <div className="error" role="alert">
        <p>{t(error || 'Изображение страницы не загрузилось.')}</p>
        <button onClick={() => { setImageFailed(false); setError(''); setAttempt(n => n + 1) }}>{t('Повторить')}</button>
      </div> : !current ? <p role="status">{t('Открываем страницу…')}</p>
        : textMode ? <div className="manual-text">{match ? <>
          {current.text.slice(0, match.start)}<mark ref={highlighted} className="manual-quote">{current.text.slice(match.start, match.end)}</mark>{current.text.slice(match.end)}
        </> : current.text}</div>
        : <img key={url + ':' + attempt} src={url + '?view=image'} alt={source.manual + ', ' + t('страница') + ' ' + number}
          onError={() => setImageFailed(true)} />}
    </div>
  </>
}
