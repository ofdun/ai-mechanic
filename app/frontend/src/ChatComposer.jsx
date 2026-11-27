import React, { useEffect, useRef, useState } from 'react'
import { Mic, Send, Square } from 'lucide-react'
import { recordSpeech } from './dictation'
import { setDictating } from './speech'
import { useUiText } from './uiLanguage'

const unsupported = 'Голосовой ввод недоступен. Введите текст.'
const noSpeech = 'Речь не распознана. Повторите или введите текст.'
const denied = 'Проверьте доступ к микрофону в настройках браузера или введите текст.'

export function ChatComposer({ input, question, setQuestion, language, disabled, pending, send }) {
  const t = useUiText()
  const active = useRef(null)
  const consent = useRef(false)
  const [listening, setListening] = useState(false)
  const [status, setStatus] = useState('')
  const [busy, setBusy] = useState(false)
  const stopRecording = useRef(null)
  useEffect(() => () => {
    active.current?.abort()
    setDictating(false)
  }, [])

  async function dictate() {
    if (active.current) {
      stopRecording.current?.()
      return
    }
    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      setStatus(unsupported)
      return
    }
    if (!consent.current && !window.confirm(t('Отправить запись на сервер «Цифрового механика» для распознавания?'))) return
    consent.current = true
    setDictating(true)
    const controller = new AbortController()
    active.current = controller
    setBusy(true)
    setStatus('')
    try {
      const audio = await recordSpeech(controller.signal, stop => {
        stopRecording.current = stop
        setListening(true)
        setStatus('Говорите. Нажмите ещё раз, чтобы остановить.')
      })
      if (controller.signal.aborted) return
      stopRecording.current = null
      setListening(false)
      setStatus('Распознаём речь…')
      const response = await fetch('/agent/audio?language=' + encodeURIComponent(language), {
        method: 'POST', body: audio, signal: controller.signal,
      })
      if (!response.ok) {
        setStatus(response.status === 422 ? noSpeech : response.status === 429 ? 'Сервис занят. Повторите через несколько секунд.' : unsupported)
        return
      }
      const { text } = await response.json()
      if (!text?.trim()) { setStatus(noSpeech); return }
      setQuestion(previous => [previous.trimEnd(), text.trim()].filter(Boolean).join(' ').slice(0, 8000))
      setStatus('Проверьте текст и нажмите «Отправить».')
    } catch (error) {
      if (!controller.signal.aborted) setStatus(['NotAllowedError', 'NotFoundError', 'NotReadableError'].includes(error.name) ? denied : unsupported)
    } finally {
      setDictating(false)
      if (!controller.signal.aborted) {
        active.current = null
        stopRecording.current = null
        setBusy(false)
        setListening(false)
      }
    }
  }

  const micLabel = t(listening ? 'Остановить диктовку' : 'Надиктовать сообщение')
  return <form onSubmit={event => {
    event.preventDefault()
    if (!disabled && !pending && !active.current && question.trim()) {
      setStatus('')
      input.current?.focus()
      send()
    }
  }}>
    <div className="compose-row">
      <input ref={input} aria-label={t('Вопрос по двигателю')} placeholder={t('Вопрос по двигателю…')} value={question}
        maxLength={8000} disabled={disabled} onChange={event => setQuestion(event.target.value)} autoComplete="off" />
      <button type="button" aria-label={micLabel} title={micLabel} aria-pressed={listening} disabled={disabled || pending || (busy && !listening)} onClick={dictate}>
        {listening ? <Square aria-hidden="true" /> : <Mic aria-hidden="true" />}
      </button>
      <button type="submit" className="submit" aria-label={t('Отправить')} disabled={disabled || !question.trim() || pending || busy}>
        <Send aria-hidden="true" /><span>{t('Отправить')}</span>
      </button>
    </div>
    <div className="dictation-status" role="status">{t(status)}</div>
  </form>
}
