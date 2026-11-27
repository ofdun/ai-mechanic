import React, { useEffect, useRef, useState } from 'react'
import { Mic, Send, Square } from 'lucide-react'
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
  useEffect(() => {
    return () => {
      const recognition = active.current
      active.current = null
      recognition?.abort()
    }
  }, [])

  function dictate() {
    if (active.current) {
      active.current.stop()
      return
    }
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition
    if (!Recognition || !window.isSecureContext) {
      setStatus(unsupported)
      return
    }
    if (!consent.current && !window.confirm(t('Браузер может отправить голос своему сервису распознавания. Разрешить диктовку?'))) return
    consent.current = true
    let recognition
    try {
      recognition = new Recognition()
      recognition.lang = language
      active.current = recognition
      let result = noSpeech
      recognition.onresult = event => {
        if (active.current !== recognition) return
        const words = Array.from(event.results).filter(item => item.isFinal).map(item => item[0].transcript.trim()).join(' ').trim()
        if (!words) return
        setQuestion(previous => [previous.trimEnd(), words].filter(Boolean).join(' ').slice(0, 8000))
        result = 'Проверьте текст и нажмите «Отправить».'
      }
      recognition.onerror = event => {
        result = ['not-allowed', 'audio-capture', 'service-not-allowed'].includes(event.error) ? denied
          : event.error === 'no-speech' ? noSpeech : unsupported
      }
      recognition.onend = () => {
        if (active.current !== recognition) return
        active.current = null
        setListening(false)
        setStatus(result)
      }
      recognition.start()
      setListening(true)
      setStatus('Говорите. Нажмите ещё раз, чтобы остановить.')
    } catch {
      active.current = null
      setListening(false)
      setStatus(unsupported)
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
      <button type="button" aria-label={micLabel} title={micLabel} aria-pressed={listening} disabled={disabled || pending} onClick={dictate}>
        {listening ? <Square aria-hidden="true" /> : <Mic aria-hidden="true" />}
      </button>
      <button type="submit" className="submit" aria-label={t('Отправить')} disabled={disabled || !question.trim() || pending || listening}>
        <Send aria-hidden="true" /><span>{t('Отправить')}</span>
      </button>
    </div>
    <div className="dictation-status" role="status">{t(status)}</div>
  </form>
}
