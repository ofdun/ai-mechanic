import React, { useEffect, useId, useMemo, useSyncExternalStore } from 'react'
import { Square, Volume2 } from 'lucide-react'
import { speechParts, speechState, startReading, stopReading, subscribe } from './speech'
import { useUiText } from './uiLanguage'

export function ReadAloud({ lines }) {
  const t = useUiText()
  const id = useId()
  const { voices, playing, dictating } = useSyncExternalStore(subscribe, speechState, speechState)
  const key = JSON.stringify(lines)
  const parts = useMemo(() => speechParts(lines, voices), [key, voices])
  useEffect(() => () => {
    if (speechState().playing === id) stopReading()
  }, [id, key])
  if (!parts) return null
  const active = playing === id
  const label = t(active ? 'Стоп' : 'Прочитать ответ')
  return <button type="button" className="read-aloud" aria-pressed={active} disabled={dictating && !active}
    onClick={() => active ? stopReading() : startReading(id, parts)}>
    {active ? <Square aria-hidden="true" /> : <Volume2 aria-hidden="true" />}{label}
  </button>
}
