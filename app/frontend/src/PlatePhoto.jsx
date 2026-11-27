import React, { useEffect, useRef, useState } from 'react'
import { Camera, X } from 'lucide-react'
import { sendImage } from './api'
import { shrinkPhoto } from './photos'
import { useUiText } from './uiLanguage'

function vehicleName({ make, model }) {
  return model?.startsWith(make || '') ? model : [make, model].filter(Boolean).join(' ')
}

// Plate recognition only suggests a vehicle. The selection changes on the mechanic's click,
// never from the recognition result itself.
export function PlatePhoto({ vehicles, disabled, onPick }) {
  const t = useUiText()
  const input = useRef(null)
  const panel = useRef(null)
  const running = useRef(null)
  const [state, setState] = useState(null)
  useEffect(() => () => running.current?.abort(), [])
  useEffect(() => { if (state && state.kind !== 'reading') panel.current?.focus() }, [state])
  const byId = new Map(vehicles.map(vehicle => [vehicle.vehicle_id, vehicle]))

  async function recognize(file) {
    running.current?.abort()
    const controller = new AbortController()
    running.current = controller
    setState({ kind: 'reading' })
    try {
      const result = await sendImage('/api/plate', await shrinkPhoto(file), controller.signal)
      if (controller.signal.aborted) return
      const candidates = result.candidates.filter(item => byId.has(item.vehicle_id))
      setState({ kind: 'result', reliable: result.reliable && candidates.length > 0, candidates, readings: result.readings })
    } catch (error) {
      if (!controller.signal.aborted) setState({ kind: 'error', message: error.message })
    } finally {
      if (running.current === controller) running.current = null
    }
  }

  function pick(vehicleId) {
    setState(null)
    onPick(vehicleId)
  }

  function label(vehicleId) {
    const vehicle = byId.get(vehicleId)
    return [vehicleId, vehicleName(vehicle), vehicle.engine_model].filter(Boolean).join(' · ')
  }

  const best = state?.kind === 'result' && state.reliable ? state.candidates[0] : null
  const others = state?.kind === 'result' ? state.candidates.filter(item => item !== best) : []
  return <>
    <button type="button" className="plate-button" disabled={disabled || state?.kind === 'reading'}
      onClick={() => input.current?.click()}><Camera aria-hidden="true" />{t('Номер по фото')}</button>
    <input ref={input} type="file" accept="image/*" capture="environment" hidden onChange={event => {
      const file = event.target.files?.[0]
      event.target.value = ''
      if (file) recognize(file)
    }} />
    {state && <section className="plate-result" ref={panel} tabIndex={-1} aria-label={t('Номер по фото')} aria-live="polite">
      <button type="button" className="plate-close" aria-label={t('Закрыть')} onClick={() => {
        running.current?.abort()
        setState(null)
      }}><X aria-hidden="true" /></button>
      {state.kind === 'reading' ? <p role="status">{t('Распознаём номер')}<span className="pending-dots">…</span></p>
        : state.kind === 'error' ? <p className="limitations">{t(state.message)}</p>
        : best ? <>
          <p>{t('Распознан номер {plate}. Проверьте, что это нужная машина.', { plate: best.reading })}</p>
          <button type="button" className="primary" disabled={disabled} onClick={() => pick(best.vehicle_id)}>
            {t('Выбрать {vehicle}', { vehicle: label(best.vehicle_id) })}</button>
        </>
        : <p className="limitations">{state.candidates.length
          ? t('Номер распознан неуверенно. Выберите машину сами.')
          : t('Номер не распознан. Выберите машину из списка.')}
          {state.readings.length > 0 && <> {t('Прочитано: {plates}.', { plates: state.readings.slice(0, 3).join(', ') })}</>}</p>}
      {others.length > 0 && <div className="plate-candidates">
        {best && <p className="meta">{t('Похожие номера')}</p>}
        {others.map(item => <button type="button" key={item.vehicle_id} disabled={disabled}
          onClick={() => pick(item.vehicle_id)}>{label(item.vehicle_id)}</button>)}
      </div>}
      {state.kind !== 'reading' && <div className="plate-manual">
        <label htmlFor="plate-manual">{t('Выбрать из списка')}</label>
        <select id="plate-manual" value="" disabled={disabled} onChange={event => event.target.value && pick(event.target.value)}>
          <option value="">{t('Выберите машину')}</option>
          {[...vehicles].sort((a, b) => a.vehicle_id.localeCompare(b.vehicle_id)).map(vehicle =>
            <option key={vehicle.vehicle_id} value={vehicle.vehicle_id}>{label(vehicle.vehicle_id)}</option>)}
        </select>
      </div>}
      <p className="meta">{t('Фото номера не сохраняется и не передаётся внешним сервисам.')}</p>
    </section>}
  </>
}
