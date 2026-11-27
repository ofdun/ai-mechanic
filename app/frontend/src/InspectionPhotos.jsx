import React, { useEffect, useRef, useState } from 'react'
import { Camera, Trash2 } from 'lucide-react'
import { conversationId, formatDate } from './api'
import { deletePhoto, inspectionKey, listPhotos, maxInspectionPhotos, savePhoto, shrinkPhoto } from './photos'
import { useUiText } from './uiLanguage'

const kinds = { part: 'Узел', stand: 'Положение стенда' }
const addLabels = { part: 'Фото узла', stand: 'Фото положения стенда' }

function Thumbnail({ blob, alt }) {
  const [url, setUrl] = useState('')
  useEffect(() => {
    const next = URL.createObjectURL(blob)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return url ? <a href={url} target="_blank" rel="noreferrer"><img src={url} alt={alt} /></a> : null
}

// Photos document the inspection for the mechanic. They stay on this device and are
// never sent to the agent, so no diagnosis is made from them.
export function InspectionPhotos({ vehicleId, sessionId, language }) {
  const t = useUiText()
  const inspection = inspectionKey(vehicleId, sessionId)
  const [photos, setPhotos] = useState(null)
  const [error, setError] = useState('')
  const [adding, setAdding] = useState(false)
  const input = useRef(null)
  const kind = useRef('part')
  useEffect(() => {
    let current = true
    listPhotos(inspection).then(list => current && setPhotos(list)).catch(() => {
      if (current) { setPhotos([]); setError('Фото недоступны: браузер не разрешает локальное хранение.') }
    })
    return () => { current = false }
  }, [inspection])

  async function add(file) {
    setAdding(true)
    setError('')
    try {
      const photo = {
        id: conversationId(), inspection, vehicle_id: vehicleId, session_id: sessionId,
        kind: kind.current, note: '', taken_at: new Date().toISOString(), blob: await shrinkPhoto(file),
      }
      await savePhoto(photo)
      setPhotos(old => [...old, photo])
    } catch (err) {
      setError(err.message || 'Не удалось сохранить фото.')
    } finally {
      setAdding(false)
    }
  }

  async function update(photo, fields) {
    const next = { ...photo, ...fields }
    setPhotos(old => old.map(item => item.id === photo.id ? next : item))
    await savePhoto(next).catch(() => setError('Не удалось сохранить фото.'))
  }

  async function remove(photo) {
    if (!window.confirm(t('Удалить фото из осмотра?'))) return
    await deletePhoto(photo.id).then(() => setPhotos(old => old.filter(item => item.id !== photo.id)))
      .catch(() => setError('Не удалось удалить фото.'))
  }

  const full = photos && photos.length >= maxInspectionPhotos
  return <section className="inspection-photos" aria-labelledby="inspection-photos-title">
    <h3 id="inspection-photos-title">{t('Фото осмотра')}</h3>
    <p className="meta">{t('Фото хранятся только в этом браузере и не передаются помощнику. Диагноз по фото не ставится.')}</p>
    <div className="footer-actions">
      {Object.entries(addLabels).map(([id, label]) => <button type="button" key={id} disabled={!photos || adding || full}
        onClick={() => { kind.current = id; input.current?.click() }}><Camera aria-hidden="true" />{t(label)}</button>)}
      <input ref={input} type="file" accept="image/*" capture="environment" hidden onChange={event => {
        const file = event.target.files?.[0]
        event.target.value = ''
        if (file) add(file)
      }} />
    </div>
    {full && <p className="meta">{t('Достигнут предел: {count} фото на осмотр.', { count: maxInspectionPhotos })}</p>}
    {error && <p className="limitations" role="alert">{t(error)}</p>}
    {adding && <p role="status">{t('Сохраняем фото')}<span className="pending-dots">…</span></p>}
    {photos?.length > 0 && <ul className="photo-grid">
      {photos.map(photo => <li key={photo.id}>
        <Thumbnail blob={photo.blob} alt={t(kinds[photo.kind])} />
        <div>
          <strong>{t(kinds[photo.kind])}</strong> <span className="meta">{formatDate(photo.taken_at, language)}</span>
          <input type="text" aria-label={t('Подпись к фото')} placeholder={t('Подпись, например: ремень генератора')}
            maxLength={200} defaultValue={photo.note} onBlur={event => {
              if (event.target.value !== photo.note) update(photo, { note: event.target.value })
            }} />
          <button type="button" className="photo-delete" onClick={() => remove(photo)}><Trash2 aria-hidden="true" />{t('Удалить')}</button>
        </div>
      </li>)}
    </ul>}
  </section>
}
