import { useEffect, useRef, useState } from 'react'
import { conversationId } from './api'

let database
function openDatabase() {
  if (!database) database = new Promise((resolve, reject) => {
    const request = indexedDB.open('mechanic-chat', 1)
    request.onupgradeneeded = () => request.result.createObjectStore('dialogs')
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
  return database
}

async function changeDialog(key, change) {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction('dialogs', 'readwrite')
    const store = tx.objectStore('dialogs')
    const read = store.get(key)
    let dialog
    read.onsuccess = () => {
      const current = read.result || { id: conversationId(), turns: [] }
      dialog = change ? change(current) : current
      if (!read.result || dialog !== current) store.put(dialog, key)
    }
    tx.oncomplete = () => resolve(dialog)
    tx.onabort = () => reject(tx.error)
  })
}

export function useChatHistory(vehicle, session) {
  const key = JSON.stringify([vehicle, session])
  const [dialog, setDialog] = useState(null)
  const [warning, setWarning] = useState('')
  const channel = useRef(null)
  const mounted = useRef(false)
  const memoryOnly = useRef(false)

  useEffect(() => {
    if (!session) return
    mounted.current = true
    const sync = () => changeDialog(key).then(saved => {
      if (mounted.current) setDialog(saved)
    }).catch(() => {
      if (!mounted.current) return
      memoryOnly.current = true
      setWarning('Хранилище браузера недоступно. Переписка сохранится только до закрытия страницы.')
      setDialog(current => current || { id: conversationId(), turns: [] })
    })
    sync()
    if (typeof BroadcastChannel !== 'undefined') {
      channel.current = new BroadcastChannel('mechanic-chat')
      channel.current.onmessage = event => { if (event.data === key) sync() }
    }
    return () => { mounted.current = false; channel.current?.close() }
  }, [key, session])

  function update(change) {
    const apply = current => current.id === dialog.id ? change(current) : current
    setDialog(apply)
    if (memoryOnly.current) return
    changeDialog(key, apply).then(() => {
      if (mounted.current) channel.current?.postMessage(key)
    }).catch(() => {
      if (mounted.current) setWarning('Не удалось сохранить изменение в хранилище браузера.')
    })
  }

  return {
    id: dialog?.id, turns: dialog?.turns || [], warning,
    setTurns: transform => update(current => ({ ...current, turns: transform(current.turns) })),
    reset: () => {
      const next = { id: conversationId(), turns: [] }
      update(() => next)
    },
  }
}
