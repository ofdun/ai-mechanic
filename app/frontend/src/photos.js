// Inspection photos live only in this browser's IndexedDB. They are not sent to the agent.
export const maxInspectionPhotos = 12

// Re-encode on a canvas: shrinks the upload and drops EXIF, including the GPS position.
export async function shrinkPhoto(file, maxSide = 1600) {
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' }).catch(() => {
    throw new Error('Не удалось открыть фото. Сделайте снимок ещё раз.')
  })
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height))
  const canvas = document.createElement('canvas')
  canvas.width = Math.round(bitmap.width * scale)
  canvas.height = Math.round(bitmap.height * scale)
  canvas.getContext('2d').drawImage(bitmap, 0, 0, canvas.width, canvas.height)
  bitmap.close()
  const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85))
  if (!blob) throw new Error('Не удалось открыть фото. Сделайте снимок ещё раз.')
  return blob
}

let database
function openDatabase() {
  if (!database) database = new Promise((resolve, reject) => {
    const request = indexedDB.open('mechanic-photos', 1)
    request.onupgradeneeded = () => {
      request.result.createObjectStore('photos', { keyPath: 'id' }).createIndex('inspection', 'inspection')
    }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
  return database
}

async function transact(mode, work) {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction('photos', mode)
    let result
    work(tx.objectStore('photos'), value => { result = value })
    tx.oncomplete = () => resolve(result)
    tx.onabort = tx.onerror = () => reject(tx.error)
  })
}

export const inspectionKey = (vehicleId, sessionId) => JSON.stringify([vehicleId, sessionId])

export function listPhotos(inspection) {
  return transact('readonly', (store, done) => {
    const request = store.index('inspection').getAll(inspection)
    request.onsuccess = () => done(request.result.sort((a, b) => a.taken_at.localeCompare(b.taken_at)))
  })
}

export function savePhoto(photo) {
  return transact('readwrite', store => store.put(photo))
}

export function deletePhoto(id) {
  return transact('readwrite', store => store.delete(id))
}
