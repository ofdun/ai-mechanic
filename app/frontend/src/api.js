export function conversationId() {
  // getRandomValues also works when a phone opens the workstation over plain LAN HTTP.
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

const serviceMessages = {
  LLM_NOT_CONFIGURED: 'Помощник сейчас недоступен. Сообщите администратору.',
  LLM_UPSTREAM_ERROR: 'Помощник сейчас не отвечает. Попробуйте ещё раз.',
  LLM_OUTPUT_INCOMPLETE: 'Не удалось получить полный ответ. Попробуйте уточнить вопрос.',
  LLM_SCHEMA_ERROR: 'Помощнику не удалось подготовить ответ. Попробуйте уточнить вопрос.',
  LLM_EVIDENCE_NOT_ALLOWED: 'Для ответа по этому руководству требуется доступ. Обратитесь к администратору.',
  RAG_INDEX_UNAVAILABLE: 'Поиск по руководствам сейчас недоступен. Попробуйте позже.',
  RAG_SCHEMA_ERROR: 'Не удалось подтвердить ответ по руководству выбранного двигателя.',
  PLATE_OCR_UNAVAILABLE: 'Распознавание номера на этом сервере не настроено. Выберите машину из списка.',
  PLATE_OCR_FAILED: 'Номер не распознан. Выберите машину из списка.',
  PLATE_IMAGE_INVALID: 'Не удалось открыть фото. Сделайте снимок ещё раз.',
}

async function send(path, init, signal) {
  const response = await fetch(path, { credentials: 'same-origin', ...init, signal }).catch(error => {
    if (signal?.aborted) throw error
    throw new Error('Нет связи с сервером. Проверьте соединение.')
  })
  const data = await response.json().catch(() => null)
  if (!response.ok || !data) {
    const detail = data?.detail
    const message = typeof detail === 'string' ? detail : serviceMessages[detail?.code] || detail?.message
    const error = new Error(message || 'Не удалось получить ответ сервера. Проверьте соединение.')
    error.code = detail?.code
    error.retryable = detail?.retryable
    throw error
  }
  return data
}

export function request(path, body, signal) {
  return send(path, body ? {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  } : { method: 'GET' }, signal)
}

export function sendImage(path, blob, signal) {
  return send(path, { method: 'POST', headers: { 'Content-Type': blob.type || 'image/jpeg' }, body: blob }, signal)
}

export const formatDate = (value, language = 'ru') => new Intl.DateTimeFormat(language, {
  day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit',
}).format(new Date(value))
