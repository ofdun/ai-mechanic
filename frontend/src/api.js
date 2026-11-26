export function conversationId() {
  // getRandomValues also works when a phone opens the workstation over plain LAN HTTP.
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) => byte.toString(16).padStart(2, '0')).join('')
}

export async function request(path, body, signal) {
  const response = await fetch(path, {
    method: body ? 'POST' : 'GET',
    credentials: 'same-origin',
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
    signal,
  }).catch(error => {
    if (signal?.aborted) throw error
    throw new Error('Нет связи с сервером. Проверьте соединение.')
  })
  const data = await response.json().catch(() => null)
  if (!response.ok || !data) {
    const detail = data?.detail
    const error = new Error(detail?.message || 'Не удалось получить ответ сервера. Проверьте соединение.')
    error.code = detail?.code
    error.retryable = detail?.retryable
    throw error
  }
  return data
}

export const formatDate = (value) => new Intl.DateTimeFormat('ru-RU', {
  day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit',
}).format(new Date(value))
