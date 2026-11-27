export async function recordSpeech(signal, onStart) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const release = () => stream.getTracks().forEach(track => track.stop())
  let timeout
  let stop
  try {
    signal.throwIfAborted()
    const recorder = new MediaRecorder(stream)
    const chunks = []
    stop = () => {
      if (recorder.state === 'recording') recorder.stop()
    }
    return await new Promise((resolve, reject) => {
      recorder.ondataavailable = event => chunks.push(event.data)
      recorder.onerror = () => reject(new Error('recording failed'))
      recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType }))
      signal.addEventListener('abort', stop, { once: true })
      recorder.start()
      timeout = setTimeout(stop, 30000)
      onStart(stop)
    })
  } finally {
    clearTimeout(timeout)
    signal.removeEventListener('abort', stop)
    release()
  }
}
