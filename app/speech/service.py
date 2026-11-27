"""Transcribe a short microphone recording. Audio and text live only in memory."""
import subprocess
from threading import Lock

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from faster_whisper import WhisperModel
from starlette.concurrency import run_in_threadpool

app = FastAPI(docs_url=None, redoc_url=None)
model = WhisperModel('/model', device='cpu', compute_type='int8', cpu_threads=1)
slot = Lock()
max_bytes = 2 * 1024 * 1024


def recognize(body, language):
    # Restrict demuxers and protocols to self-contained microphone recordings.
    args = ['ffmpeg', '-nostdin', '-loglevel', 'error', '-protocol_whitelist', 'pipe',
            '-format_whitelist', 'matroska,webm,mov,ogg,wav', '-i', 'pipe:0', '-map', '0:a:0',
            '-t', '31', '-ac', '1', '-ar', '16000', '-f', 's16le', 'pipe:1']
    try:
        pcm = subprocess.run(args, input=body, capture_output=True, timeout=6, check=True).stdout
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise HTTPException(415)
    if len(pcm) >= 31 * 32000:
        raise HTTPException(413)
    if not pcm:
        raise HTTPException(422)
    samples = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768
    segments, _ = model.transcribe(samples, language=language, beam_size=1, vad_filter=True,
                                   condition_on_previous_text=False)
    text = ' '.join(s.text.strip() for s in segments).strip()
    if not text:
        raise HTTPException(422)
    return {'text': text}


@app.post('/agent/audio')
async def transcribe(request: Request, language: str = 'ru'):
    if not slot.acquire(blocking=False):
        raise HTTPException(429)
    try:
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > max_bytes:
                raise HTTPException(413)
            data.extend(chunk)
        hint = language if language in {'ru', 'kk', 'uz', 'tg', 'hi'} else None
        return await run_in_threadpool(recognize, bytes(data), hint)
    finally:
        slot.release()
