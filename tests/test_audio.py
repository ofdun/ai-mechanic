import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from threading import Barrier
from time import sleep
from types import SimpleNamespace

import pytest

from ml.audio import MAX_AUDIO_SECONDS, SAMPLE_RATE, WhisperASR
from ml.contracts import AgentError, AudioInput


def wav(seconds):
    buffer = BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setparams((1, 2, SAMPLE_RATE, 0, 'NONE', ''))
        audio.writeframes(b'\x00\x00' * SAMPLE_RATE * seconds)
    return AudioInput(content=buffer.getvalue())


def test_audio_duration_is_checked_before_inference(monkeypatch, tmp_path):
    pytest.importorskip('av')
    pytest.importorskip('numpy')
    sample_counts = []

    def transcribe(samples, **kwargs):
        sample_counts.append(len(samples))
        return [SimpleNamespace(text='Проверь двигатель.')], None

    asr = WhisperASR('base', 'ru', tmp_path)
    monkeypatch.setattr(asr, '_model', SimpleNamespace(transcribe=transcribe))
    assert asr.transcribe(wav(1)) == 'Проверь двигатель.'
    with pytest.raises(AgentError) as error:
        asr.transcribe(wav(MAX_AUDIO_SECONDS + 1))
    assert error.value.code == 'AUDIO_TOO_LONG'
    assert sample_counts == [SAMPLE_RATE]


def test_concurrent_cold_asr_loads_a_single_model(monkeypatch, tmp_path):
    calls = []

    def load(*args, **kwargs):
        model = object()
        calls.append(model)
        sleep(0.03)  # Keep model loading active while the other request arrives.
        return model

    monkeypatch.setitem(sys.modules, 'faster_whisper', SimpleNamespace(WhisperModel=load))
    asr = WhisperASR('base', 'ru', tmp_path)
    start = Barrier(2)

    def get_model():
        start.wait(timeout=3)
        return asr.model

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = (pool.submit(get_model) for _ in range(2))
        assert first.result(timeout=3) is second.result(timeout=3)
    assert len(calls) == 1
