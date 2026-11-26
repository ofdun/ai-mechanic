from io import BytesIO
from itertools import chain
from pathlib import Path
from threading import Lock

from ml.contracts import AgentError, AudioInput


SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 60


def decode_command(audio: AudioInput):
    try:
        import av
        import numpy as np
    except ImportError as error:
        raise AgentError('ASR_NOT_INSTALLED', 'Для голосового ввода установите extra asr.') from error

    resampler = av.AudioResampler(format='s16', layout='mono', rate=SAMPLE_RATE)
    with BytesIO() as pcm, av.open(BytesIO(audio.content)) as container:
        for source_frame in chain(container.decode(audio=0), (None,)):
            for frame in resampler.resample(source_frame):
                if pcm.tell() + frame.samples * 2 > MAX_AUDIO_SECONDS * SAMPLE_RATE * 2:
                    raise AgentError('AUDIO_TOO_LONG', f'Голосовая команда должна быть не длиннее {MAX_AUDIO_SECONDS} секунд.')
                pcm.write(frame.to_ndarray().tobytes())
        samples = np.frombuffer(pcm.getvalue(), dtype=np.int16)
    if samples.size == 0:
        raise AgentError('EMPTY_TRANSCRIPT', 'В аудиофайле нет звуковых данных.')
    return samples.astype(np.float32) / 32768.0


class WhisperASR:
    def __init__(self, model: str, language: str, cache_dir: Path):
        self.model_name = model
        self.language = language
        self.cache_dir = cache_dir
        self._model = None
        self._model_lock = Lock()

    @property
    def model(self):
        with self._model_lock:
            if self._model is None:
                try:
                    from faster_whisper import WhisperModel
                except ImportError as error:
                    raise AgentError('ASR_NOT_INSTALLED', 'Для голосового ввода установите extra asr.') from error
                self._model = WhisperModel(
                    self.model_name, device='cpu', compute_type='int8', download_root=str(self.cache_dir),
                )
            return self._model

    def transcribe(self, audio: AudioInput) -> str:
        try:
            samples = decode_command(audio)
            segments, _ = self.model.transcribe(
                samples, language=self.language or None, beam_size=3, vad_filter=True,
            )
            text = ' '.join(segment.text.strip() for segment in segments).strip()
        except AgentError:
            raise
        except Exception as error:
            raise AgentError('ASR_ERROR', 'ASR не смог обработать аудио. Проверьте формат записи и доступность модели.') from error
        if not text:
            raise AgentError('EMPTY_TRANSCRIPT', 'Голосовая команда не распознана.')
        return text
