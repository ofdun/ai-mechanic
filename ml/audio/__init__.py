from functools import cached_property
from io import BytesIO
from pathlib import Path

from ml.contracts import AgentError, AudioInput


class WhisperASR:
    def __init__(self, model: str, language: str, cache_dir: Path):
        self.model_name = model
        self.language = language
        self.cache_dir = cache_dir

    @cached_property
    def model(self):
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise AgentError('ASR_NOT_INSTALLED', 'Для голосового ввода установите extra asr.') from error
        return WhisperModel(self.model_name, device='cpu', compute_type='int8', download_root=str(self.cache_dir))

    def transcribe(self, audio: AudioInput) -> str:
        try:
            segments, _ = self.model.transcribe(
                BytesIO(audio.content), language=self.language or None, beam_size=3, vad_filter=True,
            )
            text = ' '.join(segment.text.strip() for segment in segments).strip()
        except AgentError:
            raise
        except Exception as error:
            raise AgentError('ASR_ERROR', 'ASR не смог обработать аудио. Проверьте формат записи и доступность модели.') from error
        if not text:
            raise AgentError('EMPTY_TRANSCRIPT', 'Голосовая команда не распознана.')
        return text
