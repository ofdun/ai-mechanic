from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    data_dir: Path = Path('data')
    vehicles_path: Path | None = None
    artifacts_dir: Path = Path('artifacts')
    ml_api_url: str = 'http://127.0.0.1:8001'
    maintenance_api_url: str = 'http://127.0.0.1:8002'
    http_timeout_seconds: Annotated[float, Field(gt=0, le=90)] = 20
    min_diagnostic_confidence: Annotated[float, Field(ge=0, le=1)] = 0.70
    repeat_on_warning: bool = True
    planner_mode: Literal['llm', 'offline'] = 'llm'
    llm_base_url: str = 'https://openrouter.ai/api/v1'
    llm_model: str = 'z-ai/glm-5.3-flash'
    llm_api_key: SecretStr = SecretStr('')
    asr_model: str = 'base'
    asr_language: str = 'ru'
