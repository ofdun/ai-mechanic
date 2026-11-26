from pathlib import Path
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    data_dir: Path = Path('data')
    vehicles_path: Path | None = None
    artifacts_dir: Path = Path('artifacts')
    ml_api_url: AnyHttpUrl = 'http://127.0.0.1:8001'
    maintenance_api_url: AnyHttpUrl = 'http://127.0.0.1:8002'
    http_timeout_seconds: Annotated[float, Field(gt=0, le=90)] = 20
    min_diagnostic_confidence: Annotated[float, Field(ge=0, le=1)] = 0.70
    repeat_on_warning: bool = True
    planner_mode: Literal['llm', 'offline'] = 'llm'
    agent_architecture: Literal['baseline', 'a', 'b'] = 'a'
    allow_external_evidence: bool = False
    llm_base_url: AnyHttpUrl = 'https://openrouter.ai/api/v1'
    llm_model: str = 'z-ai/glm-5.3-flash'
    llm_api_key: SecretStr = SecretStr('')
    llm_timeout_seconds: Annotated[float, Field(gt=0, le=120)] = 45
    a_generator_model: str = 'deepseek/deepseek-v4.1-flash'
    a_verifier_model: str = 'z-ai/glm-5.3-flash'
    b_query_model: str = 'z-ai/glm-5.3-flash'
    b_reasoner_model: str = 'z-ai/glm-5.3'
    b_critic_model: str = 'deepseek/deepseek-v4.1-flash'
    b_repair_model: str = 'deepseek/deepseek-v4.1-flash'
    b_finalizer_model: str = 'z-ai/glm-5.3-flash'
    asr_model: str = 'base'
    asr_language: str = 'ru'

    @field_validator('ml_api_url', 'maintenance_api_url', 'llm_base_url')
    @classmethod
    def base_url_only(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.query is not None or value.fragment is not None:
            raise ValueError('Base URL must not contain a query or fragment')
        return value
