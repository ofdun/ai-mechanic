import logging
from time import monotonic
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ml.config import Settings
from ml.contracts import AgentError, CallUsage
from ml.usage import timed


logger = logging.getLogger(__name__)
Result = TypeVar('Result', bound=BaseModel)


def whole(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def amount(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


class OpenRouterClient:
    def __init__(self, http: httpx.Client, settings: Settings):
        self.http = http
        self.model = settings.llm_model
        self.url = f'{str(settings.llm_base_url).rstrip("/")}/chat/completions'
        self.api_key = settings.llm_api_key
        self.timeout = settings.llm_timeout_seconds

    def select_tool(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> tuple[str, str]:
        message = self._complete('planner', self.model, messages, tools=tools, tool_choice='required',
                                 parallel_tool_calls=False, max_tokens=1200)
        try:
            calls = message['tool_calls']
            if len(calls) != 1 or calls[0]['type'] != 'function':
                raise ValueError('Expected one function call')
            call = calls[0]['function']
            if not isinstance(call['name'], str) or not isinstance(call['arguments'], str):
                raise ValueError('Invalid function call')
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise AgentError('LLM_SCHEMA_ERROR', 'Planner получил некорректный вызов инструмента.') from error
        return call['name'], call['arguments']

    def complete_json(
        self, role: str, model: str, messages: list[dict[str, str]], response_type: type[Result],
        *, max_tokens: int = 1800, reason: bool = False,
    ) -> Result:
        options = {}
        if reason:
            options['reasoning'] = {'effort': 'low'}
        elif model == 'deepseek/deepseek-v4.1-flash':
            options['reasoning'] = {'enabled': False}
        elif model.startswith('z-ai/glm-'):
            # GLM cannot switch reasoning off, and by default it can think through the whole token limit.
            options['reasoning'] = {'effort': 'minimal'}
        message = self._complete(role, model, messages, max_tokens=max_tokens, response_format={
            'type': 'json_schema',
            'json_schema': {'name': response_type.__name__, 'strict': True, 'schema': response_type.model_json_schema()},
        }, provider={'require_parameters': True}, **options)
        try:
            return response_type.model_validate_json(message['content'])
        except (KeyError, TypeError, ValidationError) as error:
            raise AgentError('LLM_SCHEMA_ERROR', f'{role} вернул данные вне ожидаемой схемы.') from error

    def _complete(self, role: str, model: str, messages: list[dict[str, str]], **parameters: Any) -> dict[str, Any]:
        api_key = self.api_key.get_secret_value()
        if not api_key:
            raise AgentError('LLM_NOT_CONFIGURED', 'Укажите LLM_API_KEY или включите PLANNER_MODE=offline.')
        with timed(role, 'llm', model=model) as usage:
            return self._send(role, model, messages, api_key, usage, **parameters)

    def _send(self, role: str, model: str, messages: list[dict[str, str]], api_key: str, step: CallUsage,
              **parameters: Any) -> dict[str, Any]:
        started = monotonic()
        try:
            response = self.http.post(
                self.url, headers={'Authorization': f'Bearer {api_key}'}, timeout=self.timeout,
                json={'model': model, 'messages': messages, 'temperature': 0, **parameters},
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            retryable = status is None or status == 429 or status >= 500
            detail = f'HTTP {status}' if status is not None else 'сетевая ошибка или тайм-аут'
            raise AgentError('LLM_UPSTREAM_ERROR', f'{role}: {detail} от LLM.', retryable=retryable) from error

        try:
            data = response.json()
            choice = data['choices'][0]
            message = choice['message']
            if not isinstance(message, dict):
                raise ValueError('Expected an assistant message')
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise AgentError('LLM_SCHEMA_ERROR', f'{role}: ответ LLM не содержит сообщения.') from error
        usage = data.get('usage')
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get('completion_tokens_details')
        details = details if isinstance(details, dict) else {}
        step.model = data['model'] if isinstance(data.get('model'), str) else model
        step.provider = data['provider'] if isinstance(data.get('provider'), str) else None
        step.input_tokens, step.output_tokens = whole(usage.get('prompt_tokens')), whole(usage.get('completion_tokens'))
        step.reasoning_tokens, step.cost_usd = whole(details.get('reasoning_tokens')), amount(usage.get('cost'))
        logger.info('%s model=%s provider=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s cost=%s latency_ms=%d retries=0',
                    role, data.get('model', model), data.get('provider'), usage.get('prompt_tokens'),
                    usage.get('completion_tokens'), details.get('reasoning_tokens'), usage.get('cost'),
                    (monotonic() - started) * 1000)
        if choice.get('finish_reason') == 'length':
            raise AgentError('LLM_OUTPUT_INCOMPLETE', f'{role}: ответ LLM исчерпал лимит токенов.')
        return message
