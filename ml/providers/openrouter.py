import logging
from time import monotonic
from typing import Any

import httpx

from ml.config import Settings
from ml.contracts import AgentError


logger = logging.getLogger(__name__)


class OpenRouterClient:
    def __init__(self, http: httpx.Client, settings: Settings):
        self.http = http
        self.model = settings.llm_model
        self.url = f'{settings.llm_base_url.rstrip("/")}/chat/completions'
        self.api_key = settings.llm_api_key

    def select_tool(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> tuple[str, str]:
        api_key = self.api_key.get_secret_value()
        if not api_key:
            raise AgentError('LLM_NOT_CONFIGURED', 'Укажите LLM_API_KEY или включите PLANNER_MODE=offline.')
        started = monotonic()
        try:
            response = self.http.post(self.url, headers={'Authorization': f'Bearer {api_key}'}, json={
                'model': self.model,
                'messages': messages,
                'tools': tools,
                'tool_choice': 'required',
                'parallel_tool_calls': False,
                'temperature': 0,
                'max_tokens': 768,
            })
            response.raise_for_status()
        except httpx.HTTPError as error:
            status = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
            retryable = status is None or status == 429 or status >= 500
            raise AgentError('LLM_UPSTREAM_ERROR', 'Planner не получил ответ LLM.', retryable=retryable) from error

        try:
            data = response.json()
            calls = data['choices'][0]['message']['tool_calls']
            if len(calls) != 1 or calls[0]['type'] != 'function':
                raise ValueError('Expected one function call')
            call = calls[0]['function']
            if not isinstance(call['name'], str) or not isinstance(call['arguments'], str):
                raise ValueError('Invalid function call')
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise AgentError('LLM_SCHEMA_ERROR', 'Planner получил некорректный вызов инструмента.') from error
        usage = data.get('usage')
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get('completion_tokens_details')
        details = details if isinstance(details, dict) else {}
        logger.info('planner model=%s provider=%s input_tokens=%s output_tokens=%s reasoning_tokens=%s cost=%s latency_ms=%d retries=0',
                    data.get('model', self.model), data.get('provider'), usage.get('prompt_tokens'),
                    usage.get('completion_tokens'), details.get('reasoning_tokens'), usage.get('cost'),
                    (monotonic() - started) * 1000)
        return call['name'], call['arguments']
