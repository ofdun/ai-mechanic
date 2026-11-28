"""Service view metrics (#41): each step of a turn carries the tokens, cost, latency and retries its call returned."""
import httpx
import pytest
from pydantic import SecretStr

from ml.config import Settings
from ml.contracts import AgentError, CallUsage
from ml.providers.openrouter import OpenRouterClient as LLMClient
from ml.retrieval.openrouter import OpenRouterClient, OpenRouterEmbedder
from ml.usage import collect, measured, record, timed
from ml.pipeline import Pipeline
from tests.test_agent_bootstrap import (
    FixtureGenerator, FixtureRetriever, FixtureVerifier, TestUpstreams, confirmation, utterance,
    setup as bootstrap_setup,
)


USAGE = {'prompt_tokens': 1500, 'completion_tokens': 120, 'cost': 0.000135,
         'completion_tokens_details': {'reasoning_tokens': 40}}


class MeteredUpstreams(TestUpstreams):
    __test__ = False

    def __init__(self):
        super().__init__()
        self.usage = None

    def __call__(self, request):
        response = super().__call__(request)
        if self.usage is None or not request.url.path.endswith('/chat/completions'):
            return response
        return httpx.Response(200, json={**response.json(), 'model': 'z-ai/glm-5.3-flash', 'provider': 'Test',
                                         'usage': self.usage})


class MeasuredRetriever(FixtureRetriever):
    # Production retrievers carry the same decorator.
    search = measured('retrieval', 'retrieval')(FixtureRetriever.search)


@pytest.fixture
def setup(tmp_path):
    fixture = bootstrap_setup.__wrapped__(tmp_path)
    _, pipeline = next(fixture)
    upstreams = MeteredUpstreams()
    with httpx.Client(transport=httpx.MockTransport(upstreams)) as http:
        yield upstreams, Pipeline(pipeline.settings, http, retriever=MeasuredRetriever(), generator=FixtureGenerator(),
                                  verifier=FixtureVerifier())
    fixture.close()


def test_turn_reports_each_step_and_totals_from_the_calls(setup):
    upstreams, pipeline = setup
    upstreams.usage = USAGE
    response = pipeline.run(utterance(message='Запусти диагностику двигателя.'))

    usage = response.usage
    assert [(step.step, step.kind) for step in usage.steps] == [
        ('planner', 'llm'), ('diagnosis', 'service'), ('retrieval', 'retrieval'),
    ]
    planner = usage.steps[0]
    assert (planner.model, planner.provider) == ('z-ai/glm-5.3-flash', 'Test')
    assert (planner.input_tokens, planner.output_tokens, planner.reasoning_tokens) == (1500, 120, 40)
    assert planner.cost_usd == pytest.approx(0.000135) and planner.ok and planner.retries == 0
    assert (usage.input_tokens, usage.output_tokens, usage.cost_usd, usage.cost_complete) == (1500, 120, 0.000135, True)
    assert usage.latency_ms >= max(step.latency_ms for step in usage.steps)
    assert usage.failed_calls == 0
    assert 'usage' in response.model_dump(mode='json')


def test_missing_provider_values_stay_unknown(setup):
    upstreams, pipeline = setup
    response = pipeline.run(utterance(message='Запусти диагностику двигателя.'))
    planner = response.usage.steps[0]
    assert planner.input_tokens is None and planner.cost_usd is None
    # A paid call without a returned price must not read as a free turn.
    assert response.usage.cost_usd is None and not response.usage.cost_complete


def test_explicit_command_without_llm_has_no_paid_steps(setup):
    _, pipeline = setup
    response = pipeline.run(utterance(command='diagnose'))
    assert {step.kind for step in response.usage.steps} == {'service', 'retrieval'}
    assert response.usage.cost_usd is None and response.usage.cost_complete
    assert response.usage.input_tokens is None


def test_ticket_receipt_measures_recheck_and_toir(setup):
    _, pipeline = setup
    draft = pipeline.run(utterance(command='draft_ticket')).ticket_draft
    receipt = pipeline.confirm_ticket(confirmation(draft))
    assert [step.step for step in receipt.usage.steps] == ['diagnosis', 'toir_ticket']
    assert receipt.ticket_id == 'TKT-0001'


def test_failed_llm_call_is_counted():
    settings = Settings(_env_file=None, llm_base_url='https://llm.test/api/v1', llm_api_key='test')
    client = LLMClient(httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(503))), settings)
    with collect() as usage:
        with pytest.raises(AgentError):
            client.select_tool([{'role': 'user', 'content': 'x'}], [])
    summary = usage.summary()
    assert summary.failed_calls == 1 and summary.steps[0].step == 'planner' and not summary.steps[0].ok


def test_openrouter_retrieval_reports_retries_tokens_and_cost():
    calls = []

    def flaky(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(429, json={'error': {'code': 429, 'message': 'Rate limit exceeded'}})
        return httpx.Response(200, json={'data': [{'index': 0, 'embedding': [1.0]}],
                                         'usage': {'prompt_tokens': 7, 'cost': 0.0000002}})

    settings = Settings(_env_file=None, openrouter_api_key=SecretStr('key'))
    client = OpenRouterClient(settings, http=httpx.Client(transport=httpx.MockTransport(flaky)), sleep=lambda _: None)
    with collect() as usage:
        OpenRouterEmbedder('or:baai/bge-m3', client).embed_query('q')
    step, = usage.summary().steps
    assert (step.step, step.model, step.retries, step.input_tokens) == ('embeddings', 'baai/bge-m3', 2, 7)
    assert step.cost_usd == pytest.approx(0.0000002) and usage.summary().retries == 2


def test_nested_steps_keep_start_order_and_nothing_is_recorded_outside_a_turn():
    record(CallUsage(step='ignored', kind='llm'))
    with timed('outside', 'service'):
        pass
    with collect() as usage:
        with timed('retrieval', 'retrieval'):
            record(CallUsage(step='embeddings', kind='retrieval', model='m', cost_usd=0.5))
    assert [step.step for step in usage.summary().steps] == ['retrieval', 'embeddings']
    assert usage.summary().cost_usd == 0.5


def test_billed_invalid_tool_call_counts_as_failed():
    settings = Settings(_env_file=None, llm_base_url='https://llm.test/api/v1', llm_api_key='test')
    reply = {'choices': [{'message': {'tool_calls': []}}], 'usage': USAGE}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=reply))) as http:
        with collect() as usage, pytest.raises(AgentError, match='Planner'):
            LLMClient(http, settings).select_tool([], [])
    summary = usage.summary()
    assert summary.failed_calls == 1
    assert summary.cost_usd == pytest.approx(USAGE['cost'])
