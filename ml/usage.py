"""Per-turn call metrics for the service view: tokens, cost, latency and retries of each step.

``collect()`` opens a collector for one request to the agent. Providers call ``record()`` or ``timed()``;
outside a collector, for example during indexing or benchmarks, both do nothing. Values come only from the
calls themselves: a value the provider did not return stays ``None``.
"""
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import monotonic

from ml.contracts import CallUsage, TurnUsage


class Collector:
    def __init__(self) -> None:
        self.started = monotonic()
        self.steps: list[CallUsage] = []

    def summary(self) -> TurnUsage:
        def total(field: str) -> int | None:
            values = [getattr(step, field) for step in self.steps if getattr(step, field) is not None]
            return sum(values) if values else None

        # A step with a model is billed by its provider; local search and Mock ML/ТОиР are not.
        paid = [step for step in self.steps if step.model is not None and step.kind != 'service']
        costs = [step.cost_usd for step in paid if step.cost_usd is not None]
        return TurnUsage(
            latency_ms=elapsed_ms(self.started), steps=list(self.steps),
            input_tokens=total('input_tokens'), output_tokens=total('output_tokens'),
            reasoning_tokens=total('reasoning_tokens'),
            cost_usd=round(sum(costs), 8) if costs else None, cost_complete=len(costs) == len(paid),
            retries=sum(step.retries for step in self.steps), failed_calls=sum(not step.ok for step in self.steps),
        )


_current: ContextVar[Collector | None] = ContextVar('turn_usage', default=None)


def elapsed_ms(started: float) -> int:
    return round((monotonic() - started) * 1000)


@contextmanager
def collect() -> Iterator[Collector]:
    collector = Collector()
    token = _current.set(collector)
    try:
        yield collector
    finally:
        _current.reset(token)


def record(step: CallUsage) -> None:
    if (collector := _current.get()) is not None:
        collector.steps.append(step)


@contextmanager
def timed(step: str, kind: str, **fields) -> Iterator[CallUsage]:
    """Measure a block; the step keeps its start order even when nested calls finish first."""
    usage = CallUsage(step=step, kind=kind, **fields)
    record(usage)
    started = monotonic()
    try:
        yield usage
    except BaseException:
        usage.ok = False
        raise
    finally:
        usage.latency_ms = elapsed_ms(started)


def measured(step: str, kind: str):
    """Decorator form of ``timed`` for a whole method."""
    def wrap(function):
        @wraps(function)
        def call(*args, **kwargs):
            with timed(step, kind):
                return function(*args, **kwargs)
        return call
    return wrap
