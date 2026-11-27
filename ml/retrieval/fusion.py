"""Rank fusion of several result lists (queries of one plan now, BM25 and dense later)."""

from collections.abc import Sequence

Ranking = Sequence[tuple[int, float]]


def reciprocal_rank_fusion(rankings: Sequence[Ranking], *, k: int = 60) -> list[tuple[int, float]]:
    """RRF: score = Σ 1 / (k + rank). A single ranking is returned unchanged with its own scores."""
    rankings = [ranking for ranking in rankings if ranking]
    if len(rankings) == 1:
        return list(rankings[0])
    totals: dict[int, float] = {}
    for ranking in rankings:
        for rank, (index, _) in enumerate(ranking, 1):
            totals[index] = totals.get(index, 0.0) + 1 / (k + rank)
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))
