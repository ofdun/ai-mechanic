"""Rank fusion of several result lists: queries of one plan, BM25 and dense, reranker and the fused prior."""

from collections.abc import Sequence

Ranking = Sequence[tuple[int, float]]


def reciprocal_rank_fusion(
    rankings: Sequence[Ranking], *, k: int = 60, weights: Sequence[float] | None = None,
) -> list[tuple[int, float]]:
    """Weighted RRF: score = Σ w / (k + rank). A single unweighted ranking is returned unchanged with its scores."""
    weights = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError('weights must match rankings')
    pairs = [(ranking, weight) for ranking, weight in zip(rankings, weights) if ranking and weight > 0]
    if len(pairs) == 1 and pairs[0][1] == 1.0:
        return list(pairs[0][0])
    totals: dict[int, float] = {}
    for ranking, weight in pairs:
        for rank, (index, _) in enumerate(ranking, 1):
            totals[index] = totals.get(index, 0.0) + weight / (k + rank)
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))
