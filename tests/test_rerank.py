from __future__ import annotations

import pytest

attnrank = pytest.importorskip("attnrank")


def test_rerank_documents_places_rank_i_at_ith_slot() -> None:
    assert attnrank.rerank_documents(documents_by_relevance=list("ABCDE"), position_order=[4, 0, 2, 1, 3]) == [
        "B",
        "D",
        "C",
        "E",
        "A",
    ]


def test_rerank_baselines() -> None:
    documents = list("ABCDE")
    assert attnrank.rerank_baseline(documents_by_relevance=documents, strategy="descending") == documents
    assert attnrank.rerank_baseline(documents_by_relevance=documents, strategy="ascending") == list("EDCBA")
    assert attnrank.rerank_baseline(documents_by_relevance=documents, strategy="lim") == list("BDECA")
    with pytest.raises(ValueError):
        attnrank.rerank_baseline(documents_by_relevance=documents, strategy="attnrank")


def test_basin_statistics_detects_edges_above_interior() -> None:
    statistics = attnrank.basin_statistics(attention=[0.3, 0.1, 0.1, 0.1, 0.4], min_edge_ratio=1.5)
    assert statistics.is_basin
    assert statistics.edge_ratio == pytest.approx(3.0)
