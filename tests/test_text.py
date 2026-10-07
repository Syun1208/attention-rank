from __future__ import annotations

from attnrank.utils.text import bm25_scores, normalize, substring_match


def test_normalize_strips_punctuation_and_articles() -> None:
    assert normalize(text="The  Quick, brown fox!") == "quick brown fox"


def test_substring_match_is_case_and_punctuation_insensitive() -> None:
    assert substring_match(prediction="The answer is: Rome.", gold="rome")
    assert not substring_match(prediction="Paris", gold="rome")


def test_bm25_prefers_documents_sharing_query_terms() -> None:
    documents = [
        {"title": "", "text": "cats purr when happy"},
        {"title": "", "text": "the capital of italy is rome"},
        {"title": "", "text": "rome is a city in italy with many cats"},
    ]
    scores = bm25_scores(question="capital of italy", documents=documents)
    assert scores[1] > scores[2] > scores[0]
