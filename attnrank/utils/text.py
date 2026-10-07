from __future__ import annotations

import math
import re
from collections import Counter
from typing import Mapping, Sequence

PUNCTUATION = re.compile(r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]")
ARTICLES = re.compile(r"\b(a|an|the)\b")
WORD = re.compile(r"[a-z0-9]+")
BM25_K1 = 1.5
BM25_B = 0.75


def normalize(*, text: str) -> str:
    text = text.lower()
    text = PUNCTUATION.sub(" ", text)
    text = ARTICLES.sub(" ", text)
    return " ".join(text.split())


def substring_match(*, prediction: str, gold: str) -> bool:
    return normalize(text=gold) in normalize(text=prediction)


def document_text(*, document: Mapping[str, str]) -> str:
    return document.get("title", "") + ": " + document["text"]


def bm25_scores(
    *,
    question: str,
    documents: Sequence[Mapping[str, str]],
    k1: float = BM25_K1,
    b: float = BM25_B,
) -> list[float]:
    corpus = [WORD.findall(document_text(document=document).lower()) for document in documents]
    lengths = [len(tokens) for tokens in corpus]
    average = sum(lengths) / len(lengths) if lengths else 0.0
    frequencies = [Counter(tokens) for tokens in corpus]
    query = WORD.findall(question.lower())

    scores = []
    for position, counts in enumerate(frequencies):
        total = 0.0
        for token in query:
            if token not in counts:
                continue
            containing = sum(1 for other in frequencies if token in other)
            idf = math.log(1.0 + (len(corpus) - containing + 0.5) / (containing + 0.5))
            frequency = counts[token]
            length_ratio = lengths[position] / average if average else 1.0
            denominator = frequency + k1 * (1.0 - b + b * length_ratio)
            total += idf * (frequency * (k1 + 1.0)) / denominator
        scores.append(total)
    return scores


def unescape_argument(*, value: str) -> str:
    return value.encode().decode("unicode_escape")
