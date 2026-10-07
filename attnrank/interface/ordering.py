from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class RelevanceOrdering(Protocol):
    def order(self, *, question: str, documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]: ...


class AnswerScorer(Protocol):
    def score(self, *, prediction: str, reference: str) -> bool: ...


class ProgressSink(Protocol):
    def __call__(self, completed: int, total: int) -> None: ...
