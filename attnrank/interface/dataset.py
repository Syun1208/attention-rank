from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol

RowAdapter = Callable[[Mapping[str, Any]], dict[str, Any]]


class DatasetSource(Protocol):
    def rows(self) -> list[dict[str, Any]]: ...
