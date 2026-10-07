from __future__ import annotations

import pytest

from attnrank.utils.chunking import balanced_partition, chunk_context, is_table


def test_is_table_detects_pipe_rows() -> None:
    assert is_table(block="a | b | c\n1 | 2 | 3")
    assert not is_table(block="plain sentence . another sentence .")


def test_balanced_partition_covers_all_units() -> None:
    bounds = balanced_partition(lengths=[5, 5, 5, 5], count=2)
    assert bounds == [0, 2, 4]


def test_chunk_context_returns_requested_count_and_keeps_text() -> None:
    context = "first sentence here . second sentence here . third sentence here .\n\nfourth paragraph with more words ."
    chunks = chunk_context(context=context, count=3, minimum=5)
    assert len(chunks) == 3
    assert "first sentence" in " ".join(chunks)


def test_chunk_context_rejects_impossible_split() -> None:
    with pytest.raises(ValueError):
        chunk_context(context="one", count=3, minimum=5)
