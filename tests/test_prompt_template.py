from __future__ import annotations

import pytest

attnrank = pytest.importorskip("attnrank")


def test_prompt_template_renders_paper_layout() -> None:
    text = attnrank.PromptTemplate().render(question="Q?", documents=["a", {"title": "T", "text": "b"}])
    assert text.startswith("Write a high-quality answer")
    assert "\n\nDocument [1] a\nDocument [2] b\n\nQuestion: Q?\nAnswer:" in text


def test_prompt_template_titles_and_separator() -> None:
    template = attnrank.PromptTemplate(include_titles=True, document_separator="\n\n")
    text = template.render(question="Q?", documents=[{"title": "T", "text": "b"}, "c"])
    assert "Document [1] T: b\n\nDocument [2] c" in text


def test_chat_messages_wrap_the_prompt() -> None:
    profile = attnrank.AttentionProfile()
    profile.attention = [0.4, 0.2, 0.4]
    profile.recompute_position_order()
    ranker = attnrank.AttnRank(profile=profile)
    messages = ranker.chat_messages(question="Q?", documents_by_relevance=["x", "y", "z"], system_prompt="sys")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[1]["content"].endswith("Question: Q?\nAnswer:")
