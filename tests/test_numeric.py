from __future__ import annotations

from attnrank.utils.numeric import FULL_UNIT, extract_answer, is_correct, parse_number


def test_parse_number_handles_percent_and_currency() -> None:
    parsed = parse_number(text="Final answer: -$1,234.50%")
    assert parsed is not None
    assert parsed.value == -1234.5
    assert parsed.percent
    assert parsed.decimals == 2


def test_extract_answer_prefers_text_after_the_marker() -> None:
    assert extract_answer(response="revenue was 991.1 then 959.2\nFinal answer: -3.2%") == "-3.2%"
    assert extract_answer(response="values 10 and 20") == "20"


def test_is_correct_tolerance() -> None:
    assert is_correct(extracted="-3.2%", gold="-3.2%.")
    assert is_correct(extracted="0.032", gold="3.2%")
    assert not is_correct(extracted="-4.0%", gold="-3.2%.")
    assert is_correct(extracted="-3.9%", gold="-3.2%.", units=FULL_UNIT) is False
    assert is_correct(extracted="", gold="12") is False
    assert is_correct(extracted="12", gold="twelve") is None
