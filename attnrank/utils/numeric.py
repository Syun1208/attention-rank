from __future__ import annotations

import re
from dataclasses import dataclass

NUMBER = re.compile(r"[-+−]?\$?\d[\d,]*(?:\.\d+)?\s?%?")
FINAL_ANSWER_MARKER = re.compile(r"final answer\s*[:=]?", re.IGNORECASE)
HALF_UNIT = 0.5
FULL_UNIT = 1.0
RELATIVE_TOLERANCE = 0.005
TOLERANCE_SLACK = 1.0001


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    value: float
    percent: bool
    decimals: int


def parse_number(*, text: str) -> ParsedNumber | None:
    match = NUMBER.search(text)
    if match is None:
        return None
    token = match.group(0).replace("−", "-").replace("$", "").replace(",", "").strip()
    percent = token.endswith("%")
    token = token.rstrip("%").strip()
    try:
        value = float(token)
    except ValueError:
        return None
    decimals = len(token.split(".")[1]) if "." in token else 0
    return ParsedNumber(value=value, percent=percent, decimals=decimals)


def extract_answer(*, response: str) -> str:
    markers = list(FINAL_ANSWER_MARKER.finditer(response))
    if markers:
        tail = response[markers[-1].end() :]
        match = NUMBER.search(tail)
        if match:
            return match.group(0).strip()
    numbers = NUMBER.findall(response)
    return numbers[-1].strip() if numbers else ""


def is_correct(*, extracted: str, gold: str, units: float = HALF_UNIT) -> bool | None:
    target = parse_number(text=gold)
    if target is None:
        return None
    predicted = parse_number(text=extracted) if extracted else None
    if predicted is None:
        return False

    candidates = [predicted.value]
    if target.percent and not predicted.percent:
        candidates.append(predicted.value * 100.0)
    if predicted.percent and not target.percent:
        candidates.append(predicted.value / 100.0)
    tolerance = max(units * 10.0 ** (-target.decimals) * TOLERANCE_SLACK, RELATIVE_TOLERANCE * abs(target.value))
    return any(abs(candidate - target.value) <= tolerance for candidate in candidates)
