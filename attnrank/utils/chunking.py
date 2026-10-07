from __future__ import annotations

import re
from typing import Sequence

MINIMUM_UNIT_CHARACTERS = 40
SENTENCE_BOUNDARY = re.compile(r"(?<= \.) ")
PARAGRAPH_SEPARATOR = "\n\n"
TABLE_CELL_SEPARATOR = " | "


def is_table(*, block: str) -> bool:
    lines = [line for line in block.splitlines() if line.strip()]
    return len(lines) > 1 and sum(TABLE_CELL_SEPARATOR in line for line in lines) >= len(lines) - 1


def sentences(*, block: str) -> list[str]:
    parts = SENTENCE_BOUNDARY.split(block.strip())
    return [part.strip() for part in parts if part.strip()]


def split_longest(*, units: Sequence[str], minimum: int) -> list[str]:
    prose = [index for index in range(len(units)) if "\n" not in units[index] and len(units[index]) >= 2 * minimum]
    pool = prose or list(range(len(units)))
    index = max(pool, key=lambda candidate: len(units[candidate]))
    unit = units[index]

    lines = unit.splitlines()
    if len(lines) > 1:
        middle = len(lines) // 2
        halves = ["\n".join(lines[:middle]), "\n".join(lines[middle:])]
    else:
        words = unit.split(" ")
        if len(words) < 2:
            return list(units)
        middle = len(words) // 2
        halves = [" ".join(words[:middle]), " ".join(words[middle:])]
    return [*units[:index], *halves, *units[index + 1 :]]


def merge_fragments(*, units: Sequence[str], minimum: int) -> list[str]:
    merged: list[str] = []
    for unit in units:
        joinable = merged and "\n" not in unit and "\n" not in merged[-1]
        if joinable and (len(unit) < minimum or len(merged[-1]) < minimum):
            merged[-1] = f"{merged[-1]} {unit}"
        else:
            merged.append(unit)
    return merged


def balanced_partition(*, lengths: Sequence[int], count: int) -> list[int]:
    size = len(lengths)
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)

    infinity = float("inf")
    cost = [[infinity] * (size + 1) for _ in range(count + 1)]
    cut = [[0] * (size + 1) for _ in range(count + 1)]
    cost[0][0] = 0.0
    for part in range(1, count + 1):
        for stop in range(part, size - (count - part) + 1):
            for start in range(part - 1, stop):
                if cost[part - 1][start] == infinity:
                    continue
                candidate = cost[part - 1][start] + float(prefix[stop] - prefix[start]) ** 2
                if candidate < cost[part][stop]:
                    cost[part][stop] = candidate
                    cut[part][stop] = start

    bounds = [size]
    for part in range(count, 0, -1):
        bounds.append(cut[part][bounds[-1]])
    return bounds[::-1]


def split_units(*, context: str, minimum: int) -> list[str]:
    units: list[str] = []
    for block in (candidate.strip() for candidate in context.split(PARAGRAPH_SEPARATOR) if candidate.strip()):
        if len(block) < minimum and units:
            units[-1] = f"{units[-1]}\n{block}" if "\n" in units[-1] else f"{units[-1]} {block}"
        elif is_table(block=block):
            units.append(block)
        else:
            units.extend(merge_fragments(units=sentences(block=block), minimum=minimum))
    return units


def chunk_context(
    *,
    context: str,
    count: int,
    minimum: int = MINIMUM_UNIT_CHARACTERS,
) -> list[str]:
    units = split_units(context=context, minimum=minimum)
    while len(units) < count:
        grown = split_longest(units=units, minimum=minimum)
        if len(grown) == len(units):
            raise ValueError("context too short to split into the requested number of chunks")
        units = grown

    bounds = balanced_partition(lengths=[len(unit) for unit in units], count=count)
    chunks = []
    for begin, stop in zip(bounds, bounds[1:]):
        group = units[begin:stop]
        chunks.append(("\n" if any("\n" in unit for unit in group) else " ").join(group))
    return chunks
