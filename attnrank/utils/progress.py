from __future__ import annotations

from dataclasses import dataclass

from tqdm import tqdm

COUNT_BAR = (
    "{desc} ▕{bar}▏ {percentage:3.0f}% • {n_fmt}/{total_fmt} • {rate_fmt}"
    " • ⏳ {elapsed}<{remaining}{postfix}"
)
TIME_BAR = "{desc} ▕{bar}▏ {percentage:3.0f}% • ⏳ {elapsed}<{remaining}{postfix}"


@dataclass(frozen=True, slots=True)
class Stage:
    emoji: str
    title: str
    colour: str


DOWNLOAD = Stage(emoji="📥", title="Download", colour="cyan")
LOAD_MODEL = Stage(emoji="🧠", title="Load model", colour="blue")
PROBE = Stage(emoji="🔍", title="Probe", colour="green")
GENERATE = Stage(emoji="💬", title="Generate", colour="magenta")
SCORE = Stage(emoji="📊", title="Score", colour="yellow")


def progress(
    stage: Stage,
    *,
    total: float | None,
    unit: str = "it",
    show_counts: bool = True,
) -> tqdm:
    layout = COUNT_BAR if show_counts else TIME_BAR
    return tqdm(
        total=total,
        unit=unit,
        desc=f"{stage.emoji} {stage.title}",
        colour=stage.colour,
        bar_format=layout if total else None,
        dynamic_ncols=True,
    )


def finish(bar: tqdm) -> None:
    if bar.total is not None and bar.n < bar.total:
        bar.update(bar.total - bar.n)
    bar.close()
