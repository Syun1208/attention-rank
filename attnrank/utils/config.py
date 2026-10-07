from __future__ import annotations

import json
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeVar

ENV_FILE_NAME = ".env"
CONFIG_COPY_NAME = "config.yaml"

Settings = TypeVar("Settings")


def find_env_file(*, start: Path) -> Path | None:
    for directory in (start, *start.parents):
        candidate = directory / ENV_FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def load_env(*, start: Path) -> Path | None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return None

    env_file = find_env_file(start=start)
    if env_file is not None:
        load_dotenv(
            dotenv_path=env_file,
            override=False,
        )
    return env_file


def read_yaml(*, path: Path) -> dict[str, Any]:
    import yaml

    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return values


def build_settings(
    settings_type: type[Settings],
    *,
    values: Mapping[str, Any],
    overrides: Mapping[str, Any],
    source: str,
) -> Settings:
    if not is_dataclass(settings_type):
        raise TypeError(f"{settings_type.__name__} is not a dataclass")
    known = {field.name: field for field in fields(settings_type)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown keys in {source}: {sorted(unknown)}")
    merged = {**values, **{key: value for key, value in overrides.items() if value is not None}}
    converted = {key: _convert(known[key].type, value) for key, value in merged.items()}
    return settings_type(**converted)


def load_settings(
    settings_type: type[Settings],
    *,
    config_path: Path | None,
    overrides: Mapping[str, Any],
) -> Settings:
    values = read_yaml(path=config_path) if config_path is not None else {}
    source = str(config_path) if config_path is not None else "command line"
    return build_settings(
        settings_type,
        values=values,
        overrides=overrides,
        source=source,
    )


def write_settings_copy(*, settings: Any, output_dir: Path) -> Path:
    import yaml

    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / CONFIG_COPY_NAME
    target.write_text(
        yaml.safe_dump(settings_to_plain(settings=settings), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return target


def settings_to_plain(*, settings: Any) -> dict[str, Any]:
    return json.loads(json.dumps(asdict(settings), default=str))


def _convert(annotation: Any, value: Any) -> Any:
    text = str(annotation)
    if value is None:
        return None
    if "DatasetSpec" in text:
        from attnrank.data.sources import DatasetSpec

        return DatasetSpec.parse(value=value)
    if "Path" in text and isinstance(value, str):
        return Path(value)
    if "tuple" in text and isinstance(value, list):
        return tuple(value)
    return value
