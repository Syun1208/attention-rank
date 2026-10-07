from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from attnrank.utils.config import load_env

WORKSPACE_ENV = "ATTNRANK_WORKSPACE"
LOG_DIR_ENV = "ATTNRANK_LOG_DIR"
OUTPUT_DIR_ENV = "ATTNRANK_OUTPUT_DIR"
PROFILES_DIR_ENV = "ATTNRANK_PROFILES_DIR"
DATA_DIR_ENV = "ATTNRANK_DATA_DIR"
LOGS_NAME = "logs"
OUTPUTS_NAME = "outputs"
PROFILES_NAME = "profiles"
DATA_NAME = "data"
PACKAGE_PARENT = Path(__file__).resolve().parents[2]
PROJECT_MARKER = "pyproject.toml"


def default_root() -> Path:
    return PACKAGE_PARENT if (PACKAGE_PARENT / PROJECT_MARKER).exists() else Path.cwd()


@dataclass(frozen=True, slots=True)
class Workspace:
    root: Path
    logs_dir: Path
    outputs_dir: Path
    profiles_dir: Path
    data_dir: Path

    def run_output_dir(self, *, name: str) -> Path:
        return self.outputs_dir / f"{date.today():%Y-%m-%d}_{name}"

    def resolve_profile(self, *, path: Path) -> Path:
        if path.exists() or path.is_absolute():
            return path
        candidate = self.profiles_dir / path.name
        return candidate if candidate.exists() else path

    def resolve_data(self, *, path: Path) -> Path:
        if path.exists() or path.is_absolute():
            return path
        relative = path.relative_to(DATA_NAME) if path.parts and path.parts[0] == DATA_NAME else path
        candidate = self.data_dir / relative
        return candidate if candidate.exists() else path


def resolve_workspace(
    *,
    root: Path | None = None,
    logs_dir: Path | None = None,
    outputs_dir: Path | None = None,
    profiles_dir: Path | None = None,
    data_dir: Path | None = None,
) -> Workspace:
    load_env(start=Path.cwd())
    load_env(start=PACKAGE_PARENT)
    base = root or _env_path(WORKSPACE_ENV) or default_root()
    return Workspace(
        root=base,
        logs_dir=logs_dir or _env_path(LOG_DIR_ENV) or base / LOGS_NAME,
        outputs_dir=outputs_dir or _env_path(OUTPUT_DIR_ENV) or base / OUTPUTS_NAME,
        profiles_dir=profiles_dir or _env_path(PROFILES_DIR_ENV) or base / PROFILES_NAME,
        data_dir=data_dir or _env_path(DATA_DIR_ENV) or base / DATA_NAME,
    )


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None

def workspace_path(*, relative: str) -> Path:
    workspace = resolve_workspace()
    head, _, tail = relative.partition("/")
    roots = {
        LOGS_NAME: workspace.logs_dir,
        OUTPUTS_NAME: workspace.outputs_dir,
        PROFILES_NAME: workspace.profiles_dir,
        DATA_NAME: workspace.data_dir,
    }
    if head in roots:
        return roots[head] / tail if tail else roots[head]
    return workspace.root / relative
