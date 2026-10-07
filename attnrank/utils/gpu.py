from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.free,memory.total",
    "--format=csv,noheader,nounits",
]
MIB_PER_GIB = 1024.0


@dataclass(frozen=True, slots=True)
class GpuStatus:
    index: int
    name: str
    free_gib: float
    total_gib: float


def query_gpus() -> list[GpuStatus]:
    try:
        output = subprocess.run(NVIDIA_SMI_QUERY, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    statuses = []
    for line in output.strip().splitlines():
        index, name, free, total = (part.strip() for part in line.split(","))
        statuses.append(
            GpuStatus(
                index=int(index),
                name=name,
                free_gib=float(free) / MIB_PER_GIB,
                total_gib=float(total) / MIB_PER_GIB,
            )
        )
    return statuses


def require_free_memory(*, device: int, minimum_gib: float) -> None:
    statuses = {status.index: status for status in query_gpus()}
    if device < 0 or device not in statuses:
        return
    status = statuses[device]
    logger.info("gpu %d (%s): %.1f/%.1f GiB free", status.index, status.name, status.free_gib, status.total_gib)
    if status.free_gib < minimum_gib:
        raise RuntimeError(
            f"gpu {device} has {status.free_gib:.1f} GiB free, below the {minimum_gib:.1f} GiB this run needs"
        )
