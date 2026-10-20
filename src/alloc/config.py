"""Machine capacity and tuning, read from $ALLOC_HOME/config.toml with defaults for this machine."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import psutil

from .sched import RESOURCES, Res
from .units import parse_size

# CPU and GPU time is shared by the OS, so reservations may exceed them; memory and disk can't.
DEFAULT_OVERCOMMIT = Res(mem=1, cpu=2, gpu=2, disk=1)


def home() -> Path:
    return Path(os.environ.get("ALLOC_HOME") or Path.home() / ".alloc")


@dataclass(frozen=True)
class Config:
    capacity: Res
    overcommit: Res = DEFAULT_OVERCOMMIT
    warn_step: float = 0.2  # warn at 20% over the memory request, then each time it grows 20% more

    poll_interval: float = 1.0  # seconds between queue checks while waiting
    monitor_interval: float = 5.0  # seconds between memory samples while running

    @property
    def limit(self) -> Res:
        """The most that running jobs may reserve in total."""
        return Res(*(c * o for c, o in zip(self.capacity, self.overcommit)))


def _total_memory() -> int:
    try:
        return psutil.virtual_memory().total
    except (psutil.Error, OSError):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def load() -> Config:
    """Read the config file, filling anything unset with defaults for this machine."""
    path = home() / "config.toml"
    try:
        data = tomllib.loads(path.read_text()) if path.exists() else {}
        cap, over = data.get("capacity", {}), data.get("overcommit", {})
        capacity = Res(
            mem=parse_size(str(cap["mem"])) if "mem" in cap else int(0.8 * _total_memory()),
            cpu=float(cap.get("cpu", os.cpu_count() or 1)),
            gpu=float(cap.get("gpu", 1)),
            disk=parse_size(str(cap.get("disk", "100G"))),
        )
        overcommit = Res(
            *(float(over.get(name, default)) for name, default in zip(RESOURCES, DEFAULT_OVERCOMMIT))
        )
        tuning = {
            key: float(data[key])
            for key in ("warn_step", "poll_interval", "monitor_interval")
            if key in data
        }
    except (ValueError, TypeError) as e:
        raise ValueError(f"bad config {path}: {e}") from None
    return Config(capacity, overcommit, **tuning)
