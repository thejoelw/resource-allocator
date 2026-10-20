"""Parse and format resource amounts: sizes, durations, and CPU/GPU shares."""

import math
import re

GiB = 1024**3

_DURATION_UNITS = {"d": 86400, "h": 3600, "m": 60, "s": 1}


def parse_size(text: str) -> int:
    """Parse '10G', '512M' or '1.5TB' into bytes. A bare number means GiB."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([kmgt]?)(i?b)?", text.strip().lower())
    if not m:
        raise ValueError(f"invalid size {text!r} (examples: 10G, 512M)")
    number, prefix, suffix = m.groups()
    if prefix:
        scale = 1024 ** ("kmgt".index(prefix) + 1)
    elif suffix:
        scale = 1
    else:
        scale = GiB
    return int(float(number) * scale)


def parse_duration(text: str) -> float:
    """Parse '15m', '1h30m' or '90s' into seconds. A bare number means minutes."""
    s = re.sub(r"\s+", "", text.lower())
    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s) * 60
    parts = re.findall(r"(\d+(?:\.\d+)?)([dhms])", s)
    if not parts or "".join(n + u for n, u in parts) != s:
        raise ValueError(f"invalid duration {text!r} (examples: 15m, 1h30m, 90s)")
    return sum(float(n) * _DURATION_UNITS[u] for n, u in parts)


def parse_share(text: str, whole: float) -> float:
    """Parse '50%' as a share of `whole`, or a bare number as an absolute amount."""
    s = text.strip()
    try:
        value = float(s[:-1]) * whole / 100 if s.endswith("%") else float(s)
    except ValueError:
        raise ValueError(f"invalid amount {text!r} (examples: 50%, 4)") from None
    if value < 0 or math.isnan(value):
        raise ValueError(f"invalid amount {text!r}: must be zero or more")
    return value


def fmt_size(n: float) -> str:
    if n >= GiB:
        return f"{n / GiB:.1f}".removesuffix(".0") + "G"
    if n >= 1024**2:
        return f"{n / 1024**2:.0f}M"
    if n > 0:
        return f"{n / 1024:.0f}K"
    return "0"


def fmt_duration(seconds: float) -> str:
    """Format as '2d3h', '1h5m', '12m3s' or '45s', keeping the two largest units."""
    total = int(round(max(seconds, 0)))
    d, rem = divmod(total, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    units = [(d, "d"), (h, "h"), (m, "m"), (s, "s")]
    first = next((i for i, (n, _) in enumerate(units) if n), None)
    if first is None:
        return "0s"
    return "".join(f"{n}{unit}" for n, unit in units[first : first + 2] if n)
