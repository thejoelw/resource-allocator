import pytest

from alloc.units import GiB, fmt_duration, fmt_size, parse_duration, parse_share, parse_size


@pytest.mark.parametrize("text, expected", [
    ("10G", 10 * GiB),
    ("10gb", 10 * GiB),
    ("10GiB", 10 * GiB),
    ("1.5T", int(1.5 * 1024 * GiB)),
    ("512M", 512 * 1024**2),
    ("100B", 100),
    ("8", 8 * GiB),
])
def test_parse_size(text, expected):
    assert parse_size(text) == expected


@pytest.mark.parametrize("text", ["", "ten", "10X", "-1G"])
def test_parse_size_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_size(text)


@pytest.mark.parametrize("text, expected", [
    ("15m", 900),
    ("1h30m", 5400),
    ("1h 30m", 5400),
    ("90s", 90),
    ("1.5h", 5400),
    ("2d", 172800),
    ("20", 1200),
])
def test_parse_duration(text, expected):
    assert parse_duration(text) == expected


@pytest.mark.parametrize("text", ["", "soon", "15x", "m15", "1h30"])
def test_parse_duration_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_duration(text)


def test_parse_share():
    assert parse_share("50%", 10) == 5
    assert parse_share("4", 10) == 4
    assert parse_share("100%", 1) == 1
    with pytest.raises(ValueError):
        parse_share("-1", 10)
    with pytest.raises(ValueError):
        parse_share("half", 10)


def test_formatting():
    assert fmt_size(10 * GiB) == "10G"
    assert fmt_size(11.8 * GiB) == "11.8G"
    assert fmt_size(512 * 1024**2) == "512M"
    assert fmt_size(0) == "0"
    assert fmt_duration(900) == "15m"
    assert fmt_duration(723) == "12m3s"
    assert fmt_duration(3725) == "1h2m"
    assert fmt_duration(3605) == "1h"
    assert fmt_duration(7 * 86400) == "7d"
    assert fmt_duration(0) == "0s"
