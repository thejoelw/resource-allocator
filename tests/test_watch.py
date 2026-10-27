"""`alloc watch`: the screen layout, the refresh loop, and the real thing in a pseudo-terminal."""

import curses
import fcntl
import os
import pty
import re
import select
import signal
import sqlite3
import struct
import subprocess
import sys
import termios
import time

import pytest

from alloc import config, db, watch


def plain(lines) -> str:
    return "\n".join("".join(text for text, _ in line) for line in lines)


def take_snapshot(alloc, monkeypatch, who=None):
    monkeypatch.setenv("ALLOC_HOME", str(alloc.home))
    cfg = config.load()
    return watch.take_snapshot(db.connect(), cfg, who, time.time()), cfg


def make_busy(alloc):
    """Two finished jobs (1, 2), one running past its time limit (3) and one queued (4)."""
    alloc.run("run", "--who", "d", "--time", "1m", "--", "true")
    alloc.run("run", "--who", "e", "--time", "1m", "--", "false")
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "1s", "--", "sleep", "30")
    alloc.wait_for(3, "running", need_pid=True)
    alloc.start("run", "--wait", "--who", "b", "--mem", "800M", "--time", "5m", "--", "sleep", "1")
    alloc.wait_for(4, "pending")
    alloc.wait_until(lambda: alloc.status(3)["warnings"])


def test_shows_reservations_and_running_queued_and_finished_jobs(alloc, monkeypatch):
    make_busy(alloc)
    snap, cfg = take_snapshot(alloc, monkeypatch)
    lines = watch.render(snap, cfg, width=120, height=40)
    text = plain(lines)

    assert re.search(r"mem +█+░+  800M of 1G", text)
    assert "running  1" in text and "sleep 30" in text
    assert "warning: running" in text and "past its 1s time limit (1 job waiting in the queue)" in text
    assert "queued  1" in text and "sleep 1" in text
    assert "recently finished" in text and "true" in text and "false" in text

    styles = {text.strip(): style for line in lines for text, style in line}
    assert (styles["exit 0"], styles["exit 1"]) == ("green", "red")
    assert any(text.strip().endswith("/1s") and style == "red" for line in lines for text, style in line)


def test_who_filters_jobs_but_not_reservations(alloc, monkeypatch):
    make_busy(alloc)
    snap, cfg = take_snapshot(alloc, monkeypatch, who="a")
    text = plain(watch.render(snap, cfg, width=120, height=40, who="a"))
    assert "alloc watch · a" in text
    assert "sleep 30" in text
    assert "sleep 1" not in text and "false" not in text
    assert "800M of 1G" in text


def test_fits_the_window(alloc, monkeypatch):
    make_busy(alloc)
    snap, cfg = take_snapshot(alloc, monkeypatch)

    short = watch.render(snap, cfg, width=120, height=12)
    assert len(short) == 12
    assert "more lines; make the window taller" in plain(short[-1:])

    narrow = plain(watch.render(snap, cfg, width=60, height=40)).splitlines()
    row = next(line for line in narrow if line.startswith("  3 "))
    assert len(row) <= 60 and row.endswith("…")


def test_needs_a_terminal(alloc):
    r = alloc.run("watch")
    assert r.returncode == 2
    assert "needs a terminal" in r.stderr


class FakeScreen:
    """Stands in for a curses window, checking that nothing is drawn outside it."""

    def __init__(self, height, width, keys):
        self.height, self.width, self.keys = height, width, list(keys)
        self.frames: list[dict] = []

    def getmaxyx(self):
        return self.height, self.width

    def timeout(self, ms):
        self.timeout_ms = ms

    def erase(self):
        self.frames.append({})

    def addstr(self, y, x, text, attr):
        assert 0 <= y < self.height and 0 <= x and x + len(text) <= self.width, (y, x, text)
        self.frames[-1][y] = self.frames[-1].get(y, "").ljust(x) + text

    def refresh(self):
        pass

    def getch(self):
        key = self.keys.pop(0)
        if isinstance(key, BaseException):
            raise key
        return key

    def text(self, frame=-1) -> str:
        lines = self.frames[frame]
        return "\n".join(lines.get(y, "") for y in range(self.height))


def watch_on(screen, alloc, monkeypatch, interval=0.5) -> int:
    """Run watch.run on a fake screen instead of the terminal."""
    monkeypatch.setenv("ALLOC_HOME", str(alloc.home))
    monkeypatch.setattr(curses, "wrapper", lambda func, *args: func(screen, *args))
    monkeypatch.setattr(curses, "curs_set", lambda visibility: None)
    monkeypatch.setattr(curses, "set_escdelay", lambda ms: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    return watch.run(config.load(), None, interval)


def test_redraws_until_q_and_never_changes_the_queue(alloc, monkeypatch):
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "5m", "--", "sleep", "30")
    alloc.wait_for(1, "running", need_pid=True)
    waiter = alloc.start("run", "--wait", "--who", "b", "--mem", "800M", "--time", "5m", "--", "true")
    alloc.wait_for(2, "pending")
    # Leave a queued job whose `alloc run` is gone: any command that writes would clean it up.
    waiter.kill()
    waiter.wait()

    screen = FakeScreen(14, 70, keys=[-1, curses.KEY_RESIZE, ord("q")])
    assert watch_on(screen, alloc, monkeypatch) == 0
    assert len(screen.frames) == 3 and screen.timeout_ms == 500
    assert "alloc watch" in screen.text() and "sleep 30" in screen.text()

    with sqlite3.connect(alloc.home / "alloc.db") as conn:
        assert conn.execute("SELECT state FROM jobs WHERE id = 2").fetchone()[0] == "pending"
    assert alloc.status(2)["state"] == "lost"  # whereas status does clean up


def test_ctrl_c_quits_quietly(alloc, monkeypatch):
    screen = FakeScreen(30, 100, keys=[KeyboardInterrupt()])
    assert watch_on(screen, alloc, monkeypatch) == 0
    assert "recently finished" in screen.text()


def can_open_a_terminal() -> bool:
    try:
        master, slave = os.openpty()
    except OSError:
        return False
    os.close(master)
    os.close(slave)
    return True


needs_a_terminal = pytest.mark.skipif(not can_open_a_terminal(),
                                      reason="can't open a pseudo-terminal here (sandboxed?)")


def run_in_terminal(alloc, *args, wait_for: str, then=None) -> tuple[str, int]:
    """Run `alloc watch` in a 40x120 pseudo-terminal until `wait_for` shows, then call `then(proc, fd)`.

    Returns what it drew, escape sequences removed, and its exit status.
    """
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
    proc = subprocess.Popen([sys.executable, "-m", "alloc", "watch", *args],
                            stdin=slave, stdout=slave, stderr=slave, cwd=alloc.cwd,
                            env={**alloc.env, "TERM": "xterm-256color"})
    os.close(slave)
    output, screen, acted = b"", "", False
    deadline = time.time() + 15
    try:
        while time.time() < deadline:
            if not acted and wait_for in screen:
                then(proc, master)
                acted = True
            if select.select([master], [], [], 0.1)[0]:
                try:
                    chunk = os.read(master, 1 << 16)
                except OSError:  # EIO: watch exited and the terminal closed
                    break
                if not chunk:
                    break
                output += chunk
                screen = re.sub(r"\x1b(\[[0-9;?]*[ -/]*[@-~]|[()][0-9A-Za-z]|[=>])", "",
                                output.decode(errors="replace"))
            elif proc.poll() is not None:
                break
        return screen, proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        os.close(master)


@needs_a_terminal
def test_runs_in_a_real_terminal_until_q(alloc):
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "5m", "--", "sleep", "30")
    alloc.wait_for(1, "running", need_pid=True)

    screen, status = run_in_terminal(alloc, "-n", "0.2", wait_for="sleep 30",
                                     then=lambda proc, fd: os.write(fd, b"q"))
    assert status == 0
    assert "alloc watch" in screen and "800M of 1G" in screen


@needs_a_terminal
def test_ctrl_c_quits_a_real_terminal_cleanly(alloc):
    screen, status = run_in_terminal(alloc, wait_for="recently finished",
                                     then=lambda proc, fd: proc.send_signal(signal.SIGINT))
    assert status == 0
