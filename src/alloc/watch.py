"""`alloc watch`: a live view of running, queued and recently finished jobs.

It only reads the database (the connection is query_only), so watching never changes the
queue or holds up `alloc run`.
"""

import curses
import math
import time
from dataclasses import dataclass

import psutil

from . import db
from .cli import (
    FINISHED_HEADER,
    QUEUED_HEADER,
    RUNNING_HEADER,
    finished_cells,
    fmt_amount,
    job_info,
    overview_rows,
    queued_cells,
    running_cells,
)
from .config import Config
from .sched import RESOURCES, Res
from .units import fmt_size

RECENT = 50  # finished jobs to fetch; as many as fit are shown

# A screen line is a list of (text, style) segments. Styles are names that become curses
# attributes only when drawn, so rendering can be tested without a terminal.
Cell = tuple[str, str]
Line = list[Cell]


@dataclass
class Snapshot:
    now: float
    used: Res  # reserved by running jobs, whoever they belong to
    jobs_mem: float  # what running jobs last measured using
    rows: list  # jobs to list, from overview_rows
    infos: dict  # job_info for each row, by ID
    mem_used: int  # the whole machine
    mem_total: int
    cpu_percent: float  # the whole machine, since the previous snapshot


def take_snapshot(conn, cfg: Config, who: str | None, now: float) -> Snapshot:
    with db.snapshot(conn):
        starts = db.predicted_starts(conn, cfg, now)
        queued = db.count_queued(conn)
        used = db.in_use(conn)
        jobs_mem = conn.execute(
            "SELECT COALESCE(SUM(mem_now), 0) FROM jobs WHERE state = 'running'"
        ).fetchone()[0]
        rows = overview_rows(conn, who, now, recent=RECENT)
    infos = {row["id"]: job_info(row, cfg, now, starts, queued) for row in rows}
    memory = psutil.virtual_memory()
    return Snapshot(now, used, jobs_mem, rows, infos, memory.total - memory.available, memory.total,
                    psutil.cpu_percent())


# --- layout --------------------------------------------------------------------------------


def render(snap: Snapshot, cfg: Config, width: int, height: int, who: str | None = None,
           interval: float = 1.0) -> list[Line]:
    """Lay out one screenful: what's reserved, then running, queued and finished jobs."""
    title = "alloc watch" + (f" · {who}" if who else "")
    clock = f"{time.strftime('%H:%M:%S', time.localtime(snap.now))} · every {interval:g}s · q quits"
    lines: list[Line] = [[(title, "bold"), (clock.rjust(max(width - len(title), len(clock) + 2)), "dim")], []]
    lines += resource_lines(snap, cfg, width)

    running = [r for r in snap.rows if r["state"] == "running"]
    pending = [r for r in snap.rows if r["state"] == "pending"]
    finished = [r for r in snap.rows if r["state"] not in ("running", "pending")]
    lines += section("running", len(running), RUNNING_HEADER,
                     [running_row(r, cfg, snap.now) for r in running], width,
                     notes=[snap.infos[r["id"]]["warnings"] for r in running])
    lines += section("queued", len(pending), QUEUED_HEADER,
                     [queued_row(r, snap.infos[r["id"]], snap.now) for r in pending], width)
    # Finished jobs get whatever room is left below the blank line, title and header.
    room = max(height - len(lines) - 3, 1)
    lines += section("recently finished", None, FINISHED_HEADER,
                     [finished_row(r, snap.now) for r in finished[:room]], width)

    if len(lines) > height:
        hidden = len(lines) - height + 1
        lines = lines[: max(height - 1, 0)]
        lines.append([(f"  … {hidden} more lines; make the window taller to see them", "dim")])
    return lines


def resource_lines(snap: Snapshot, cfg: Config, width: int) -> list[Line]:
    bar = max(10, min(30, width // 4))
    amounts = {
        name: f"{fmt_amount(name, u)} of {fmt_amount(name, limit)}"
        for name, u, limit in zip(RESOURCES, snap.used, cfg.limit)
    }
    amount_width = max(len(text) for text in amounts.values())
    measured = {
        "mem": f"jobs {fmt_size(snap.jobs_mem)}, machine {fmt_size(snap.mem_used)} of {fmt_size(snap.mem_total)}",
        "cpu": f"machine {snap.cpu_percent:.0f}% busy",
    }
    lines = [[(" " * 8 + "reserved".ljust(bar + amount_width + 4), "dim"), ("measured", "dim")]]
    for name, u, limit in zip(RESOURCES, snap.used, cfg.limit):
        share = u / limit if limit else 0
        filled = min(bar, math.ceil(share * bar - 1e-9))  # any reservation shows at least one block
        style = "green" if share < 0.75 else "yellow" if share < 1 else "red"
        lines.append([
            (f"  {name:<6}", "bold"), ("█" * filled, style), ("░" * (bar - filled), "dim"),
            (f"  {amounts[name]:<{amount_width}}  ", ""), (measured.get(name, ""), "dim"),
        ])
    return lines


def section(title: str, count: int | None, header: list[str], rows: list[list[Cell]], width: int,
            notes: list[list[str]] | None = None) -> list[Line]:
    """A titled table, with each row's `notes` (warnings) on lines under it."""
    lines: list[Line] = [[], [(title, "bold")] + ([(f"  {count}", "dim")] if count is not None else [])]
    if not rows:
        return lines + [[("  (none)", "dim")]]
    head, *body = table(header, rows, width)
    lines.append(head)
    for i, line in enumerate(body):
        lines.append(line)
        lines += [[(f"      warning: {note}", "yellow")] for note in (notes[i] if notes else [])]
    return lines


def table(header: list[str], rows: list[list[Cell]], width: int) -> list[Line]:
    """Columns two spaces apart. The last one gets what's left of the width, cut short with …"""
    widths = [max(len(text) for text in column)
              for column in zip(header, *([text for text, _ in row] for row in rows))]
    room = width - 2 - sum(w + 2 for w in widths[:-1])

    def layout(cells: list[Cell]) -> Line:
        line = [("  ", "")]
        for (text, style), w in zip(cells[:-1], widths):
            line += [(text.ljust(w), style), ("  ", "")]
        text, style = cells[-1]
        if len(text) > room:
            text = text[: max(room - 1, 0)] + "…"
        return line + [(text, style)]

    return [layout([(text, "dim") for text in header]), *(layout(row) for row in rows)]


def running_row(r, cfg: Config, now: float) -> list[Cell]:
    over_time = now - r["started_at"] > r["time_limit"]
    over_mem = r["mem"] and r["mem_peak"] and r["mem_peak"] >= r["mem"] * (1 + cfg.warn_step)
    styles = ["bold", "", "red" if over_time else "", "red" if over_mem else "", "", "", "", ""]
    return list(zip(running_cells(r, now, command_width=None), styles))


def queued_row(r, info: dict, now: float) -> list[Cell]:
    styles = ["bold", "", "", "cyan", "", "", "", "", "", ""]
    return list(zip(queued_cells(r, info, now, command_width=None), styles))


def finished_row(r, now: float) -> list[Cell]:
    if r["state"] == "done":
        outcome = "green" if r["returncode"] == 0 else "red"
    else:
        outcome = "yellow" if r["state"] == "cancelled" else "magenta"
    styles = ["", "", outcome, outcome, "", "", "", ""]
    return list(zip(finished_cells(r, now, command_width=None), styles))


# --- terminal ------------------------------------------------------------------------------


def run(cfg: Config, who: str | None, interval: float) -> int:
    conn = db.connect()
    conn.execute("PRAGMA query_only = ON")  # look, don't touch
    psutil.cpu_percent()  # starts the measurement; the first call always returns 0
    try:
        curses.wrapper(_loop, conn, cfg, who, interval)
    except KeyboardInterrupt:
        pass
    return 0


def _loop(screen, conn, cfg: Config, who: str | None, interval: float) -> None:
    styles = _styles()
    try:
        curses.curs_set(0)
    except curses.error:
        pass  # the terminal can't hide the cursor
    curses.set_escdelay(25)
    screen.timeout(round(interval * 1000))
    while True:
        height, width = screen.getmaxyx()
        snap = take_snapshot(conn, cfg, who, time.time())
        draw(screen, render(snap, cfg, width, height, who, interval), styles)
        # Wait for the next refresh. Any key, or a resize, refreshes now.
        if screen.getch() in (ord("q"), ord("Q"), 27):  # 27: Esc
            return


def _styles() -> dict[str, int]:
    styles = {"": curses.A_NORMAL, "bold": curses.A_BOLD, "dim": curses.A_DIM}
    colors = {"red": curses.COLOR_RED, "yellow": curses.COLOR_YELLOW, "green": curses.COLOR_GREEN,
              "cyan": curses.COLOR_CYAN, "magenta": curses.COLOR_MAGENTA}
    if not curses.has_colors():
        return styles | {name: curses.A_BOLD for name in colors}
    try:
        curses.use_default_colors()
        background = -1
    except curses.error:
        background = curses.COLOR_BLACK
    for pair, (name, color) in enumerate(colors.items(), start=1):
        curses.init_pair(pair, color, background)
        styles[name] = curses.color_pair(pair)
    return styles


def draw(screen, lines: list[Line], styles: dict[str, int]) -> None:
    height, width = screen.getmaxyx()
    screen.erase()
    for y, line in enumerate(lines[:height]):
        x = 0
        for text, style in line:
            text = text[: max(width - x, 0)]
            if not text:
                continue
            try:
                screen.addstr(y, x, text, styles[style])
            except curses.error:
                pass  # writing the screen's last cell moves the cursor off it; the text still shows
            x += len(text)
    screen.refresh()
