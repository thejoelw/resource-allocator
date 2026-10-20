"""Shared state in SQLite: the queue, running reservations, and job history.

There's no daemon. Every `alloc` process opens the same database and makes scheduling
decisions inside an exclusive transaction, so they never race.
"""

import fcntl
import sqlite3
from contextlib import contextmanager

from . import procs
from .config import Config, home
from .sched import Res, first_fit, predict_starts

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id               INTEGER PRIMARY KEY,  -- not AUTOINCREMENT: a refused `run` frees its id
    who              TEXT NOT NULL,
    command          TEXT NOT NULL,  -- JSON array
    cwd              TEXT,
    mem              REAL NOT NULL,  -- bytes
    cpu              REAL NOT NULL,  -- cores
    gpu              REAL NOT NULL,  -- GPUs
    disk             REAL NOT NULL,  -- bytes
    time_limit       REAL NOT NULL,  -- seconds
    state            TEXT NOT NULL,  -- pending | running | done | cancelled | lost
    submitted_at     REAL NOT NULL,
    started_at       REAL,
    ended_at         REAL,
    seen_at          REAL,           -- last time the job was known to be alive
    wrapper_pid      INTEGER,        -- the `alloc run` process
    wrapper_ctime    REAL,
    child_pid        INTEGER,        -- the command it runs
    child_ctime      REAL,
    returncode       INTEGER,        -- negative: killed by that signal
    mem_now          REAL,
    mem_peak         REAL,
    cancel_requested INTEGER NOT NULL DEFAULT 0  -- signal `alloc cancel` asked for (0: none)
);
CREATE INDEX IF NOT EXISTS jobs_state ON jobs (state);
"""


def connect() -> sqlite3.Connection:
    path = home()
    path.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path / "alloc.db", timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # Switching to WAL fails rather than waits if another process is setting up the database
    # at the same moment, so serialize setup with a lock file. WAL mode persists in the file,
    # so after the first time this only reads it.
    with open(path / "setup.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if conn.execute("PRAGMA journal_mode").fetchone()[0] != "wal":
            conn.execute("PRAGMA journal_mode = WAL")
        with transaction(conn):
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def res(row) -> Res:
    return Res(row["mem"], row["cpu"], row["gpu"], row["disk"])


def get(conn, job_id: int):
    return conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()


def jobs_in(conn, state: str) -> list:
    return conn.execute("SELECT * FROM jobs WHERE state = ? ORDER BY id", (state,)).fetchall()


def in_use(conn) -> Res:
    return sum((res(row) for row in jobs_in(conn, "running")), Res())


def reap(conn, now: float) -> None:
    """Close out jobs whose processes died without recording it (e.g. `alloc` was SIGKILLed)."""
    for row in conn.execute("SELECT * FROM jobs WHERE state IN ('pending', 'running')").fetchall():
        if procs.alive(row["wrapper_pid"], row["wrapper_ctime"]):
            continue
        if row["state"] == "pending":
            conn.execute("UPDATE jobs SET state = 'lost', ended_at = ? WHERE id = ?", (now, row["id"]))
        elif procs.alive(row["child_pid"], row["child_ctime"]):
            # The command outlived its `alloc run` process, so it keeps its reservation.
            conn.execute("UPDATE jobs SET seen_at = ? WHERE id = ?", (now, row["id"]))
        else:
            conn.execute(
                "UPDATE jobs SET state = ?, ended_at = COALESCE(seen_at, started_at, ?) WHERE id = ?",
                ("cancelled" if row["cancel_requested"] else "lost", now, row["id"]),
            )


def schedule(conn, cfg: Config, now: float) -> None:
    """Reap dead jobs, then admit queued jobs in FIFO order, skipping any that don't fit yet."""
    reap(conn, now)
    pending = jobs_in(conn, "pending")
    admitted, _ = first_fit(in_use(conn), cfg.limit, [(row["id"], res(row)) for row in pending])
    conn.executemany(
        "UPDATE jobs SET state = 'running', started_at = ? WHERE id = ?",
        [(now, job_id) for job_id in admitted],
    )


def predicted_starts(conn, cfg: Config, now: float) -> dict[int, float | None]:
    """Estimated start time of each queued job, assuming every job runs its full time limit."""
    running = [(res(row), row["started_at"] + row["time_limit"]) for row in jobs_in(conn, "running")]
    pending = jobs_in(conn, "pending")
    starts = predict_starts(now, cfg.limit, running, [(res(row), row["time_limit"]) for row in pending])
    return {row["id"]: start for row, start in zip(pending, starts)}


def count_overdue(conn, now: float) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM jobs WHERE state = 'running' AND started_at + time_limit < ?", (now,)
    ).fetchone()[0]


def count_queued(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM jobs WHERE state = 'pending'").fetchone()[0]
