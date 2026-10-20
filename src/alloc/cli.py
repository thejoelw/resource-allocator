"""The `alloc` command line: run, status, cancel, usage."""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime

from . import db, procs
from .config import Config, home, load as load_config
from .sched import RESOURCES, Res
from .units import GiB, fmt_duration, fmt_size, parse_duration, parse_share, parse_size

EXIT_USAGE = 2
EXIT_UNAVAILABLE = 75  # EX_TEMPFAIL: the resources are busy, try again later

DESCRIPTION = """\
Reserve memory, CPU, GPU and disk on this machine, run a command, and release the
reservation when the command exits, however it exits.

examples:
  alloc run --who agent-7 --mem 10G --cpu 50% --gpu 100% --time 15m -- python train.py
  alloc run --wait --who agent-7 --mem 10G --time 2h -- python train.py > train.log 2>&1 &
  alloc status              what's running and queued, and when queued jobs should start
  alloc status 42           one job: memory use, time, warnings, exit status
  alloc cancel 42           stop a job (SIGTERM) or take it out of the queue
  alloc usage --since 1d    resources used per agent
"""

RUN_DESCRIPTION = """\
Reserve resources and run COMMAND. The reservation is released when COMMAND exits.

Without --wait, COMMAND starts now if the resources are free. Otherwise alloc prints an
estimated start time and exits with status 75 without running anything. With --wait,
the job joins a FIFO queue and starts as soon as it fits; a job further back may start
first if it fits in what's free.

Once COMMAND runs, alloc exits with COMMAND's exit status. alloc writes lines starting
with "[alloc]" to stderr: the job ID when it starts, a warning when the job uses 20%
more memory than requested or runs past --time (repeated as it grows), and a summary
when it exits. alloc never stops a job for exceeding its request; that's your call.
"""

RUN_EPILOG = """\
amounts:
  --mem, --disk   sizes like 10G or 512M. On Apple Silicon the GPU shares system
                  memory, so --mem must include GPU memory too.
  --cpu           cores (4) or a share of the machine (50%)
  --gpu           a share of the GPU (100%, 50%)
  --time          15m, 1h30m, 90s; a bare number is minutes. Only used to estimate
                  start times for queued jobs; nothing is killed when it runs out.
  Resources you leave out default to 0.

long runs:
  If your shell commands time out, run alloc in the background and check on it:
    alloc run --wait --who agent-7 --mem 10G --time 2h -- python train.py > train.log 2>&1 &
    alloc status --who agent-7
"""


class UsageError(Exception):
    pass


def say(message: str) -> None:
    print(f"[alloc] {message}", file=sys.stderr, flush=True)


def plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def fmt_amount(name: str, value: float) -> str:
    return fmt_size(value) if name in ("mem", "disk") else f"{value:g}"


def fmt_res(r: Res) -> str:
    parts = [f"{name} {fmt_amount(name, v)}" for name, v in zip(RESOURCES, r) if v]
    return ", ".join(parts) or "no resources"


def fmt_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return "within a minute"
    return "in ~" + fmt_duration(round(seconds / 60) * 60)


def fmt_exit(returncode: int | None) -> str:
    if returncode is None:
        return "no exit status"
    if returncode < 0:
        try:
            return f"killed by {signal.Signals(-returncode).name}"
        except ValueError:
            return f"killed by signal {-returncode}"
    return f"exit {returncode}"


def fmt_time(t: float | None) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else "-"


def iso(t: float | None) -> str | None:
    return datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds") if t else None


def waiting_text(queued: int) -> str:
    return f"{plural(queued, 'job')} waiting in the queue" if queued else "no jobs waiting"


# --- run -----------------------------------------------------------------------------------


def parse_request(args, cfg: Config) -> tuple[Res, float]:
    try:
        req = Res(
            mem=parse_size(args.mem) if args.mem else 0,
            cpu=parse_share(args.cpu, cfg.capacity.cpu) if args.cpu else 0,
            gpu=parse_share(args.gpu, cfg.capacity.gpu) if args.gpu else 0,
            disk=parse_size(args.disk) if args.disk else 0,
        )
        time_limit = parse_duration(args.time)
    except ValueError as e:
        raise UsageError(str(e)) from None
    if time_limit <= 0:
        raise UsageError("--time must be more than zero")
    too_big = [
        f"{name} {fmt_amount(name, want)} > {fmt_amount(name, have)}"
        for name, want, have in zip(RESOURCES, req, cfg.limit)
        if not Res(**{name: want}).fits(Res(**{name: have}))
    ]
    if too_big:
        raise UsageError(
            f"request is more than this machine offers jobs ({', '.join(too_big)}); "
            f"capacity is set in {home() / 'config.toml'}"
        )
    return req, time_limit


def cmd_run(args, cfg: Config) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise UsageError("no command given (usage: alloc run [options] -- COMMAND [ARGS...])")
    who = args.who or os.environ.get("ALLOC_WHO")
    if not who:
        raise UsageError("--who is required (or set ALLOC_WHO)")
    req, time_limit = parse_request(args, cfg)

    conn = db.connect()
    now = time.time()
    with db.transaction(conn):
        job_id = conn.execute(
            "INSERT INTO jobs (who, command, cwd, mem, cpu, gpu, disk, time_limit, state,"
            " submitted_at, wrapper_pid, wrapper_ctime)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)",
            (who, json.dumps(command), os.getcwd(), *req, time_limit, now,
             os.getpid(), procs.create_time(os.getpid())),
        ).lastrowid
        db.schedule(conn, cfg, now)
        admitted = db.get(conn, job_id)["state"] == "running"
        if not admitted:
            ahead = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE state = 'pending' AND id < ?", (job_id,)
            ).fetchone()[0]
            start = db.predicted_starts(conn, cfg, now)[job_id]
            overdue = db.count_overdue(conn, now)
            if not args.wait:
                conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    if not admitted:
        eta = fmt_eta(None if start is None else start - now)
        if overdue:
            eta += f" (rough guess: {plural(overdue, 'running job')} already past the time limit)"
        if not args.wait:
            say(f"not available now. Estimated start {eta}, {plural(ahead, 'job')} queued ahead. "
                "Try again later, or pass --wait to join the queue.")
            return EXIT_UNAVAILABLE
        say(f"job {job_id} queued behind {plural(ahead, 'job')}, estimated start {eta}. Waiting...")
        if not wait_for_turn(conn, cfg, job_id):
            return 1
    return supervise(conn, cfg, job_id, req, time_limit, command)


def _exit_on_signal(signum, _frame):
    raise SystemExit(128 + signum)


def wait_for_turn(conn, cfg: Config, job_id: int) -> bool:
    """Poll the queue until the job is admitted. Returns False if it was cancelled meanwhile."""
    signal.signal(signal.SIGTERM, _exit_on_signal)
    signal.signal(signal.SIGHUP, _exit_on_signal)
    try:
        while True:
            time.sleep(cfg.poll_interval)
            with db.transaction(conn):
                db.schedule(conn, cfg, time.time())
                state = db.get(conn, job_id)["state"]
            if state == "running":
                return True
            if state != "pending":
                say(f"job {job_id} was {state} while queued")
                return False
    except BaseException:
        # Interrupted: leave the queue, or give back the reservation if it was just granted.
        with db.transaction(conn):
            conn.execute(
                "UPDATE jobs SET state = 'cancelled', ended_at = ?"
                " WHERE id = ? AND state IN ('pending', 'running')",
                (time.time(), job_id),
            )
        raise


def supervise(conn, cfg: Config, job_id: int, req: Res, time_limit: float, command: list[str]) -> int:
    """Run the command, report on it while it runs, and release the reservation when it exits."""
    if db.get(conn, job_id)["cancel_requested"]:
        finish(conn, cfg, job_id, None)
        say(f"job {job_id} was cancelled before it started")
        return 1
    # Announce before starting, so the job ID comes first in the log even if the command is quick.
    say(f"job {job_id} started ({fmt_res(req)}, time {fmt_duration(time_limit)}). "
        f"Check on it with: alloc status {job_id}")
    try:
        child = subprocess.Popen(command, env={**os.environ, "ALLOC_JOB_ID": str(job_id)})
    except OSError as e:
        finish(conn, cfg, job_id, 127)
        say(f"could not run {command[0]}: {e.strerror}")
        return 127
    started = time.time()
    # Ctrl-C reaches the command directly through the process group. Other signals sent to
    # alloc are passed on, so the command can shut down cleanly and we still record its exit.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, lambda signum, _frame: child.send_signal(signum))
    with db.transaction(conn):
        conn.execute(
            "UPDATE jobs SET child_pid = ?, child_ctime = ?, started_at = ?, seen_at = ? WHERE id = ?",
            (child.pid, procs.create_time(child.pid), started, started, job_id),
        )

    peak = 0
    next_mem_warning = req.mem * (1 + cfg.warn_step)
    next_time_warning = time_limit
    cancel_sent = 0
    while True:
        try:
            returncode = child.wait(timeout=cfg.monitor_interval)
            break
        except subprocess.TimeoutExpired:
            pass
        now = time.time()
        mem = procs.tree_memory(child.pid)
        peak = max(peak, mem or 0)
        with db.transaction(conn):
            conn.execute(
                "UPDATE jobs SET mem_now = ?, mem_peak = ?, seen_at = ? WHERE id = ?",
                (mem, peak or None, now, job_id),
            )
            queued = db.count_queued(conn)
            cancel = db.get(conn, job_id)["cancel_requested"]
        if cancel and cancel != cancel_sent:
            # `alloc cancel` asks us to send the signal: a sandbox may not let it signal our child.
            cancel_sent = cancel
            say(f"job {job_id} cancelled with alloc cancel; sending {signal.Signals(cancel).name}")
            if cancel == signal.SIGKILL:
                procs.kill_tree(child.pid, None)
            else:
                child.send_signal(cancel)
        if cancel_sent:
            continue  # no point warning about a job that's being stopped
        if req.mem and peak >= next_mem_warning:
            next_mem_warning = peak * (1 + cfg.warn_step)
            say(f"warning: job {job_id} has used {fmt_size(peak)} of memory, "
                f"{peak / req.mem - 1:.0%} more than its {fmt_size(req.mem)} request ({waiting_text(queued)})")
        elapsed = now - started
        if elapsed >= next_time_warning:
            next_time_warning = elapsed * (1 + cfg.warn_step)
            say(f"warning: job {job_id} has run for {fmt_duration(elapsed)}, past its "
                f"{fmt_duration(time_limit)} time limit ({waiting_text(queued)})")

    finish(conn, cfg, job_id, returncode, peak or None)
    summary = (f"job {job_id} finished: {fmt_exit(returncode)} after {fmt_duration(time.time() - started)}"
               f" (time limit {fmt_duration(time_limit)})")
    if peak:
        summary += f", peak memory {fmt_size(peak)} (requested {fmt_size(req.mem)})"
    say(summary)
    return returncode if returncode >= 0 else 128 - returncode


def finish(conn, cfg: Config, job_id: int, returncode: int | None, peak: float | None = None) -> None:
    """Record how the job ended and hand its resources to the queue."""
    now = time.time()
    with db.transaction(conn):
        conn.execute(
            "UPDATE jobs SET state = CASE WHEN cancel_requested THEN 'cancelled' ELSE 'done' END,"
            " ended_at = ?, seen_at = ?, returncode = ?, mem_now = NULL,"
            " mem_peak = COALESCE(?, mem_peak) WHERE id = ?",
            (now, now, returncode, peak, job_id),
        )
        db.schedule(conn, cfg, now)


# --- status --------------------------------------------------------------------------------


def json_amounts(r: Res) -> dict:
    """Resource amounts for JSON output, with sizes as whole bytes."""
    return {name: int(v) if name in ("mem", "disk") else v for name, v in zip(RESOURCES, r)}


def job_warnings(row, cfg: Config, now: float, queued: int) -> list[str]:
    warnings = []
    mem, peak = row["mem"], row["mem_peak"]
    if mem and peak and peak >= mem * (1 + cfg.warn_step):
        warnings.append(f"memory peaked at {fmt_size(peak)}, {peak / mem - 1:.0%} more than "
                        f"the {fmt_size(mem)} request")
    if row["state"] == "running":
        elapsed = now - row["started_at"]
        if elapsed > row["time_limit"]:
            warnings.append(f"running {fmt_duration(elapsed - row['time_limit'])} past its "
                            f"{fmt_duration(row['time_limit'])} time limit ({waiting_text(queued)})")
        if row["child_pid"] and not procs.alive(row["wrapper_pid"], row["wrapper_ctime"]):
            warnings.append("its `alloc run` process is gone, so memory isn't being tracked; "
                            "the command is still running and keeps its reservation")
    return warnings


def job_info(row, cfg: Config, now: float, starts: dict, queued: int) -> dict:
    started, ended = row["started_at"], row["ended_at"]
    start = starts.get(row["id"])
    return {
        "id": row["id"],
        "who": row["who"],
        "state": row["state"],
        "command": json.loads(row["command"]),
        "cwd": row["cwd"],
        "pid": row["child_pid"],
        "requested": {**json_amounts(db.res(row)), "time_limit": row["time_limit"]},
        "submitted_at": iso(row["submitted_at"]),
        "started_at": iso(started),
        "ended_at": iso(ended),
        "elapsed_seconds": None if started is None else round((ended or now) - started, 1),
        "estimated_start_in_seconds": None if start is None else round(max(start - now, 0)),
        "mem_now": int(row["mem_now"]) if row["state"] == "running" and row["mem_now"] else None,
        "mem_peak": int(row["mem_peak"]) if row["mem_peak"] else None,
        "returncode": row["returncode"],
        "warnings": job_warnings(row, cfg, now, queued),
    }


def print_table(title: str, header: list[str], rows: list[list[str]], first: bool = False) -> None:
    print(title if first else f"\n{title}")
    if not rows:
        print("  (none)")
        return
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows)]
    for row in (header, *rows):
        print("  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())


def short_command(row, width: int = 50) -> str:
    text = shlex.join(json.loads(row["command"]))
    return text if len(text) <= width else text[: width - 1] + "…"


def print_overview(cfg: Config, used: Res, rows: list, infos: dict, now: float) -> None:
    print("in use:  " + "   ".join(
        f"{name} {fmt_amount(name, u)} of {fmt_amount(name, limit)}"
        for name, u, limit in zip(RESOURCES, used, cfg.limit)
    ))
    running = [r for r in rows if r["state"] == "running"]
    pending = [r for r in rows if r["state"] == "pending"]
    finished = [r for r in rows if r["state"] not in ("running", "pending")]

    print_table("running", ["ID", "WHO", "ELAPSED", "MEM NOW/PEAK/REQ", "CPU", "GPU", "DISK", "COMMAND", "NOTES"], [
        [
            str(r["id"]), r["who"],
            f"{fmt_duration(now - r['started_at'])}/{fmt_duration(r['time_limit'])}",
            "/".join(fmt_size(v) if v else "-" for v in (r["mem_now"], r["mem_peak"], r["mem"])),
            f"{r['cpu']:g}", f"{r['gpu']:g}", fmt_size(r["disk"]), short_command(r),
            "WARNING (see alloc status ID)" if infos[r["id"]]["warnings"] else "",
        ]
        for r in running
    ])
    print_table("queued", ["ID", "WHO", "WAITED", "EST. START", "MEM", "CPU", "GPU", "DISK", "TIME", "COMMAND"], [
        [
            str(r["id"]), r["who"], fmt_duration(now - r["submitted_at"]),
            fmt_eta(infos[r["id"]]["estimated_start_in_seconds"]),
            fmt_size(r["mem"]), f"{r['cpu']:g}", f"{r['gpu']:g}", fmt_size(r["disk"]),
            fmt_duration(r["time_limit"]), short_command(r),
        ]
        for r in pending
    ])
    print_table("recently finished", ["ID", "WHO", "STATE", "RESULT", "RAN", "PEAK MEM", "ENDED", "COMMAND"], [
        [
            str(r["id"]), r["who"], r["state"], fmt_exit(r["returncode"]),
            fmt_duration(r["ended_at"] - r["started_at"]) if r["started_at"] else "-",
            fmt_size(r["mem_peak"]) if r["mem_peak"] else "-",
            fmt_duration(now - r["ended_at"]) + " ago", short_command(r),
        ]
        for r in finished
    ])


def print_job(row, info: dict, now: float) -> None:
    state = row["state"]
    lines = [
        ("who", row["who"]),
        ("command", shlex.join(info["command"])),
        ("cwd", row["cwd"]),
        ("pid", str(row["child_pid"] or "-")),
        ("requested", f"{fmt_res(db.res(row))}, time {fmt_duration(row['time_limit'])}"),
        ("submitted", fmt_time(row["submitted_at"])),
    ]
    if state == "pending":
        lines.append(("est. start", fmt_eta(info["estimated_start_in_seconds"])))
    if row["started_at"]:
        waited = fmt_duration(row["started_at"] - row["submitted_at"])
        lines.append(("started", f"{fmt_time(row['started_at'])} (waited {waited})"))
        lines.append(("elapsed", f"{fmt_duration(info['elapsed_seconds'])} of "
                                 f"{fmt_duration(row['time_limit'])} time limit"))
    if row["ended_at"]:
        lines.append(("ended", fmt_time(row["ended_at"])))
    if state in ("done", "cancelled", "lost") and row["started_at"]:
        lines.append(("result", fmt_exit(row["returncode"])))
    if row["mem_now"] or row["mem_peak"]:
        now_text = f"{fmt_size(row['mem_now'])} now, " if state == "running" and row["mem_now"] else ""
        lines.append(("memory", f"{now_text}{fmt_size(row['mem_peak'] or 0)} peak"))
    for warning in info["warnings"]:
        lines.append(("WARNING", warning))
    print(f"job {row['id']}: {state}")
    for label, value in lines:
        print(f"  {label:<11}{value}")


def cmd_status(args, cfg: Config) -> int:
    conn = db.connect()
    now = time.time()
    with db.transaction(conn):
        db.schedule(conn, cfg, now)  # reap dead jobs first so the answer is current
        starts = db.predicted_starts(conn, cfg, now)
        queued = db.count_queued(conn)
        used = db.in_use(conn)
        if args.job_id is not None:
            rows = [db.get(conn, args.job_id)]
            if rows[0] is None:
                raise UsageError(f"no job {args.job_id}")
        else:
            who = "" if args.who is None else " AND who = :who"
            rows = conn.execute(
                f"SELECT * FROM jobs WHERE state IN ('running', 'pending'){who} ORDER BY id",
                {"who": args.who},
            ).fetchall()
            rows += conn.execute(
                f"SELECT * FROM jobs WHERE state NOT IN ('running', 'pending'){who}"
                " AND ended_at > :since ORDER BY ended_at DESC LIMIT 10",
                {"who": args.who, "since": now - 86400},
            ).fetchall()
    infos = {row["id"]: job_info(row, cfg, now, starts, queued) for row in rows}

    if args.json:
        if args.job_id is not None:
            output = infos[args.job_id]
        else:
            output = {
                "in_use": json_amounts(used),
                "available_to_jobs": json_amounts(cfg.limit),
                "jobs": list(infos.values()),
            }
        print(json.dumps(output, indent=2))
    elif args.job_id is not None:
        print_job(rows[0], infos[args.job_id], now)
    else:
        print_overview(cfg, used, rows, infos, now)
    return 0


# --- cancel --------------------------------------------------------------------------------


def cmd_cancel(args, cfg: Config) -> int:
    sig = signal.SIGKILL if args.force else signal.SIGTERM
    conn = db.connect()
    now = time.time()
    with db.transaction(conn):
        db.reap(conn, now)
        row = db.get(conn, args.job_id)
        if row is None:
            raise UsageError(f"no job {args.job_id}")
        if row["state"] == "pending":
            conn.execute("UPDATE jobs SET state = 'cancelled', ended_at = ? WHERE id = ?", (now, row["id"]))
        elif row["state"] == "running":
            conn.execute("UPDATE jobs SET cancel_requested = ? WHERE id = ?", (int(sig), row["id"]))

    job_id, pid = row["id"], row["child_pid"]
    if row["state"] == "pending":
        say(f"job {job_id} removed from the queue")
        return 0
    if row["state"] != "running":
        say(f"job {job_id} already ended ({row['state']})")
        return 0
    if pid and not procs.alive(row["wrapper_pid"], row["wrapper_ctime"]):
        # Nobody is supervising the command any more, so signal it directly.
        try:
            if args.force:
                procs.kill_tree(pid, row["child_ctime"])
            else:
                procs.send_signal(pid, row["child_ctime"], sig)
        except PermissionError:
            say(f"not allowed to signal job {job_id} (pid {pid}), probably because of a sandbox. "
                f"Its `alloc run` process is gone; kill pid {pid} from where it was started.")
            return 1
    # Otherwise the job's `alloc run` process sends the signal on its next check.
    deadline = time.time() + cfg.monitor_interval + 10
    while time.time() < deadline:
        with db.transaction(conn):
            db.schedule(conn, cfg, time.time())
            row = db.get(conn, job_id)
        if row["state"] != "running":
            result = "" if row["returncode"] is None else f" ({fmt_exit(row['returncode'])})"
            say(f"job {job_id} stopped{result}")
            return 0
        time.sleep(0.2)
    say(f"sent {sig.name} to job {job_id}, but it's still shutting down. Check with "
        f"`alloc status {job_id}`" + ("." if args.force else ", or use --force to kill it."))
    return 0


# --- usage ---------------------------------------------------------------------------------


def cmd_usage(args, cfg: Config) -> int:
    try:
        window = parse_duration(args.since)
    except ValueError as e:
        raise UsageError(str(e)) from None
    conn = db.connect()
    now = time.time()
    since = now - window
    rows = conn.execute(
        "SELECT * FROM jobs WHERE started_at IS NOT NULL AND COALESCE(ended_at, :now) > :since",
        {"now": now, "since": since},
    ).fetchall()
    totals: dict[str, dict] = {}
    for row in rows:
        # Charge actual run time (within the window), not the requested time limit.
        hours = max(0.0, (row["ended_at"] or now) - max(row["started_at"], since)) / 3600
        t = totals.setdefault(row["who"], {"jobs": 0, "hours": 0.0, "mem_gb_hours": 0.0,
                                            "cpu_core_hours": 0.0, "gpu_hours": 0.0})
        t["jobs"] += 1
        t["hours"] += hours
        t["mem_gb_hours"] += row["mem"] / GiB * hours
        t["cpu_core_hours"] += row["cpu"] * hours
        t["gpu_hours"] += row["gpu"] * hours

    ranked = sorted(totals.items(), key=lambda item: -item[1]["hours"])
    if args.json:
        print(json.dumps({who: {k: round(v, 3) for k, v in t.items()} for who, t in ranked}, indent=2))
        return 0
    print_table(f"usage over the last {fmt_duration(window)} (actual run time x requested resources)",
                ["WHO", "JOBS", "RUN TIME", "MEM GB·H", "CPU CORE·H", "GPU·H"], [
        [who, str(t["jobs"]), fmt_duration(t["hours"] * 3600), f"{t['mem_gb_hours']:.1f}",
         f"{t['cpu_core_hours']:.1f}", f"{t['gpu_hours']:.2f}"]
        for who, t in ranked
    ], first=True)
    return 0


# --- entry point ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    raw = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(prog="alloc", description=DESCRIPTION, formatter_class=raw)
    sub = parser.add_subparsers(metavar="COMMAND", required=True)

    run = sub.add_parser("run", help="reserve resources and run a command",
                         description=RUN_DESCRIPTION, epilog=RUN_EPILOG, formatter_class=raw)
    run.add_argument("--who", help="your name or agent ID (default: $ALLOC_WHO)")
    run.add_argument("--mem", metavar="SIZE", help="memory, including GPU memory (e.g. 10G)")
    run.add_argument("--cpu", metavar="AMOUNT", help="cores (4) or share of the machine (50%%)")
    run.add_argument("--gpu", metavar="AMOUNT", help="share of the GPU (100%%, 50%%)")
    run.add_argument("--disk", metavar="SIZE", help="disk space the job will write (e.g. 20G)")
    run.add_argument("--time", required=True, metavar="DURATION",
                     help="how long you expect it to run (15m, 1h30m)")
    run.add_argument("--wait", action="store_true",
                     help="if the resources are busy, queue and block until they're free")
    run.add_argument("command", nargs=argparse.REMAINDER, help="the command to run, after --")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="show running, queued and recent jobs, or one job",
                            description="Show running, queued and recently finished jobs, or one job in detail.")
    status.add_argument("job_id", nargs="?", type=int, help="show just this job")
    status.add_argument("--who", help="only show jobs from this agent")
    status.add_argument("--json", action="store_true", help="machine-readable output (sizes in bytes)")
    status.set_defaults(func=cmd_status)

    cancel = sub.add_parser("cancel", help="stop a running job or remove a queued one",
                            description="Send SIGTERM to a running job, or remove a queued one.")
    cancel.add_argument("job_id", type=int)
    cancel.add_argument("--force", action="store_true",
                        help="SIGKILL the job and its child processes instead")
    cancel.set_defaults(func=cmd_cancel)

    usage = sub.add_parser("usage", help="resources used per agent",
                           description="Resources used per agent: actual run time x requested resources.")
    usage.add_argument("--since", default="7d", metavar="DURATION", help="time window (default: 7d)")
    usage.add_argument("--json", action="store_true", help="machine-readable output")
    usage.set_defaults(func=cmd_usage)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config()
    except ValueError as e:
        say(f"error: {e}")
        sys.exit(EXIT_USAGE)
    try:
        sys.exit(args.func(args, cfg))
    except UsageError as e:
        say(f"error: {e}")
        sys.exit(EXIT_USAGE)
    except KeyboardInterrupt:
        sys.exit(130)
