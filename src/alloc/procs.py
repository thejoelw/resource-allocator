"""Process liveness, memory measurement, and signalling."""

import ctypes
import os
import signal
import sys

import psutil


def create_time(pid: int) -> float | None:
    try:
        return psutil.Process(pid).create_time()
    except (psutil.Error, OSError):
        return None


def alive(pid: int | None, ctime: float | None = None) -> bool:
    """Whether `pid` is running and is the same process that started at `ctime`."""
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_ZOMBIE:
            return False
        # A different start time means the pid was reused by an unrelated process.
        return ctime is None or abs(proc.create_time() - ctime) < 1
    except psutil.NoSuchProcess:
        return False
    except (psutil.Error, OSError):
        # Can't inspect it (e.g. inside a sandbox): fall back to a plain existence check.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True


if sys.platform == "darwin":

    class _RUsageInfoV0(ctypes.Structure):
        _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
            (name, ctypes.c_uint64)
            for name in (
                "user_time",
                "system_time",
                "pkg_idle_wkups",
                "interrupt_wkups",
                "pageins",
                "wired_size",
                "resident_size",
                "phys_footprint",
                "proc_start_abstime",
                "proc_exit_abstime",
            )
        ]

    _libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    _libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(_RUsageInfoV0)]
    _libc.proc_pid_rusage.restype = ctypes.c_int

    def _footprint(pid: int) -> int | None:
        info = _RUsageInfoV0()
        if _libc.proc_pid_rusage(pid, 0, ctypes.byref(info)) != 0:  # 0 = RUSAGE_INFO_V0
            return None
        return info.phys_footprint

else:

    def _footprint(pid: int) -> int | None:
        return None


def tree_memory(pid: int) -> int | None:
    """Memory used by `pid` and all its descendants, in bytes.

    On macOS this is the physical footprint (Activity Monitor's "Memory" column), which,
    unlike RSS, includes GPU allocations. Elsewhere it falls back to RSS.
    """
    try:
        root = psutil.Process(pid)
        tree = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return None
    total, measured = 0, False
    for proc in tree:
        size = _footprint(proc.pid)
        if size is None:
            try:
                size = proc.memory_info().rss
            except (psutil.Error, OSError):
                continue
        total += size
        measured = True
    return total if measured else None


def send_signal(pid: int | None, ctime: float | None, sig: int) -> bool:
    if not alive(pid, ctime):
        return False
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        return False
    return True


def kill_tree(pid: int | None, ctime: float | None) -> bool:
    """SIGKILL `pid` and all its descendants. Raises PermissionError if not allowed to."""
    if not alive(pid, ctime):
        return False
    try:
        root = psutil.Process(pid)
        tree = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return send_signal(pid, ctime, signal.SIGKILL)
    for proc in tree:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as e:
            raise PermissionError(str(e)) from e
    return True
