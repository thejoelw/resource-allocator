"""The `alloc` fixture: real `alloc` processes sharing a database in a temp directory."""

import json
import os
import subprocess
import sys
import time

import psutil
import pytest

CONFIG = """
poll_interval = 0.1
monitor_interval = 0.1

[capacity]
mem = "1G"
cpu = 4
gpu = 1
disk = "10G"

[overcommit]
cpu = 1
gpu = 1
"""


class Alloc:
    def __init__(self, tmp_path):
        home = tmp_path / "alloc-home"
        home.mkdir()
        (home / "config.toml").write_text(CONFIG)
        self.home = home
        self.cwd = tmp_path
        self.env = {**os.environ, "ALLOC_HOME": str(home)}
        self.env.pop("ALLOC_WHO", None)
        self.background: list[subprocess.Popen] = []

    def run(self, *args) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, "-m", "alloc", *args], env=self.env, cwd=self.cwd,
                              capture_output=True, text=True, timeout=30)

    def start(self, *args) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-m", "alloc", *args], env=self.env, cwd=self.cwd,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.background.append(proc)
        return proc

    def status(self, job_id=None) -> dict:
        args = ["status", "--json"] + ([str(job_id)] if job_id is not None else [])
        result = self.run(*args)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)

    def wait_until(self, predicate, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        raise AssertionError("timed out waiting for condition")

    def wait_for(self, job_id, state, need_pid=False):
        def check():
            jobs = {j["id"]: j for j in self.status()["jobs"]}
            job = jobs.get(job_id)
            return job is not None and job["state"] == state and (job["pid"] or not need_pid)
        self.wait_until(check)
        return self.status(job_id)


@pytest.fixture
def alloc(tmp_path):
    a = Alloc(tmp_path)
    yield a
    for proc in a.background:
        if proc.poll() is None:
            for child in psutil.Process(proc.pid).children(recursive=True):
                child.kill()
            proc.kill()
        proc.communicate(timeout=10)
