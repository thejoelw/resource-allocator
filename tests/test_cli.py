"""End-to-end tests: real `alloc` processes sharing a database in a temp directory."""

import json
import os
import signal
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


def test_runs_command_and_passes_through_exit_status(alloc):
    r = alloc.run("run", "--who", "a", "--time", "1m", "--",
                  sys.executable, "-c", "import os, sys; print(os.environ['ALLOC_JOB_ID']); sys.exit(3)")
    assert r.returncode == 3
    assert r.stdout == "1\n"
    assert "[alloc] job 1 started" in r.stderr
    assert "exit 3" in r.stderr
    job = alloc.status(1)
    assert (job["state"], job["returncode"], job["who"]) == ("done", 3, "a")


def test_who_can_come_from_the_environment(alloc):
    alloc.env["ALLOC_WHO"] = "env-agent"
    assert alloc.run("run", "--time", "1m", "true").returncode == 0
    assert alloc.status(1)["who"] == "env-agent"


def test_rejects_requests_bigger_than_the_machine(alloc):
    r = alloc.run("run", "--who", "a", "--mem", "2G", "--time", "1m", "--", "true")
    assert r.returncode == 2
    assert "mem 2G > 1G" in r.stderr


def test_missing_command_reports_127(alloc):
    r = alloc.run("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "no-such-command-xyz")
    assert r.returncode == 127
    assert alloc.status(1)["state"] == "done"
    # The reservation was released.
    assert alloc.run("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "true").returncode == 0


def test_busy_without_wait_prints_estimate_and_runs_nothing(alloc):
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "10m", "--", "sleep", "30")
    alloc.wait_for(1, "running", need_pid=True)

    r = alloc.run("run", "--who", "b", "--mem", "500M", "--time", "1m", "--", "touch", "ran")
    assert r.returncode == 75
    assert "Estimated start in ~10m" in r.stderr
    assert not (alloc.cwd / "ran").exists()
    assert [j["id"] for j in alloc.status()["jobs"]] == [1]  # the attempt didn't stay queued


def test_wait_queues_until_resources_are_released(alloc):
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "sleep", "1")
    alloc.wait_for(1, "running", need_pid=True)

    r = alloc.run("run", "--wait", "--who", "b", "--mem", "500M", "--time", "1m", "--", "true")
    assert r.returncode == 0
    assert "queued behind 0 jobs" in r.stderr
    first, second = alloc.status(1), alloc.status(2)
    assert first["state"] == "done"
    assert second["started_at"] >= first["ended_at"]


def test_smaller_jobs_start_ahead_of_a_queued_job_that_does_not_fit(alloc):
    alloc.start("run", "--who", "a", "--gpu", "100%", "--time", "1m", "--", "sleep", "30")
    alloc.wait_for(1, "running", need_pid=True)
    alloc.start("run", "--wait", "--who", "b", "--gpu", "100%", "--time", "1m", "--", "true")
    alloc.wait_for(2, "pending")

    r = alloc.run("run", "--wait", "--who", "c", "--mem", "100M", "--time", "1m", "--", "true")
    assert r.returncode == 0
    assert alloc.status(2)["state"] == "pending"
    assert alloc.status(2)["estimated_start_in_seconds"] > 50


def test_queue_is_fifo_for_jobs_that_need_the_same_resources(alloc):
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "sleep", "1")
    alloc.wait_for(1, "running", need_pid=True)
    second = alloc.start("run", "--wait", "--who", "b", "--mem", "800M", "--time", "1m", "--", "sleep", "0.5")
    alloc.wait_for(2, "pending")
    third = alloc.start("run", "--wait", "--who", "c", "--mem", "800M", "--time", "1m", "--", "true")
    alloc.wait_for(3, "pending")

    assert second.wait(timeout=10) == 0
    assert third.wait(timeout=10) == 0
    jobs = [alloc.status(i) for i in (1, 2, 3)]
    assert jobs[0]["ended_at"] <= jobs[1]["started_at"]
    assert jobs[1]["ended_at"] <= jobs[2]["started_at"]


def test_command_that_outlives_alloc_keeps_its_reservation_until_it_exits(alloc):
    wrapper = alloc.start("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "sleep", "30")
    pid = alloc.wait_for(1, "running", need_pid=True)["pid"]
    wrapper.kill()
    wrapper.wait()

    assert alloc.run("run", "--who", "b", "--mem", "500M", "--time", "1m", "--", "true").returncode == 75
    assert any("process is gone" in w for w in alloc.status(1)["warnings"])

    # With no `alloc run` left to pass it on, cancel signals the command directly.
    r = alloc.run("cancel", "1")
    assert r.returncode == 0, r.stderr
    assert "job 1 stopped" in r.stderr
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert alloc.status(1)["state"] == "cancelled"
    assert alloc.run("run", "--who", "b", "--mem", "500M", "--time", "1m", "--", "true").returncode == 0


def test_command_that_outlives_alloc_is_marked_lost_when_it_exits(alloc):
    wrapper = alloc.start("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "sleep", "30")
    pid = alloc.wait_for(1, "running", need_pid=True)["pid"]
    wrapper.kill()
    wrapper.wait()
    os.kill(pid, signal.SIGKILL)
    alloc.wait_until(lambda: alloc.status(1)["state"] == "lost")


def test_killed_while_queued_leaves_the_queue(alloc):
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "sleep", "30")
    alloc.wait_for(1, "running", need_pid=True)
    waiter = alloc.start("run", "--wait", "--who", "b", "--mem", "800M", "--time", "1m", "--", "true")
    alloc.wait_for(2, "pending")

    waiter.send_signal(signal.SIGTERM)
    assert waiter.wait(timeout=10) == 128 + signal.SIGTERM
    assert alloc.status(2)["state"] == "cancelled"


def test_cancel_queued_and_running_jobs(alloc):
    holder = alloc.start("run", "--who", "a", "--mem", "800M", "--time", "1m", "--", "sleep", "30")
    alloc.wait_for(1, "running", need_pid=True)
    waiter = alloc.start("run", "--wait", "--who", "b", "--mem", "800M", "--time", "1m", "--", "true")
    alloc.wait_for(2, "pending")

    assert alloc.run("cancel", "2").returncode == 0
    assert waiter.wait(timeout=10) == 1
    r = alloc.run("cancel", "1")
    assert "job 1 stopped (killed by SIGTERM)" in r.stderr
    assert holder.wait(timeout=10) == 128 + signal.SIGTERM
    assert "sending SIGTERM" in holder.communicate()[1]
    assert [alloc.status(i)["state"] for i in (1, 2)] == ["cancelled", "cancelled"]


def test_cancel_force_kills_the_whole_process_tree(alloc):
    holder = alloc.start("run", "--who", "a", "--time", "1m", "--",
                         "sh", "-c", "trap '' TERM; sleep 30 & wait")
    job = alloc.wait_for(1, "running", need_pid=True)
    alloc.wait_until(lambda: len(psutil.Process(job["pid"]).children()) == 1)
    grandchild = psutil.Process(job["pid"]).children()[0]

    r = alloc.run("cancel", "1", "--force")
    assert "killed by SIGKILL" in r.stderr
    assert holder.wait(timeout=10) == 128 + signal.SIGKILL
    alloc.wait_until(lambda: not grandchild.is_running() or grandchild.status() == psutil.STATUS_ZOMBIE)


def test_warns_about_overruns_but_lets_the_job_finish(alloc):
    r = alloc.run("run", "--who", "a", "--mem", "10M", "--time", "1s", "--", sys.executable, "-c",
                  "import time; x = bytearray(100 * 2**20); time.sleep(2)")
    assert r.returncode == 0, r.stderr
    assert "more than its 10M request" in r.stderr
    assert "past its 1s time limit" in r.stderr
    job = alloc.status(1)
    assert job["mem_peak"] > 100 * 2**20
    assert any("more than the 10M request" in w for w in job["warnings"])


def test_usage_charges_actual_run_time(alloc):
    alloc.run("run", "--who", "a", "--gpu", "100%", "--time", "1h", "--", "true")
    usage = json.loads(alloc.run("usage", "--json").stdout)
    assert usage["a"]["jobs"] == 1
    assert usage["a"]["gpu_hours"] < 0.01  # a fraction of a second, not the hour requested


def test_status_text_output(alloc):
    alloc.start("run", "--who", "a", "--mem", "800M", "--time", "5m", "--", "sleep", "30")
    alloc.wait_for(1, "running", need_pid=True)
    alloc.start("run", "--wait", "--who", "b", "--mem", "800M", "--time", "5m", "--", "true")
    alloc.wait_for(2, "pending")

    overview = alloc.run("status").stdout
    assert "mem 800M of 1G" in overview
    assert "sleep 30" in overview
    assert "in ~5m" in overview
    detail = alloc.run("status", "1").stdout
    assert detail.startswith("job 1: running")
