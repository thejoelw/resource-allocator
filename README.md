# alloc

Reserve memory, CPU, GPU and disk on a shared machine, run a command, and release the
reservation when the command exits. Built for ML training agents sharing one box.

```
alloc run --who agent-7 --mem 10G --cpu 50% --gpu 100% --time 15m -- python train.py
```

- If the resources are free, the command runs and `alloc` exits with its exit status.
- If not, `alloc` prints an estimated start time and exits with status 75, or with
  `--wait` it joins a queue and blocks until the job fits.
- The reservation is released when the command exits, however it exits, so a run that
  crashes after one second only holds resources for one second.

## Install

```
uv tool install --editable .
```

This puts `alloc` on your PATH regardless of which virtualenv an agent is in.

## Commands

| Command | What it does |
|---|---|
| `alloc run [--wait] --who ID --time T [--mem] [--cpu] [--gpu] [--disk] -- CMD...` | reserve and run |
| `alloc status [JOB_ID] [--who ID] [--json]` | running, queued and recently finished jobs, or one job |
| `alloc watch [--who ID] [-n SECONDS]` | the same, live in your terminal, plus what's reserved and measured (q quits) |
| `alloc cancel JOB_ID [--force]` | SIGTERM a running job (`--force`: SIGKILL its process tree), or remove a queued one |
| `alloc usage [--since 7d] [--json]` | per-agent usage: actual run time × requested resources |

`alloc <command> --help` has the details.

Exit statuses of `alloc run`: the command's own status once it runs (128+N if it was
killed by signal N), `75` if the resources are busy and `--wait` wasn't given, `127` if
the command couldn't be started, `2` for bad arguments, `1` if a queued job was cancelled.

## How it works

**Scheduling.** Jobs that can't start immediately wait in a FIFO queue. Whenever
something changes, alloc walks the queue in order and starts every job that fits in the
resources not yet reserved, so a small job can start ahead of a big one that's still
waiting. A job "fits" if, for every resource, what's reserved plus the request is within
capacity × overcommit.

**Estimates** assume every job runs exactly its `--time`. They're guesses: jobs that
finish early make them pessimistic, jobs that overrun make them optimistic.

**No enforcement.** alloc never kills a job for using more than it asked for. It writes
`[alloc] warning:` lines to the job's stderr when the job's memory goes 20% over its
request or the job runs past `--time` (repeated as the overrun grows, along with how many
jobs are waiting), and `alloc status` shows the same warnings. Deciding what to do about
it is up to the agent.

**Memory** is measured as the physical footprint of the command's whole process tree,
the number Activity Monitor shows. On Apple Silicon this includes GPU (MPS) memory, which
comes out of the same pool as everything else, so `--mem` should cover both.

**No daemon.** State lives in SQLite at `~/.alloc/alloc.db` (override with `$ALLOC_HOME`).
Each `alloc run` process supervises its own command; waiting jobs poll the database once
a second. If an `alloc run` process is killed outright, the next `alloc` invocation
notices: a queued job is dropped, and a running job keeps its reservation until its
command exits too. The exception is `alloc watch`, which only reads the database, so
leaving it open never affects jobs.

## Configuration

Optional, at `~/.alloc/config.toml`. Defaults shown:

```toml
warn_step = 0.2          # warn at 20% over the memory request, then each time it grows 20% more
poll_interval = 1.0      # seconds between queue checks while waiting
monitor_interval = 5.0   # seconds between memory samples while running

[capacity]               # what jobs may reserve in total
mem = "51.2G"            # 80% of physical memory
cpu = 10                 # number of cores
gpu = 1
disk = "100G"

[overcommit]             # reservations may exceed capacity by this factor
mem = 1
cpu = 2
gpu = 2
disk = 1
```

## Claude Code skill

`skills/alloc/SKILL.md` teaches Claude Code agents when and how to use alloc: sizing
requests, running long jobs in the background, reading warnings, and deciding whether to
stop an overrunning job. Install it for every project by linking it into your personal
skills folder (edits to it here then take effect everywhere):

```
ln -s /Users/joel/proj/resource-allocator/skills/alloc ~/.claude/skills/alloc
```

## Sandboxed agents

Claude Code's sandbox only lets commands write inside the project and temp folders, so
allow `~/.alloc` in `~/.claude/settings.json`:

```json
"sandbox": {
  "enabled": true,
  "filesystem": { "allowWrite": ["~/.alloc"] }
}
```

`alloc cancel` works across sandboxed commands. The sandbox won't let one command signal
a process that another command started, so instead of signalling the job directly,
`alloc cancel` flags it in the database and the job's own `alloc run` process sends the
signal (within `monitor_interval`, 5 seconds by default).

## Instructions for other agents

For agents that don't load Claude Code skills, paste something like this into their
instructions:

> Before running anything that needs significant memory, CPU or GPU on this machine,
> wrap it in `alloc run`:
>
>     alloc run --who <your-id> --mem 10G --gpu 100% --time 30m -- python train.py
>
> - `--mem` must cover everything the job uses, including GPU tensors (the GPU shares
>   system memory). `--time` is your best estimate of the run time; it's used for
>   scheduling estimates, and nothing is killed when it runs out.
> - If the resources are busy, `alloc` exits with status 75 and prints an estimated start
>   time. Do other work and retry, or add `--wait` to queue.
> - For runs longer than your shell timeout, run it in the background and check on it:
>
>       alloc run --wait --who <your-id> ... -- python train.py > train.log 2>&1 &
>       alloc status --who <your-id>
>
> - Watch for `[alloc] warning:` lines in the output and the warnings in `alloc status`.
>   If your job is well over its memory request or time limit, decide whether to let it
>   finish or stop it with `alloc cancel <job-id>`, and request more next time.

## Development

```
uv run pytest
```
