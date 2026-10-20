---
name: alloc
description: Reserve memory, CPU, GPU and disk with the `alloc` CLI before running ML training or any other heavy job on this shared machine, so agents don't overload it or starve each other. Use this whenever you're about to launch a training or fine-tuning run, an evaluation, a hyperparameter sweep, a benchmark, data preprocessing, or anything else that needs several GB of memory, the GPU (PyTorch MPS, MLX), several CPU cores for more than a minute, or lots of disk, even if the user doesn't mention alloc or resources. Also use it when checking on, sizing, or stopping such jobs.
---

# Running heavy jobs with alloc

Several agents share this machine. `alloc` keeps them from overcommitting it: you declare
what your job needs, alloc starts it when that fits, and it releases the reservation
when the job exits, however it exits. This only works if everyone uses it. A heavy job
started outside alloc is invisible to it, so alloc will admit other jobs into memory your
job is already using and the machine starts swapping, which slows everyone down.

Use alloc for anything that needs more than ~2 GB of memory, the GPU, several cores for
more than a minute, or more than a few GB of disk. Quick things (installing packages, unit
tests, small scripts, inspecting data) don't need it.

## Quick start

```bash
alloc run --who <your-id> --mem 12G --gpu 100% --cpu 4 --time 30m -- python train.py --epochs 3
```

- **`--who`** identifies you for usage tracking. Use the ID your instructions give you;
  if there isn't one, use `claude-<project-directory-name>`. Keep it the same across runs.
- **The command** goes after `--`. alloc runs it directly rather than through a shell, so
  wrap shell syntax inside the job: `-- sh -c 'cd exp && python train.py'`. Redirecting
  alloc's own output (`alloc run ... > train.log 2>&1`) is fine.
- **If the resources are free**, the command starts right away and alloc exits with the
  command's exit status.
- **If they aren't**, alloc prints an estimated start time and exits with status **75**
  without running anything. See "When the machine is busy" below.

`alloc status` shows what's running and queued and how much the machine offers jobs.

## Sizing a request

Ask for what the job will actually use. Asking for too much keeps other agents (and your
own next job) waiting. Asking for too little lets alloc admit other jobs into memory
you're using.

| Flag | Takes | Guidance |
|---|---|---|
| `--mem` | size, e.g. `12G` | Peak memory of the whole job. On Apple Silicon the GPU shares system memory, so include everything on MPS (weights, gradients, optimizer state, activations) as well as CPU memory and dataloader workers. Rough guide for fp32 training with Adam: ~16 bytes per parameter, plus activations, plus ~2G of Python/PyTorch overhead. |
| `--gpu` | share, e.g. `100%`, `50%` | `100%` for a run that keeps the GPU busy, less for light inference. |
| `--cpu` | cores (`4`) or share of the machine (`50%`) | The main process plus dataloader workers. For CPU-bound work, also cap threads (`torch.set_num_threads(n)` or `OMP_NUM_THREADS=n`) so the job stays near what you asked for. |
| `--disk` | size, e.g. `20G` | What the job will write: checkpoints × how many you keep, downloaded datasets, logs. |
| `--time` | `15m`, `1h30m`, `90s` | Your honest estimate. It's only used to estimate start times for queued jobs; nothing is killed when it runs out. |

Resources you leave out default to 0.

**The best way to size memory is to measure it.** Run a short version first (a few steps,
same model and batch size) with a generous request. alloc's last line reports the peak:

```
[alloc] job 12 finished: exit 0 after 1m4s (time limit 5m), peak memory 9.3G (requested 16G)
```

Request that peak plus ~20% for the full run, plus more if the full run does things the
short one didn't (evaluation passes, longer sequences). `alloc status --who <your-id>`
lists your recent jobs with their peak memory, which helps when rerunning something similar.

## When the machine is busy

```
[alloc] not available now. Estimated start in ~25m, 1 job queued ahead. Try again later, or pass --wait to join the queue.
```

- **`--wait`** joins the queue and blocks until the job fits. This is usually the right
  choice if the job is what you're waiting on anyway.
- **Do other useful work** and retry later. Don't retry in a tight loop.
- **Smaller jobs can start sooner.** When resources free up, alloc goes through the queue
  in order and starts every job that fits, so a job that fits in the free space can run
  ahead of a bigger one. Don't shrink a request below what the job needs to take
  advantage of this, though.

Estimates assume every running job runs its full `--time`, so treat them as rough.

## Long runs: start them in the background

Run anything that might take more than a couple of minutes as a background command, with
output going to a log. That way you aren't blocked, and a timeout or interruption of your
shell command doesn't take the job down with it. In Claude Code, use the Bash tool with
`run_in_background: true`:

```bash
alloc run --wait --who <your-id> --mem 12G --gpu 100% --time 2h -- python train.py > train.log 2>&1
```

Elsewhere, `nohup alloc run ... > train.log 2>&1 &` works too. The job ID is on the
`[alloc] job N started` line in the log, or find it with `alloc status --who <your-id>`.
The job's environment has `ALLOC_JOB_ID` set, which is handy for naming logs and
checkpoints.

## Keeping an eye on your job

alloc writes `[alloc]` lines to the job's stderr, so they end up in your log:

```
[alloc] job 12 started (mem 12G, cpu 4, gpu 1, time 2h). Check on it with: alloc status 12
[alloc] warning: job 12 has used 15G of memory, 25% more than its 12G request (2 jobs waiting in the queue)
[alloc] warning: job 12 has run for 2h24m, past its 2h time limit (no jobs waiting)
[alloc] job 12 finished: exit 0 after 2h31m (time limit 2h), peak memory 15.2G (requested 12G)
```

Memory warnings start at 20% over the request and repeat each time usage grows another
20%. Time warnings repeat as the overrun grows. When you check a log, grep for
`[alloc] warning`.

`alloc status 12` shows the same information on demand: state, elapsed time vs. limit,
current and peak memory, warnings, and exit status. Add `--json` to parse it.

## When your job overruns

alloc never stops a job for exceeding its request. Deciding what to do is your job, and
it's part of sharing the machine well. Weigh:

- **How far over, and is it still growing?** Memory 25% over and flat means you sized it
  wrong. Memory climbing steadily may be a leak and will only get worse.
- **Is anyone waiting?** Each warning says how many jobs are queued. Overrunning when
  nobody is waiting costs nobody anything.
- **How close is it to done?** A run that's 90% through should usually finish, because
  stopping it throws the work away. A run that's barely started and already far over is
  better stopped, fixed, and resubmitted with a better request.

To stop a job, run `alloc cancel <job-id>`. It sends SIGTERM, so a training script that
saves a checkpoint on SIGTERM loses little work; that's worth adding to long-running
scripts. `--force` sends SIGKILL to the whole process tree instead. Either way, request
what you actually saw next time.

## Other commands

- `alloc cancel <job-id>` also removes a queued job. Cancel queued jobs you no longer need
  so they don't hold a place in line.
- `alloc usage --since 1d` shows resource usage per agent.
- `alloc <command> --help` has the details.

## Exit statuses of `alloc run`

| Status | Meaning |
|---|---|
| the command's own | The command ran. 128+N means it was killed by signal N (143 = SIGTERM). |
| 75 | The resources are busy and you didn't pass `--wait`. Nothing ran. |
| 2 | Bad arguments, or a request bigger than the machine offers jobs. |
| 127 | The command couldn't be started (e.g. not found). |
| 1 | A queued job was cancelled before it started. |

## If alloc itself fails

- **`alloc: command not found`**: it's installed at `~/.local/bin/alloc`. If it isn't
  there, tell the user.
- **Errors mentioning `~/.alloc`** (such as "unable to open database file" or "Operation
  not permitted") usually mean a sandbox is blocking writes to `~/.alloc`. Tell the user;
  the fix is in `/Users/joel/proj/resource-allocator/README.md`. Don't work around it by
  running the job without alloc.
