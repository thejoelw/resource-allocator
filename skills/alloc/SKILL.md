---
name: alloc
description: Reserve memory, CPU, GPU and disk with the `alloc` CLI before running heavy jobs on this shared machine, so agents don't overload it or starve each other. Use it before any training or fine-tuning run, evaluation, sweep, benchmark, data preprocessing, or other job that needs several GB of memory, the GPU (PyTorch MPS, MLX), several cores for over a minute, or lots of disk, even if the user doesn't mention alloc. Also use it to check on, size, or stop such jobs.
---

# alloc: reserve resources for heavy jobs

Several agents share this machine. You declare what your job needs; alloc starts it when it
fits and releases the reservation when it exits. A heavy job run outside alloc is invisible
to it, so other jobs get admitted on top of yours and the machine swaps.

Use it for anything needing more than ~2 GB of memory, the GPU, several cores for over a
minute, or more than a few GB of disk. Skip it for package installs, unit tests and small
scripts.

## Run

```bash
alloc run --who <your-id> --mem 12G --gpu 100% --cpu 4 --time 30m -- python train.py
```

- `--who`: the ID your instructions give you, else `claude-<project-directory-name>`. Keep
  it the same across runs.
- The command after `--` runs without a shell, so wrap shell syntax:
  `-- sh -c 'cd exp && python train.py'`.
- If resources are free, it runs now and alloc exits with the command's status. If not,
  alloc prints an estimated start and exits **75** without running anything. Add `--wait`
  to queue instead.
- For anything over a couple of minutes, use the Bash tool with `run_in_background: true`
  and send output to a log (`> train.log 2>&1`). The job's environment has `ALLOC_JOB_ID`.

`alloc status` shows running and queued jobs and what the machine offers.

## Sizing

Ask for what the job will actually use: too much keeps others waiting, too little lets
alloc overcommit the machine. Resources you leave out default to 0.

| Flag | Example | Notes |
|---|---|---|
| `--mem` | `12G` | Peak of the whole job. On Apple Silicon the GPU shares system memory, so include weights, gradients, optimizer state, activations and dataloader workers. fp32 + Adam ≈ 16 bytes/param + activations + ~2G overhead. |
| `--gpu` | `100%`, `50%` | `100%` if it keeps the GPU busy, less for light inference. |
| `--cpu` | `4` or `50%` | Main process plus workers. Cap threads (`OMP_NUM_THREADS`, `torch.set_num_threads`) to match. |
| `--disk` | `20G` | Checkpoints kept, downloaded datasets, logs. |
| `--time` | `90s`, `1h30m` | Honest estimate. Only used for queue estimates; never enforced. |

Measure memory rather than guess: run a few steps with a generous request, and alloc's last
line reports the peak.

```
[alloc] job 12 finished: exit 0 after 1m4s (time limit 5m), peak memory 9.3G (requested 16G)
```

Request that peak plus ~20%, and more if the full run does things the short one didn't
(evaluation passes, longer sequences). `alloc status --who <your-id>` lists your recent
jobs' peaks.

## When the machine is busy

- Use `--wait` if you're blocked on the job anyway. Otherwise do other work and retry
  later, not in a tight loop.
- The queue starts every job that fits, in order, so a smaller job can start sooner. Don't
  under-request to take advantage of this.
- Start estimates assume every job runs its full `--time`, so they're rough.

## Monitoring and overruns

alloc writes `[alloc]` lines to the job's stderr. Grep your log for `[alloc] warning`:

```
[alloc] warning: job 12 has used 15G of memory, 25% more than its 12G request (2 jobs waiting in the queue)
[alloc] warning: job 12 has run for 2h24m, past its 2h time limit (no jobs waiting)
```

`alloc status <job-id>` (add `--json` to parse) shows state, elapsed time vs. limit, current
and peak memory, and warnings.

alloc never stops a job for overrunning; you decide. Weigh:

- **Trend.** Flat memory means you mis-sized it; steadily climbing may be a leak.
- **Who's waiting.** Each warning says how many jobs are queued. With nobody waiting, an
  overrun costs nothing.
- **Progress.** A run that's 90% done should usually finish. One that's barely started and
  far over is better stopped, fixed and resubmitted.

`alloc cancel <job-id>` sends SIGTERM (so long scripts should checkpoint on SIGTERM);
`--force` SIGKILLs the whole process tree. It also removes queued jobs; cancel ones you no
longer need.

## Reference

- `alloc usage --since 1d` shows usage per agent; `alloc <command> --help` has details.
- Exit statuses of `alloc run`: the command's own (128+N means killed by signal N);
  **75** busy and nothing ran; **2** bad arguments or a request bigger than the machine
  offers; **127** command couldn't start; **1** queued job cancelled before it started.
