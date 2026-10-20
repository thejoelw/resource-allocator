"""Scheduling policy: FIFO first-fit admission, and start-time estimates."""

from dataclasses import dataclass

RESOURCES = ("mem", "cpu", "gpu", "disk")


@dataclass(frozen=True)
class Res:
    """An amount of each resource: memory and disk in bytes, CPU in cores, GPU in GPUs."""

    mem: float = 0
    cpu: float = 0
    gpu: float = 0
    disk: float = 0

    def __iter__(self):
        return iter((self.mem, self.cpu, self.gpu, self.disk))

    def __add__(self, other: "Res") -> "Res":
        return Res(*(a + b for a, b in zip(self, other)))

    def fits(self, limit: "Res") -> bool:
        # The tolerance keeps float sums like 0.1 + 0.2 from failing an exact fit.
        return all(a <= b * (1 + 1e-9) + 1e-9 for a, b in zip(self, limit))


def first_fit(used: Res, limit: Res, queue: list[tuple[object, Res]]) -> tuple[list, Res]:
    """Walk the queue in order, admitting every job that fits in what's left.

    `queue` holds (key, request) pairs. Returns the admitted keys and the new amount in use.
    """
    admitted = []
    for key, req in queue:
        if (used + req).fits(limit):
            used = used + req
            admitted.append(key)
    return admitted, used


def predict_starts(
    now: float,
    limit: Res,
    running: list[tuple[Res, float]],
    queue: list[tuple[Res, float]],
) -> list[float | None]:
    """Predict when each queued job will start, assuming every job runs exactly its time limit.

    `running` holds (request, expected end) pairs; jobs already past their end are assumed
    to finish now. `queue` holds (request, time limit) pairs in FIFO order. Returns a start
    time for each queued job, or None for one that can never fit.
    """
    active = [(max(end, now), req) for req, end in running]
    waiting = list(range(len(queue)))
    starts: list[float | None] = [None] * len(queue)
    t = now
    while waiting:
        used = sum((req for _, req in active), Res())
        admitted, _ = first_fit(used, limit, [(i, queue[i][0]) for i in waiting])
        for i in admitted:
            starts[i] = t
            active.append((t + queue[i][1], queue[i][0]))
        waiting = [i for i in waiting if i not in admitted]
        if not active:
            break  # nothing left to finish, so whatever is still waiting can never fit
        t = min(end for end, _ in active)
        active = [(end, req) for end, req in active if end > t]
    return starts
