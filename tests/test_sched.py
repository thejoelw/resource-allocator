from alloc.sched import Res, first_fit, predict_starts


def test_first_fit_admits_in_order_and_skips_what_doesnt_fit():
    queue = [("big", Res(mem=8)), ("small", Res(mem=2)), ("medium", Res(mem=3)), ("last", Res(mem=1))]
    admitted, used = first_fit(Res(mem=5), Res(mem=10), queue)
    assert admitted == ["small", "medium"]
    assert used == Res(mem=10)


def test_first_fit_checks_every_resource():
    limit = Res(mem=10, cpu=4, gpu=1)
    queue = [("gpu-hog", Res(mem=1, gpu=1)), ("cpu-hog", Res(mem=1, cpu=4))]
    admitted, _ = first_fit(Res(gpu=0.5), limit, queue)
    assert admitted == ["cpu-hog"]


def test_fits_tolerates_float_rounding():
    assert (Res(cpu=0.1) + Res(cpu=0.2)).fits(Res(cpu=0.3))


def test_predict_starts():
    limit = Res(mem=10)
    running = [(Res(mem=6), 100.0)]
    queue = [(Res(mem=8), 50.0), (Res(mem=4), 30.0), (Res(mem=4), 10.0)]
    # The 8 waits for the running job to end at 100. The first 4 fits beside it now;
    # the second 4 starts when the first one ends at 30.
    assert predict_starts(0.0, limit, running, queue) == [100.0, 0.0, 30.0]


def test_predict_starts_assumes_overdue_jobs_end_now():
    starts = predict_starts(50.0, Res(mem=10), [(Res(mem=10), 20.0)], [(Res(mem=5), 10.0)])
    assert starts == [50.0]


def test_predict_starts_for_a_job_that_never_fits():
    starts = predict_starts(0.0, Res(mem=10), [], [(Res(mem=11), 10.0), (Res(mem=1), 10.0)])
    assert starts == [None, 0.0]
