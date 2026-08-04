from __future__ import annotations

import threading
import time

import pytest

from astronomicAL.platform.jobs import JobManager


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_job_runs_and_calls_on_done(jobs: JobManager) -> None:
    done = threading.Event()
    results = []

    def work(*, cancel_token, value):
        assert cancel_token is not None
        return value * 2

    def on_done(result):
        results.append(result)
        done.set()

    jobs.submit(work, title="double", on_done=on_done, value=21)

    assert done.wait(2)
    assert results == [42]


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_job_error_calls_on_error(jobs: JobManager) -> None:
    done = threading.Event()
    errors = []

    def work(*, cancel_token):
        raise RuntimeError("intentional failure")

    def on_error(error):
        errors.append(error)
        done.set()

    jobs.submit(work, title="broken", on_error=on_error)

    assert done.wait(2)
    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "intentional failure"


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_duplicate_job_key_reuses_inflight_job_and_adds_callback(jobs: JobManager) -> None:
    first_done = threading.Event()
    second_done = threading.Event()
    results = []
    calls = []

    def work(*, cancel_token):
        calls.append("work-started")
        time.sleep(0.05)
        return "shared-result"

    handle_a = jobs.submit(
        work,
        title="shared",
        key="same-key",
        on_done=lambda result: (results.append(("a", result)), first_done.set()),
    )
    handle_b = jobs.submit(
        work,
        title="shared",
        key="same-key",
        on_done=lambda result: (results.append(("b", result)), second_done.set()),
    )

    assert handle_a.job_id == handle_b.job_id

    assert first_done.wait(2)
    assert second_done.wait(2)

    assert calls == ["work-started"]
    assert sorted(results) == [
        ("a", "shared-result"),
        ("b", "shared-result"),
    ]


@pytest.mark.unit
@pytest.mark.timeout(5)
def test_job_cancel_sets_cooperative_cancel_token(jobs: JobManager) -> None:
    done = threading.Event()
    results = []

    def work(*, cancel_token):
        while not cancel_token.cancelled():
            time.sleep(0.01)
        return "cancelled"

    handle = jobs.submit(
        work,
        title="cancellable",
        on_done=lambda result: (results.append(result), done.set()),
    )

    handle.cancel()

    assert done.wait(2)
    assert results == ["cancelled"]