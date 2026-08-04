from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, List, Optional
import threading
import time
import traceback
import uuid


DoneCallback = Callable[[Any], None]
ErrorCallback = Callable[[BaseException], None]


def job_debug(label: str, **values: Any) -> None:
    try:
        parts = " ".join(f"{key}={value!r}" for key, value in values.items())
        print(
            f"[AL_DEBUG][JobManager][{label}] "
            f"thread={threading.current_thread().name} {parts}",
            flush=True,
        )
    except Exception:
        print(f"[AL_DEBUG][JobManager][{label}] ", flush=True)


def _current_document() -> Any | None:
    """Return the current Panel/Bokeh document when one is available."""
    try:
        import panel as pn  # local import to avoid hard dependency during tests

        return pn.state.curdoc
    except Exception:
        return None


def _call_on_ui_thread(
    fn: Callable[[], None],
    *,
    document: Any | None = None,
) -> None:
    """Best-effort UI-thread marshalling for Panel/Bokeh.

    ``Future`` completion callbacks execute on executor threads. Looking up
    ``pn.state.curdoc`` from those threads usually returns ``None``, even when
    the job was submitted from a live Panel session. Callers should therefore
    pass the session document captured at submission time.

    When no document is available, the callback runs immediately. This keeps
    the job manager usable in tests and non-Panel contexts.
    """
    target_document = document if document is not None else _current_document()
    if target_document is None:
        fn()
        return

    try:
        target_document.add_next_tick_callback(fn)
    except Exception:
        fn()


class CancellationToken:
    """Cooperative cancellation token."""

    def __init__(self) -> None:
        self._evt = threading.Event()

    def cancel(self) -> None:
        self._evt.set()

    def cancelled(self) -> bool:
        return self._evt.is_set()


@dataclass
class JobHandle:
    job_id: str
    title: str
    future: Future
    token: CancellationToken
    key: Optional[str] = None
    submitted_at: float = 0.0

    def cancel(self) -> bool:
        """Best-effort cancellation.

        Cancels the future when it has not started and always sets the
        cooperative cancellation token.
        """
        self.token.cancel()
        return self.future.cancel()


@dataclass(frozen=True)
class JobSnapshot:
    job_id: str
    title: str
    key: Optional[str]
    status: str
    submitted_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    elapsed: float
    cancelled: bool
    done: bool
    error: Optional[str] = None


@dataclass
class _JobState:
    job_id: str
    title: str
    key: Optional[str]
    submitted_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    status: str = "queued"
    cancelled: bool = False
    error: Optional[str] = None


class JobManager:
    """Central background-job runner.

    - Uses a shared ``ThreadPoolExecutor`` by default.
    - Supports deduplication through a job key.
    - Tracks keyed and unkeyed active jobs.
    - Stores recent completed jobs for runtime diagnostics.
    - Marshals ``on_done`` and ``on_error`` to the Panel/Bokeh session document
      captured when each callback was registered.
    """

    def __init__(
        self,
        max_workers: int = 16,
        *,
        history_limit: int = 500,
    ) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()

        self._inflight: Dict[str, JobHandle] = {}
        self._active_by_id: Dict[str, JobHandle] = {}
        self._states: Dict[str, _JobState] = {}
        self._history: Deque[JobSnapshot] = deque(maxlen=history_limit)

    def _snapshot_from_state(
        self,
        state: _JobState,
        *,
        now: Optional[float] = None,
    ) -> JobSnapshot:
        if now is None:
            now = time.time()

        end = state.finished_at or now
        start = state.started_at or state.submitted_at
        elapsed = max(0.0, end - start)
        return JobSnapshot(
            job_id=state.job_id,
            title=state.title,
            key=state.key,
            status=state.status,
            submitted_at=state.submitted_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
            elapsed=elapsed,
            cancelled=state.cancelled,
            done=state.finished_at is not None,
            error=state.error,
        )

    def active_jobs(self) -> List[JobSnapshot]:
        """Return snapshots for all currently active jobs."""
        with self._lock:
            snapshots = [
                self._snapshot_from_state(state)
                for state in self._states.values()
                if state.finished_at is None
            ]

        snapshots.sort(key=lambda item: item.submitted_at)
        return snapshots

    def recent_jobs(self, n: int = 50) -> List[JobSnapshot]:
        """Return recently completed job snapshots."""
        with self._lock:
            if n <= 0:
                return []
            return list(self._history)[-n:]

    def submit(
        self,
        fn: Callable[..., Any],
        *,
        title: str = "Job",
        key: Optional[str] = None,
        on_done: Optional[DoneCallback] = None,
        on_error: Optional[ErrorCallback] = None,
        **kwargs: Any,
    ) -> JobHandle:
        """Submit work to the thread pool.

        If ``key`` identifies an active job, that job is reused. Any callbacks
        supplied by the late joiner are attached to the existing future and
        retain the late joiner's own session document.

        ``kwargs`` are passed to ``fn`` together with the reserved
        ``cancel_token`` keyword argument.
        """
        callback_document = _current_document()

        with self._lock:
            active_keys = list(self._inflight.keys())
        job_debug(
            "submit ENTER",
            title=title,
            key=key,
            inflight_keys=active_keys,
        )

        with self._lock:
            if key and key in self._inflight:
                handle = self._inflight[key]

                if handle.future.done():
                    self._inflight.pop(key, None)
                else:
                    if on_done or on_error:

                        def _late_join_callback(_f: Future) -> None:
                            try:
                                result = _f.result()
                                if on_done:
                                    _call_on_ui_thread(
                                        lambda result=result: on_done(result),
                                        document=callback_document,
                                    )
                            except BaseException as exc:
                                if on_error:
                                    _call_on_ui_thread(
                                        lambda exc=exc: on_error(exc),
                                        document=callback_document,
                                    )
                                else:
                                    traceback.print_exc()

                        handle.future.add_done_callback(_late_join_callback)

                    job_debug(
                        "submit DEDUPED",
                        title=title,
                        key=key,
                        existing_job_id=getattr(handle, "job_id", None),
                    )
                    return handle

            job_id = uuid.uuid4().hex
            token = CancellationToken()
            submitted_at = time.time()
            state = _JobState(
                job_id=job_id,
                title=title,
                key=key,
                submitted_at=submitted_at,
            )

        def _runner() -> Any:
            started = time.time()

            with self._lock:
                state.started_at = started
                state.status = "running"
            job_debug("runner START", title=title, key=key, job_id=job_id)

            try:
                result = fn(cancel_token=token, **kwargs)
                job_debug(
                    "runner DONE",
                    title=title,
                    key=key,
                    job_id=job_id,
                    elapsed=round(time.time() - started, 3),
                )
                return result
            except BaseException as exc:
                with self._lock:
                    state.status = "error"
                    state.error = repr(exc)

                job_debug(
                    "runner ERROR",
                    title=title,
                    key=key,
                    job_id=job_id,
                    elapsed=round(time.time() - started, 3),
                    error=repr(exc),
                )
                raise

        future = self._executor.submit(_runner)
        handle = JobHandle(
            job_id=job_id,
            title=title,
            future=future,
            token=token,
            key=key,
            submitted_at=submitted_at,
        )

        with self._lock:
            self._active_by_id[job_id] = handle
            self._states[job_id] = state
            if key:
                self._inflight[key] = handle

        def _cleanup_callback(_f: Future) -> None:
            job_debug(
                "cleanup_callback ENTER",
                title=title,
                key=key,
                job_id=job_id,
                cancelled=_f.cancelled(),
            )

            finished_at = time.time()
            with self._lock:
                if _f.cancelled():
                    state.status = "cancelled"
                    state.cancelled = True
                    state.error = None
                else:
                    try:
                        exception = _f.exception()
                    except CancelledError:
                        exception = None
                        state.status = "cancelled"
                        state.cancelled = True

                    if state.status != "cancelled":
                        if exception is None:
                            if state.status != "error":
                                state.status = "finished"
                        else:
                            state.status = "error"
                            state.error = repr(exception)

                state.finished_at = finished_at
                self._active_by_id.pop(job_id, None)
                self._states.pop(job_id, None)

                if key:
                    current = self._inflight.get(key)
                    if current is not None and current.job_id == job_id:
                        self._inflight.pop(key, None)

                self._history.append(
                    self._snapshot_from_state(state, now=finished_at)
                )

            try:
                result = _f.result()
                if on_done:
                    _call_on_ui_thread(
                        lambda result=result: on_done(result),
                        document=callback_document,
                    )
            except BaseException as exc:
                if on_error:
                    _call_on_ui_thread(
                        lambda exc=exc: on_error(exc),
                        document=callback_document,
                    )
                else:
                    traceback.print_exc()

        future.add_done_callback(_cleanup_callback)
        return handle

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            for handle in list(self._active_by_id.values()):
                try:
                    handle.cancel()
                except Exception:
                    pass

            self._inflight.clear()
            self._active_by_id.clear()
            self._states.clear()

        self._executor.shutdown(wait=wait)