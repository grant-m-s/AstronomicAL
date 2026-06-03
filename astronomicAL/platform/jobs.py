# astronomicAL/platform/jobs.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Dict
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import uuid
import traceback

import time

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
        print(f"[AL_DEBUG][JobManager][{label}] <print failed>", flush=True)


def _call_on_ui_thread(fn: Callable[[], None]) -> None:
    """
    Best-effort UI thread marshalling for Panel/Bokeh.

    If Panel is available and we have a current document, schedule on next tick.
    Otherwise, run immediately in the current thread.
    """
    try:
        import panel as pn  # local import to avoid hard dependency during non-UI tests
        doc = pn.state.curdoc
    except Exception:
        doc = None

    if doc is None:
        fn()
    else:
        try:
            doc.add_next_tick_callback(fn)
        except Exception:
            # As a last resort, just run it (better than dropping callbacks)
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

    def cancel(self) -> bool:
        """
        Best-effort cancel:
        - Cancels if not started (future.cancel()).
        - Always sets cancellation token for cooperative checks.
        """
        self.token.cancel()
        return self.future.cancel()


class JobManager:
    """
    Central job runner.

    - Uses a shared ThreadPoolExecutor by default.
    - Supports dedupe via `key`: if key already in-flight, returns same job handle.
    - Ensures on_done/on_error callbacks are invoked on the Panel/Bokeh UI thread when available.
    """

    def __init__(self, max_workers: int = 16) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._inflight: Dict[str, JobHandle] = {}

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
        """
        Submit work to threadpool.

        If `key` is provided and a job with that key is running, re-use it.
        In that case, any provided on_done/on_error will be *added* as extra callbacks
        to the existing job's future (so late-joiners still get notified).

        kwargs are passed into fn, plus a reserved kwarg `cancel_token`.
        """

        job_debug(
            "submit ENTER",
            title=title,
            key=key,
            inflight_keys=list(self._inflight.keys()),
        )

        with self._lock:
            if key and key in self._inflight:
                handle = self._inflight[key]

                # Late join: attach additional callbacks for this submitter (if any)
                if on_done or on_error:
                    def _late_join_callback(_f: Future) -> None:
                        try:
                            res = _f.result()
                            if on_done:
                                _call_on_ui_thread(lambda res=res: on_done(res))
                        except BaseException as e:
                            if on_error:
                                _call_on_ui_thread(lambda e=e: on_error(e))
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

            def _runner() -> Any:
                started = time.time()
                job_debug("runner START", title=title, key=key)
                try:
                    result = fn(cancel_token=token, **kwargs)
                    job_debug(
                        "runner DONE",
                        title=title,
                        key=key,
                        elapsed=round(time.time() - started, 3),
                    )
                    return result
                except BaseException as exc:
                    job_debug(
                        "runner ERROR",
                        title=title,
                        key=key,
                        elapsed=round(time.time() - started, 3),
                        error=repr(exc),
                    )
                    raise

            fut = self._executor.submit(_runner)
            handle = JobHandle(job_id=job_id, title=title, future=fut, token=token)

            if key:
                self._inflight[key] = handle

            # One canonical completion callback handles cleanup + the first submitter's callbacks.
            def _cleanup_callback(_f: Future) -> None:

                job_debug(
                    "cleanup_callback ENTER",
                    title=title,
                    key=key,
                    job_id=job_id,
                    cancelled=_f.cancelled(),
                )
                # Remove inflight on completion
                if key:
                    with self._lock:
                        self._inflight.pop(key, None)

                try:
                    res = _f.result()
                    if on_done:
                        _call_on_ui_thread(lambda res=res: on_done(res))
                except BaseException as e:
                    if on_error:
                        _call_on_ui_thread(lambda e=e: on_error(e))
                    else:
                        traceback.print_exc()

            fut.add_done_callback(_cleanup_callback)
            return handle

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            self._inflight.clear()
        self._executor.shutdown(wait=wait)