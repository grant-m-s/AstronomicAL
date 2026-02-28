# astronomicAL/platform/jobs.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Dict
from concurrent.futures import Future, ThreadPoolExecutor
import threading
import uuid
import traceback


DoneCallback = Callable[[Any], None]
ErrorCallback = Callable[[BaseException], None]


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

    - Uses a shared ThreadPoolExecutor by default (good for I/O).
    - Supports dedupe via `key`: if key already in-flight, returns same job handle.
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
        kwargs are passed into fn, plus a reserved kwarg `cancel_token`.
        """
        with self._lock:
            if key and key in self._inflight:
                return self._inflight[key]

            job_id = uuid.uuid4().hex
            token = CancellationToken()

            def _runner() -> Any:
                return fn(cancel_token=token, **kwargs)

            fut = self._executor.submit(_runner)
            handle = JobHandle(job_id=job_id, title=title, future=fut, token=token)

            if key:
                self._inflight[key] = handle

            def _cleanup_callback(_f: Future) -> None:
                # Remove inflight on completion
                if key:
                    with self._lock:
                        self._inflight.pop(key, None)

                try:
                    res = _f.result()
                    if on_done:
                        on_done(res)
                except BaseException as e:
                    if on_error:
                        on_error(e)
                    else:
                        traceback.print_exc()

            fut.add_done_callback(_cleanup_callback)
            return handle

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            self._inflight.clear()
        self._executor.shutdown(wait=wait)